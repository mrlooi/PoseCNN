import _init_paths
# from fcn.test import test_net_images
from fcn.config import cfg, cfg_from_file
from datasets.factory import get_imdb

from utils.blob import pad_im, unpad_im # im_list_to_blob

from transforms3d.quaternions import quat2mat, mat2quat
import time, os, sys
import tensorflow as tf
import os.path as osp
import numpy as np
import cv2

def load_cfg(args):
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id
    device_name = '/gpu:{:d}'.format(args.gpu_id)
    print(device_name)

    cfg.TRAIN.NUM_STEPS = 1
    cfg.TRAIN.GRID_SIZE = cfg.TEST.GRID_SIZE
    cfg.TRAIN.TRAINABLE = False

    cfg.RIG = args.rig_name
    cfg.CAD = args.cad_name
    cfg.POSE = args.pose_name
    cfg.BACKGROUND = args.background_name
    cfg.IS_TRAIN = False

def get_lov2d_args():
    class Args():
        pass

    args = Args()
    args.gpu_id = 0
    args.weights = None # "data/imagenet_models/vgg16.npy"
    args.model = "data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt"
    args.cfg_file = "experiments/cfgs/lov_color_2d.yml"
    args.wait = True
    args.imdb_name = "lov_keyframe"
    args.network_name = "vgg16_convs"
    args.rig_name = "data/LOV/camera.json"
    args.cad_name = "data/LOV/models.txt"
    args.kfusion = False
    args.pose_name = "data/LOV/poses.txt"
    args.background_name = "data/cache/backgrounds.pkl"

    return args

def _get_image_blob(im):

    assert len(cfg.TEST.SCALES_BASE) == 1
    
    im_orig = im.astype(np.float32, copy=True)
    # mask the color image according to depth
    if cfg.EXP_DIR == 'rgbd_scene':
        I = np.where(im_depth == 0)
        im_orig[I[0], I[1], :] = 0

    processed_ims_rescale = []
    im_scale = cfg.TEST.SCALES_BASE[0]
    im_rescale = cv2.resize(im_orig / 127.5 - 1, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_ims_rescale.append(im_rescale)

    im_orig -= cfg.PIXEL_MEANS
    processed_ims = []
    im_scale_factors = []
    
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

    # Create a blob to hold the input images
    # blob = im_list_to_blob(processed_ims, 3)
    # blob_rescale = im_list_to_blob(processed_ims_rescale, 3)
    blob = np.array(processed_ims)
    blob_rescale = np.array(processed_ims_rescale)

    return blob, blob_rescale, np.array(im_scale_factors)

# extract vertmap for vertex predication
def _extract_vertmap(labels, vertex_pred, num_classes):
    height = labels.shape[0]
    width = labels.shape[1]
    vertmap = np.zeros((height, width, 3), dtype=np.float32)

    for i in xrange(1, num_classes):
        I = np.where(labels == i)
        if len(I[0]) > 0:
            start = 3 * i
            end = 3 * i + 3
            print(start)
            print(end)
            vertmap[I[0], I[1], :] = vertex_pred[I[0], I[1], start:end]
    vertmap[:, :, 2] = np.exp(vertmap[:, :, 2])
    return vertmap

def vis_segmentations_vertmaps_detection(im, im_depth, im_labels, colors, center_map, 
  labels, rois, poses, intrinsic_matrix, num_classes, classes, points):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(3, 3, 1)
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    plt.imshow(im)
    ax.set_title('input image')

    # show depth
    ax = fig.add_subplot(3, 3, 2)
    plt.imshow(im_depth)
    ax.set_title('input depth')

    # show class label
    ax = fig.add_subplot(3, 3, 3)
    plt.imshow(im_labels)
    ax.set_title('class labels')      

    if cfg.TEST.VERTEX_REG_2D:
        # show centers
        for i in xrange(rois.shape[0]):
            if rois[i, 1] == 0:
                continue
            cx = (rois[i, 2] + rois[i, 4]) / 2
            cy = (rois[i, 3] + rois[i, 5]) / 2
            w = rois[i, 4] - rois[i, 2]
            h = rois[i, 5] - rois[i, 3]
            if not np.isinf(cx) and not np.isinf(cy):
                plt.plot(cx, cy, 'yo')

                # show boxes
                plt.gca().add_patch(
                    plt.Rectangle((cx-w/2, cy-h/2), w, h, fill=False,
                                   edgecolor='g', linewidth=3))
        
    # show vertex map
    ax = fig.add_subplot(3, 3, 4)
    plt.imshow(center_map[:,:,0])
    ax.set_title('centers x')

    ax = fig.add_subplot(3, 3, 5)
    plt.imshow(center_map[:,:,1])
    ax.set_title('centers y')
    
    ax = fig.add_subplot(3, 3, 6)
    plt.imshow(center_map[:,:,2])
    ax.set_title('centers z')

    # show projection of the poses
    if cfg.TEST.POSE_REG:

        ax = fig.add_subplot(3, 3, 7, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        for i in xrange(rois.shape[0]):
            cls = int(rois[i, 1])
            if cls > 0:
                # extract 3D points
                x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                x3d[0, :] = points[cls,:,0]
                x3d[1, :] = points[cls,:,1]
                x3d[2, :] = points[cls,:,2]

                # projection
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(poses[i, :4])
                RT[:, 3] = poses[i, 4:7]
                print classes[cls]
                print RT
                print '\n'
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.5)
                # plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[cls], 255.0), s=10)

        ax.set_title('projection of model points')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

    plt.show()


if __name__ == '__main__':
    # import pprint

    args = get_lov2d_args()
    # print(args)
    load_cfg(args)
    # pprint.pprint(cfg)

    assert(cfg.INPUT == 'COLOR')

    weights_filename = os.path.splitext(os.path.basename(args.model))[0]
    imdb = get_imdb(args.imdb_name)

    # construct the filenames
    root = 'data/demo_images/'
    rgb_filenames = sorted([root + f for f in os.listdir(root) if f.endswith("color.png")])
    depth_filenames = sorted([root + f for f in os.listdir(root) if f.endswith("depth.png")])
    print(rgb_filenames)
    print(depth_filenames)

    # construct meta data
    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    meta_data = dict({'intrinsic_matrix': K, 'factor_depth': 10000.0})
    print (meta_data)

    from networks.factory import get_network
    network = get_network(args.network_name)
    net = network
    print ('Use network `{:s}` in training'.format(args.network_name))

    # start a session
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    saver.restore(sess, args.model)
    print ('Loading model weights from {:s}').format(args.model)

    # START
    idx = 0
    im = pad_im(cv2.imread(rgb_filenames[idx], cv2.IMREAD_UNCHANGED), 16)

    im_blob, im_rescale_blob, im_scale_factors = _get_image_blob(im)
    im_scale = im_scale_factors[0]

    extents = imdb._extents
    points = imdb._points_all
    symmetry = imdb._symmetry
    num_classes = imdb.num_classes

    K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
    K[2, 2] = 1
    Kinv = np.linalg.pinv(K)
    mdata = np.zeros(48, dtype=np.float32)
    mdata[0:9] = K.flatten()
    mdata[9:18] = Kinv.flatten()
    # mdata[18:30] = pose_world2live.flatten()
    # mdata[30:42] = pose_live2world.flatten()
    # mdata[42] = voxelizer.step_x
    # mdata[43] = voxelizer.step_y
    # mdata[44] = voxelizer.step_z
    # mdata[45] = voxelizer.min_x
    # mdata[46] = voxelizer.min_y
    # mdata[47] = voxelizer.min_z
    if cfg.FLIP_X:
        mdata[0] = -1 * mdata[0]
        mdata[9] = -1 * mdata[9]
        mdata[11] = -1 * mdata[11]
    print(meta_data)

    meta_data_blob = np.zeros((1, 1, 1, 48), dtype=np.float32)
    meta_data_blob[0,0,0,:] = mdata

    height = int(im.shape[0] * im_scale)
    width = int(im.shape[1] * im_scale)
    label_blob = np.ones((1, height, width), dtype=np.int32)
    vertex_target_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)
    vertex_weight_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)
    pose_blob = np.zeros((1, 13), dtype=np.float32)


    feed_dict = {net.data: im_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                 net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                 net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.symmetry: symmetry, net.poses: pose_blob}

    sess.run(net.enqueue_op, feed_dict=feed_dict)

    labels_2d, probs, vertex_pred, rois, poses_init, poses_pred = \
        sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), \
                  net.get_output('rois'), net.get_output('poses_init'), net.get_output('poses_tanh')])

    # non-maximum suppression
    from utils.nms import nms

    keep = nms(rois, 0.5)
    rois = rois[keep, :]
    poses_init = poses_init[keep, :]
    poses_pred = poses_pred[keep, :]
    print keep
    print rois

    # combine poses
    poses = poses_init
    for i in xrange(rois.shape[0]):
        class_id = int(rois[i, 1])
        if class_id >= 0:
            poses[i, :4] = poses_pred[i, 4*class_id:4*class_id+4]

    vertex_pred = vertex_pred[0, :, :, :]

    labels = labels_2d[0,:,:].astype(np.int32)
    probs = probs[0,:,:,:]

    labels = unpad_im(labels, 16)
    roi_classes = [imdb._classes[int(c)] for c in rois[:,1]]

    # build the label image
    im_label = imdb.labels_to_image(im, labels)

    labels_new = cv2.resize(labels, None, None, fx=1.0/im_scale, fy=1.0/im_scale, interpolation=cv2.INTER_NEAREST)

    pose_data = [{"name": roi_classes[ix], "pose": p.tolist()} for ix, p in enumerate(poses)]

    SAVE = True
    if SAVE:
        import json
        j_file = rgb_filenames[idx].replace("color.png", "pred_pose.json")
        with open(j_file, "w") as f:
            json.dump(pose_data, f)
            print("Saved pose data to %s"%(j_file))

    if cfg.TEST.VISUALIZE:
        im_depth = pad_im(cv2.imread(depth_filenames[idx], cv2.IMREAD_UNCHANGED), 16)

        vertmap = _extract_vertmap(labels, vertex_pred, num_classes)
        vis_segmentations_vertmaps_detection(im, im_depth, im_label, imdb._class_colors, vertmap, 
            labels, rois, poses, meta_data['intrinsic_matrix'], imdb.num_classes, imdb._classes, imdb._points_all)
        # from tools.open3d_test2 import render_object_pose
        # object_model_dir = osp.join(imdb._get_default_path(), "models")
        # print(pose_data)
        # render_object_pose(im, im_depth, meta_data, pose_data, object_model_dir)
    
