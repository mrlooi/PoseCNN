import _init_paths
# from fcn.test import test_net_images
from datasets.factory import get_imdb

from utils.blob import pad_im, unpad_im # im_list_to_blob

import time, os, sys
import os.path as osp
import numpy as np
import cv2

import torch
import torch.nn as nn

# mgc = get_ipython().magic
# mgc('%matplotlib WXAgg')

from demo import _get_image_blob


if __name__ == '__main__':
    # import pprint
    from utils.nms import nms
    from convert_to_pth import PoseCNN, get_meta_info

    imdb_name = "lov_keyframe"
    imdb = get_imdb(imdb_name)

    # variables
    extents = imdb._extents
    points = imdb._points_all
    symmetry = imdb._symmetry
    num_classes = imdb.num_classes

    im_scale = 1.0

    extents, poses, meta_data = get_meta_info(num_classes)
    K = meta_data[0,:9]
    factor_depth = 10000
    # K[2,2] = 1

    # construct the filenames
    # demo_dir = 'data/demo_images/'
    demo_dir = 'data/LOV/data/0000/'
    rgb_filenames = sorted([demo_dir + f for f in os.listdir(demo_dir) if f.endswith("color.png")])
    depth_filenames = sorted([demo_dir + f for f in os.listdir(demo_dir) if f.endswith("depth.png")])
    print(rgb_filenames)
    print(depth_filenames)

    # load network
    # model_file = "posecnn.pth"
    model_file = "output/lov/lov_debug/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_40.pth"
    model = PoseCNN(64, num_classes)
    model.load_state_dict(torch.load(model_file))
    print("Loaded model %s"%model_file)

    model.eval()
    model.cuda()


    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

    FCT = torch.cuda.FloatTensor

    # START
    for idx in range(len(rgb_filenames)):
        im_file = rgb_filenames[idx]
        depth_file = depth_filenames[idx]
        img = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)
        im = pad_im(img, 16)

        im_blob, _, _ = _get_image_blob(im, im_scale, PIXEL_MEANS)

        im_blob = FCT(np.transpose(im_blob, [0,3,1,2]))
        _, labels_2d, vertex_pred, hough_outputs, poses_pred = model.forward(im_blob, FCT(extents), FCT(poses), FCT(meta_data))
        rois, poses_init = hough_outputs[:2]

        labels_2d = labels_2d.data.cpu().numpy()
        vertex_pred = vertex_pred.data.cpu().numpy()
        rois = rois.data.cpu().numpy()
        poses_init = poses_init.data.cpu().numpy()
        poses_pred = poses_pred.data.cpu().numpy()

        # non-maximum suppression
        keep = nms(rois, 0.5)
        rois = rois[keep, :]
        poses_init = poses_init[keep, :]
        poses_pred = poses_pred[keep, :]
        # print keep
        # print rois

        # combine poses
        poses = poses_init
        for i in xrange(rois.shape[0]):
            class_id = int(rois[i, 1])
            if class_id >= 0:
                poses[i, :4] = poses_pred[i, 4*class_id:4*class_id+4]

        labels = labels_2d[0,:,:].astype(np.int32)
        labels = unpad_im(labels, 16)

        roi_classes = [imdb._classes[int(c)] for c in rois[:,1]]

        # build the label image
        im_label = imdb.labels_to_image(im, labels)

        labels_new = cv2.resize(labels, None, None, fx=1.0/im_scale, fy=1.0/im_scale, interpolation=cv2.INTER_NEAREST)

        pose_data = [{"name": roi_classes[ix], "pose": p.tolist()} for ix, p in enumerate(poses)]

        SAVE = True

        im_file_prefix = im_file.replace("color.png","")
        # np.save(im_file_prefix + "label2d.npy", labels_2d)
        # np.save(im_file_prefix + "vert_pred.npy", vertex_pred)

        if SAVE:
            import json
            j_file = im_file_prefix + "pred_pose.json"
            with open(j_file, "w") as f:
                j_data = {"poses": pose_data, "meta": {'intrinsic_matrix': K.tolist(), 'factor_depth': factor_depth}}
                json.dump(j_data, f)
                print("Saved pose data to %s"%(j_file))

        VIS = True
        # if cfg.TEST.VISUALIZE:
        if VIS:
            color_m = np.zeros(img.shape, dtype=np.uint8)
            for c in xrange(num_classes-1):
                cls = c + 1
                color = np.random.randint(0,255,size=(3))
                color_m[labels==cls] = color
            cv2.imshow("img", img)
            cv2.imshow("m", color_m)
            cv2.waitKey(100)

            im_depth = pad_im(cv2.imread(depth_file, cv2.IMREAD_UNCHANGED), 16)

            from vis_utils import extract_vertmap, vis_segmentations_vertmaps_detection

            vp = np.transpose(vertex_pred[0], [1,2,0])
            vertmap = extract_vertmap(labels, vp, num_classes)
            K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
            vis_segmentations_vertmaps_detection(im, im_depth, im_label, imdb._class_colors, vertmap, 
                labels, rois, poses, K, imdb.num_classes, imdb._classes, imdb._points_all)
