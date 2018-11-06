import time, os, sys
import tensorflow as tf
import os.path as osp
import numpy as np
# import cv2

import _init_paths
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
from fcn.train import SolverWrapper


def load_cfg(args):
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id
    device_name = '/gpu:{:d}'.format(args.gpu_id)
    print(device_name)

    cfg.RIG = args.rig_name
    cfg.CAD = args.cad_name
    cfg.POSE = args.pose_name
    cfg.IS_TRAIN = True

    cfg.USE_FLIPPED = False

def get_lov2d_args():
    class Args():
        pass

    args = Args()
    args.gpu_id = 0
    args.max_iters = 40000
    args.pretrained_model = "data/imagenet_models/vgg16.npy"
    args.pretrained_ckpt = "data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt"
    args.cfg_file = "experiments/cfgs/lov_color_2d.yml"
    args.imdb_name = "lov_debug"
    args.randomize = False
    args.network_name = "vgg16_convs"
    args.rig_name = None # "data/LOV/camera.json"
    args.cad_name = "data/LOV/models.txt"
    args.pose_name = "data/LOV/poses.txt"

    return args

def get_network():
    from model1 import vgg16_convs
    return vgg16_convs(cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, \
                                                     cfg.TRAIN.THRESHOLD_LABEL, cfg.TRAIN.VOTING_THRESHOLD, \
                                                     cfg.TRAIN.VERTEX_REG_2D, \
                                                     cfg.TRAIN.POSE_REG, cfg.TRAIN.TRAINABLE, cfg.IS_TRAIN)


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    return imdb.roidb

def load_and_enqueue(sess, net, data_layer, coord):

    iter_ = 0
    while not coord.should_stop():
        blobs = data_layer.forward(iter_)
        iter_ += 1

        feed_dict={net.data: blobs['data_image_color'], net.gt_label_2d: blobs['data_label'], net.keep_prob: 0.5, \
                   net.vertex_targets: blobs['data_vertex_targets'], net.vertex_weights: blobs['data_vertex_weights'], \
                   net.poses: blobs['data_pose'], net.extents: blobs['data_extents'], net.meta_data: blobs['data_meta_data'], \
                   net.points: blobs['data_points'], net.symmetry: blobs['data_symmetry']}

        sess.run(net.enqueue_op, feed_dict=feed_dict)


class PoseCNNSolver(SolverWrapper):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """
    def __init__(self, sess, network, output_dir, pretrained_model=None, pretrained_ckpt=None):
        super(PoseCNNSolver, self).__init__(sess, network, None, None, output_dir, pretrained_model, pretrained_ckpt)

    def train_model_vertex_pose(self, sess, train_op, loss, loss_cls, loss_vertex, loss_pose, learning_rate, max_iters, data_layer):
        """Network training loop."""
        # add summary
        # tf.summary.scalar('loss', loss)
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(self.output_dir, sess.graph)

        coord = tf.train.Coordinator()
        if cfg.TRAIN.VISUALIZE:
            load_and_enqueue(sess, self.net, data_layer, coord)
        else:
            t = threading.Thread(target=load_and_enqueue, args=(sess, self.net, data_layer, coord))
            t.start()

        # intialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        if self.pretrained_ckpt is not None:
            print ('Loading pretrained ckpt '
                   'weights from {:s}').format(self.pretrained_ckpt)
            self.restore(sess, self.pretrained_ckpt)

        tf.get_default_graph().finalize()

        # tf.train.write_graph(sess.graph_def, self.output_dir, 'model.pbtxt')

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):

            timer.tic()
            loss_value, loss_cls_value, loss_vertex_value, loss_pose_value, lr, _ = sess.run([loss, 
                    loss_cls, loss_vertex, loss_pose, learning_rate, train_op])
            # train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %.4f, loss_cls: %.4f, loss_vertex: %.4f, loss_pose: %.4f, lr: %.8f,  time: %.2f' %\
                    (iter+1, max_iters, loss_value, loss_cls_value, loss_vertex_value, loss_pose_value, lr, timer.diff)

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

        sess.run(self.net.close_queue_op)
        coord.request_stop()
        coord.join([t])


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, pretrained_ckpt=None, max_iters=40000):
    from fcn.train import smooth_l1_loss_vertex, loss_cross_entropy_single_frame
    # from gt_single_data_layer.layer import GtSingleDataLayer
    # from gt_synthesize_layer.layer import GtSynthesizeLayer
    from gt_posecnn_layer.layer import GtPoseCNNLayer

    # if cfg.TRAIN.SINGLE_FRAME:
    scores = network.get_output('prob')
    labels = network.get_output('gt_label_weight')
    loss_cls = loss_cross_entropy_single_frame(scores, labels)
    loss = loss_cls
    if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
        vertex_pred = network.get_output('vertex_pred')
        vertex_targets = network.get_output('vertex_targets')
        vertex_weights = network.get_output('vertex_weights')
        # loss_vertex = tf.div( tf.reduce_sum(tf.multiply(vertex_weights, tf.abs(tf.subtract(vertex_pred, vertex_targets)))), tf.reduce_sum(vertex_weights) + 1e-10 )
        loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights)

        loss += loss_vertex
        if cfg.TRAIN.POSE_REG:
            loss_pose = cfg.TRAIN.POSE_W * network.get_output('loss_pose')[0]
            loss += loss_pose

    loss_regu = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
    loss += loss_regu

    # optimizer
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
    momentum = cfg.TRAIN.MOMENTUM
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # # data layer
        # if cfg.TRAIN.SINGLE_FRAME:
        #     data_layer = GtSynthesizeLayer(roidb, imdb.num_classes, imdb._extents, imdb._points_all, imdb._symmetry, imdb.cache_path, imdb.name, imdb.data_queue, cfg.CAD, cfg.POSE)
        # else:
        #     data_layer = GtDataLayer(roidb, imdb.num_classes)
        sw = PoseCNNSolver(sess, network, output_dir, pretrained_model=pretrained_model, pretrained_ckpt=pretrained_ckpt)

        data_layer = GtPoseCNNLayer(roidb, imdb.num_classes, imdb._extents)

        print('Training...')
        sw.train_model_vertex_pose(sess, train_op, loss, loss_cls, loss_vertex, loss_pose, learning_rate, max_iters, data_layer)
        print('Training complete')

if __name__ == '__main__':
    # from fcn.train import train_net#, get_training_roidb

    args = get_lov2d_args()
    load_cfg(args)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    imdb = get_imdb(args.imdb_name)

    imdb.data_queue = []
    # if cfg.TRAIN.SYNTHESIZE and cfg.TRAIN.SYN_ONLINE:
    #     pass

    output_dir = get_output_dir(imdb, None)
    print('Output will be saved to `{:s}`'.format(output_dir))

    roidb = get_training_roidb(imdb)

    # network = get_network()

    # train_net(network, imdb, roidb, output_dir,
    #               pretrained_model=args.pretrained_model,
    #               pretrained_ckpt=args.pretrained_ckpt,
    #               max_iters=args.max_iters)


    # DEBUG DATA LAYER
    num_classes = imdb.num_classes
    extents = imdb._extents
    db_inds = [0,1,2]
    minibatch_db = [roidb[i] for i in db_inds]
    roidb = minibatch_db
        
    import scipy.io as sio
    import cv2
    num_images = len(roidb)
    i = 0
    meta_data = sio.loadmat(roidb[i]['meta_data'])
    K = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    def pad_im(im, x):
        return im

    rgba = pad_im(cv2.imread(roidb[i]['image'], cv2.IMREAD_UNCHANGED), 16)
    if rgba.shape[2] == 4:
        im = np.copy(rgba[:,:,:3])
        alpha = rgba[:,:,3]
        I = np.where(alpha == 0)
        im[I[0], I[1], :] = 0
    else:
        im = rgba

    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_scale = 1.0
    # im = im_orig

    im = cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED)

    meta_data = scipy.io.loadmat(roidb[i]['meta_data'])
    class_colors = roidb[i]['class_colors']
    class_weights = roidb[i]['class_weights']
    im_cls, im_labels = _process_label_image(im, class_colors, class_weights)

    from transforms3d.quaternions import mat2quat

    poses = meta_data['poses']
    num = poses.shape[2]
    qt = np.zeros((num, 13), dtype=np.float32)
    for j in xrange(num):
        R = poses[:, :3, j]
        T = poses[:, 3, j]

        qt[j, 0] = i
        qt[j, 1] = meta_data['cls_indexes'][j, 0]
        qt[j, 2:6] = 0  # fill box later, roidb[i]['boxes'][j, :]
        qt[j, 6:10] = mat2quat(R)
        qt[j, 10:] = T

    pose_blob = np.concatenate((pose_blob, qt), axis=0)

    labels = cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED)
    center = meta_data['center']
    cls_indexes = meta_data['cls_indexes']
