import _init_paths
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb

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
    args.imdb_name = "lov_train"
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


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, pretrained_ckpt=None, max_iters=40000):
    from fcn.train import SolverWrapper    

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
        # data layer
        if cfg.TRAIN.SINGLE_FRAME:
            data_layer = GtSynthesizeLayer(roidb, imdb.num_classes, imdb._extents, imdb._points_all, imdb._symmetry, imdb.cache_path, imdb.name, imdb.data_queue, cfg.CAD, cfg.POSE)
        else:
            data_layer = GtDataLayer(roidb, imdb.num_classes)

        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model, pretrained_ckpt=pretrained_ckpt)

        print('Solving...')
        if cfg.TRAIN.VERTEX_REG_2D or cfg.TRAIN.VERTEX_REG_3D:
            if cfg.TRAIN.POSE_REG:
                sw.train_model_vertex_pose(sess, train_op, loss, loss_cls, loss_vertex, loss_pose, learning_rate, max_iters, data_layer)
            else:
                sw.train_model_vertex(sess, train_op, loss, loss_cls, loss_vertex, loss_regu, learning_rate, max_iters, data_layer)
        else:
            sw.train_model(sess, train_op, loss, learning_rate, max_iters, data_layer)
        print('done solving')

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

    network = get_network()

    # train_net(network, imdb, roidb, output_dir,
    #               pretrained_model=args.pretrained_model,
    #               pretrained_ckpt=args.pretrained_ckpt,
    #               max_iters=args.max_iters)

