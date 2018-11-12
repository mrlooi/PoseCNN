import time, os, sys
import os.path as osp
import numpy as np

import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


sys.path.append("my_tools")

import _init_paths
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
from gt_posecnn_layer.layer import GtPoseCNNLayer
from model1 import vgg16_convs
from convert_to_pth import PoseCNN, posecnn_pth_to_tf_mapping

sys.path.append("lib/model")
from average_distance_loss_pth.modules.average_distance_loss import AverageDistanceLoss
from hard_label.modules.hard_label import HardLabel

def get_network_tf():
    return vgg16_convs(cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, \
                                                     cfg.TRAIN.THRESHOLD_LABEL, cfg.TRAIN.VOTING_THRESHOLD, \
                                                     cfg.TRAIN.VERTEX_REG_2D, \
                                                     cfg.TRAIN.POSE_REG, cfg.TRAIN.TRAINABLE, cfg.IS_TRAIN)

def get_network_pth():
    return PoseCNN(cfg.TRAIN.NUM_UNITS, cfg.TRAIN.NUM_CLASSES, \
                                                     500, cfg.TRAIN.VOTING_THRESHOLD, cfg.IS_TRAIN)

# TF LOSSES
def loss_cross_entropy_single_frame_tf(scores, labels):
    """
    scores: a tensor [batch_size, height, width, num_classes]
    labels: a tensor [batch_size, height, width, num_classes]
    """
    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(labels * scores, reduction_indices=[3])
        loss = tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(labels)+1e-10)

    return loss

# PTH LOSSES
def loss_cross_entropy_single_frame(scores, labels):
    """
    scores: a tensor [batch_size, num_classes, height, width]
    labels: a tensor [batch_size, num_classes, height, width]
    """
    cross_entropy = -torch.sum(labels * scores, 1)
    loss = torch.div( torch.sum(cross_entropy), torch.sum(labels) + 1e-10)

    return loss

def smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights, sigma=1.0):

    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = torch.mul(vertex_weights, vertex_diff)
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).clone().float()
    smoothL1_sign.detach_()
    # smoothL1_sign = smoothL1_sign.float()
    # smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1. / sigma_2)))
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss = torch.div( torch.sum(in_loss), torch.sum(vertex_weights) + 1e-10 )
    return loss

def hard_label_func(prob_normalized, gt_label_2d, threshold=1.0):
    return HardLabel(threshold)(gt_label_2d, prob_normalized)

def average_distance_loss_func(poses_pred, poses_target, poses_weight, points, symmetry, num_classes, margin=0.01):
    return AverageDistanceLoss(num_classes, margin)(poses_pred, poses_target, poses_weight, points, symmetry)

def transpose_BHWC_to_BCHW(x):
    return np.transpose(x, [0,3,1,2])

def CT(x, dtype=torch.float32, requires_grad=False):
    return torch.tensor(x, dtype=dtype, requires_grad=requires_grad, device="cuda")
def FCT(x, requires_grad=False):
    return CT(x, requires_grad=requires_grad)
def ICT(x, requires_grad=False):
    return CT(x, dtype=torch.int32, requires_grad=requires_grad)

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

    cfg.TRAIN.LEARNING_RATE = 0.001
    cfg.TRAIN.SNAPSHOT_ITERS = 100

def get_lov2d_args():
    class Args():
        pass

    args = Args()
    args.gpu_id = 0
    args.max_iters = 1000
    # args.pretrained_model = "/data/models/pytorch/vgg16.pth"
    # args.pretrained_ckpt = None#"posecnn.pth"
    args.pth_ckpt = "posecnn.pth"
    args.tf_ckpt = "data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt"
    args.cfg_file = "experiments/cfgs/lov_color_2d.yml"
    args.imdb_name = "lov_debug"
    # args.randomize = False
    args.rig_name = None # "data/LOV/camera.json"
    args.cad_name = "data/LOV/models.txt"
    args.pose_name = "data/LOV/poses.txt"

    return args

def get_pth_tensor_as_np(x):
    return x.data.cpu().numpy()

def transpose_4d_tensor(x, t=[0,1,2,3]):
    if isinstance(x, np.ndarray):
        return np.transpose(x, t)
    elif isinstance(x, torch.Tensor):
        return x.clone().permute(t)
    else:
        raise NotImplementedError
def transpose_BHWC_to_BCHW(x):
    return transpose_4d_tensor(x, [0,3,1,2])
def transpose_BCHW_to_BHWC(x):
    return transpose_4d_tensor(x, [0,2,3,1])

def get_error(d1, d2):
    x = get_pth_tensor_as_np(d1) if isinstance(d1, torch.Tensor) else d1
    y = get_pth_tensor_as_np(d2) if isinstance(d2, torch.Tensor) else d2
    e = np.abs(x-y)
    return np.sum(e)

def get_tf_outputs(sess, net, feed_dict, output_names):
    sess.run(net.enqueue_op, feed_dict=feed_dict)
    if len(output_names) == 0:
        return []
    if isinstance(output_names, str):
        return sess.run(net.layers[output_names])
    
    if len(output_names) > 1:
        layer_ops = [net.layers[l] for l in output_names]
    else:
        layer_ops = net.layers[output_names[0]]
    return sess.run(layer_ops)


if __name__ == '__main__':
    
    args = get_lov2d_args()
    load_cfg(args)
    imdb = get_imdb(args.imdb_name)

    # LOAD PTH
    print("Loading PTH model...")
    model = get_network_pth()
    model.load_state_dict(torch.load(args.pth_ckpt))
    print("Loaded PTH model %s"%(args.pth_ckpt))
    model.cuda()
    model.train()

    # LOAD TF
    print("Loading TF model...")
    net_tf = get_network_tf()
    net = net_tf
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    saver.restore(sess, args.tf_ckpt)
    print("Loaded TF model %s"%(args.tf_ckpt))

    # LOAD DATA
    print("Loading data...")
    data_layer = GtPoseCNNLayer(imdb.roidb, imdb.num_classes, imdb._extents, imdb._points_all, imdb._symmetry)
    blobs = data_layer.forward(0)

    # stage PTH data
    print("Staging PTH data...")
    blob_im = FCT(transpose_BHWC_to_BCHW(blobs['data_image_color']))
    vertex_weights = FCT(transpose_BHWC_to_BCHW(blobs['data_vertex_weights']))
    vertex_targets = FCT(transpose_BHWC_to_BCHW(blobs['data_vertex_targets']))

    blob_points = FCT(blobs['data_points'])
    blob_labels = ICT(blobs['data_label'])
    blob_extents = FCT(blobs['data_extents'])
    blob_poses = FCT(blobs['data_pose'])
    blob_meta_data = FCT(blobs['data_meta_data'])
    blob_sym = FCT(blobs['data_symmetry'])

    print("Running PTH Net...")
    conv1 = model.conv1(blob_im)
    conv2 = model.conv2(model.max_pool2d(conv1))
    conv3 = model.conv3(model.max_pool2d(conv2))
    conv4 = model.conv4(model.max_pool2d(conv3))
    conv5 = model.conv5(model.max_pool2d(conv4))
    sc_conv4 = model.score_conv4(conv4)
    sc_conv5 = model.score_conv5(conv5)
    upsc_conv5 = model.upscore_conv5(sc_conv5)
    add_score = torch.add(sc_conv4, upsc_conv5)
    upsc = model.upscore(add_score)
    sc = model.score(upsc)
    sm = F.softmax(sc, 1)

    prob = F.log_softmax(sc, 1).permute((0,2,3,1)).clone() # permute for hard_label_func, which is in BHWC format
    prob_n = sm.permute((0,2,3,1)).clone() # permute for hard_label_func, which is in BHWC format
    hard_labels = hard_label_func(prob_n, blob_labels, threshold=1.0)
    loss_cls_pth = loss_cross_entropy_single_frame(prob, hard_labels)

    # stage TF data
    print("Staging TF data...")
    feed_dict={net_tf.data: blobs['data_image_color'], net_tf.gt_label_2d: blobs['data_label'], net_tf.keep_prob: 1.0, \
           net_tf.vertex_targets: blobs['data_vertex_targets'], net_tf.vertex_weights: blobs['data_vertex_weights'], \
           net_tf.poses: blobs['data_pose'], net_tf.extents: blobs['data_extents'], net_tf.meta_data: blobs['data_meta_data'], \
           net_tf.points: blobs['data_points'], net_tf.symmetry: blobs['data_symmetry']}

    print("Running TF Net...")
    loss_cls = loss_cross_entropy_single_frame_tf(net_tf.get_output('prob'), net_tf.get_output('gt_label_weight'))

    sess.run(net.enqueue_op, feed_dict=feed_dict)
    loss_cls_tf, score, conv5_3, upscore_conv5, poses_targets, poses_weights, vertex_pred, poses_pred = sess.run([loss_cls, net_tf.layers['score'], 
            net_tf.layers['conv5_3'], net.layers['upscore_conv5'],
            net_tf.layers['poses_target'], net_tf.layers['poses_weight'], net_tf.layers['vertex_pred'], 
            net_tf.layers['poses_pred']], feed_dict=feed_dict)
    
    def get_tf_out(x):
        return get_tf_outputs(sess, net, feed_dict, x)


    # get_error(transpose_BCHW_to_BHWC(conv5), conv5_3)
    # x = get_pth_tensor_as_np(transpose_BCHW_to_BHWC(sc)); x2 = get_tf_out("score")
    # get_error(transpose_BCHW_to_BHWC(F.log_softmax(sc, 1)), upscore_conv5)    
    get_error(transpose_BCHW_to_BHWC(sm), get_tf_out("prob_normalized"))
    get_error(hard_labels, get_tf_out("gt_label_weight"))

