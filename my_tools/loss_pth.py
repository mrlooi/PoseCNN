import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf

sys.path.append("lib/model")
from average_distance_loss.modules.average_distance_loss import AverageDistanceLoss
from hard_label.modules.hard_label import HardLabel
from hough_voting.modules.hough_voting import HoughVoting

sys.path.append("lib")
import hard_label_layer.hard_label_op as hard_label_op
import hard_label_layer.hard_label_op_grad


def loss_cross_entropy_single_frame(scores, labels):
    """
    scores: a tensor [batch_size, num_classes, height, width]
    labels: a tensor [batch_size, num_classes, height, width]
    """
    cross_entropy = -torch.sum(labels * scores, 3)
    loss = torch.div( torch.sum(cross_entropy), torch.sum(labels).float())

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
    loss = torch.div( torch.sum(in_loss), torch.sum(vertex_weights).float() )
    return loss

def hard_label_func_tf(prob_normalized, gt_label_2d, threshold=1.0):
    return hard_label_op.hard_label(prob_normalized, gt_label_2d, threshold, name="hard_label")

def hard_label_func(prob_normalized, gt_label_2d, threshold=1.0):
    return HardLabel(threshold)(gt_label_2d, prob_normalized)

def run_hough_voting(num_classes, label_2d, vertex_pred, extents, poses, mdata):
    vote_threshold = -1.0
    vote_percentage = 0.02
    skip_pixels = 20
    label_threshold = 500
    inlier_threshold = 0.9
    is_train = True
    hv = HoughVoting(num_classes, vote_threshold, vote_percentage, label_threshold=label_threshold, 
            inlier_threshold=inlier_threshold, skip_pixels=skip_pixels, is_train=is_train)
    return hv(label_2d, vertex_pred, extents, poses, mdata)

def average_distance_loss_func(poses_pred, poses_target, poses_weight, points, symmetry, num_classes, margin=0.01):

    return AverageDistanceLoss(num_classes, margin)(poses_pred, poses_target, poses_weight, points, symmetry)

def T(x, dtype=torch.float32, requires_grad=False):
    return torch.tensor(x, dtype=dtype, requires_grad=requires_grad)
def FT(x, requires_grad=False):
    return T(x, requires_grad=requires_grad)
def IT(x, requires_grad=False):
    return T(x, torch.int32, requires_grad=requires_grad)

def CT(x, dtype=torch.float32, requires_grad=False):
    return torch.tensor(x, dtype=dtype, requires_grad=requires_grad, device="cuda")
def FCT(x, requires_grad=False):
    return CT(x, requires_grad=requires_grad)
def ICT(x, requires_grad=False):
    return CT(x, dtype=torch.int32, requires_grad=requires_grad)

def get_error(d1, d2):
    e = np.abs(d1-d2)
    print(np.sum(e))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
gpu_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
cpu_config = tf.ConfigProto(device_count = {'GPU': 0})

if __name__ == '__main__':


    labels = np.load("labels.npy").astype(np.int32)
    vertex_targets = np.load("vertex_targets.npy")
    vertex_weights = np.load("vertex_weights.npy")
    poses_targets = np.load("poses_targets.npy")
    poses_weights = np.load("poses_weights.npy")

    score = np.load("score.npy")
    vertex_pred = np.load("vertex_pred.npy")
    poses_pred = np.load("poses_pred.npy")

    points = np.load("points_all.npy")
    poses_gt =  np.load("pose_gt.npy")
    extents =  np.load("extents.npy")
    meta_data =  np.load("meta_data.npy")

    symmetry = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    # labels = np.squeeze(labels, axis=0)

    VERTEX_W = 1.0
    POSE_W = 1.0

    num_classes = 22
    width = 640
    height = 480

    t_score = FCT(score, True)
    prob = F.log_softmax(t_score, 3)
    prob_normalized = F.softmax(t_score, 3)
    label_2d = torch.argmax(prob_normalized, dim=3)
    # label_2d = FCT(label_2d, True)
    hard_label_tf = hard_label_func_tf(prob_normalized.clone().detach().cpu().numpy(), labels)

    sess = tf.Session(config=cpu_config)
    sess.run(tf.global_variables_initializer())
    hl_tf = sess.run(hard_label_tf)
    hl_pth = hard_label_func(prob_normalized, ICT(labels)).detach().cpu().numpy()
    get_error(hl_tf, hl_pth)

    # loss cross entropy
    t_hl = FCT(hl_tf, True)
    # loss vertex
    t_vp = FCT(vertex_pred, True)
    t_vt = FCT(vertex_targets)
    t_vw = FCT(vertex_weights)
    # loss hough vote
    t_pgt = FCT(poses_gt)
    t_extents = FCT(extents)
    t_meta = FCT(meta_data)
    # loss pose
    t_pp = FCT(poses_pred, True)
    t_pt = FCT(poses_targets)
    t_pw = FCT(poses_weights)
    t_pts = FCT(points)
    t_sym = FCT(symmetry)

    hough_outputs = run_hough_voting(num_classes, label_2d, t_vp, t_extents, t_pgt, t_meta)
    t_pt, t_pw = hough_outputs[2:4]

    # loss_cls = loss_cross_entropy_single_frame(prob, t_hl)
    # loss_vertex = VERTEX_W * smooth_l1_loss_vertex(t_vp, t_vt, t_vw)
    loss_pose = POSE_W * average_distance_loss_func(t_pp, t_pt, t_pw, t_pts, t_sym, num_classes, margin=0.01)

    loss_pose.backward()
    g_tvp = t_vp.grad.cpu().numpy()
    assert np.sum(g_tvp) == 0