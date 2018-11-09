import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("lib/model")
from average_distance_loss.modules.average_distance_loss import AverageDistanceLoss

sys.path.append("lib")
import hard_label_layer.hard_label_op as hard_label_op
import hard_label_layer.hard_label_op_grad

class persistent_locals(object):
    def __init__(self, func):
        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event=='return':
                l = frame.f_locals.copy()
                self._locals = l
                for k,v in l.items():
                    globals()[k] = v

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
            
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals

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

def hard_label_func(prob_normalized, gt_label_2d, threshold=1.0):
    return hard_label_op.hard_label(prob_normalized, gt_label_2d, threshold, name="hard_label")

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

if __name__ == '__main__':
    import tensorflow as tf

    labels = np.load("labels.npy")
    vertex_targets = np.load("vertex_targets.npy")
    vertex_weights = np.load("vertex_weights.npy")
    poses_targets = np.load("poses_targets.npy")
    poses_weights = np.load("poses_weights.npy")

    score = np.load("score.npy")
    vertex_pred = np.load("vertex_pred.npy")
    poses_pred = np.load("poses_pred.npy")

    points = np.load("points_all.npy")
    symmetry = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    # labels = np.squeeze(labels, axis=0)

    VERTEX_W = 1.0
    POSE_W = 1.0

    num_classes = 22
    width = 640
    height = 480


    prob = F.log_softmax(T(score), 3)
    prob_normalized = F.softmax(T(score), 3)
    hard_label_tf = hard_label_func(prob_normalized.detach().numpy(), labels)

    sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
    hard_label = sess.run(hard_label_tf)

    t_hl = FT(hard_label, True)
    t_vp = FT(vertex_pred, True)
    t_vt = FT(vertex_targets)
    t_vw = FT(vertex_weights)
    t_pp = FCT(poses_pred, True)
    t_pt = FCT(poses_targets)
    t_pw = FCT(poses_weights)
    t_pts = FCT(points)
    t_sym = FCT(symmetry)

    # loss_cls = loss_cross_entropy_single_frame(prob, t_hl)
    # loss_vertex = VERTEX_W * smooth_l1_loss_vertex(t_vp, t_vt, t_vw)
    loss_pose = POSE_W * average_distance_loss_func(t_pp, t_pt, t_pw, t_pts, t_sym, num_classes, margin=0.01)

