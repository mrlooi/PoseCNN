import numpy as np
import sys

import tensorflow as tf

sys.path.append("lib")
import average_distance_loss.average_distance_loss_op as average_distance_loss_op
import average_distance_loss.average_distance_loss_op_grad
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
    scores: a tensor [batch_size, height, width, num_classes]
    labels: a tensor [batch_size, height, width, num_classes]
    """
    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(labels * scores, reduction_indices=[3])
        loss = tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(labels)+1e-10)

    return loss

def smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights, sigma=1.0):
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = tf.multiply(vertex_weights, vertex_diff)
    abs_diff = tf.abs(diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1. / sigma_2)))
    in_loss = tf.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss = tf.div( tf.reduce_sum(in_loss), tf.reduce_sum(vertex_weights) + 1e-10 )
    return loss

def hard_label_func(prob_normalized, gt_label_2d, threshold=1.0):
    return hard_label_op.hard_label(prob_normalized, gt_label_2d, threshold, name="hard_label")

def average_distance_loss_func(poses_pred, poses_target, poses_weight, points, symmetry, margin=0.01):
    return average_distance_loss_op.average_distance_loss(poses_pred, poses_target, poses_weight, points, symmetry, margin, name="loss_pose")

if __name__ == '__main__':
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

    num_classes = 22
    width = 640
    height = 480

    # GT
    labels_ph = tf.placeholder(tf.int32, shape=[None, height, width])  
    vertex_targets_ph = tf.placeholder(tf.float32, shape=[None, height, width, num_classes*3])
    vertex_weights_ph = tf.placeholder(tf.float32, shape=[None, height, width, num_classes*3])  
    poses_targets_ph = tf.placeholder(tf.float32, shape=[None, 4 * num_classes])
    poses_weights_ph = tf.placeholder(tf.float32, shape=[None, 4 * num_classes])
    # points_ph = tf.placeholder(tf.float32, shape=[num_classes, None, 3])
    # symmetry_ph = tf.placeholder(tf.float32, shape=[num_classes])

    # OUT
    score_ph = tf.placeholder(tf.float32, shape=[None, height, width, num_classes])  #  
    vertex_pred_ph = tf.placeholder(tf.float32, shape=[None, height, width, num_classes*3])  
    poses_pred_ph = tf.placeholder(tf.float32, shape=[None, 4 * num_classes]) 


    prob = tf.nn.log_softmax(score_ph, 3)
    prob_normalized = tf.nn.softmax(score_ph, 3)
    hard_label = hard_label_func(prob_normalized, labels_ph)

    # ALL LOSSES

    VERTEX_W = 1.0
    POSE_W = 1.0

    loss_cls = loss_cross_entropy_single_frame(prob, hard_label)
    loss_vertex = VERTEX_W * smooth_l1_loss_vertex(vertex_pred_ph, vertex_targets_ph, vertex_weights_ph)
    loss_pose = POSE_W * average_distance_loss_func(poses_pred_ph, poses_targets_ph, poses_weights_ph, points, symmetry, margin=0.01)[0]
    # loss_regu = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
    loss = loss_cls + loss_vertex + loss_pose #+ loss_regu

    # # # # iter: 500 / 500, loss: 0.2680, loss_cls: 0.0258, loss_vertex: 0.0407, loss_pose: 0.0061, lr: 0.00010000,  time: 0.32

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict = {
        labels_ph: labels,
        vertex_targets_ph: vertex_targets,
        vertex_weights_ph: vertex_weights,
        poses_targets_ph: poses_targets,
        poses_weights_ph: poses_weights,
        
        score_ph: score,
        vertex_pred_ph: vertex_pred,
        poses_pred_ph: poses_pred
    }

    g, lcls, lvertex, lpose = sess.run([tf.gradients(loss_pose, vertex_pred_ph), loss_cls, loss_vertex, loss_pose], feed_dict=feed_dict)
    
