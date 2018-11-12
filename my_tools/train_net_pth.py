import time, os, sys
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append("lib/model")
from average_distance_loss.modules.average_distance_loss import AverageDistanceLoss
from hard_label.modules.hard_label import HardLabel

import _init_paths
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
from gt_posecnn_layer.layer import GtPoseCNNLayer

# from fcn.train import SolverWrapper


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

    cfg.TRAIN.LEARNING_RATE = 0.0001

def get_lov2d_args():
    class Args():
        pass

    args = Args()
    args.gpu_id = 0
    args.max_iters = 40
    args.pretrained_model = "/data/models/vgg16.pth"
    args.pretrained_ckpt = "posecnn.pth"
    args.cfg_file = "experiments/cfgs/lov_color_2d.yml"
    args.imdb_name = "lov_debug"
    args.randomize = False
    args.rig_name = None # "data/LOV/camera.json"
    args.cad_name = "data/LOV/models.txt"
    args.pose_name = "data/LOV/poses.txt"

    return args

def get_network():
    from convert_to_pth import PoseCNN
    return PoseCNN(cfg.TRAIN.NUM_UNITS, cfg.TRAIN.NUM_CLASSES, \
                                                     cfg.TRAIN.THRESHOLD_LABEL, cfg.TRAIN.VOTING_THRESHOLD, cfg.IS_TRAIN)


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    return imdb.roidb

def save_net(network, output_dir, iter):
    infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
    filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix + '_iter_{:d}'.format(iter+1) + '.pth')
    filename = os.path.join(output_dir, filename)
    torch.save(network.state_dict(), filename)
    print("Saved to %s"%(filename))


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

def get_losses(network, blobs, num_classes):
    blob_im = FCT(transpose_BHWC_to_BCHW(blobs['data_image_color']))
    vertex_weights = FCT(transpose_BHWC_to_BCHW(blobs['data_vertex_weights']))
    vertex_targets = FCT(transpose_BHWC_to_BCHW(blobs['data_vertex_targets']))

    blob_points = FCT(blobs['data_points'])
    blob_labels = ICT(blobs['data_label'])
    blob_extents = FCT(blobs['data_extents'])
    blob_poses = FCT(blobs['data_pose'])
    blob_meta_data = FCT(blobs['data_meta_data'])
    blob_sym = FCT(blobs['data_symmetry'])

    scores, label_2d, vertex_pred, hough_outputs, poses_tanh = network.forward(blob_im, blob_extents, blob_poses, blob_meta_data)
    poses_target, poses_weight = hough_outputs[2:4]

    # cls loss
    prob = F.log_softmax(scores, 1).permute((0,2,3,1)) # permute for hard_label_func, which is in BHWC format
    prob_n = F.softmax(scores, 1).permute((0,2,3,1)) # permute for hard_label_func, which is in BHWC format
    hard_labels = hard_label_func(prob_n, blob_labels)
    loss_cls = loss_cross_entropy_single_frame(prob, hard_labels)

    # vertex loss
    loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights)

    # pose loss
    poses_mul = torch.mul(poses_tanh, poses_weight)
    poses_pred = F.normalize(poses_mul, p=2, dim=1)

    loss_pose = cfg.TRAIN.POSE_W * average_distance_loss_func(poses_pred, poses_target, poses_weight, blob_points, blob_sym, num_classes, margin=0.01)

    loss_regu = 0 # TODO: tf.add_n(tf.losses.get_regularization_losses(), 'regu')

    loss = loss_cls + loss_vertex + loss_pose + loss_regu

    return loss, loss_cls, loss_vertex, loss_pose


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, pretrained_ckpt=None, max_iters=40000):

    num_classes = imdb.num_classes
    # LOAD DATA LAYER
    data_layer = GtPoseCNNLayer(roidb, num_classes, imdb._extents, imdb._points_all, imdb._symmetry)

    # LOAD PRETRAINED MODEL OR CKPT
    if pretrained_ckpt is not None:
        print ('Loading pretrained ckpt weights from %s'%(pretrained_ckpt))
        ckpt = torch.load(pretrained_ckpt)
        network.load_state_dict(ckpt)
    elif pretrained_model is not None:
        print ('Loading pretrained model weights from %s'%(pretrained_model))
        network.load_pretrained(pretrained_model)

    network.cuda()
    network.train()

    # optimizer
    start_lr = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,network.parameters()), lr=start_lr)
    lr = start_lr

    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                            cfg.TRAIN.STEPSIZE, 0.1, staircase=True)

    get_tensor_np = lambda x: x.data.cpu().numpy()

    print('Training...')
    last_snapshot_iter = 0
    for iter in xrange(max_iters):
        blobs = data_layer.forward(iter)

        loss, loss_cls, loss_vertex, loss_pose = get_losses(network, blobs, num_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss_value, loss_cls_value, loss_vertex_value, loss_pose_value = get_tensor_np(loss), 
        #         get_tensor_np(loss_cls), get_tensor_np(loss_vertex), get_tensor_np(loss_pose)

        print('iter: %d / %d, loss: %.4f, loss_cls: %.4f, loss_vertex: %.4f, loss_pose: %.4f, lr: %.6f' %\
            (iter+1, max_iters, loss, loss_cls, loss_vertex, loss_pose, lr))

        if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
            last_snapshot_iter = iter
            save_net(network, output_dir, iter)

    if last_snapshot_iter != iter:
        save_net(network, output_dir, iter)

    print('Training complete')

if __name__ == '__main__':
    # from fcn.train import train_net#, get_training_roidb

    args = get_lov2d_args()
    load_cfg(args)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    imdb = get_imdb(args.imdb_name)

    output_dir = get_output_dir(imdb, None)
    print('Output will be saved to `{:s}`'.format(output_dir))

    roidb = get_training_roidb(imdb)

    network = get_network()

    train_net(network, imdb, roidb, output_dir,
                  pretrained_model=args.pretrained_model,
                  pretrained_ckpt=args.pretrained_ckpt,
                  max_iters=args.max_iters)
