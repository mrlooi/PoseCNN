import time, os, sys
import os.path as osp
import numpy as np

from Queue import Queue
from threading import Thread

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
import cv2
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


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

    cfg.TRAIN.LEARNING_RATE = 0.0005
    cfg.TRAIN.SNAPSHOT_ITERS = 500
    cfg.TRAIN.USE_FLIPPED = False
    cfg.TRAIN.IMS_PER_BATCH = 1
    # cfg.TRAIN.SNAPSHOT_PREFIX = "vgg16"
    cfg.TRAIN.SNAPSHOT_PREFIX = "resnet50"

def get_lov2d_args():
    class Args():
        pass

    args = Args()
    args.gpu_id = 0
    args.max_iters = 200
    # args.pretrained_model = "/data/models/vgg16.pth"
    args.pretrained_model = "/data/models/resnet50.pth"
    args.pretrained_ckpt = None#"posecnn.pth"
    # args.pretrained_ckpt = "output/lov/lov_debug/resnet50_lov_iter_100.pth"
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
                                                     500, cfg.TRAIN.VOTING_THRESHOLD, cfg.IS_TRAIN, 0.5)


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    # if cfg.TRAIN.USE_FLIPPED:
    #     print 'Appending horizontally-flipped training examples...'
    #     imdb.append_flipped_images()
    #     print 'done'

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

def fetch_data(q, data_layer, sleep_time=None):
    while True:
        blobs = data_layer.forward()
        blobs['data_image_color'] = FCT(transpose_BHWC_to_BCHW(blobs['data_image_color']))
        blobs['data_vertex_weights'] = FCT(transpose_BHWC_to_BCHW(blobs['data_vertex_weights']))
        blobs['data_vertex_targets'] = FCT(transpose_BHWC_to_BCHW(blobs['data_vertex_targets']))

        blobs['data_points'] = FCT(blobs['data_points'])
        blobs['data_label'] = ICT(blobs['data_label'])
        blobs['data_extents'] = FCT(blobs['data_extents'])
        blobs['data_pose'] = FCT(blobs['data_pose'])
        blobs['data_meta_data'] = FCT(blobs['data_meta_data'])
        blobs['data_symmetry'] = FCT(blobs['data_symmetry'])

        q.put(blobs)

        if sleep_time is not None and sleep_time != 0:
            time.sleep(sleep_time)

def get_losses(network, blobs, num_classes, include_pose_loss=False):
    blob_im = blobs['data_image_color']
    vertex_weights = blobs['data_vertex_weights']
    vertex_targets = blobs['data_vertex_targets']

    blob_points = blobs['data_points']
    blob_labels = blobs['data_label']
    blob_extents = blobs['data_extents']
    blob_poses = blobs['data_pose']
    blob_meta_data = blobs['data_meta_data']
    blob_sym = blobs['data_symmetry']
    

    # pose loss
    if include_pose_loss:
        scores, label_2d, vertex_pred, hough_outputs, poses_tanh = network.forward(blob_im, blob_extents, blob_poses, blob_meta_data)
        poses_target, poses_weight = hough_outputs[2:4]
        poses_mul = torch.mul(poses_tanh, poses_weight)
        poses_pred = F.normalize(poses_mul, p=2, dim=1)
        loss_pose = cfg.TRAIN.POSE_W * average_distance_loss_func(poses_pred, poses_target, poses_weight, blob_points, blob_sym, num_classes, margin=0.01)
    else:
        scores, label_2d, vertex_pred = network.forward_image(blob_im)
        loss_pose = FCT(0)
        
    # cls loss
    prob = F.log_softmax(scores, 1).permute((0,2,3,1)).clone() # permute for hard_label_func, which is in BHWC format
    prob_n = F.softmax(scores, 1).permute((0,2,3,1)).clone() # permute for hard_label_func, which is in BHWC format
    hard_labels = hard_label_func(prob_n, blob_labels, threshold=1.0)
    loss_cls = loss_cross_entropy_single_frame(prob, hard_labels)

    # vertex loss
    loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights)

    loss = loss_cls + loss_vertex + loss_pose

    # loss_regu = 0 # TODO: tf.add_n(tf.losses.get_regularization_losses(), 'regu')

    return loss, loss_cls, loss_vertex, loss_pose


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, pretrained_ckpt=None, max_iters=40000):

    num_classes = imdb.num_classes
    # LOAD DATA LAYER
    data_layer = GtPoseCNNLayer(roidb, num_classes, imdb._extents, imdb._points_all, imdb._symmetry)
    q = Queue(maxsize=10)
    sleep_time_seconds = None
    num_data_workers = 3
    for i in xrange(num_data_workers):
        worker = Thread(target=fetch_data, args=(q,data_layer,sleep_time_seconds,))
        worker.setDaemon(True)
        worker.start()

    # LOAD PRETRAINED MODEL OR CKPT
    if pretrained_ckpt is not None:
        print ('Loading pretrained ckpt weights from %s'%(pretrained_ckpt))
        ckpt = torch.load(pretrained_ckpt)
        network.load_state_dict(ckpt)
    elif pretrained_model is not None:
        print ('Loading pretrained model weights from %s'%(pretrained_model))
        network.load_pretrained(pretrained_model)
    torch.cuda.empty_cache()

    network.cuda()
    network.train()

    # optimizer
    start_lr = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    optimizer = optim.Adam(network.parameters(), lr=start_lr, weight_decay=cfg.TRAIN.WEIGHT_REG)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad,network.parameters()), lr=start_lr, momentum=momentum, weight_decay=cfg.TRAIN.WEIGHT_REG)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad,network.parameters()), lr=start_lr, momentum=momentum)
    lr = start_lr

    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                            cfg.TRAIN.STEPSIZE, 0.1, staircase=True)

    get_tensor_np = lambda x: x.data.cpu().numpy()

    print('Training...')
    last_snapshot_iter = 0
    iter = 0

    loss_cls = loss_vertex = 1e5
    while iter < max_iters:
        if q.empty():
            sleep_s = 0.5
            print("Data queue empty, sleeping for %.2f seconds.."%(sleep_s))
            time.sleep(sleep_s)
            continue

        blobs = q.get()
        q.task_done()

        include_pose_loss = loss_cls < 0.4 and loss_vertex < 0.2 
        loss, loss_cls, loss_vertex, loss_pose = get_losses(network, blobs, num_classes, include_pose_loss=include_pose_loss)

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

        iter += 1

    iter -= 1
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

