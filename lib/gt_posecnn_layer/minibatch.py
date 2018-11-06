# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import sys
import numpy as np
import numpy.random as npr
import cv2
from fcn.config import cfg
from utils.blob import im_list_to_blob, chromatic_transform
from utils.se3 import *
import scipy.io
# from normals import gpu_normals
from transforms3d.quaternions import mat2quat, quat2mat

from gt_single_data_layer.minibatch import _scale_vertmap, _unscale_vertmap, _get_bb3D, _vote_centers

def get_minibatch(roidb, num_classes, extents):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)

    # Get the input image blob, formatted for tensorflow
    random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
    im_blob, _, im_scales = _get_image_blob(roidb, random_scale_ind)

    # build the label blob
    label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, \
        pose_blob = _get_label_blob(roidb, num_classes, im_scales)

    # For debug visualizations
    if cfg.TRAIN.VISUALIZE:
        _vis_minibatch(im_blob, None, label_blob, meta_data_blob, vertex_target_blob, pose_blob, extents)

    symmetry_blob = symmetry if is_symmetric else np.zeros_like(symmetry)

    blobs = {'data_image_color': im_blob,
             # 'data_image_depth': im_depth_blob,
             # 'data_image_color_rescale': im_rescale_blob,
             # 'data_image_normal': im_normal_blob,
             'data_label': label_blob,
             # 'data_depth': depth_blob,
             'data_meta_data': meta_data_blob,
             'data_vertex_targets': vertex_target_blob,
             'data_vertex_weights': vertex_weight_blob,
             'data_pose': pose_blob,
             'data_extents': extents,
             'data_points': point_blob,
             'data_symmetry': symmetry_blob,
             }

    return blobs

def _get_image_blob(roidb, scale_ind):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    # processed_ims_depth = []
    # processed_ims_normal = []
    im_scales = []

    for i in xrange(num_images):
        # rgba
        rgba = cv2.imread(roidb[i]['image'], cv2.IMREAD_UNCHANGED)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        # chromatic transform
        if cfg.TRAIN.CHROMATIC:
            label = cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED)
            im = chromatic_transform(im, label)

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scales.append(im_scale)
        processed_ims.append(im)

        # # depth
        # im_depth_raw = cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED)
        # height = im_depth_raw.shape[0]
        # width = im_depth_raw.shape[1]

        # im_depth = im_depth_raw.astype(np.float32, copy=True) / float(im_depth_raw.max()) * 255
        # im_depth = np.tile(im_depth[:,:,np.newaxis], (1,1,3)) # turn to (H,W,3), the last channel is the same

        # if roidb[i]['flipped']:
        #     im_depth = im_depth[:, ::-1]

        # im_orig = im_depth.astype(np.float32, copy=True)
        # im_orig -= cfg.PIXEL_MEANS
        # im_depth = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # processed_ims_depth.append(im_depth)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    # blob_depth = im_list_to_blob(processed_ims_depth, 3)

    return blob, [], im_scales


def _process_label_image(label_image, class_colors, class_weights):
    """
    change label image to label index
    """
    height = label_image.shape[0]
    width = label_image.shape[1]
    num_classes = len(class_colors)
    label_index = np.zeros((height, width, num_classes), dtype=np.float32)
    labels = np.zeros((height, width), dtype=np.float32)

    if len(label_image.shape) == 3:
        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in xrange(len(class_colors)):
            color = class_colors[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            label_index[I[0], I[1], i] = class_weights[i]
            labels[I[0], I[1]] = i
    else:
        for i in xrange(len(class_colors)):
            I = np.where(label_image == i)
            label_index[I[0], I[1], i] = class_weights[i]
            labels[I[0], I[1]] = i
    
    return label_index, labels


def _get_label_blob(roidb, num_classes, im_scales):
    """ build the label blob """

    num_images = len(roidb)
    processed_depth = []
    processed_label = []
    processed_meta_data = []
    processed_vertex_targets = []
    processed_vertex_weights = []

    for i in xrange(num_images):
        im_scale = im_scales[i]

        # load meta data
        meta_data = scipy.io.loadmat(roidb[i]['meta_data'])
        # im_depth = cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED)

        # read label image
        im = cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED)
        height = im.shape[0]
        width = im.shape[1]
        # mask the label image according to depth
        if roidb[i]['flipped']:
            if len(im.shape) == 2:
                im = im[:, ::-1]
            else:
                im = im[:, ::-1, :]

        if im_scale != 1:
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)

        cls_indexes = meta_data['cls_indexes']
        # if num_classes == 2:
        #     I = np.where(im > 0)
        #     im[I[0], I[1]] = 1
        #     for j in xrange(len(meta_data['cls_indexes'])):
        #         meta_data['cls_indexes'][j] = 1
        im_cls, im_labels = _process_label_image(im, roidb[i]['class_colors'], roidb[i]['class_weights'])
        processed_label.append(im_cls)

        # vertex regression targets and weights
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))

        center_targets, center_weights = _vote_centers(im, cls_indexes, im_scale * meta_data['center'], poses, num_classes)
        processed_vertex_targets.append(center_targets)
        processed_vertex_weights.append(center_weights)

        num = poses.shape[2]
        qt = np.zeros((num, 13), dtype=np.float32)
        for j in xrange(num):
            R = poses[:, :3, j]
            T = poses[:, 3, j]

            qt[j, 0] = i
            qt[j, 1] = cls_indexes[j, 0]
            qt[j, 2:6] = 0  # fill box later, roidb[i]['boxes'][j, :]
            qt[j, 6:10] = mat2quat(R)
            qt[j, 10:] = T

        pose_blob = qt

        # # depth
        # if roidb[i]['flipped']:
        #     im_depth = im_depth[:, ::-1]
        # depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
        # depth = cv2.resize(depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # processed_depth.append(depth)

        K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        mdata = np.zeros(48, dtype=np.float32)
        mdata[0:9] = K.flatten()
        mdata[9:18] = Kinv.flatten()

        if cfg.FLIP_X:
            mdata[0] = -1 * mdata[0]
            mdata[9] = -1 * mdata[9]
            mdata[11] = -1 * mdata[11]
        processed_meta_data.append(mdata)

    # construct the blobs
    # height = processed_depth[0].shape[0]
    # width = processed_depth[0].shape[1]
    # depth_blob = np.zeros((num_images, height, width, 1), dtype=np.float32)
    label_blob = np.zeros((num_images, height, width, num_classes), dtype=np.float32)
    meta_data_blob = np.zeros((num_images, 1, 1, 48), dtype=np.float32)

    vertex_target_blob = np.zeros((num_images, height, width, 3 * num_classes), dtype=np.float32)
    vertex_weight_blob = np.zeros((num_images, height, width, 3 * num_classes), dtype=np.float32)

    for i in xrange(num_images):
        # depth_blob[i,:,:,0] = processed_depth[i]
        label_blob[i,:,:,:] = processed_label[i]
        meta_data_blob[i,0,0,:] = processed_meta_data[i]

        vertex_target_blob[i,:,:,:] = processed_vertex_targets[i]
        vertex_weight_blob[i,:,:,:] = processed_vertex_weights[i]
    
    return label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob


def _vis_minibatch(im_blob, depth_blob, label_blob, meta_data_blob, vertex_target_blob, pose_blob, extents):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    class_bb3d = [_get_bb3D(ext) for ext in extents]

    for i in xrange(im_blob.shape[0]):
        fig = plt.figure()
        # show image
        im = im_blob[i, :, :, :].copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        fig.add_subplot(231)
        plt.imshow(im)

        # project the 3D box to image
        metadata = meta_data_blob[i, 0, 0, :]
        intrinsic_matrix = metadata[:9].reshape((3,3))
        for j in xrange(pose_blob.shape[0]):
            if pose_blob[j, 0] != i:
                continue

            class_id = int(pose_blob[j, 1])
            bb3d = class_bb3d[class_id]
            x3d = np.ones((4, 8), dtype=np.float32)
            x3d[0:3, :] = bb3d
            
            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(pose_blob[j, 6:10])
            RT[:, 3] = pose_blob[j, 10:]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

            x1 = np.min(x2d[0, :])
            x2 = np.max(x2d[0, :])
            y1 = np.min(x2d[1, :])
            y2 = np.max(x2d[1, :])
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3))

        # show depth image
        if depth_blob is not None and len(depth_blob) > 0:
            depth = depth_blob[i, :, :, 0]
            fig.add_subplot(232)
            plt.imshow(abs(depth))

        # show label
        label = label_blob[i, :, :, :]
        height = label.shape[0]
        width = label.shape[1]
        num_classes = label.shape[2]
        l = np.zeros((height, width), dtype=np.int32)
        if cfg.TRAIN.VERTEX_REG:
            vertex_target = vertex_target_blob[i, :, :, :]
            center = np.zeros((height, width, 3), dtype=np.float32)
        for k in xrange(num_classes):
            index = np.where(label[:,:,k] > 0)
            l[index] = k
            if cfg.TRAIN.VERTEX_REG and len(index[0]) > 0 and k > 0:
                center[index[0], index[1], :] = vertex_target[index[0], index[1], 3*k:3*k+3]
        fig.add_subplot(233)
        if cfg.TRAIN.VERTEX_REG:
            plt.imshow(l)
            fig.add_subplot(234)
            plt.imshow(center[:,:,0])
            fig.add_subplot(235)
            plt.imshow(center[:,:,1])
            fig.add_subplot(236)
            plt.imshow(center[:,:,2])
        else:
            plt.imshow(l)

        plt.show()

