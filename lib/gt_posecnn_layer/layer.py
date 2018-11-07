# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""The data layer used during training to train a FCN for single frames.
"""

from fcn.config import cfg
# from gt_single_data_layer.layer import GtSingleDataLayer
from gt_posecnn_layer.minibatch import get_minibatch
import numpy as np
# from utils.voxelizer import Voxelizer

class GtPoseCNNLayer(object):
    def __init__(self, roidb, num_classes, extents, points, symmetry):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        self._extents = extents;
        # self._voxelizer = Voxelizer(cfg.TRAIN.GRID_SIZE, num_classes)
        self._points = points
        self._symmetry = symmetry

        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH

        return db_inds
        
    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]

        is_symmetric = True
        return get_minibatch(minibatch_db, self._num_classes, self._extents, self._points, self._symmetry, is_symmetric)


    def forward(self, iter_):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        return blobs
