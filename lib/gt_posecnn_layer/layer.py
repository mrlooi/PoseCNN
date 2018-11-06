# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""The data layer used during training to train a FCN for single frames.
"""

from fcn.config import cfg
from gt_single_data_layer.layer import GtSingleDataLayer
from gt_posecnn_layer.minibatch import get_minibatch
import numpy as np
# from utils.voxelizer import Voxelizer

class GtPoseCNNLayer(GtSingleDataLayer):
    def __init__(self, roidb, num_classes, extents):
        super(GtPoseCNNLayer, self).__init__(roidb, num_classes, extents)

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes, self._extents)

