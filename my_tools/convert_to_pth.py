import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import vgg16

FT = torch.FloatTensor
FCT = torch.cuda.FloatTensor

import sys
sys.path.append("my_tools")
sys.path.append("lib/model")
from roi_pooling.modules.roi_pool import _RoIPooling
from hough_voting.modules.hough_voting import HoughVoting
from vgg16 import VGG16
from resnet import ResNet50

def _get_conv_transpose2d(in_cn, out_cn, kernel_size, stride=(1,1), padding=0, trainable=False):
    x = nn.ConvTranspose2d(in_cn, out_cn, kernel_size=kernel_size, stride=stride, padding=padding)
    x.bias.data.zero_()
    # if not trainable:
    #     for p in x.parameters():
    #         p.requires_grad = False
    return x

def _get_conv_transpose2d_by_factor(in_cn, out_cn, factor, trainable=False):
    """
    Maintain output_size = input_size * factor (multiple of 2)
    """
    # stride = int(1.0/spatial_scale)
    assert factor >= 2 and factor % 2 == 0
    stride = factor
    k = stride * 2
    kernel_size = (k,k)
    padding = stride / 2
    stride = (stride, stride)
    return _get_conv_transpose2d(in_cn, out_cn, kernel_size, stride, padding, trainable=trainable)

def _get_fc_layer(in_cn, out_cn):
    x = nn.Linear(in_cn, out_cn)
    x.bias.data.zero_()
    nn.init.normal_(x.weight, 0.0, 0.001)
    return x

class SemanticMaskHead(nn.Module):
    def __init__(self, conv0_dim_in, conv1_dim_in, conv0_spatial_scale, conv1_spatial_scale, dim_out, num_units=64):
        super(SemanticMaskHead, self).__init__()

        inplace = True

        self.score_conv4 = nn.Sequential(
            nn.Conv2d(conv0_dim_in, num_units, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace)
        )   
        self.score_conv5 = nn.Sequential(
            nn.Conv2d(conv1_dim_in, num_units, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace)
        )
        self.upscore_conv5 = _get_conv_transpose2d_by_factor(num_units, num_units, factor=int(conv0_spatial_scale/conv1_spatial_scale), trainable=False)
        self.upscore = _get_conv_transpose2d_by_factor(num_units, num_units, factor=int(1.0/conv0_spatial_scale), trainable=False)
        self.score = nn.Sequential(
            nn.Conv2d(num_units, dim_out, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace)
        )

        self.dim_out = dim_out

    def forward(self, conv0, conv1):
        sc_conv4 = self.score_conv4(conv0)
        sc_conv5 = self.score_conv5(conv1)
        upsc_conv5 = self.upscore_conv5(sc_conv5)
        add_score = torch.add(sc_conv4, upsc_conv5)
        upscore = self.upscore(add_score)
        score = self.score(upscore)
        softmax = F.softmax(score, 1)
        label2d = torch.argmax(softmax, dim=1)
        return score, label2d

class VertexRegHead(nn.Module):
    def __init__(self, conv0_dim_in, conv1_dim_in, conv0_spatial_scale, conv1_spatial_scale, dim_out, num_units=128):
        super(VertexRegHead, self).__init__()

        inplace = True

        self.dim_out = dim_out * 3  # xyz reg

        self.score_conv4_vertex = nn.Conv2d(conv0_dim_in, num_units, kernel_size=(1, 1), stride=(1, 1))
        self.score_conv5_vertex = nn.Conv2d(conv1_dim_in, num_units, kernel_size=(1, 1), stride=(1, 1))
        self.upscore_conv5_vertex = _get_conv_transpose2d_by_factor(num_units, num_units, factor=int(conv0_spatial_scale/conv1_spatial_scale), trainable=False)
        self.upscore_vertex = _get_conv_transpose2d_by_factor(num_units, num_units, factor=int(1.0/conv0_spatial_scale), trainable=False)
        self.vertex_pred = nn.Conv2d(num_units, self.dim_out, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, conv0, conv1):
        sc_conv4 = self.score_conv4_vertex(conv0)
        sc_conv5 = self.score_conv5_vertex(conv1)
        upsc_conv5 = self.upscore_conv5_vertex(sc_conv5)
        add_score = torch.add(sc_conv4, upsc_conv5)
        upscore = self.upscore_vertex(add_score)
        vertex_pred = self.vertex_pred(upscore)
        return vertex_pred

class RoiPoolLayer(nn.Module):
    def __init__(self, conv0_spatial_scale, conv1_spatial_scale, roi_pool_dims=(7,7)):
        super(RoiPoolLayer, self).__init__()    
        if isinstance(roi_pool_dims, int):
            roi_pool_dims = (roi_pool_dims, roi_pool_dims)
        self.roi_pool_dims = roi_pool_dims
    
        h, w = self.roi_pool_dims
        self.roi_pool1 = _RoIPooling(h,w,conv0_spatial_scale) 
        self.roi_pool2 = _RoIPooling(h,w,conv1_spatial_scale) 

    def forward(self, conv0, conv1, rois_raw):
        if rois_raw.shape[0] == 0:
            rois = rois_raw.new(1,5).zero_() #.new(0,5)
        else:
            n, rs = rois_raw.shape
            if rs == 5:
                rois = rois_raw
            elif rs == 7:      # rois_raw from hough_voting: batch_index, cls, x1, y1, x2, y2, max_hough_idx
                rois = rois_raw.new(n,5)
                rois[:,0] = rois_raw[:,0]
                rois[:,1:] = rois_raw[:,2:-1]
            else:
                raise NotImplementedError("ROI dimension must be of shape (N,5) or (N,7), but received shape: (%d,%d)"%(n, rs))

        rp1 = self.roi_pool1(conv0, rois)  # B,C,H,W  (B,512,7,7)
        rp2 = self.roi_pool2(conv1, rois)  # B,C,H,W  (B,512,7,7)
        rp = torch.add(rp1, rp2)
        return rp

class PoseRegHead(nn.Module):
    def __init__(self, dim_in, dim_out, num_units=4096):
        super(PoseRegHead, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out * 4
        self.num_units = num_units

        self.poses_fc1 = _get_fc_layer(self.dim_in, num_units)
        self.poses_fc2 = _get_fc_layer(num_units, num_units)
        self.poses_fc3 = _get_fc_layer(num_units, self.dim_out)

    def forward(self, x, drop_prob=0.0, is_train=False):
        x_flat = x.view(-1, self.dim_in)

        fc1 = self.poses_fc1(x_flat)
        fc1 = F.normalize(fc1, p=2, dim=1)
        fc1 = F.dropout(F.relu(fc1, inplace=True), drop_prob, training=is_train)
        fc2 = self.poses_fc2(fc1)
        fc2 = F.normalize(fc2, p=2, dim=1)
        fc2 = F.dropout(F.relu(fc2, inplace=True), drop_prob, training=is_train)
        fc3 = self.poses_fc3(fc2)
        fc3 = F.normalize(fc3, p=2, dim=1)

        return F.tanh(fc3)

class HoughVoteLayer(nn.Module):
    def __init__(self, dim_in, threshold_vote, threshold_percentage, label_threshold=500, inlier_threshold=0.9, skip_pixels=1, is_train=False):
        super(HoughVoteLayer, self).__init__()

        self.dim_in = dim_in
        self.label_threshold = int(label_threshold)
        self.inlier_threshold = float(inlier_threshold)
        self.threshold_vote = float(threshold_vote)
        self.threshold_percentage = float(threshold_percentage)
        self.skip_pixels = int(skip_pixels)

        self.is_train = is_train

        self.hough_voting = HoughVoting(self.dim_in, self.threshold_vote, self.threshold_percentage, label_threshold=self.label_threshold, 
            inlier_threshold=self.inlier_threshold, skip_pixels=self.skip_pixels, is_train=self.is_train)

    def forward(self, label_2d, vertex_pred, extents, poses, mdata):
        outputs = self.hough_voting(label_2d, vertex_pred.permute((0,2,3,1)).clone(), extents, poses, mdata)
        # rois_raw = outputs[0]

        return outputs

class PoseCNN(nn.Module):
    def __init__(self, num_units, num_classes, label_threshold=500, vote_threshold=-1.0, is_train=False, keep_prob=1.0):
        super(PoseCNN, self).__init__()

        self.num_units = num_units
        self.num_classes = num_classes
        self.is_train = is_train

        self.label_threshold = label_threshold
        self.vote_threshold = vote_threshold
        self.vote_percentage = 0.02
        self.skip_pixels = 20
        self.inlier_threshold = 0.9

        inplace = True

        drop_prob = 1.0 - keep_prob if is_train else 0.0
        self.drop_prob = drop_prob

        # self.features = VGG16()
        self.features = ResNet50()
        dim_outs = self.features.conv_dim_out
        dim_out_0, dim_out_1 = dim_outs[-2:]
        spatial_scales = self.features.conv_spatial_scale
        spatial_scale_0, spatial_scale_1 = spatial_scales[-2:]

        """
        SEMANTIC MASK LAYER
        """
        self.mask_head = SemanticMaskHead(dim_out_0, dim_out_1, spatial_scale_0, spatial_scale_1, num_classes, num_units)

        """
        VERTEX REG LAYER
        """
        self.vertex_head = VertexRegHead(dim_out_0, dim_out_1, spatial_scale_0, spatial_scale_1, num_classes, num_units*2)

        """
        HOUGH VOTE LAYER
        """
        self.hough_voting = HoughVoteLayer(self.num_classes, self.vote_threshold, self.vote_percentage, label_threshold=self.label_threshold, 
            inlier_threshold=self.inlier_threshold, skip_pixels=self.skip_pixels, is_train=self.is_train)

        """
        ROI POOL LAYER
        """
        self.conv_filter_0 = lambda x: x
        self.conv_filter_1 = lambda x: x
        max_num_units = 512
        if dim_out_0 != max_num_units:
            self.conv_filter_0 = nn.Conv2d(dim_out_0, max_num_units, kernel_size=(1, 1), stride=(1, 1))
        if dim_out_1 != max_num_units:
            self.conv_filter_1 = nn.Conv2d(dim_out_1, max_num_units, kernel_size=(1, 1), stride=(1, 1))
        self.roi_pool_dims = (7,7)
        self.roi_pool = RoiPoolLayer(spatial_scale_0,spatial_scale_1,self.roi_pool_dims)

        """
        POSE REG LAYER
        """
        h, w = self.roi_pool_dims
        self.pose_head = PoseRegHead(h*w*max_num_units, num_classes, 1024)

    def forward_image(self, x):
        c4, c5 = self.features(x)
        score, label_2d = self.mask_head(c4, c5)
        vertex_pred = self.vertex_head(c4, c5)

        return score, label_2d, vertex_pred

    def forward(self, x, extents, poses, mdata):
        c4, c5 = self.features(x)
        score, label_2d = self.mask_head(c4, c5)
        vertex_pred = self.vertex_head(c4, c5)
        hough_outputs = self.hough_voting(label_2d, vertex_pred, extents, poses, mdata)
        rois = hough_outputs[0]
        rois_out = self.roi_pool(self.conv_filter_0(c4), self.conv_filter_1(c5), rois)
        pose_reg = self.pose_head(rois_out, self.drop_prob, self.is_train)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return score, label_2d, vertex_pred, hough_outputs, pose_reg

    def load_pretrained(self, model_file):
        self.features.load_pretrained(model_file)

def posecnn_pth_to_tf_mapping():
    return {
        'conv1.0.weight': 'conv1_1/weights:0',
        'conv1.0.bias': 'conv1_1/biases:0',
        'conv1.2.weight': 'conv1_2/weights:0',
        'conv1.2.bias': 'conv1_2/biases:0',
        'conv2.0.weight': 'conv2_1/weights:0',
        'conv2.0.bias': 'conv2_1/biases:0',
        'conv2.2.weight': 'conv2_2/weights:0',
        'conv2.2.bias': 'conv2_2/biases:0',
        'conv3.0.weight': 'conv3_1/weights:0',
        'conv3.0.bias': 'conv3_1/biases:0',
        'conv3.2.weight': 'conv3_2/weights:0',
        'conv3.2.bias': 'conv3_2/biases:0',
        'conv3.4.weight': 'conv3_3/weights:0',
        'conv3.4.bias': 'conv3_3/biases:0',
        'conv4.0.weight': 'conv4_1/weights:0',
        'conv4.0.bias': 'conv4_1/biases:0',
        'conv4.2.weight': 'conv4_2/weights:0',
        'conv4.2.bias': 'conv4_2/biases:0',
        'conv4.4.weight': 'conv4_3/weights:0',
        'conv4.4.bias': 'conv4_3/biases:0',
        'conv5.0.weight': 'conv5_1/weights:0',
        'conv5.0.bias': 'conv5_1/biases:0',
        'conv5.2.weight': 'conv5_2/weights:0',
        'conv5.2.bias': 'conv5_2/biases:0',
        'conv5.4.weight': 'conv5_3/weights:0',
        'conv5.4.bias': 'conv5_3/biases:0',
        'score_conv4.0.weight': 'score_conv4/weights:0',
        'score_conv4.0.bias': 'score_conv4/biases:0',
        'score_conv5.0.weight': 'score_conv5/weights:0',
        'score_conv5.0.bias': 'score_conv5/biases:0',
        'upscore_conv5.weight': 'upscore_conv5/weights:0',
        'upscore.weight': 'upscore/weights:0',
        'score.0.weight': 'score/weights:0',
        'score.0.bias': 'score/biases:0',
        'score_conv4_vertex.weight': 'score_conv4_vertex/weights:0',
        'score_conv4_vertex.bias': 'score_conv4_vertex/biases:0',
        'score_conv5_vertex.weight': 'score_conv5_vertex/weights:0',
        'score_conv5_vertex.bias': 'score_conv5_vertex/biases:0',
        'upscore_conv5_vertex.weight': 'upscore_conv5_vertex/weights:0',
        'upscore_vertex.weight': 'upscore_vertex/weights:0',
        'vertex_pred.weight': 'vertex_pred/weights:0',
        'vertex_pred.bias': 'vertex_pred/biases:0',
        'poses_fc.0.weight': 'fc6/weights:0',
        'poses_fc.0.bias': 'fc6/biases:0',
        'poses_fc.2.weight': 'fc7/weights:0',
        'poses_fc.2.bias': 'fc7/biases:0',
        'poses_fc.4.weight': 'fc8/weights:0',
        'poses_fc.4.bias': 'fc8/biases:0'
    }

def get_random_image_blob():
    from demo import _get_image_blob

    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

    im_file = 'data/demo_images/000005-color.png'
    img = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)
    im_blob, _, _ = _get_image_blob(img, 1, PIXEL_MEANS)
    return img, im_blob

def test_model(model, num_classes):
    from vis_utils import extract_vertmap

    model.eval()

    img, im_blob = get_random_image_blob()
    im_blob = torch.FloatTensor(np.transpose(im_blob, [0,3,1,2]))

    mask, vert_pred = model.forward_image(im_blob.cuda())
    m = mask.data.cpu().numpy()[0]
    vp = vert_pred.data.cpu().numpy()[0]

    labels = m
    vert_map = extract_vertmap(labels, np.transpose(vp, [1,2,0]), num_classes)
    cv2.imshow("centers x", vert_map[:,:,0])
    cv2.imshow("centers y", vert_map[:,:,1])
    cv2.imshow("centers z", vert_map[:,:,2])

    color_m = np.zeros(img.shape, dtype=np.uint8)
    for c in xrange(num_classes-1):
        cls = c + 1
        color = np.random.randint(0,255,size=(3))
        color_m[labels==cls] = color

    cv2.imshow("img", img)
    cv2.imshow("m", color_m)
    cv2.waitKey(0)

def get_meta_info(num_classes):
    im_scale = 1
    batch_sz = 1
    root_dir = "/home/vincent/Documents/deep_learning/PoseCNN/data"
    # extents blob
    extent_file = root_dir + "/LOV/extents.txt"
    extents = np.zeros((num_classes, 3), dtype=np.float32)
    extents[1:, :] = np.loadtxt(extent_file)

    poses = np.zeros((batch_sz, 13), dtype=np.float32)

    K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
    factor_depth = 10000.0
    meta_data = dict({'intrinsic_matrix': K, 'factor_depth': factor_depth})
    K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
    K[2, 2] = 1
    Kinv = np.linalg.pinv(K)
    mdata = np.zeros((batch_sz,48), dtype=np.float32)
    mdata[:,0:9] = K.flatten()
    mdata[:,9:18] = Kinv.flatten()

    return extents, poses, mdata

if __name__ == '__main__':
    import tensorflow as tf

    num_classes = 22
    num_units = 64
    height = 480
    width = 640

    model = PoseCNN(num_units, num_classes)
    # model.load_state_dict(torch.load("posecnn.pth"))

    from model2 import vgg_test_net
    model_file = "data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt"
    net = vgg_test_net(num_classes, num_units, trainable=True, is_train=True) #, threshold_label, voting_threshold, vertex_reg_2d, pose_reg, trainable, is_train)

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    saver.restore(sess, model_file)

    def load_model(model):
        state_dict = model.state_dict()
        sdk = state_dict.keys()

        # conv_vars = [v.name for v in tf.global_variables() if v.name.startswith('conv')]

        from collections import OrderedDict
        mapping = posecnn_pth_to_tf_mapping()

        for ix,k in enumerate(mapping):
            tf_k = mapping[k]
            tf_d = sess.run(tf_k)  # shape: k_h, k_w, in_channels, out_channels
            pth_d = state_dict[k]  # shape: out_channels, in_channels, k_h, k_w
            print("%s -> tf shape: %s, pytorch shape: %s"%(tf_k, str(tf_d.shape), str(pth_d.shape)))
                        
            if tf_k == 'fc6/weights:0':
                # shape: (7*7*512,4096)
                tf_d = np.reshape(tf_d, [7,7,512,4096])  
                tf_d = np.transpose(tf_d, [2,0,1,3])  # change to pytorch, channels first (512,7,7,4096)
                tf_d = np.reshape(tf_d, [7*7*512,4096])  
                tf_d = np.transpose(tf_d, [1,0])
            elif 'bias' not in k:
                if len(tf_d.shape) == 4:   # conv layers
                    tf_d = np.transpose(tf_d, [3,2,0,1])
                elif len(tf_d.shape) == 2:  # fc layers
                    tf_d = np.transpose(tf_d, [1,0])
            state_dict[k] = torch.Tensor(tf_d)

        model.load_state_dict(state_dict)
        # torch.save(state_dict, "posecnn.pth")

    model.load_state_dict(torch.load("posecnn.pth"))
    # load_model(model)
    model.cuda()
    model.eval()

    # # test_model(model, num_classes)

    label_blob = np.random.randint(0, num_classes, size=(1, height, width))
    extents, poses, meta_data = get_meta_info(num_classes)

    img, im_blob_tf = get_random_image_blob()
    im_blob = torch.FloatTensor(np.transpose(im_blob_tf, [0,3,1,2])).cuda()

    feed_dict = {net.data: im_blob_tf, net.gt_label_2d: label_blob, net.keep_prob: 1.0, net.extents: extents, net.poses: poses, net.meta_data: meta_data}
    conv1_2, conv2_2, conv3_3, conv4_3, conv5_3, score_conv4, upscore_conv5, upscore, score, pn, label_2d_tf, \
            score_conv4_vertex, upscore_conv5_vertex, vertex_pred_tf, rp1_tf, rp2_tf, pool_score, fc6 = net.get_outputs(sess, feed_dict, 
                    ['conv1_2', 'conv2_2','conv3_3', 'conv4_3', 'conv5_3', 'score_conv4', 'upscore_conv5', 'upscore', 'score', 'prob_normalized', 'label_2d',
                     'score_conv4_vertex', 'upscore_conv5_vertex', 'vertex_pred', 'roi_pool1', "roi_pool2", "pool_score", 'fc6'])

    # np.save("conv5_3.npy", conv5_3)
    # np.save("conv4_3.npy", conv4_3)

    conv1_2 = np.transpose(conv1_2[0], [2,0,1])
    conv2_2 = np.transpose(conv2_2[0], [2,0,1])
    conv3_3 = np.transpose(conv3_3[0], [2,0,1])
    conv4_3 = np.transpose(conv4_3[0], [2,0,1])
    conv5_3 = np.transpose(conv5_3[0], [2,0,1])
    score_conv4 = np.transpose(score_conv4[0], [2,0,1])
    upscore_conv5 = np.transpose(upscore_conv5[0], [2,0,1])
    upscore = np.transpose(upscore[0], [2,0,1])
    score = np.transpose(score[0], [2,0,1])
    sm_tf = np.transpose(pn[0], [2,0,1])
    score_conv4_vertex = np.transpose(score_conv4_vertex[0], [2,0,1])
    upscore_conv5_vertex = np.transpose(upscore_conv5_vertex[0], [2,0,1])
    vertex_pred_tf = np.transpose(vertex_pred_tf[0], [2,0,1])
    rp1_tf = np.transpose(rp1_tf[0], [0,3,1,2])
    rp2_tf = np.transpose(rp2_tf[0], [0,3,1,2])

    conv1 = model.conv1(im_blob)
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
    sm = F.softmax(sc)
    sc_conv4_vert = model.score_conv4_vertex(conv4)
    sc_conv5_vert = model.score_conv5_vertex(conv5)
    upsc_conv5_vert = model.upscore_conv5_vertex(sc_conv5_vert)
    add_score_vert = torch.add(sc_conv4_vert, upsc_conv5_vert)
    upsc_vert = model.upscore_vertex(add_score_vert)
    vertex_pred = model.vertex_pred(upsc_vert)

    def get_error(pth_l, tf_l):
        x = pth_l.data.cpu().numpy()
        e = np.abs(x-tf_l)
        print(np.sum(e))

    # # get_error(upsc_conv5_vert[0], upscore_conv5_vertex)
    # # get_error(sc_conv4_vert[0], score_conv4_vertex)
    label_2d, vertex_pred = model.forward_image(im_blob)
    # get_error(vertex_pred[0], vertex_pred_tf)
    # get_error(label_2d, label_2d_tf)
    hough_outputs = model.run_hough_voting(label_2d, vertex_pred, FCT(extents), FCT(poses), FCT(meta_data))
    rois_raw = hough_outputs[0]
    print(rois_raw)
    rois = rois_raw.new(len(rois_raw),5).zero_()
    rois[:,0] = rois_raw[:,0]
    rois[:,1:] = rois_raw[:,2:-1]
    rp1 = model.roi_pool1(conv5, rois)
    rp2 = model.roi_pool2(conv4, rois)
