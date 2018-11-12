import numpy as np
import cv2

import tensorflow as tf
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

class PoseCNN(nn.Module):
    def __init__(self, num_units, num_classes, label_threshold=500, vote_threshold=-1.0, is_train=False):
        super(PoseCNN, self).__init__()

        self.num_units = num_units
        self.num_classes = num_classes
        self.is_train = False

        self.label_threshold = label_threshold
        self.vote_threshold = vote_threshold
        self.vote_percentage = 0.02
        self.skip_pixels = 20
        self.inlier_threshold = 0.9

        inplace = True

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace),
        )

        # self.features = nn.Sequential(
        #     self.conv1, self.max_pool2d, 
        #     self.conv2, self.max_pool2d, 
        #     self.conv3, self.max_pool2d,
        #     self.conv4, self.max_pool2d,
        #     self.conv5
        # )
        
        """
        SEMANTIC MASK LAYER
        """
        self.score_conv4 = nn.Sequential(
            nn.Conv2d(512, num_units, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace)
        )   
        self.score_conv5 = nn.Sequential(
            nn.Conv2d(512, num_units, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace)
        )
        self.upscore_conv5 = self._get_conv_transpose2d(num_units, num_units, kernel_size=(4,4), stride=(2,2), padding=(1,1), trainable=False)
        self.upscore = self._get_conv_transpose2d(self.num_units, self.num_units, kernel_size=(16,16), stride=(8,8), padding=(4,4), trainable=False)
        self.score = nn.Sequential(
            nn.Conv2d(self.num_units, self.num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace)
        )

        """
        VERTEX REG LAYER
        """
        self.score_conv4_vertex = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.score_conv5_vertex = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.upscore_conv5_vertex = self._get_conv_transpose2d(128, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1), trainable=False)
        self.upscore_vertex = self._get_conv_transpose2d(128, 128, kernel_size=(16,16), stride=(8,8), padding=(4,4), trainable=False)
        self.vertex_pred = nn.Conv2d(128, 3 * self.num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.hough_voting = HoughVoting(self.num_classes, self.vote_threshold, self.vote_percentage, label_threshold=self.label_threshold, 
            inlier_threshold=self.inlier_threshold, skip_pixels=self.skip_pixels, is_train=self.is_train)

        """
        POSE REG LAYER
        """
        self.roi_pool_dims = (7,7)
        h, w = self.roi_pool_dims
        self.roi_pool1 = _RoIPooling(h,w,1.0/16)  # from conv5_3
        self.roi_pool2 = _RoIPooling(h,w,1.0/8)  # from conv4_3
        self.poses_fc = nn.Sequential(
            nn.Linear(h*w*512,4096),
            nn.ReLU(inplace),
            nn.Linear(4096,4096),
            nn.ReLU(inplace),
            nn.Linear(4096,4 * self.num_classes)
        )


    def features(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.max_pool2d(conv1))
        conv3 = self.conv3(self.max_pool2d(conv2))
        conv4 = self.conv4(self.max_pool2d(conv3))
        conv5 = self.conv5(self.max_pool2d(conv4))
        return conv4, conv5

    def _get_conv_transpose2d(self, in_cn, out_cn, kernel_size, stride=(1,1), padding=0, trainable=False):
        x = nn.ConvTranspose2d(in_cn, out_cn, kernel_size=kernel_size, stride=stride, padding=padding)
        x.bias.data.zero_()
        if not trainable:
            for p in x.parameters():
                p.requires_grad = False
        return x

    def semantic_mask_layer(self, conv4, conv5):
        sc_conv4 = self.score_conv4(conv4)
        sc_conv5 = self.score_conv5(conv5)
        upsc_conv5 = self.upscore_conv5(sc_conv5)
        add_score = torch.add(sc_conv4, upsc_conv5)
        upscore = self.upscore(add_score)
        score = self.score(upscore)
        softmax = F.softmax(score, 1)
        label2d = torch.argmax(softmax, dim=1)
        return score, label2d

    def vertex_reg_layer(self, conv4, conv5):
        sc_conv4 = self.score_conv4_vertex(conv4)
        sc_conv5 = self.score_conv5_vertex(conv5)
        upsc_conv5 = self.upscore_conv5_vertex(sc_conv5)
        add_score = torch.add(sc_conv4, upsc_conv5)
        upscore = self.upscore_vertex(add_score)
        vertex_pred = self.vertex_pred(upscore)
        return vertex_pred

    def run_hough_voting(self, label_2d, vertex_pred, extents, poses, mdata):
        # need to permute vertex_pred from (B,C,H,W) to (B,H,W,C)
        return self.hough_voting(label_2d, vertex_pred.permute((0,2,3,1)).clone(), extents, poses, mdata)

    def pose_reg_layer(self, conv4, conv5, rois_raw):
        # rois_raw from hough_voting: batch_index, cls, x1, y1, x2, y2, max_hough_idx
        h, w = self.roi_pool_dims
        if rois_raw.shape[0] == 0:
            rois = rois_raw.new(1,5).zero_() #.new(0,5)
        else:
            n, rs = rois_raw.shape
            assert rs == 7
            rois = rois_raw.new(n,5)
            rois[:,0] = rois_raw[:,0]
            rois[:,1:] = rois_raw[:,2:-1]

        rp1 = self.roi_pool1(conv5, rois)  # B,C,H,W  (B,512,7,7)
        rp2 = self.roi_pool2(conv4, rois)  # B,C,H,W  (B,512,7,7)
        rp = torch.add(rp1, rp2)
        # print(rp.shape)
        rp_flat = rp.view(-1, h*w*512)
        fc = self.poses_fc(rp_flat)
        return F.tanh(fc)

    def forward_image(self, x):
        c4, c5 = self.features(x)
        _, label_2d = self.semantic_mask_layer(c4, c5)
        vertex_pred = self.vertex_reg_layer(c4, c5)

        return label_2d, vertex_pred

    def forward(self, x, extents, poses, mdata):
        c4, c5 = self.features(x)
        score, label_2d = self.semantic_mask_layer(c4, c5)
        vertex_pred = self.vertex_reg_layer(c4, c5)
        hough_outputs = self.run_hough_voting(label_2d, vertex_pred, extents, poses, mdata)
        rois = hough_outputs[0]
        pose_reg = self.pose_reg_layer(c4, c5, rois)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return score, label_2d, vertex_pred, hough_outputs, pose_reg

    def load_pretrained(self, model_file):
        m = torch.load(model_file)
        mk = m.keys()[:26]
        sd = self.state_dict()
        sdk = sd.keys()

        print("Loading pretrained model %s..."%(model_file))
        for ix,k in enumerate(mk):
            md = m[k]
            sk = sdk[ix]
            d = sd[sk]
            assert d.shape == md.shape
            print("%s -> %s [%s]"%(k, sk, str(d.shape)))
            sd[sk] = md
        self.load_state_dict(sd)
        print("Loaded pretrained model %s"%(model_file))

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
    root_dir = "/home/vincent/Documents/py/ml/PoseCNN/data"
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

    num_classes = 22
    num_units = 64
    height = 480
    width = 640

    model = PoseCNN(num_units, num_classes)
    # model.load_state_dict(torch.load("posecnn.pth"))

    from model2 import vgg_test_net
    model_file = "data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt"
    net = vgg_test_net(num_classes, num_units)#, threshold_label, voting_threshold, vertex_reg_2d, pose_reg, trainable, is_train)

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

    load_model(model)
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
