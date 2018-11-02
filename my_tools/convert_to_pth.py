import numpy as np

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import vgg16

class PoseCNN(nn.Module):
    def __init__(self, num_units, num_classes):
        super(PoseCNN, self).__init__()

        self.num_units = num_units
        self.num_classes = num_classes

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
        softmax = F.softmax(score)
        label2d = torch.argmax(softmax, dim=1)
        return softmax, label2d

    def vertex_reg_layer(self, conv4, conv5):
        sc_conv4 = self.score_conv4_vertex(conv4)
        sc_conv5 = self.score_conv5_vertex(conv5)
        upsc_conv5 = self.upscore_conv5_vertex(sc_conv5)
        add_score = torch.add(sc_conv4, upsc_conv5)
        upscore = self.upscore_vertex(add_score)
        vertex_pred = self.vertex_pred(upscore)
        return vertex_pred

    def forward(self, x):
        c4, c5 = self.features(x)
        softmax, mask = self.semantic_mask_layer(c4, c5)
        vertex_pred = self.vertex_reg_layer(c4, c5)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return mask, vertex_pred

def test_model(model, num_classes):
    import cv2
    from demo import _get_image_blob
    from utils.blob import pad_im, unpad_im # im_list_to_blob
    from vis_utils import extract_vertmap

    model.eval()

    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

    im_file = 'data/demo_images/000001-color.png'
    img = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)
    im = pad_im(img, 16)    
    im_blob, _, _ = _get_image_blob(im, 1, PIXEL_MEANS)
    im_blob = torch.FloatTensor(np.transpose(im_blob, [0,3,1,2]))

    mask, vert_pred = model.forward(im_blob.cuda())
    m = mask.data.cpu().numpy()[0]
    vp = vert_pred.data.cpu().numpy()[0]

    labels = unpad_im(m, 16)

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
  
if __name__ == '__main__':
    from model2 import vgg_test_net

    model_file = "data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt"
    num_classes = 22
    num_units = 64
    height = 480
    width = 640

    net = vgg_test_net(num_classes, num_units)#, threshold_label, voting_threshold, vertex_reg_2d, pose_reg, trainable, is_train)

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    saver.restore(sess, model_file)

    def load_model(model):
        state_dict = model.state_dict()
        sdk = state_dict.keys()

        # nl = sorted([k for k in net.layers.keys() if k.startswith('conv')])
        conv_vars = [v.name for v in tf.global_variables() if v.name.startswith('conv')]
        t_conv_vars = sdk[:26]

        for ix,k in enumerate(t_conv_vars):
            tf_k = conv_vars[ix]
            tf_d = sess.run(tf_k)  # shape: k_h, k_w, in_channels, out_channels
            pth_d = state_dict[k]  # shape: out_channels, in_channels, k_h, k_w
            print("%s -> tf shape: %s, pytorch shape: %s"%(tf_k, str(tf_d.shape), str(pth_d.shape)))
            
            if 'bias' not in k:
                tf_d = np.transpose(tf_d, [3,2,0,1])
            state_dict[k] = torch.Tensor(tf_d)
            # assert(tuple(state_dict[k].shape) == tf_d.shape)

        score_vars = [v.name for v in tf.global_variables() if 'score' in v.name or 'vertex_pred' in v.name]
        score_vars = score_vars
        t_score_vars = sdk[26:]
        t_score_vars = [v for v in t_score_vars if not ('upscore' in v and v.endswith('bias'))]

        for ix,k in enumerate(t_score_vars):
            tf_k = score_vars[ix]
            tf_d = sess.run(tf_k)  # shape: k_h, k_w, in_channels, out_channels
            pth_d = state_dict[k]  # shape: out_channels, in_channels, k_h, k_w
            print("%s -> tf shape: %s, pytorch shape: %s"%(tf_k, str(tf_d.shape), str(pth_d.shape)))
            
            if 'bias' not in k:
                tf_d = np.transpose(tf_d, [3,2,0,1])
            state_dict[k] = torch.Tensor(tf_d)

        model.load_state_dict(state_dict)

    model = PoseCNN(num_units, num_classes)
    load_model(model)
    model.cuda()
    model.eval()

    test_model(model, num_classes)

    rand_blob = np.random.randint(0, 10, size=(1,height,width,num_classes))
    label_blob = np.random.randint(0, num_classes, size=(1, height, width))
    extents_blob = np.zeros((num_classes, 3), dtype=np.float32)
    pose_blob = np.zeros((1, 13), dtype=np.float32)
    meta_blob = np.zeros((1,1,1,48), dtype=np.float32)

    im_blob_tf = np.ones((1,height,width,3)) 
    im_blob = torch.ones(1,3,height,width).cuda()

    feed_dict = {net.data: im_blob_tf, net.gt_label_2d: label_blob, net.keep_prob: 1.0, net.extents: extents_blob, net.poses: pose_blob, net.meta_data: meta_blob}
    conv1_2, conv2_2, conv3_3, conv4_3, conv5_3, score_conv4, upscore_conv5, upscore, score, pn, label_2d, \
            score_conv4_vertex, upscore_conv5_vertex, vertex_pred_tf = net.get_outputs(sess, feed_dict, 
                    ['conv1_2', 'conv2_2','conv3_3', 'conv4_3', 'conv5_3', 'score_conv4', 'upscore_conv5', 'upscore', 'score', 'prob_normalized', 'label_2d',
                     'score_conv4_vertex', 'upscore_conv5_vertex', 'vertex_pred'])

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
        x = pth_l.data.cpu().numpy()[0]
        e = np.abs(x-tf_l)
        print(np.sum(e))

    # get_error(upsc_conv5_vert, upscore_conv5_vertex)
    # get_error(sc_conv4_vert, score_conv4_vertex)
    get_error(vertex_pred, vertex_pred_tf)
