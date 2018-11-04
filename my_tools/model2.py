
import tensorflow as tf

import _init_paths
from network import Network
import numpy as np

class vgg_test_net(Network):
    def __init__(self, num_classes, num_units, trainable=False, is_train=False):
        # self.inputs = []
        self.num_classes = num_classes
        self.num_units = num_units
        self.trainable = trainable
        self.scale = 1.0
        self.threshold_label = 1.0

        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.gt_label_2d = tf.placeholder(tf.int32, shape=[None, None, None])
        self.keep_prob = tf.placeholder(tf.float32)

        # needed during training, for hough voting layer
        self.meta_data = tf.placeholder(tf.float32, shape=[None, 48])
        self.poses = tf.placeholder(tf.float32, shape=[None, 13])
        self.extents = tf.placeholder(tf.float32, shape=[num_classes, 3])

        # if vote_threshold < 0, only detect single instance (default). 
        # Otherwise, multiple instances are detected if hough voting score larger than the threshold
        self.is_train = is_train
        self.vote_threshold = -1.0
        self.skip_pixels = 10
        self.vote_percentage = 0.02

        # if self.is_train:
        #     self.points = tf.placeholder(tf.float32, shape=[num_classes, None, 3])
        #     self.symmetry = tf.placeholder(tf.float32, shape=[num_classes])


        queue_size = 25
        q = tf.FIFOQueue(queue_size, [tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32])
        self.enqueue_op = q.enqueue([self.data, self.gt_label_2d, self.keep_prob, self.poses, self.extents, self.meta_data])
        data, gt_label_2d, self.keep_prob_queue, poses, extents, meta_data = q.dequeue()

        self.layers = dict({'data': data, 'gt_label_2d': gt_label_2d, 'poses': poses, 'extents': extents, 'meta_data': meta_data})

        self.close_queue_op = q.close(cancel_pending_enqueues=True)
        self.queue_size = q.size()

        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', c_i=3, trainable=self.trainable)
             .conv(3, 3, 64, 1, 1, name='conv1_2', c_i=64, trainable=self.trainable)
             .max_pool(2, 2, 2, 2, name='pool1')  # down x2
             .conv(3, 3, 128, 1, 1, name='conv2_1', c_i=64, trainable=self.trainable)
             .conv(3, 3, 128, 1, 1, name='conv2_2', c_i=128, trainable=self.trainable)
             .max_pool(2, 2, 2, 2, name='pool2')  # down x4
             .conv(3, 3, 256, 1, 1, name='conv3_1', c_i=128, trainable=self.trainable)
             .conv(3, 3, 256, 1, 1, name='conv3_2', c_i=256, trainable=self.trainable)
             .conv(3, 3, 256, 1, 1, name='conv3_3', c_i=256, trainable=self.trainable)
             .max_pool(2, 2, 2, 2, name='pool3')  # down x8
             .conv(3, 3, 512, 1, 1, name='conv4_1', c_i=256, trainable=self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv4_2', c_i=512, trainable=self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv4_3', c_i=512, trainable=self.trainable)
             .max_pool(2, 2, 2, 2, name='pool4')  # down x16
             .conv(3, 3, 512, 1, 1, name='conv5_1', c_i=512, trainable=self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv5_2', c_i=512, trainable=self.trainable)
             .conv(3, 3, 512, 1, 1, name='conv5_3', c_i=512, trainable=self.trainable))

        """
        SEMANTIC MASK LAYER
        """
        (self.feed('conv4_3')
             .conv(1, 1, self.num_units, 1, 1, name='score_conv4', c_i=512))

        (self.feed('conv5_3')
             .conv(1, 1, self.num_units, 1, 1, name='score_conv5', c_i=512)
             .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv5', trainable=False))  # down x16 to down x8

        (self.feed('score_conv4', 'upscore_conv5')
             .add(name='add_score')
             .dropout(self.keep_prob_queue, name='dropout')
             .deconv(int(16*self.scale), int(16*self.scale), self.num_units, int(8*self.scale), int(8*self.scale), name='upscore', trainable=False)) # down x8 to original

        (self.feed('upscore')
             .conv(1, 1, self.num_classes, 1, 1, name='score', c_i=self.num_units))  

        (self.feed('score')
             .softmax(3, name='prob_normalized')   # per pixel softmax
             .argmax_2d(name='label_2d'))  # per-pixel max to get the predicted per-pixel class label

        # (self.feed('prob_normalized', 'gt_label_2d')
        #      .hard_label(threshold=self.threshold_label, name='gt_label_weight'))  # just setting all the gt pixels to ones...

        """
        VERTEX REG LAYER
        """
        (self.feed('conv4_3')
             .conv(1, 1, 128, 1, 1, name='score_conv4_vertex', relu=False, c_i=512))

        (self.feed('conv5_3')
             .conv(1, 1, 128, 1, 1, name='score_conv5_vertex', relu=False, c_i=512)
             .deconv(4, 4, 128, 2, 2, name='upscore_conv5_vertex', trainable=False))    # down x16 to down x8
        
        (self.feed('score_conv4_vertex', 'upscore_conv5_vertex')
             .add(name='add_score_vertex')
             .dropout(self.keep_prob_queue, name='dropout_vertex')
             .deconv(int(16*self.scale), int(16*self.scale), 128, int(8*self.scale), int(8*self.scale), name='upscore_vertex', trainable=False)  # down x8 to original
             .conv(1, 1, 3 * self.num_classes, 1, 1, name='vertex_pred', relu=False, c_i=128))  # regress x direction to center, y direction to center, z distance of center per class

        # hough voting layer after vertex pred
        (self.feed('label_2d', 'vertex_pred', 'extents', 'meta_data', 'poses')
             .hough_voting_gpu(self.is_train, self.vote_threshold, self.vote_percentage, self.skip_pixels, name='hough'))

        rois, poses_init, poses_target, poses_weight = self.get_output('hough')[:4]
        self.layers['rois'] = rois
        self.layers['poses_init'] = poses_init
        self.layers['poses_target'] = poses_target
        self.layers['poses_weight'] = poses_weight

        """
        POSE REG LAYER
        """
        (self.feed('conv5_3', 'rois')
             .roi_pool(7, 7, 1.0 / 16.0, 0, name='roi_pool1'))
             #.crop_pool_new(16.0, pool_size=7, name='roi_pool1'))
             
        (self.feed('conv4_3', 'rois')
             .roi_pool(7, 7, 1.0 / 8.0, 0, name='roi_pool2'))

        (self.feed('roi_pool1', 'roi_pool2')
             .add(name='pool_score')
             .fc(4096, height=7, width=7, channel=512, name='fc6')
             .dropout(self.keep_prob_queue, name='drop6')
             .fc(4096, num_in=4096, name='fc7')
             .dropout(self.keep_prob_queue, name='drop7')
             .fc(4 * self.num_classes, relu=False, name='fc8')
             .tanh(name='poses_tanh'))

        # (self.feed('poses_tanh', 'poses_weight')
        #      .multiply(name='poses_mul')
        #      .l2_normalize(dim=1, name='poses_pred'))

        # if self.is_train:
        #     (self.feed('poses_pred', 'poses_target', 'poses_weight', 'points', 'symmetry')
        #          .average_distance_loss(margin=0.01, name='loss_pose'))

        
    def check_output_shapes(self, sess, feed_dict, keyword=None):
        sess.run(self.enqueue_op, feed_dict=feed_dict)
        layer_names = sorted([k for k in self.layers.keys()]) # if keyword in k])
        if type(keyword) == str:
            layer_names = [k for k in layer_names if keyword in k]
        layer_ops = [self.layers[l] for l in layer_names]
        values = sess.run(layer_ops)
        for ix,l in enumerate(layer_names):
            print("%s: %s"%(l, str(values[ix].shape)))

    def get_outputs(self, sess, feed_dict, output_names):
        sess.run(self.enqueue_op, feed_dict=feed_dict)
        layer_ops = [self.layers[l] for l in output_names]
        return sess.run(layer_ops)

class test_net(Network):
    def __init__(self, num_classes, num_units, trainable=False):
        # self.inputs = []
        self.num_classes = num_classes

        self.data = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])
        self.gt_label_2d = tf.placeholder(tf.int32, shape=[None, None, None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.trainable = trainable

        self.threshold_label = 1
        self.layers = {'data': self.data, 'gt_label_2d': self.gt_label_2d}
        self.setup()

    def setup(self):
        # (self.feed('data')
        #      .conv(1, 1, self.num_classes, 1, 1, name='conv1_1', c_i=3, trainable=self.trainable))
        (self.feed('data')
            .softmax_high_dimension(self.num_classes, name='prob_normalized')   # per pixel softmax
            .argmax_2d(name='label_2d'))
        (self.feed('prob_normalized', 'gt_label_2d')
             .hard_label(threshold=self.threshold_label, name='gt_label_weight'))

    # def run(self, sess, im_blob, gt_label_2d_blob):
    #     feed_dict = {self.data : im_blob, self.gt_label_2d : gt_label_2d_blob}
    #     return sess.run(, feed_dict=feed_dict)

class hough_net(Network):
    def __init__(self, num_classes, is_train=False):
        im_width = 640
        im_height = 480

        self.num_classes = num_classes
 
        self.label_2d = tf.placeholder(tf.int32, shape=[None, None, None]) # batch, height, width
        self.vertex_pred = tf.placeholder(tf.float32, shape=[None, None, None, num_classes * 3]) # batch, height, width, N*3
        self.meta_data = tf.placeholder(tf.float32, shape=[48])
        self.extents = tf.placeholder(tf.float32, shape=[num_classes, 3])
        self.poses = tf.placeholder(tf.float32, shape=[None, 13])

        self.is_train = is_train
        self.vote_threshold = -1.0
        self.vote_percentage = 0.02
        self.skip_pixels = 20

        self.layers = dict({'label_2d': self.label_2d, 'vertex_pred': self.vertex_pred, 'extents': self.extents, 'poses': self.poses, 'meta_data': self.meta_data})

        self.setup()

    def setup(self):
        (self.feed('label_2d', 'vertex_pred', 'extents', 'meta_data', 'poses')
             .hough_voting_gpu(self.is_train, self.vote_threshold, self.vote_percentage, self.skip_pixels, name='hough'))

    def test(self, sess):
        im_scale = 1

        # extents blob
        extent_file = "data/LOV/extents.txt"
        extents = np.zeros((self.num_classes, 3), dtype=np.float32)
        extents[1:, :] = np.loadtxt(extent_file)

        # meta blob
        K = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])
        factor_depth = 10000.0
        meta_data = dict({'intrinsic_matrix': K, 'factor_depth': factor_depth})
        K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        mdata = np.zeros(48, dtype=np.float32)
        mdata[0:9] = K.flatten()
        mdata[9:18] = Kinv.flatten()

        # label2d blob
        label_2d_file = "data/demo_images/000005-label2d.npy"
        label_2d = np.load(label_2d_file)

        # vertex pred blob
        vertex_pred_file = "data/demo_images/000005-vert_pred.npy"
        vertex_pred = np.load(vertex_pred_file)
        assert label_2d.shape == vertex_pred.shape[:3]
        assert vertex_pred.shape[-1] == self.num_classes * 3

        # pose blob
        pose_blob = np.zeros((len(label_2d), 13), dtype=np.float32)

        feed_dict = {self.label_2d: label_2d, self.vertex_pred: vertex_pred, self.poses: pose_blob, self.extents: extents, self.meta_data: mdata}
        hough = sess.run(self.get_output('hough'), feed_dict)
        return hough

class pose_reg_net(Network):
    def __init__(self, num_classes):
        im_width = 640
        im_height = 480

        self.num_classes = num_classes
 
        # self.roi_pool1 = tf.placeholder(tf.float32, shape=[None, None, None]) 
        # self.roi_pool2 = tf.placeholder(tf.float32, shape=[None, None, None, num_classes * 3]) 
        self.rois = tf.placeholder(tf.float32, shape=[None, 7]) 
        self.conv5_3 = tf.placeholder(tf.float32, shape=[None, 30, 40, 512])
        self.conv4_3 = tf.placeholder(tf.float32, shape=[None, 60, 80, 512])

        self.layers = dict({'rois': self.rois, 'conv5_3': self.conv5_3, 'conv4_3': self.conv4_3})

        self.setup()

    def setup(self):
        (self.feed('conv5_3', 'rois')
             .roi_pool(7, 7, 1.0 / 16.0, 0, name='roi_pool1'))
             #.crop_pool_new(16.0, pool_size=7, name='roi_pool1'))
             
        (self.feed('conv4_3', 'rois')
             .roi_pool(7, 7, 1.0 / 8.0, 0, name='roi_pool2'))

        (self.feed('roi_pool1', 'roi_pool2')
             .add(name='pool_score')
             .fc(4096, height=7, width=7, channel=512, name='fc6')
             .fc(4096, num_in=4096, name='fc7')
             .fc(4 * self.num_classes, relu=False, name='fc8')
             .tanh(name='poses_tanh'))



if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    gpu_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    cpu_config = tf.ConfigProto(device_count = {'GPU': 0})

    model_file = "data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt"

    def hough_net_demo():
        sess = tf.Session(config=gpu_config)
        sess.run(tf.global_variables_initializer())

        net = hough_net(22)
        hough = net.test(sess)
        return hough

    # hough_outputs = hough_net_demo()
    # rois = hough_outputs[0]

    def vgg_net_demo():
        num_classes = 22
        height = 480
        width = 640

        im_blob = np.ones((1,height,width,3)) # num_classes))

        net = vgg_test_net(num_classes, 64)    
        rand_blob = np.random.randint(0, 10, size=(1,height,width,num_classes))
        label_blob = np.random.randint(0, num_classes, size=(1, height, width))
        extents_blob = np.zeros((num_classes, 3), dtype=np.float32)
        pose_blob = np.zeros((1, 13), dtype=np.float32)
        meta_blob = np.zeros((1,1,1,48), dtype=np.float32)

        feed_dict = {net.data: im_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, net.extents: extents_blob, net.poses: pose_blob, net.meta_data: meta_blob}

        sess = tf.Session(config=gpu_config)
        saver = tf.train.Saver()
        saver.restore(sess, model_file)
        # sess.run(tf.global_variables_initializer())

        net.check_output_shapes(sess, feed_dict, 'conv')
        # rois, poses_init = net.get_outputs(sess, feed_dict, ['rois','poses_init'])

        roi_pool1, roi_pool2, rois, conv5_3 = net.get_outputs(sess, feed_dict, ['roi_pool1', "roi_pool2", "rois", "conv5_3"])

    # def pose_reg_demo():
    num_classes = 22
    net = pose_reg_net(num_classes)

    sess = tf.Session(config=gpu_config)
    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    rois = np.load("rois.npy")
    conv5_3 = np.load("conv5_3.npy")
    conv4_3 = np.load("conv4_3.npy")
    feed_dict = {net.rois: rois, net.conv4_3: conv4_3, net.conv5_3: conv5_3}
    rp1, rp2, pool_score, fc6, fc7, fc8, poses_tanh = sess.run([net.get_output('roi_pool1'), net.get_output('roi_pool2'), net.get_output('pool_score'), 
            net.get_output('fc6'),net.get_output('fc7'),net.get_output('fc8'), net.get_output('poses_tanh')], feed_dict)
    rp1_tf = rp1[0]
    rp2_tf = rp2[0]

    # def get_error(pth_l, tf_l):
    #     x = pth_l.data.cpu().numpy()
    #     e = np.abs(x-tf_l)
    #     print(np.sum(e))
    def get_error(d1, d2):
        # x = pth_l.data.cpu().numpy()
        e = np.abs(d1-d2)
        print(np.sum(e))

    def get_pth_rois(rois_raw):
        r = np.zeros((len(rois_raw),5), dtype=np.float32)
        r[:,0] = rois_raw[:,0]
        r[:,1:] = rois_raw[:,2:-1]
        return r

    from convert_to_pth import PoseCNN
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    FCT = torch.cuda.FloatTensor
    model = PoseCNN(64, num_classes)
    model.load_state_dict(torch.load("posecnn.pth"))
    model.cuda()
    model.eval()

    pth_rois = get_pth_rois(rois)

    rp1 = model.roi_pool1(FCT(np.transpose(conv5_3, [0,3,1,2])), FCT(pth_rois))
    rp2 = model.roi_pool2(FCT(np.transpose(conv4_3, [0,3,1,2])), FCT(pth_rois))

    get_error(rp1.data.cpu().numpy(), np.transpose(rp1_tf, [0,3,1,2]))
    get_error(rp2.data.cpu().numpy(), np.transpose(rp2_tf, [0,3,1,2]))

    rp = torch.add(rp1, rp2)
    rp_flat = rp.view(-1, 7*7*512)
    pose_reg = model.poses_fc(rp_flat)
    get_error(F.tanh(pose_reg).data.cpu().numpy(), poses_tanh)

