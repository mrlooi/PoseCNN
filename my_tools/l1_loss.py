import numpy as np
import tensorflow as tf

import torch

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

def smooth_l1_loss_vertex_pth(vertex_pred, vertex_targets, vertex_weights, sigma=1.0):

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
    loss = torch.div( torch.sum(in_loss), torch.sum(vertex_weights).float() )
    return loss


def T(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, requires_grad=True)
def FT(x):
    return T(x)
def IT(x):
    return T(x, torch.int32)

vert_p = np.arange(100).reshape((5,20)).astype(np.float32)
vert_t = np.arange(100).reshape((5,20)).astype(np.float32) + 1
vert_w = np.ones((5,20)).astype(np.float32)


def get_error(d1, d2):
    e = np.abs(d1-d2)
    print(np.sum(e))

# TF HERE
tvp = tf.Variable(vert_p)
tvt = tf.Variable(vert_t)
tvw = tf.Variable(vert_w)
loss_tf = smooth_l1_loss_vertex(tvp, tvt, tvw)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
lt = sess.run(loss_tf)
tvpg, tvtg, tvwg = sess.run(tf.gradients(loss_tf, [tvp,tvt,tvw]))

# PYTORCH HERE
fvp = FT(vert_p)
fvt = FT(vert_t)
fvw = FT(vert_w)
lp = smooth_l1_loss_vertex_pth(fvp, fvt, fvw)
lp.backward()
fvpg = fvp.grad.numpy()
fvtg = fvt.grad.numpy()
fvwg = fvw.grad.numpy()

get_error(tvpg, fvpg)
get_error(tvtg, fvtg)
get_error(tvwg, fvwg)



# WORKS
x = FT([4])
z = (x**2).float()  # works with .double() as well, but not int()/long()!
# z = (x**2).int()    # THIS WILL FAIL ON THE BACKWARD PASS, perhaps since int destroys integrity of float values
z += x.clone().detach_()**2  # MAKE SURE clone first then detach, otherwise all previous grads will be detached from tensor
z.backward()
x.grad  # 6
x.grad.data.zero_()  # otherwise will accumulate. this step is usually done across vars via optimizer.zero_grad()

# WORKS
x = IT([3])
z = (x**2).float()  # WORKS HERE (perhaps since double/float maintains integrity of int)
z.backward()
x.grad


xph = tf.placeholder(tf.float32, shape=[None])
# z_tf = tf.stop_gradient(tf.to_float(xph**2))
z_tf = tf.to_float(xph**2)   # SAME AS pytorch .float(), gradient WORKS 
sess.run(tf.gradients(z_tf, xph), feed_dict={xph:[3]})
