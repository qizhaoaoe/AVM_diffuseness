import tensorflow as tf
import numpy as np

def kaiming_initializer_3d(kernel_size, in_ch):
    std = np.sqrt(2./(kernel_size**3*in_ch))
    return tf.truncated_normal_initializer(stddev=std)

def normal_initializer(std):
    return tf.truncated_normal_initializer(stddev=std)

def get_initializer(init_method, kernel_size, in_ch, std=0.1):
    # in_ch is the channel number of input
    if init_method == 'normal':
        return tf.truncated_normal_initializer(stddev=std)
    elif init_method == 'kaiming':
        return kaiming_initializer_3d(kernel_size, in_ch)
    return None


def resize_image3d(input_layer, scale_factor=(2,2,2)):
    '''
    scale factor is three item tuple or list, scale factor in direction of x, y, z
    '''
    df, hf, wf = scale_factor
    shape = input_layer.get_shape()
    b, d, h, w, c = shape
    x = tf.reshape(input_layer, [b, d, h, w*c])
    x = tf.image.resize_images(x, [d*df, h*hf], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = tf.reshape(x, [b, d*df, h*hf, w, c])
    x = tf.transpose(x, [0, 3, 2, 1, 4])
    x = tf.reshape(x, [b, w, h*hf, d*df*c])
    x = tf.image.resize_images(x, [w*wf, h*hf], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = tf.reshape(x, [b, w*wf, h*hf,d*df, c])
    x = tf.transpose(x, [0,3,2,1,4])
    return x


def z_score_normalization(x):
    # x dims : [batch, h, w, d]
    mean, var = tf.nn.moments(x,axes=[1,2,3])
    res = (x - mean)/var
    return res

def linear_normalization(x):
    minv = tf.reduce_min(x)
    maxv = tf.reduce_max(x)
    return (x - minv) / (maxv - minv)


def Group_norm3d(x, G=16, name_scope='', reuse=tf.AUTO_REUSE, esp=1e-5):
    with tf.variable_scope(name_scope, reuse=reuse):
        # normalize
        # tranpose: [bs, h, w, d, c] to [bs, c, d,h, w] following the paper
        x = tf.transpose(x, [0, 4, 3, 1, 2])
        N, C, D, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, D, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4, 5])
        mean = tf.reshape(mean, [-1, G, 1, 1, 1, 1])
        var = tf.reshape(var, [-1, G, 1, 1, 1, 1])
        x = (x - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        gamma = tf.get_variable('gamma', [C],
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C],
                               initializer=tf.constant_initializer(0.0))
        gamma = tf.reshape(gamma, [1, C, 1, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1, 1])

        output = tf.reshape(x, [-1, C, D, H, W]) * gamma + beta
        # tranpose: [bs, c, d, h, w, c] to [bs, h, w, d, c] following the paper
        output = tf.transpose(output, [0, 3, 4, 2, 1])
    return output
