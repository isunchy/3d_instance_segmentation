import os
import sys
import tensorflow as tf

assert(os.path.isdir('ocnn/tensorflow'))
sys.path.append('ocnn/tensorflow')
sys.path.append('ocnn/tensorflow/script')

from libs import *
from ocnn import *


def predict_module(data, num_output, num_hidden, n_layer, training=True, reuse=False):
  with tf.variable_scope('predict_%d' % num_output, reuse=reuse):
    for i in range(n_layer):
      with tf.variable_scope('conv{}'.format(i)):
        data = octree_conv1x1_bn_lrelu(data, num_hidden, training)
    with tf.variable_scope('conv{}'.format(n_layer)):
      logit = octree_conv1x1(data, num_output, use_bias=True)
  logit = tf.transpose(tf.squeeze(logit, [0, 3])) # (1, C, H, 1) -> (H, C)
  data = tf.transpose(tf.squeeze(data, [0, 3])) # (1, C, H, 1) -> (H, C)
  return logit, data


def feature_aggregation(point_feature, point_predict_prob, point_batch_index, batch_size):
  point_aggregation_feature_list = []
  for i in range(batch_size):
    shape_point_index = tf.math.equal(point_batch_index, tf.constant(i, dtype=tf.int32))
    shape_point_predict_prob = tf.boolean_mask(point_predict_prob, shape_point_index) # [Ni, n_part]
    shape_point_feature = tf.boolean_mask(point_feature, shape_point_index) # [Ni, n_feat]
    shape_part_feature = tf.matmul(tf.transpose(shape_point_predict_prob), shape_point_feature) # [n_part, n_feat]
    part_point_prob_sum = tf.reduce_sum(shape_point_predict_prob, axis=0) # [n_part]
    shape_part_feature = tf.math.divide_no_nan(shape_part_feature, tf.reshape(part_point_prob_sum, [-1, 1])) # [n_part, n_feat]
    shape_point_aggregation_feature = tf.matmul(shape_point_predict_prob, shape_part_feature) # [Ni, n_feat]
    point_aggregation_feature_list.append(shape_point_aggregation_feature)
  point_aggregation_feature = tf.concat(point_aggregation_feature_list, axis=0) # [N, n_feat]
  return point_aggregation_feature


def predict_module_offset(data, point_predict_prob, point_predict_prob_1, point_predict_prob_2, point_batch_index, node_position, batch_size, num_hidden, n_layer, training=True, reuse=False):
  with tf.variable_scope('predict_offset', reuse=reuse):
    for i in range(n_layer):
      with tf.variable_scope('conv{}'.format(i)):
        data = octree_conv1x1_bn_lrelu(data, num_hidden, training)
    point_feature = tf.transpose(tf.squeeze(data, [0, 3])) # (1, C, H, 1) -> (H, C)

    point_aggregation_feature = feature_aggregation(point_feature, point_predict_prob, point_batch_index, batch_size) # [H, C]
    point_aggregation_feature = tf.expand_dims(tf.expand_dims(tf.transpose(point_aggregation_feature), axis=0), axis=-1) # [1, C, H, 1]
    with tf.variable_scope('convtransfer'):
      point_aggregation_feature = octree_conv1x1_bn_lrelu(point_aggregation_feature, num_hidden, training)
    point_aggregation_feature = tf.transpose(tf.squeeze(point_aggregation_feature, [0, 3])) # (1, C, H, 1) -> (H, C)

    point_aggregation_feature_1 = feature_aggregation(point_feature, point_predict_prob_1, point_batch_index, batch_size) # [H, C]
    point_aggregation_feature_1 = tf.expand_dims(tf.expand_dims(tf.transpose(point_aggregation_feature_1), axis=0), axis=-1) # [1, C, H, 1]
    with tf.variable_scope('convtransfer_1'):
      point_aggregation_feature_1 = octree_conv1x1_bn_lrelu(point_aggregation_feature_1, num_hidden, training)
    point_aggregation_feature_1 = tf.transpose(tf.squeeze(point_aggregation_feature_1, [0, 3])) # (1, C, H, 1) -> (H, C)

    point_aggregation_feature_2 = feature_aggregation(point_feature, point_predict_prob_2, point_batch_index, batch_size) # [H, C]
    point_aggregation_feature_2 = tf.expand_dims(tf.expand_dims(tf.transpose(point_aggregation_feature_2), axis=0), axis=-1) # [1, C, H, 1]
    with tf.variable_scope('convtransfer_2'):
      point_aggregation_feature_2 = octree_conv1x1_bn_lrelu(point_aggregation_feature_2, num_hidden, training)
    point_aggregation_feature_2 = tf.transpose(tf.squeeze(point_aggregation_feature_2, [0, 3])) # (1, C, H, 1) -> (H, C)

    point_feature = tf.concat([point_feature, point_aggregation_feature, point_aggregation_feature_1,
        point_aggregation_feature_2, node_position[:, :3]], axis=1) # [H, C*4+3]
    point_feature = tf.expand_dims(tf.expand_dims(tf.transpose(point_feature), axis=0), axis=-1) # [1, C*4+3, H, 1]
    with tf.variable_scope('convfusion'):
      point_feature = octree_conv1x1_bn_lrelu(point_feature, num_hidden, training)
    with tf.variable_scope('conv{}'.format(n_layer)):
      offset = octree_conv1x1(point_feature, 6, use_bias=True)
  offset = tf.transpose(tf.squeeze(offset, [0, 3])) # (1, C, H, 1) -> (H, C)
  return offset


def extract_pts_feature_from_octree_node(inputs, octree, pts, depth):
  # pts shape: [n_pts, 4]
  xyz, ids = tf.split(pts, [3, 1], axis=1)
  xyz = xyz + 1.0                                             # [0, 2]
  pts_input = tf.concat([xyz * (2**(depth-1)), ids], axis=1)
  feature = octree_bilinear_v3(pts_input, inputs, octree, depth=depth)
  return feature


def network_unet_two_decoder(octree, depth, channel=4, training=True, reuse=False):
  nout = [512, 256, 256, 256, 256, 128, 64, 32, 16, 16, 16]
  with tf.variable_scope('ocnn_unet', reuse=reuse):    
    with tf.variable_scope('signal'):
      data = octree_property(octree, property_name='feature', dtype=tf.float32,
                             depth=depth, channel=channel)
      data = tf.abs(data)
      data = tf.reshape(data, [1, channel, -1, 1])

    ## encoder
    convd = [None]*11
    convd[depth+1] = data
    for d in range(depth, 1, -1):
      with tf.variable_scope('encoder_d%d' % d):
        # downsampling
        dd = d if d == depth else d + 1
        stride = 1 if d == depth else 2
        kernel_size = [3] if d == depth else [2]
        convd[d] = octree_conv_bn_relu(convd[d+1], octree, dd, nout[d], training,
                                       stride=stride, kernel_size=kernel_size)
        # resblock
        for n in range(0, 3):
          with tf.variable_scope('resblock_%d' % n):
            convd[d] = octree_resblock(convd[d], octree, d, nout[d], 1, training)

    ## decoder
    deconv_seg = convd[2]
    for d in range(3, depth + 1):
      with tf.variable_scope('decoder_seg_d%d' % d):
        # upsampling
        deconv_seg = octree_deconv_bn_relu(deconv_seg, octree, d-1, nout[d], training, 
                                       kernel_size=[2], stride=2, fast_mode=False)
        deconv_seg = convd[d] + deconv_seg # skip connections

        # resblock
        for n in range(0, 3):
          with tf.variable_scope('resblock_%d' % n):
            deconv_seg = octree_resblock(deconv_seg, octree, d, nout[d], 1, training)

    ## decoder
    deconv_offset = convd[2]
    for d in range(3, depth + 1):
      with tf.variable_scope('decoder_offset_d%d' % d):
        # upsampling
        deconv_offset = octree_deconv_bn_relu(deconv_offset, octree, d-1, nout[d], training, 
                                       kernel_size=[2], stride=2, fast_mode=False)
        deconv_offset = convd[d] + deconv_offset # skip connections

        # resblock
        for n in range(0, 3):
          with tf.variable_scope('resblock_%d' % n):
            deconv_offset = octree_resblock(deconv_offset, octree, d, nout[d], 1, training)

  return deconv_seg, deconv_offset
