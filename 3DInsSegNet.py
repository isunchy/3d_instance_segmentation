import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append('util')
import vis_pointcloud
from config import *
from dataset import *
from network import *
from numeric_function import *
from cluster import semantic_mean_shift_cluster
from instance_metric import per_shape_mean_ap, per_part_mean_ap
from scipy.stats import entropy
from tensorflow import set_random_seed
set_random_seed(2)


# instance_center_bytes: [bs]
def decode_instance_center(instance_center_bytes, batch_size):
  instance_center_list = []
  for i in range(batch_size):
    instance_center_list.append(tf.reshape(tf.decode_raw(instance_center_bytes[i], out_type=tf.float32), [-1, 3]))
  instance_center = tf.concat(instance_center_list, axis=0)
  return instance_center

def get_training_input_data(dataset_path, batch_size, depth=6):
  with tf.name_scope('input_data_training'):
    [octree, inbox_points_bytes_1, inbox_points_bytes_2, inbox_points_bytes_3, points_flag,
        inbox_instance_points_bytes_1, inbox_instance_points_bytes_2, inbox_instance_points_bytes_3,
        instance_center_bytes_1, semantic_center_bytes_1, instance_center_bytes_2,
        semantic_center_bytes_2, instance_center_bytes_3, semantic_center_bytes_3] = \
            points_dataset(dataset_path, batch_size, depth=depth, test=False)

    inbox_node_position = points_property(inbox_points_bytes_1, property_name='xyz', channel=4) # [n_point, 4]
    inbox_batch_index = tf.cast(inbox_node_position[:, -1], dtype=tf.int32) # [n_point]

    inbox_gt_part_index_1 = points_property(inbox_points_bytes_1, property_name='label', channel=1) # [n_point, 1]
    inbox_gt_part_index_1 = tf.reshape(tf.cast(inbox_gt_part_index_1, dtype=tf.int32), [-1]) # [n_point]
    inbox_gt_part_index_2 = points_property(inbox_points_bytes_2, property_name='label', channel=1) # [n_point, 1]
    inbox_gt_part_index_2 = tf.reshape(tf.cast(inbox_gt_part_index_2, dtype=tf.int32), [-1]) # [n_point]
    inbox_gt_part_index_3 = points_property(inbox_points_bytes_3, property_name='label', channel=1) # [n_point, 1]
    inbox_gt_part_index_3 = tf.reshape(tf.cast(inbox_gt_part_index_3, dtype=tf.int32), [-1]) # [n_point]

    points_flag = tf.cast(tf.reshape(points_flag, [-1]), tf.float32) # [bs], labeled data flag
    instance_center_1 = decode_instance_center(instance_center_bytes_1, batch_size) # [n_point, 3]
    semantic_center_1 = decode_instance_center(semantic_center_bytes_1, batch_size) # [n_point, 3]
    instance_center_2 = decode_instance_center(instance_center_bytes_2, batch_size) # [n_point, 3]
    semantic_center_2 = decode_instance_center(semantic_center_bytes_2, batch_size) # [n_point, 3]
    instance_center_3 = decode_instance_center(instance_center_bytes_3, batch_size) # [n_point, 3]
    semantic_center_3 = decode_instance_center(semantic_center_bytes_3, batch_size) # [n_point, 3]

    inbox_gt_instance_index_1 = points_property(inbox_instance_points_bytes_1, property_name='label', channel=1) # [n_point, 1]
    inbox_gt_instance_index_1 = tf.reshape(tf.cast(inbox_gt_instance_index_1, dtype=tf.int32), [-1]) # [n_point]
    inbox_gt_instance_index_2 = points_property(inbox_instance_points_bytes_2, property_name='label', channel=1) # [n_point, 1]
    inbox_gt_instance_index_2 = tf.reshape(tf.cast(inbox_gt_instance_index_2, dtype=tf.int32), [-1]) # [n_point]
    inbox_gt_instance_index_3 = points_property(inbox_instance_points_bytes_3, property_name='label', channel=1) # [n_point, 1]
    inbox_gt_instance_index_3 = tf.reshape(tf.cast(inbox_gt_instance_index_3, dtype=tf.int32), [-1]) # [n_point]

  return [octree, inbox_node_position, inbox_batch_index, inbox_gt_part_index_1, inbox_gt_part_index_2, inbox_gt_part_index_3,
      inbox_gt_instance_index_1, inbox_gt_instance_index_2, inbox_gt_instance_index_3, points_flag,
      instance_center_1, semantic_center_1, instance_center_2, semantic_center_2, instance_center_3, semantic_center_3]

def get_test_input_data(dataset_path, batch_size, depth=6):
  with tf.name_scope('input_data_test'):
    [octree, points_bytes_1, instance_center_bytes_1, instance_points_bytes_1, semantic_center_bytes_1,
        points_bytes_2, instance_center_bytes_2, instance_points_bytes_2, semantic_center_bytes_2,
        points_bytes_3, instance_center_bytes_3, instance_points_bytes_3, semantic_center_bytes_3] = \
            points_dataset(dataset_path, batch_size, depth=depth, test=True)

    node_position = points_property(points_bytes_1, property_name='xyz', channel=4) # [n_point, 4]
    point_batch_index = tf.cast(node_position[:, -1], dtype=tf.int32) # [n_point]

    point_gt_part_index_1 = points_property(points_bytes_1, property_name='label', channel=1) # [n_point, 1]
    point_gt_part_index_1 = tf.reshape(tf.cast(point_gt_part_index_1, dtype=tf.int32), [-1]) # [n_point]
    point_gt_part_index_2 = points_property(points_bytes_2, property_name='label', channel=1) # [n_point, 1]
    point_gt_part_index_2 = tf.reshape(tf.cast(point_gt_part_index_2, dtype=tf.int32), [-1]) # [n_point]
    point_gt_part_index_3 = points_property(points_bytes_3, property_name='label', channel=1) # [n_point, 1]
    point_gt_part_index_3 = tf.reshape(tf.cast(point_gt_part_index_3, dtype=tf.int32), [-1]) # [n_point]

    instance_center_1 = decode_instance_center(instance_center_bytes_1, batch_size) # [n_point, 3]
    semantic_center_1 = decode_instance_center(semantic_center_bytes_1, batch_size) # [n_point, 3]
    instance_center_2 = decode_instance_center(instance_center_bytes_2, batch_size) # [n_point, 3]
    semantic_center_2 = decode_instance_center(semantic_center_bytes_2, batch_size) # [n_point, 3]
    instance_center_3 = decode_instance_center(instance_center_bytes_3, batch_size) # [n_point, 3]
    semantic_center_3 = decode_instance_center(semantic_center_bytes_3, batch_size) # [n_point, 3]

    point_gt_instance_index_1 = points_property(instance_points_bytes_1, property_name='label', channel=1) # [n_point, 1]
    point_gt_instance_index_1 = tf.reshape(tf.cast(point_gt_instance_index_1, dtype=tf.int32), [-1]) # [n_point]
    point_gt_instance_index_2 = points_property(instance_points_bytes_2, property_name='label', channel=1) # [n_point, 1]
    point_gt_instance_index_2 = tf.reshape(tf.cast(point_gt_instance_index_2, dtype=tf.int32), [-1]) # [n_point]
    point_gt_instance_index_3 = points_property(instance_points_bytes_3, property_name='label', channel=1) # [n_point, 1]
    point_gt_instance_index_3 = tf.reshape(tf.cast(point_gt_instance_index_3, dtype=tf.int32), [-1]) # [n_point]

  return [octree, node_position, point_gt_part_index_1, point_gt_part_index_2, point_gt_part_index_3,
      point_batch_index, instance_center_1, instance_center_2, instance_center_3, point_gt_instance_index_1,
      point_gt_instance_index_2, point_gt_instance_index_3, semantic_center_1, semantic_center_2, semantic_center_3]

# node_position: [n_point, 4]
def backbone(octree, node_position, depth=6, training=True, reuse=False):
  node_feature_seg, node_feature_offset = network_unet_two_decoder(octree, depth, training=training, reuse=reuse) # [1, C, n_node, 1]
  point_feature_seg = extract_pts_feature_from_octree_node(node_feature_seg, octree, node_position, depth) # [1, C, n_point, 1]
  point_feature_offset = extract_pts_feature_from_octree_node(node_feature_offset, octree, node_position, depth) # [1, C, n_point, 1]
  return point_feature_seg, point_feature_offset

# point_feature: [1, C, n_point, 1]
def seg_header(point_feature, n_part, n_hidden=128, n_layer=2, training=True, reuse=False):
  point_predict_logits, point_hidden_feature = predict_module(point_feature, n_part,
      n_hidden, n_layer, training=training, reuse=reuse) # [n_point, n_part], [n_point, n_hidden]
  point_predict_prob = tf.nn.softmax(point_predict_logits) # [n_point, n_part]
  return point_predict_logits, point_predict_prob, point_hidden_feature

# point_feature: [1, C, n_point, 1]
def offset_header(point_feature, point_predict_prob, point_predict_prob_1, point_predict_prob_2, point_batch_index, node_position,
    batch_size=1, n_hidden=128, n_layer=2, training=True, reuse=False):
  if FLAGS.stop_gradient:
    point_predict_prob = tf.stop_gradient(point_predict_prob)
    point_predict_prob_1 = tf.stop_gradient(point_predict_prob_1)
    point_predict_prob_2 = tf.stop_gradient(point_predict_prob_2)
  point_predict_offset = predict_module_offset(point_feature, point_predict_prob, point_predict_prob_1, point_predict_prob_2, point_batch_index, node_position, batch_size, n_hidden, n_layer, training=training, reuse=reuse) # [n_point, 6]
  return point_predict_offset

# point_predict_offset: [n_point, 3]
# node_position: [n_point, 3]
# instance_center: [n_point, 3]
# inbox_batch_index: [n_point]
def compute_offset_loss(point_predict_offset, node_position, instance_center, inbox_batch_index, point_gt_part_index, delete_0=False):
  with tf.name_scope('offset_loss'):
    point_gt_offset = instance_center - node_position # [n_point, 3]
    offset_loss = tf.reduce_mean(tf.abs(point_predict_offset - point_gt_offset), axis=1) # [n_point]
    if delete_0:
      non_zero_mask = tf.cast(point_gt_part_index > 0, dtype=tf.float32) # [n_point]
      offset_loss = tf.math.divide_no_nan(
          tf.segment_sum(tf.multiply(offset_loss, non_zero_mask), inbox_batch_index),
          tf.segment_sum(non_zero_mask, inbox_batch_index)) # [bs]
    else:
      offset_loss = tf.segment_mean(offset_loss, inbox_batch_index) # [bs]
  return tf.reduce_mean(offset_loss)

# point_predict_logits: [n_point, n_part]
# point_gt_part_index: [n_point]
# inbox_batch_index: [n_point]
def compute_segmentation_loss(point_predict_logits, point_gt_part_index, inbox_batch_index, n_part, delete_0=False):
  with tf.name_scope('segmentation_loss'):
    seg_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(point_gt_part_index, n_part, dtype=tf.float32),
        logits=point_predict_logits) # [n_point]
    if delete_0:
      non_zero_mask = tf.cast(point_gt_part_index > 0, dtype=tf.float32) # [n_point]
      seg_loss = tf.math.divide_no_nan(
          tf.segment_sum(tf.multiply(seg_loss, non_zero_mask), inbox_batch_index),
          tf.segment_sum(non_zero_mask, inbox_batch_index)) # [bs]
    else:
      seg_loss = tf.segment_mean(seg_loss, inbox_batch_index) # [bs]
  return tf.reduce_mean(seg_loss)

# point_predict_part_index: [n_point]
# point_gt_part_index: [n_point]
# point_batch_index: [n_point]
def compute_point_match_accuracy(point_predict_part_index, point_gt_part_index, point_batch_index, delete_0=False):
  with tf.name_scope('point_match_accuracy'):
    point_predict_match = tf.cast(tf.math.equal(point_predict_part_index, point_gt_part_index), dtype=tf.float32) # [n_point]
    if delete_0:
      non_zero_mask = tf.cast(point_gt_part_index > 0, dtype=tf.float32) # [n_point]
      point_match_accuracy = tf.math.divide_no_nan(
          tf.segment_sum(tf.multiply(point_predict_match, non_zero_mask), point_batch_index),
          tf.segment_sum(non_zero_mask, point_batch_index)) # [bs]
    else:
      point_match_accuracy = tf.segment_mean(point_predict_match, point_batch_index) # [bs]
  return point_match_accuracy

# point_predict_prob: [n_point, n_part]
# point_gt_part_index: [n_point]
# point_batch_index: [n_point]
def compute_segmentation_accuracy(point_predict_prob, point_gt_part_index, point_batch_index, delete_0=False):
  with tf.name_scope('segmentation_accuracy'):
    point_predict_part_index = tf.argmax(point_predict_prob, axis=-1, output_type=tf.int32) # [n_point]
    point_match_accuracy = compute_point_match_accuracy(point_predict_part_index,
        point_gt_part_index, point_batch_index, delete_0=delete_0) # [bs]
  return tf.reduce_mean(point_match_accuracy), point_predict_part_index


def train_network():
  [
      octree,                           # string
      inbox_node_position,              # [n_point, 4]
      inbox_batch_index,                # [n_point]
      inbox_gt_part_index_1,            # [n_point]
      inbox_gt_part_index_2,            # [n_point]
      inbox_gt_part_index_3,            # [n_point]
      inbox_gt_instance_index_1,        # [n_point]
      inbox_gt_instance_index_2,        # [n_point]
      inbox_gt_instance_index_3,        # [n_point]
      points_flag,                      # [bs]
      instance_center_1,                # [n_point, 3]
      semantic_center_1,                # [n_point, 3]
      instance_center_2,                # [n_point, 3]
      semantic_center_2,                # [n_point, 3]
      instance_center_3,                # [n_point, 3]
      semantic_center_3,                # [n_point, 3]
  ] = get_training_input_data(FLAGS.train_data, FLAGS.train_batch_size, depth=FLAGS.depth)

  # get point feature
  with tf.variable_scope('seg'):
    inbox_point_feature_seg, inbox_point_feature_offset = backbone(octree, inbox_node_position, depth=FLAGS.depth, training=True, reuse=False) # [1, C, n_point, 1], [1, C, n_point, 1]
    with tf.variable_scope('level_1'):
      inbox_point_predict_logits_1, inbox_point_predict_prob_1, inbox_point_hidden_feature_1 = seg_header(
          inbox_point_feature_seg, n_part_1, training=True, reuse=False) # [n_point, n_part_1], [n_point, n_part_1], [n_point, 128]
    with tf.variable_scope('level_2'):
      inbox_point_predict_logits_2, inbox_point_predict_prob_2, inbox_point_hidden_feature_2 = seg_header(
          inbox_point_feature_seg, n_part_2, training=True, reuse=False) # [n_point, n_part_2], [n_point, n_part_2], [n_point, 128]
    with tf.variable_scope('level_3'):
      inbox_point_predict_logits_3, inbox_point_predict_prob_3, inbox_point_hidden_feature_3 = seg_header(
          inbox_point_feature_seg, n_part_3, training=True, reuse=False) # [n_point, n_part_3], [n_point, n_part_3], [n_point, 128]
    with tf.name_scope('level_3-2'):
      with tf.name_scope('mapping'):
        label_mapping_32 = get_label_mapping(FLAGS.category, slevel=3, dlevel=2) # [n_part_3]
        label_mapping_32_matrix = tf.one_hot(label_mapping_32, n_part_2, dtype=tf.float32) # [n_part_3, n_part_2]
      with tf.name_scope('prob_merge'):
        inbox_point_predict_prob_32 = tf.matmul(inbox_point_predict_prob_3, label_mapping_32_matrix) # [n_point, n_part_2]
    with tf.name_scope('level_2-1'):
      with tf.name_scope('mapping'):
        label_mapping_21 = get_label_mapping(FLAGS.category, slevel=2, dlevel=1) # [n_part_2]
        label_mapping_21_matrix = tf.one_hot(label_mapping_21, n_part_1, dtype=tf.float32) # [n_part_2, n_part_1]
      with tf.name_scope('prob_merge'):
        inbox_point_predict_prob_21 = tf.matmul(inbox_point_predict_prob_2, label_mapping_21_matrix) # [n_point, n_part_1]

    with tf.variable_scope('level_1'):
      inbox_point_predict_offset_1 = offset_header(inbox_point_feature_offset, inbox_point_predict_prob_1, inbox_point_predict_prob_2, inbox_point_predict_prob_3, inbox_batch_index, inbox_node_position, FLAGS.train_batch_size, training=True, reuse=False)
    with tf.variable_scope('level_2'):
      inbox_point_predict_offset_2 = offset_header(inbox_point_feature_offset, inbox_point_predict_prob_2, inbox_point_predict_prob_1, inbox_point_predict_prob_3, inbox_batch_index, inbox_node_position, FLAGS.train_batch_size, training=True, reuse=False)
    with tf.variable_scope('level_3'):
      inbox_point_predict_offset_3 = offset_header(inbox_point_feature_offset, inbox_point_predict_prob_3, inbox_point_predict_prob_1, inbox_point_predict_prob_2, inbox_batch_index, inbox_node_position, FLAGS.train_batch_size, training=True, reuse=False)

    inbox_point_predict_offset_1, inbox_point_predict_offset_sem_1 = tf.split(inbox_point_predict_offset_1, [3, 3], axis=1) # [n_point, 3], [n_point, 3]
    inbox_point_predict_offset_2, inbox_point_predict_offset_sem_2 = tf.split(inbox_point_predict_offset_2, [3, 3], axis=1) # [n_point, 3], [n_point, 3]
    inbox_point_predict_offset_3, inbox_point_predict_offset_sem_3 = tf.split(inbox_point_predict_offset_3, [3, 3], axis=1) # [n_point, 3], [n_point, 3]

  # offset loss
  with tf.name_scope('offset_loss'):
    offset_loss_1 = compute_offset_loss(inbox_point_predict_offset_1, inbox_node_position[:, :-1], instance_center_1,
        inbox_batch_index, inbox_gt_part_index_1, delete_0=FLAGS.delete_0) # scalar
    offset_loss_2 = compute_offset_loss(inbox_point_predict_offset_2, inbox_node_position[:, :-1], instance_center_2,
        inbox_batch_index, inbox_gt_part_index_2, delete_0=FLAGS.delete_0) # scalar
    offset_loss_3 = compute_offset_loss(inbox_point_predict_offset_3, inbox_node_position[:, :-1], instance_center_3,
        inbox_batch_index, inbox_gt_part_index_3, delete_0=FLAGS.delete_0) # scalar

  # sem offset loss
  with tf.name_scope('sem_offset_loss'):
    sem_offset_loss_1 = compute_offset_loss(inbox_point_predict_offset_sem_1, inbox_node_position[:, :-1], semantic_center_1,
        inbox_batch_index, inbox_gt_part_index_1, delete_0=FLAGS.delete_0) # scalar
    sem_offset_loss_2 = compute_offset_loss(inbox_point_predict_offset_sem_2, inbox_node_position[:, :-1], semantic_center_2,
        inbox_batch_index, inbox_gt_part_index_2, delete_0=FLAGS.delete_0) # scalar
    sem_offset_loss_3 = compute_offset_loss(inbox_point_predict_offset_sem_3, inbox_node_position[:, :-1], semantic_center_3,
        inbox_batch_index, inbox_gt_part_index_3, delete_0=FLAGS.delete_0) # scalar

  # segmentation loss
  with tf.name_scope('seg_loss'):
    with tf.name_scope('loss'):
      seg_loss_1 = compute_segmentation_loss(inbox_point_predict_logits_1, inbox_gt_part_index_1, inbox_batch_index, n_part_1, delete_0=FLAGS.delete_0) # scalar
      seg_loss_2 = compute_segmentation_loss(inbox_point_predict_logits_2, inbox_gt_part_index_2, inbox_batch_index, n_part_2, delete_0=FLAGS.delete_0) # scalar
      seg_loss_3 = compute_segmentation_loss(inbox_point_predict_logits_3, inbox_gt_part_index_3, inbox_batch_index, n_part_3, delete_0=FLAGS.delete_0) # scalar
    with tf.name_scope('accuracy'):
      seg_accuracy_1, inbox_point_predict_part_index_1 = compute_segmentation_accuracy(inbox_point_predict_prob_1, inbox_gt_part_index_1,
          inbox_batch_index, delete_0=FLAGS.delete_0) # scalar, [n_point]
      seg_accuracy_2, inbox_point_predict_part_index_2 = compute_segmentation_accuracy(inbox_point_predict_prob_2, inbox_gt_part_index_2,
          inbox_batch_index, delete_0=FLAGS.delete_0) # scalar, [n_point]
      seg_accuracy_3, inbox_point_predict_part_index_3 = compute_segmentation_accuracy(inbox_point_predict_prob_3, inbox_gt_part_index_3,
          inbox_batch_index, delete_0=FLAGS.delete_0) # scalar, [n_point]

  # level loss
  level_1_loss = FLAGS.seg_loss_weight * seg_loss_1 + \
      FLAGS.offset_weight * offset_loss_1 + \
      FLAGS.sem_offset_weight * sem_offset_loss_1
  level_2_loss = FLAGS.seg_loss_weight * seg_loss_2 + \
      FLAGS.offset_weight * offset_loss_2 + \
      FLAGS.sem_offset_weight * sem_offset_loss_2
  level_3_loss = FLAGS.seg_loss_weight * seg_loss_3 + \
      FLAGS.offset_weight * offset_loss_3 + \
      FLAGS.sem_offset_weight * sem_offset_loss_3

  train_loss = FLAGS.level_1_weight * level_1_loss + \
      FLAGS.level_2_weight * level_2_loss + \
      FLAGS.level_3_weight * level_3_loss

  # optimizer
  with tf.name_scope('optimizer'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      tvars = tf.trainable_variables()
      with tf.name_scope('weight_decay'):
        regularizer = tf.add_n([tf.nn.l2_loss(v) for v in tvars])
      with tf.name_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False)
        if FLAGS.decay_policy =='step':
          boundaries = [int(max_iter*0.5), int(max_iter*0.75)]
          values = [i*FLAGS.learning_rate for i in [1, 0.1, 0.01]]
          lr = tf.train.piecewise_constant(global_step, boundaries, values)
        elif FLAGS.decay_policy == 'poly':
          lr = tf.train.polynomial_decay(FLAGS.learning_rate, global_step, max_iter, end_learning_rate=0.0, power=0.9)
        else:
          lr = FLAGS.learning_rate

      if FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
      else:
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)

      solver = optimizer.minimize(train_loss + regularizer * FLAGS.weight_decay, var_list=tvars, global_step=global_step)

  with tf.name_scope('train_summary'):
    with tf.name_scope('loss'):
      summary_train_loss = tf.summary.scalar('train_loss', train_loss)
      summary_regularizer = tf.summary.scalar('regularizer', regularizer)
      summary_offset_loss_1 = tf.summary.scalar('offset_loss_1', offset_loss_1)
      summary_sem_offset_loss_1 = tf.summary.scalar('sem_offset_loss_1', sem_offset_loss_1)
      summary_seg_loss_1 = tf.summary.scalar('seg_loss_1', seg_loss_1)
      summary_seg_accuracy_1 = tf.summary.scalar('seg_accuracy_1', seg_accuracy_1)
      summary_offset_loss_2 = tf.summary.scalar('offset_loss_2', offset_loss_2)
      summary_sem_offset_loss_2 = tf.summary.scalar('sem_offset_loss_2', sem_offset_loss_2)
      summary_seg_loss_2 = tf.summary.scalar('seg_loss_2', seg_loss_2)
      summary_seg_accuracy_2 = tf.summary.scalar('seg_accuracy_2', seg_accuracy_2)
      summary_offset_loss_3 = tf.summary.scalar('offset_loss_3', offset_loss_3)
      summary_sem_offset_loss_3 = tf.summary.scalar('sem_offset_loss_3', sem_offset_loss_3)
      summary_seg_loss_3 = tf.summary.scalar('seg_loss_3', seg_loss_3)
      summary_seg_accuracy_3 = tf.summary.scalar('seg_accuracy_3', seg_accuracy_3)
    with tf.name_scope('misc'):
      summary_lr_scheme = tf.summary.scalar('learning_rate', lr)
      summary_points_flag = tf.summary.scalar('points_flag', tf.reduce_mean(points_flag))
    train_merged = tf.summary.merge([
        summary_train_loss,
        summary_regularizer,
        summary_offset_loss_1,
        summary_sem_offset_loss_1,
        summary_seg_loss_1,
        summary_seg_accuracy_1,
        summary_offset_loss_2,
        summary_sem_offset_loss_2,
        summary_seg_loss_2,
        summary_seg_accuracy_2,
        summary_offset_loss_3,
        summary_sem_offset_loss_3,
        summary_seg_loss_3,
        summary_seg_accuracy_3,
        summary_lr_scheme,
        summary_points_flag
    ])

  return train_merged, solver


def test_network(test_data, visual=True):
  [
      octree,                           # string
      node_position,                    # [n_point, 4]
      point_gt_part_index_1,            # [n_point]
      point_gt_part_index_2,            # [n_point]
      point_gt_part_index_3,            # [n_point]
      point_batch_index,                # [n_point]
      instance_center_1,                # [n_point, 3]
      instance_center_2,                # [n_point, 3]
      instance_center_3,                # [n_point, 3]
      point_gt_instance_index_1,        # [n_point]
      point_gt_instance_index_2,        # [n_point]
      point_gt_instance_index_3,        # [n_point]
      semantic_center_1,                # [n_point, 3]
      semantic_center_2,                # [n_point, 3]
      semantic_center_3,                # [n_point, 3]
  ] = get_test_input_data(test_data, FLAGS.test_batch_size, depth=FLAGS.depth)


  # for segmentation
  with tf.variable_scope('seg'):
    point_feature_seg, point_feature_offset = backbone(octree, node_position, depth=FLAGS.depth, training=False, reuse=True) # [1, C, n_point, 1], [1, C, n_point, 1]
    with tf.variable_scope('level_1'):
      point_predict_logits_1, point_predict_prob_1, point_hidden_feature_1 = seg_header(
          point_feature_seg, n_part_1, training=False, reuse=True) # [n_point, n_part_1], [n_point, n_part_1], [n_point, 128]
    with tf.variable_scope('level_2'):
      point_predict_logits_2, point_predict_prob_2, point_hidden_feature_2 = seg_header(
          point_feature_seg, n_part_2, training=False, reuse=True) # [n_point, n_part_2], [n_point, n_part_2], [n_point, 128]
    with tf.variable_scope('level_3'):
      point_predict_logits_3, point_predict_prob_3, point_hidden_feature_3 = seg_header(
          point_feature_seg, n_part_3, training=False, reuse=True) # [n_point, n_part_3], [n_point, n_part_3], [n_point, 128]
    with tf.name_scope('level_3-2'):
      with tf.name_scope('mapping'):
        label_mapping_32 = get_label_mapping(FLAGS.category, slevel=3, dlevel=2) # [n_part_3]
        label_mapping_32_matrix = tf.one_hot(label_mapping_32, n_part_2, dtype=tf.float32) # [n_part_3, n_part_2]
      with tf.name_scope('prob_merge'):
        point_predict_prob_32 = tf.matmul(point_predict_prob_3, label_mapping_32_matrix) # [n_point, n_part_2]
    with tf.name_scope('level_2-1'):
      with tf.name_scope('mapping'):
        label_mapping_21 = get_label_mapping(FLAGS.category, slevel=2, dlevel=1) # [n_part_2]
        label_mapping_21_matrix = tf.one_hot(label_mapping_21, n_part_1, dtype=tf.float32) # [n_part_2, n_part_1]
      with tf.name_scope('prob_merge'):
        point_predict_prob_21 = tf.matmul(point_predict_prob_2, label_mapping_21_matrix) # [n_point, n_part_1]

    with tf.variable_scope('level_1'):
      point_predict_offset_1 = offset_header(point_feature_offset, point_predict_prob_1, point_predict_prob_2, point_predict_prob_3, point_batch_index, node_position, training=False, reuse=True)
    with tf.variable_scope('level_2'):
      point_predict_offset_2 = offset_header(point_feature_offset, point_predict_prob_2, point_predict_prob_1, point_predict_prob_3, point_batch_index, node_position, training=False, reuse=True)
    with tf.variable_scope('level_3'):
      point_predict_offset_3 = offset_header(point_feature_offset, point_predict_prob_3, point_predict_prob_1, point_predict_prob_2, point_batch_index, node_position, training=False, reuse=True)

    point_predict_offset_1, point_predict_offset_sem_1 = tf.split(point_predict_offset_1, [3, 3], axis=1) # [n_point, 3], [n_point, 3]
    point_predict_offset_2, point_predict_offset_sem_2 = tf.split(point_predict_offset_2, [3, 3], axis=1) # [n_point, 3], [n_point, 3]
    point_predict_offset_3, point_predict_offset_sem_3 = tf.split(point_predict_offset_3, [3, 3], axis=1) # [n_point, 3], [n_point, 3]

  # offset loss
  with tf.name_scope('offset_loss'):
    offset_loss_1 = compute_offset_loss(point_predict_offset_1, node_position[:, :-1], instance_center_1,
        point_batch_index, point_gt_part_index_1, delete_0=FLAGS.delete_0) # scalar
    offset_loss_2 = compute_offset_loss(point_predict_offset_2, node_position[:, :-1], instance_center_2,
        point_batch_index, point_gt_part_index_2, delete_0=FLAGS.delete_0) # scalar
    offset_loss_3 = compute_offset_loss(point_predict_offset_3, node_position[:, :-1], instance_center_3,
        point_batch_index, point_gt_part_index_3, delete_0=FLAGS.delete_0) # scalar

  # sem offset loss
  with tf.name_scope('sem_offset_loss'):
    sem_offset_loss_1 = compute_offset_loss(point_predict_offset_sem_1, node_position[:, :-1], semantic_center_1,
        point_batch_index, point_gt_part_index_1, delete_0=FLAGS.delete_0) # scalar
    sem_offset_loss_2 = compute_offset_loss(point_predict_offset_sem_2, node_position[:, :-1], semantic_center_2,
        point_batch_index, point_gt_part_index_2, delete_0=FLAGS.delete_0) # scalar
    sem_offset_loss_3 = compute_offset_loss(point_predict_offset_sem_3, node_position[:, :-1], semantic_center_3,
        point_batch_index, point_gt_part_index_3, delete_0=FLAGS.delete_0) # scalar

  # segmentation loss
  with tf.name_scope('seg_loss'):
    with tf.name_scope('loss'):
      seg_loss_1 = compute_segmentation_loss(point_predict_logits_1, point_gt_part_index_1, point_batch_index, n_part_1, delete_0=FLAGS.delete_0) # scalar
      seg_loss_2 = compute_segmentation_loss(point_predict_logits_2, point_gt_part_index_2, point_batch_index, n_part_2, delete_0=FLAGS.delete_0) # scalar
      seg_loss_3 = compute_segmentation_loss(point_predict_logits_3, point_gt_part_index_3, point_batch_index, n_part_3, delete_0=FLAGS.delete_0) # scalar
      seg_accuracy_1, point_predict_part_index_1 = compute_segmentation_accuracy(point_predict_prob_1, point_gt_part_index_1,
          point_batch_index, delete_0=FLAGS.delete_0) # scalar, [n_point]
      seg_accuracy_2, point_predict_part_index_2 = compute_segmentation_accuracy(point_predict_prob_2, point_gt_part_index_2,
          point_batch_index, delete_0=FLAGS.delete_0) # scalar, [n_point]
      seg_accuracy_3, point_predict_part_index_3 = compute_segmentation_accuracy(point_predict_prob_3, point_gt_part_index_3,
          point_batch_index, delete_0=FLAGS.delete_0) # scalar, [n_point]

  # level loss
  level_1_loss = FLAGS.seg_loss_weight * seg_loss_1 + \
      FLAGS.offset_weight * offset_loss_1 + \
      FLAGS.sem_offset_weight * sem_offset_loss_1
  level_2_loss = FLAGS.seg_loss_weight * seg_loss_2 + \
      FLAGS.offset_weight * offset_loss_2 + \
      FLAGS.sem_offset_weight * sem_offset_loss_2
  level_3_loss = FLAGS.seg_loss_weight * seg_loss_3 + \
      FLAGS.offset_weight * offset_loss_3 + \
      FLAGS.sem_offset_weight * sem_offset_loss_3

  test_loss = FLAGS.level_1_weight * level_1_loss + \
      FLAGS.level_2_weight * level_2_loss + \
      FLAGS.level_3_weight * level_3_loss

  with tf.name_scope('test_summary'):
    average_test_loss = tf.placeholder(tf.float32)
    average_offset_loss_1 = tf.placeholder(tf.float32)
    average_sem_offset_loss_1 = tf.placeholder(tf.float32)
    average_seg_loss_1 = tf.placeholder(tf.float32)
    average_seg_accuracy_1 = tf.placeholder(tf.float32)
    average_miou_v1_1 = tf.placeholder(tf.float32)
    average_miou_v2_1 = tf.placeholder(tf.float32)
    average_miou_v3_1 = tf.placeholder(tf.float32)
    average_offset_loss_2 = tf.placeholder(tf.float32)
    average_sem_offset_loss_2 = tf.placeholder(tf.float32)
    average_seg_loss_2 = tf.placeholder(tf.float32)
    average_seg_accuracy_2 = tf.placeholder(tf.float32)
    average_miou_v1_2 = tf.placeholder(tf.float32)
    average_miou_v2_2 = tf.placeholder(tf.float32)
    average_miou_v3_2 = tf.placeholder(tf.float32)
    average_offset_loss_3 = tf.placeholder(tf.float32)
    average_sem_offset_loss_3 = tf.placeholder(tf.float32)
    average_seg_loss_3 = tf.placeholder(tf.float32)
    average_seg_accuracy_3 = tf.placeholder(tf.float32)
    average_miou_v1_3 = tf.placeholder(tf.float32)
    average_miou_v2_3 = tf.placeholder(tf.float32)
    average_miou_v3_3 = tf.placeholder(tf.float32)
    summary_test_loss = tf.summary.scalar('test_loss', average_test_loss)
    summary_offset_loss_1 = tf.summary.scalar('offset_loss_1', average_offset_loss_1)
    summary_sem_offset_loss_1 = tf.summary.scalar('sem_offset_loss_1', average_sem_offset_loss_1)
    summary_seg_loss_1 = tf.summary.scalar('seg_loss_1', average_seg_loss_1)
    summary_seg_accuracy_1 = tf.summary.scalar('seg_accuracy_1', average_seg_accuracy_1)
    summary_miou_v1_1 = tf.summary.scalar('miou_v1_1', average_miou_v1_1)
    summary_miou_v2_1 = tf.summary.scalar('miou_v2_1', average_miou_v2_1)
    summary_miou_v3_1 = tf.summary.scalar('miou_v3_1', average_miou_v3_1)
    summary_offset_loss_2 = tf.summary.scalar('offset_loss_2', average_offset_loss_2)
    summary_sem_offset_loss_2 = tf.summary.scalar('sem_offset_loss_2', average_sem_offset_loss_2)
    summary_seg_loss_2 = tf.summary.scalar('seg_loss_2', average_seg_loss_2)
    summary_seg_accuracy_2 = tf.summary.scalar('seg_accuracy_2', average_seg_accuracy_2)
    summary_miou_v1_2 = tf.summary.scalar('miou_v1_2', average_miou_v1_2)
    summary_miou_v2_2 = tf.summary.scalar('miou_v2_2', average_miou_v2_2)
    summary_miou_v3_2 = tf.summary.scalar('miou_v3_2', average_miou_v3_2)
    summary_offset_loss_3 = tf.summary.scalar('offset_loss_3', average_offset_loss_3)
    summary_sem_offset_loss_3 = tf.summary.scalar('sem_offset_loss_3', average_sem_offset_loss_3)
    summary_seg_loss_3 = tf.summary.scalar('seg_loss_3', average_seg_loss_3)
    summary_seg_accuracy_3 = tf.summary.scalar('seg_accuracy_3', average_seg_accuracy_3)
    summary_miou_v1_3 = tf.summary.scalar('miou_v1_3', average_miou_v1_3)
    summary_miou_v2_3 = tf.summary.scalar('miou_v2_3', average_miou_v2_3)
    summary_miou_v3_3 = tf.summary.scalar('miou_v3_3', average_miou_v3_3)
    test_merged = tf.summary.merge([
        summary_test_loss,
        summary_offset_loss_1,
        summary_sem_offset_loss_1,
        summary_seg_loss_1,
        summary_seg_accuracy_1,
        summary_miou_v1_1,
        summary_miou_v2_1,
        summary_miou_v3_1,
        summary_offset_loss_2,
        summary_sem_offset_loss_2,
        summary_seg_loss_2,
        summary_seg_accuracy_2,
        summary_miou_v1_2,
        summary_miou_v2_2,
        summary_miou_v3_2,
        summary_offset_loss_3,
        summary_sem_offset_loss_3,
        summary_seg_loss_3,
        summary_seg_accuracy_3,
        summary_miou_v1_3,
        summary_miou_v2_3,
        summary_miou_v3_3,
    ])
  return_list = [
      test_merged,
      average_test_loss,
      average_offset_loss_1,
      average_sem_offset_loss_1,
      average_seg_loss_1,
      average_seg_accuracy_1,
      average_miou_v1_1,
      average_miou_v2_1,
      average_miou_v3_1,
      average_offset_loss_2,
      average_sem_offset_loss_2,
      average_seg_loss_2,
      average_seg_accuracy_2,
      average_miou_v1_2,
      average_miou_v2_2,
      average_miou_v3_2,
      average_offset_loss_3,
      average_sem_offset_loss_3,
      average_seg_loss_3,
      average_seg_accuracy_3,
      average_miou_v1_3,
      average_miou_v2_3,
      average_miou_v3_3,
      test_loss,
      offset_loss_1,
      sem_offset_loss_1,
      seg_loss_1,
      seg_accuracy_1,
      offset_loss_2,
      sem_offset_loss_2,
      seg_loss_2,
      seg_accuracy_2,
      offset_loss_3,
      sem_offset_loss_3,
      seg_loss_3,
      seg_accuracy_3,
      node_position,
      point_predict_offset_1,
      point_predict_offset_sem_1,
      instance_center_1,
      semantic_center_1,
      point_gt_part_index_1,
      point_predict_part_index_1,
      point_predict_prob_1,
      point_gt_instance_index_1,
      point_predict_offset_2,
      point_predict_offset_sem_2,
      instance_center_2,
      semantic_center_2,
      point_gt_part_index_2,
      point_predict_part_index_2,
      point_predict_prob_2,
      point_gt_instance_index_2,
      point_predict_offset_3,
      point_predict_offset_sem_3,
      instance_center_3,
      semantic_center_3,
      point_gt_part_index_3,
      point_predict_part_index_3,
      point_predict_prob_3,
      point_gt_instance_index_3
  ]

  return return_list


def main(argv=None):

  train_summary, solver = train_network()

  [
      test_summary,
      average_test_loss,
      average_offset_loss_1,
      average_sem_offset_loss_1,
      average_seg_loss_1,
      average_seg_accuracy_1,
      average_miou_v1_1,
      average_miou_v2_1,
      average_miou_v3_1,
      average_offset_loss_2,
      average_sem_offset_loss_2,
      average_seg_loss_2,
      average_seg_accuracy_2,
      average_miou_v1_2,
      average_miou_v2_2,
      average_miou_v3_2,
      average_offset_loss_3,
      average_sem_offset_loss_3,
      average_seg_loss_3,
      average_seg_accuracy_3,
      average_miou_v1_3,
      average_miou_v2_3,
      average_miou_v3_3,
      test_loss,
      offset_loss_1,
      sem_offset_loss_1,
      seg_loss_1,
      seg_accuracy_1,
      offset_loss_2,
      sem_offset_loss_2,
      seg_loss_2,
      seg_accuracy_2,
      offset_loss_3,
      sem_offset_loss_3,
      seg_loss_3,
      seg_accuracy_3,
      node_position,
      point_predict_offset_1,
      point_predict_offset_sem_1,
      instance_center_1,
      semantic_center_1,
      point_gt_part_index_1,
      point_predict_part_index_1,
      point_predict_prob_1,
      point_gt_instance_index_1,
      point_predict_offset_2,
      point_predict_offset_sem_2,
      instance_center_2,
      semantic_center_2,
      point_gt_part_index_2,
      point_predict_part_index_2,
      point_predict_prob_2,
      point_gt_instance_index_2,
      point_predict_offset_3,
      point_predict_offset_sem_3,
      instance_center_3,
      semantic_center_3,
      point_gt_part_index_3,
      point_predict_part_index_3,
      point_predict_prob_3,
      point_gt_instance_index_3
  ] = test_network(FLAGS.test_data, visual=False)

  # checkpoint
  ckpt = tf.train.latest_checkpoint(FLAGS.ckpt)
  start_iters = 0 if not ckpt else int(ckpt[ckpt.find('iter') + 4:-5]) + 1
  print('start_iters: ', start_iters)

  # saver
  allvars = tf.all_variables()

  tf_saver = tf.train.Saver(var_list=allvars, max_to_keep=1)
  if ckpt:
    assert(os.path.exists(FLAGS.ckpt))
    tf_restore_saver = tf.train.Saver(var_list=allvars, max_to_keep=1)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    # tf summary
    summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

    # initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    if ckpt:
      tf_restore_saver.restore(sess, ckpt)

    obj_dir = os.path.join('obj', FLAGS.cache_folder)
    if not os.path.exists(obj_dir): os.makedirs(obj_dir, exist_ok=True)

    if FLAGS.phase == 'train':
      # start training
      for i in tqdm(range(start_iters, max_iter + 1)):

        if i % FLAGS.test_every_iter == 0 and i != 0:
          avg_test_loss = 0
          avg_offset_loss_1 = 0
          avg_sem_offset_loss_1 = 0
          avg_seg_loss_1 = 0
          avg_seg_accuracy_1 = 0
          avg_offset_loss_2 = 0
          avg_sem_offset_loss_2 = 0
          avg_seg_loss_2 = 0
          avg_seg_accuracy_2 = 0
          avg_offset_loss_3 = 0
          avg_sem_offset_loss_3 = 0
          avg_seg_loss_3 = 0
          avg_seg_accuracy_3 = 0
          all_point_gt_part_index_1 = np.empty([test_iter*n_test_point], dtype=np.int32)
          all_point_predict_part_index_1 = np.empty([test_iter*n_test_point], dtype=np.int32)
          all_point_predict_prob_entropy_1 = np.empty([test_iter*n_test_point], dtype=np.float32)
          all_point_predict_instance_index_1 = np.empty([test_iter*n_test_point], dtype=np.int32)
          all_point_gt_instance_index_1 = np.empty([test_iter*n_test_point], dtype=np.int32)
          all_point_gt_part_index_2 = np.empty([test_iter*n_test_point], dtype=np.int32)
          all_point_predict_part_index_2 = np.empty([test_iter*n_test_point], dtype=np.int32)
          all_point_predict_prob_entropy_2 = np.empty([test_iter*n_test_point], dtype=np.float32)
          all_point_predict_instance_index_2 = np.empty([test_iter*n_test_point], dtype=np.int32)
          all_point_gt_instance_index_2 = np.empty([test_iter*n_test_point], dtype=np.int32)
          all_point_gt_part_index_3 = np.empty([test_iter*n_test_point], dtype=np.int32)
          all_point_predict_part_index_3 = np.empty([test_iter*n_test_point], dtype=np.int32)
          all_point_predict_prob_entropy_3 = np.empty([test_iter*n_test_point], dtype=np.float32)
          all_point_predict_instance_index_3 = np.empty([test_iter*n_test_point], dtype=np.int32)
          all_point_gt_instance_index_3 = np.empty([test_iter*n_test_point], dtype=np.int32)
          n_point_count = 0
          all_point_shape_index = np.empty([test_iter*n_test_point], dtype=np.int32)
          for it in tqdm(range(test_iter)):
            [
                test_loss_value,
                offset_loss_1_value,
                sem_offset_loss_1_value,
                seg_loss_1_value,
                seg_accuracy_1_value,
                point_gt_part_index_1_value,
                point_predict_part_index_1_value,
                point_predict_prob_1_value,
                offset_loss_2_value,
                sem_offset_loss_2_value,
                seg_loss_2_value,
                seg_accuracy_2_value,
                point_gt_part_index_2_value,
                point_predict_part_index_2_value,
                point_predict_prob_2_value,
                offset_loss_3_value,
                sem_offset_loss_3_value,
                seg_loss_3_value,
                seg_accuracy_3_value,
                point_gt_part_index_3_value,
                point_predict_part_index_3_value,
                point_predict_prob_3_value,
                node_position_value,
                point_predict_offset_1_value,
                point_predict_offset_sem_1_value,
                point_gt_instance_index_1_value,
                point_predict_offset_2_value,
                point_predict_offset_sem_2_value,
                point_gt_instance_index_2_value,
                point_predict_offset_3_value,
                point_predict_offset_sem_3_value,
                point_gt_instance_index_3_value
            ] = sess.run(
                [
                    test_loss,
                    offset_loss_1,
                    sem_offset_loss_1,
                    seg_loss_1,
                    seg_accuracy_1,
                    point_gt_part_index_1,
                    point_predict_part_index_1,
                    point_predict_prob_1,
                    offset_loss_2,
                    sem_offset_loss_2,
                    seg_loss_2,
                    seg_accuracy_2,
                    point_gt_part_index_2,
                    point_predict_part_index_2,
                    point_predict_prob_2,
                    offset_loss_3,
                    sem_offset_loss_3,
                    seg_loss_3,
                    seg_accuracy_3,
                    point_gt_part_index_3,
                    point_predict_part_index_3,
                    point_predict_prob_3,
                    node_position,
                    point_predict_offset_1,
                    point_predict_offset_sem_1,
                    point_gt_instance_index_1,
                    point_predict_offset_2,
                    point_predict_offset_sem_2,
                    point_gt_instance_index_2,
                    point_predict_offset_3,
                    point_predict_offset_sem_3,
                    point_gt_instance_index_3
                ]
            )

            n_shape_point = point_gt_part_index_1_value.size
            assert n_point_count + n_shape_point <= test_iter*n_test_point, 'Test point number {} > {}={}*{}'.format(n_point_count + n_shape_point, test_iter*n_test_point, test_iter, n_test_point)
            avg_test_loss += test_loss_value
            avg_offset_loss_1 += offset_loss_1_value
            avg_sem_offset_loss_1 += sem_offset_loss_1_value
            avg_seg_loss_1 += seg_loss_1_value
            avg_seg_accuracy_1 += seg_accuracy_1_value
            avg_offset_loss_2 += offset_loss_2_value
            avg_sem_offset_loss_2 += sem_offset_loss_2_value
            avg_seg_loss_2 += seg_loss_2_value
            avg_seg_accuracy_2 += seg_accuracy_2_value
            avg_offset_loss_3 += offset_loss_3_value
            avg_sem_offset_loss_3 += sem_offset_loss_3_value
            avg_seg_loss_3 += seg_loss_3_value
            avg_seg_accuracy_3 += seg_accuracy_3_value
            all_point_gt_part_index_1[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_1_value
            all_point_predict_part_index_1[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_1_value
            all_point_gt_part_index_2[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_2_value
            all_point_predict_part_index_2[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_2_value
            all_point_gt_part_index_3[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_3_value
            all_point_predict_part_index_3[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_3_value
            all_point_gt_instance_index_1[n_point_count:n_point_count+n_shape_point] = point_gt_instance_index_1_value
            all_point_gt_instance_index_2[n_point_count:n_point_count+n_shape_point] = point_gt_instance_index_2_value
            all_point_gt_instance_index_3[n_point_count:n_point_count+n_shape_point] = point_gt_instance_index_3_value
            all_point_shape_index[n_point_count:n_point_count+n_shape_point] = it
            n_point_count += n_shape_point
          all_point_gt_part_index_1 = all_point_gt_part_index_1[:n_point_count]
          all_point_predict_part_index_1 = all_point_predict_part_index_1[:n_point_count]
          all_point_gt_part_index_2 = all_point_gt_part_index_2[:n_point_count]
          all_point_predict_part_index_2 = all_point_predict_part_index_2[:n_point_count]
          all_point_gt_part_index_3 = all_point_gt_part_index_3[:n_point_count]
          all_point_predict_part_index_3 = all_point_predict_part_index_3[:n_point_count]
          all_point_gt_instance_index_1 = all_point_gt_instance_index_1[:n_point_count]
          all_point_gt_instance_index_2 = all_point_gt_instance_index_2[:n_point_count]
          all_point_gt_instance_index_3 = all_point_gt_instance_index_3[:n_point_count]
          all_point_shape_index = all_point_shape_index[:n_point_count]
          avg_test_loss /= test_iter
          avg_offset_loss_1 /= test_iter
          avg_sem_offset_loss_1 /= test_iter
          avg_seg_loss_1 /= test_iter
          avg_seg_accuracy_1 /= test_iter
          avg_offset_loss_2 /= test_iter
          avg_sem_offset_loss_2 /= test_iter
          avg_seg_loss_2 /= test_iter
          avg_seg_accuracy_2 /= test_iter
          avg_offset_loss_3 /= test_iter
          avg_sem_offset_loss_3 /= test_iter
          avg_seg_loss_3 /= test_iter
          avg_seg_accuracy_3 /= test_iter
          miou_v1_1, _ = compute_iou_v1(all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, n_part_1, delete_0=FLAGS.delete_0)
          miou_v2_1, _ = compute_iou_v2(all_point_predict_part_index_1, all_point_gt_part_index_1, n_part_1, delete_0=FLAGS.delete_0)
          miou_v3_1, _ = compute_iou_v3(all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, n_part_1, delete_0=FLAGS.delete_0)
          miou_v1_2, _ = compute_iou_v1(all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, n_part_2, delete_0=FLAGS.delete_0)
          miou_v2_2, _ = compute_iou_v2(all_point_predict_part_index_2, all_point_gt_part_index_2, n_part_2, delete_0=FLAGS.delete_0)
          miou_v3_2, _ = compute_iou_v3(all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, n_part_2, delete_0=FLAGS.delete_0)
          miou_v1_3, _ = compute_iou_v1(all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, n_part_3, delete_0=FLAGS.delete_0)
          miou_v2_3, _ = compute_iou_v2(all_point_predict_part_index_3, all_point_gt_part_index_3, n_part_3, delete_0=FLAGS.delete_0)
          miou_v3_3, _ = compute_iou_v3(all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, n_part_3, delete_0=FLAGS.delete_0)
          summary = sess.run(test_summary,
              feed_dict={
                  average_test_loss: avg_test_loss,
                  average_offset_loss_1: avg_offset_loss_1,
                  average_sem_offset_loss_1: avg_sem_offset_loss_1,
                  average_seg_loss_1: avg_seg_loss_1,
                  average_seg_accuracy_1: avg_seg_accuracy_1,
                  average_miou_v1_1: miou_v1_1*100.0,
                  average_miou_v2_1: miou_v2_1*100.0,
                  average_miou_v3_1: miou_v3_1*100.0,
                  average_offset_loss_2: avg_offset_loss_2,
                  average_sem_offset_loss_2: avg_sem_offset_loss_2,
                  average_seg_loss_2: avg_seg_loss_2,
                  average_seg_accuracy_2: avg_seg_accuracy_2,
                  average_miou_v1_2: miou_v1_2*100.0,
                  average_miou_v2_2: miou_v2_2*100.0,
                  average_miou_v3_2: miou_v3_2*100.0,
                  average_offset_loss_3: avg_offset_loss_3,
                  average_sem_offset_loss_3: avg_sem_offset_loss_3,
                  average_seg_loss_3: avg_seg_loss_3,
                  average_seg_accuracy_3: avg_seg_accuracy_3,
                  average_miou_v1_3: miou_v1_3*100.0,
                  average_miou_v2_3: miou_v2_3*100.0,
                  average_miou_v3_3: miou_v3_3*100.0,
              })
          summary_writer.add_summary(summary, i)
          tf_saver.save(sess, os.path.join(FLAGS.logdir, 'model/iter{:06d}.ckpt'.format(i)))

          result_string = '\nIteration {}:\n Level1:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n seg accu    : {:6.4f}\n seg loss    : {:6.4f}\n off loss    : {:6.4f}\n sem loss    : {:6.4f}\n Level2:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n seg accu    : {:6.4f}\n seg loss    : {:6.4f}\n off loss    : {:6.4f}\n sem loss    : {:6.4f}\n Level3:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n seg accu    : {:6.4f}\n seg loss    : {:6.4f}\n off loss    : {:6.4f}\n sem loss    : {:6.4f}\n'.format(i, 
              miou_v1_1*100.0, miou_v2_1*100.0, miou_v3_1*100.0, avg_seg_accuracy_1, avg_seg_loss_1, avg_offset_loss_1, avg_sem_offset_loss_1,
              miou_v1_2*100.0, miou_v2_2*100.0, miou_v3_2*100.0, avg_seg_accuracy_2, avg_seg_loss_2, avg_offset_loss_2, avg_sem_offset_loss_2,
              miou_v1_3*100.0, miou_v2_3*100.0, miou_v3_3*100.0, avg_seg_accuracy_3, avg_seg_loss_3, avg_offset_loss_3, avg_sem_offset_loss_3)
          print(result_string); sys.stdout.flush()

        summary, _ = sess.run([train_summary, solver])
        summary_writer.add_summary(summary, i)

    else:
      assert(FLAGS.phase == 'test')
      # run_time
      avg_test_loss = 0
      avg_offset_loss_1 = 0
      avg_sem_offset_loss_1 = 0
      avg_seg_loss_1 = 0
      avg_seg_accuracy_1 = 0
      all_point_gt_part_index_1 = np.empty([test_iter*n_test_point], dtype=np.int32)
      all_point_predict_part_index_1 = np.empty([test_iter*n_test_point], dtype=np.int32)
      all_point_predict_semantic_instance_index_bandwidth_1 = np.empty([test_iter*n_test_point], dtype=np.int32)
      all_point_predict_prob_entropy_1 = np.empty([test_iter*n_test_point], dtype=np.float32)
      avg_offset_loss_2 = 0
      avg_sem_offset_loss_2 = 0
      avg_seg_loss_2 = 0
      avg_seg_accuracy_2 = 0
      all_point_gt_part_index_2 = np.empty([test_iter*n_test_point], dtype=np.int32)
      all_point_predict_part_index_2 = np.empty([test_iter*n_test_point], dtype=np.int32)
      all_point_predict_semantic_instance_index_bandwidth_2 = np.empty([test_iter*n_test_point], dtype=np.int32)
      all_point_predict_prob_entropy_2 = np.empty([test_iter*n_test_point], dtype=np.float32)
      avg_offset_loss_3 = 0
      avg_sem_offset_loss_3 = 0
      avg_seg_loss_3 = 0
      avg_seg_accuracy_3 = 0
      all_point_gt_part_index_3 = np.empty([test_iter*n_test_point], dtype=np.int32)
      all_point_predict_part_index_3 = np.empty([test_iter*n_test_point], dtype=np.int32)
      all_point_predict_semantic_instance_index_bandwidth_3 = np.empty([test_iter*n_test_point], dtype=np.int32)
      all_point_predict_prob_entropy_3 = np.empty([test_iter*n_test_point], dtype=np.float32)
      all_point_gt_instance_index_1 = np.empty([test_iter*n_test_point], dtype=np.int32)
      all_point_gt_instance_index_2 = np.empty([test_iter*n_test_point], dtype=np.int32)
      all_point_gt_instance_index_3 = np.empty([test_iter*n_test_point], dtype=np.int32)
      n_point_count = 0
      all_point_shape_index = np.empty([test_iter*n_test_point], dtype=np.int32)

      for it in tqdm(range(test_iter)):
        [
            test_loss_value,
            offset_loss_1_value,
            sem_offset_loss_1_value,
            seg_loss_1_value,
            seg_accuracy_1_value,
            point_gt_part_index_1_value,
            point_predict_part_index_1_value,
            point_predict_prob_1_value,
            offset_loss_2_value,
            sem_offset_loss_2_value,
            seg_loss_2_value,
            seg_accuracy_2_value,
            point_gt_part_index_2_value,
            point_predict_part_index_2_value,
            point_predict_prob_2_value,
            offset_loss_3_value,
            sem_offset_loss_3_value,
            seg_loss_3_value,
            seg_accuracy_3_value,
            point_gt_part_index_3_value,
            point_predict_part_index_3_value,
            point_predict_prob_3_value,
            node_position_value,
            point_predict_offset_1_value,
            point_predict_offset_sem_1_value,
            instance_center_1_value,
            semantic_center_1_value,
            point_gt_instance_index_1_value,
            point_predict_offset_2_value,
            point_predict_offset_sem_2_value,
            instance_center_2_value,
            semantic_center_2_value,
            point_gt_instance_index_2_value,
            point_predict_offset_3_value,
            point_predict_offset_sem_3_value,
            instance_center_3_value,
            semantic_center_3_value,
            point_gt_instance_index_3_value
        ] = sess.run(
            [
                test_loss,
                offset_loss_1,
                sem_offset_loss_1,
                seg_loss_1,
                seg_accuracy_1,
                point_gt_part_index_1,
                point_predict_part_index_1,
                point_predict_prob_1,
                offset_loss_2,
                sem_offset_loss_2,
                seg_loss_2,
                seg_accuracy_2,
                point_gt_part_index_2,
                point_predict_part_index_2,
                point_predict_prob_2,
                offset_loss_3,
                sem_offset_loss_3,
                seg_loss_3,
                seg_accuracy_3,
                point_gt_part_index_3,
                point_predict_part_index_3,
                point_predict_prob_3,
                node_position,
                point_predict_offset_1,
                point_predict_offset_sem_1,
                instance_center_1,
                semantic_center_1,
                point_gt_instance_index_1,
                point_predict_offset_2,
                point_predict_offset_sem_2,
                instance_center_2,
                semantic_center_2,
                point_gt_instance_index_2,
                point_predict_offset_3,
                point_predict_offset_sem_3,
                instance_center_3,
                semantic_center_3,
                point_gt_instance_index_3
            ]
        )

        n_shape_point = point_gt_part_index_1_value.size
        assert n_point_count + n_shape_point <= test_iter*n_test_point, 'Test point number {} > {}={}*{}'.format(n_point_count + n_shape_point, test_iter*n_test_point, test_iter, n_test_point)
        avg_test_loss += test_loss_value
        avg_offset_loss_1 += offset_loss_1_value
        avg_sem_offset_loss_1 += sem_offset_loss_1_value
        avg_seg_loss_1 += seg_loss_1_value
        avg_seg_accuracy_1 += seg_accuracy_1_value
        all_point_gt_part_index_1[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_1_value
        all_point_predict_part_index_1[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_1_value
        all_point_gt_instance_index_1[n_point_count:n_point_count+n_shape_point] = point_gt_instance_index_1_value
        avg_offset_loss_2 += offset_loss_2_value
        avg_sem_offset_loss_2 += sem_offset_loss_2_value
        avg_seg_loss_2 += seg_loss_2_value
        avg_seg_accuracy_2 += seg_accuracy_2_value
        all_point_gt_part_index_2[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_2_value
        all_point_predict_part_index_2[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_2_value
        all_point_gt_instance_index_2[n_point_count:n_point_count+n_shape_point] = point_gt_instance_index_2_value
        avg_offset_loss_3 += offset_loss_3_value
        avg_sem_offset_loss_3 += sem_offset_loss_3_value
        avg_seg_loss_3 += seg_loss_3_value
        avg_seg_accuracy_3 += seg_accuracy_3_value
        all_point_gt_part_index_3[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_3_value
        all_point_predict_part_index_3[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_3_value
        all_point_gt_instance_index_3[n_point_count:n_point_count+n_shape_point] = point_gt_instance_index_3_value
        all_point_shape_index[n_point_count:n_point_count+n_shape_point] = it

        # point entropy
        point_predict_prob_1_value = np.reshape(point_predict_prob_1_value, [-1, n_part_1])
        point_predict_prob_entropy_1 = entropy(point_predict_prob_1_value, axis=1)
        all_point_predict_prob_entropy_1[n_point_count:n_point_count+n_shape_point] = point_predict_prob_entropy_1
        point_predict_prob_2_value = np.reshape(point_predict_prob_2_value, [-1, n_part_2])
        point_predict_prob_entropy_2 = entropy(point_predict_prob_2_value, axis=1)
        all_point_predict_prob_entropy_2[n_point_count:n_point_count+n_shape_point] = point_predict_prob_entropy_2
        point_predict_prob_3_value = np.reshape(point_predict_prob_3_value, [-1, n_part_3])
        point_predict_prob_entropy_3 = entropy(point_predict_prob_3_value, axis=1)
        all_point_predict_prob_entropy_3[n_point_count:n_point_count+n_shape_point] = point_predict_prob_entropy_3

        # point predict instance label level 1
        predict_offset_node_position_1 = np.copy(node_position_value)
        predict_offset_node_position_1[:, :-1] += point_predict_offset_1_value
        predict_semantic_offset_node_position_bandwidth_1 = np.copy(predict_offset_node_position_1)
        predict_semantic_offset_node_position_bandwidth_1[:, :-1] += FLAGS.semantic_center_offset*(point_predict_offset_1_value - point_predict_offset_sem_1_value)/np.linalg.norm(point_predict_offset_1_value - point_predict_offset_sem_1_value, axis=1, keepdims=True)
        point_predict_semantic_instance_index_bandwidth_1 = semantic_mean_shift_cluster(point_predict_part_index_1_value.flatten(), predict_semantic_offset_node_position_bandwidth_1[:, :-1], bandwidth=FLAGS.bandwidth)
        all_point_predict_semantic_instance_index_bandwidth_1[n_point_count:n_point_count+n_shape_point] = point_predict_semantic_instance_index_bandwidth_1

        # point predict instance label level 2
        predict_offset_node_position_2 = np.copy(node_position_value)
        predict_offset_node_position_2[:, :-1] += point_predict_offset_2_value
        predict_semantic_offset_node_position_bandwidth_2 = np.copy(predict_offset_node_position_2)
        predict_semantic_offset_node_position_bandwidth_2[:, :-1] += FLAGS.semantic_center_offset*(point_predict_offset_2_value - point_predict_offset_sem_2_value)/np.linalg.norm(point_predict_offset_2_value - point_predict_offset_sem_2_value, axis=1, keepdims=True)
        point_predict_semantic_instance_index_bandwidth_2 = semantic_mean_shift_cluster(point_predict_part_index_2_value.flatten(), predict_semantic_offset_node_position_bandwidth_2[:, :-1], bandwidth=FLAGS.bandwidth)
        all_point_predict_semantic_instance_index_bandwidth_2[n_point_count:n_point_count+n_shape_point] = point_predict_semantic_instance_index_bandwidth_2

        # point predict instance label level 3
        predict_offset_node_position_3 = np.copy(node_position_value)
        predict_offset_node_position_3[:, :-1] += point_predict_offset_3_value
        predict_semantic_offset_node_position_bandwidth_3 = np.copy(predict_offset_node_position_3)
        predict_semantic_offset_node_position_bandwidth_3[:, :-1] += FLAGS.semantic_center_offset*(point_predict_offset_3_value - point_predict_offset_sem_3_value)/np.linalg.norm(point_predict_offset_3_value - point_predict_offset_sem_3_value, axis=1, keepdims=True)
        point_predict_semantic_instance_index_bandwidth_3 = semantic_mean_shift_cluster(point_predict_part_index_3_value.flatten(), predict_semantic_offset_node_position_bandwidth_3[:, :-1], bandwidth=FLAGS.bandwidth)
        all_point_predict_semantic_instance_index_bandwidth_3[n_point_count:n_point_count+n_shape_point] = point_predict_semantic_instance_index_bandwidth_3

        if FLAGS.test_visual and it < test_iter_visual:

          pc_filename = os.path.join(obj_dir, 'pc_L1_gt_seg_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_gt_part_index_1_value, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L1_pred_seg_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_predict_part_index_1_value, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L1_gt_ins_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_gt_instance_index_1_value+1, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L1_pred_ins_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_predict_semantic_instance_index_bandwidth_1+1, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L1_pred_ins_shifted_{:06d}_{:04d}.obj'.format(0, it))
          valid_index = np.ones(point_predict_semantic_instance_index_bandwidth_1.size, dtype=bool)
          for ins_index in np.unique(point_predict_semantic_instance_index_bandwidth_1):
            if np.sum(point_predict_semantic_instance_index_bandwidth_1 == ins_index) < 50:
              valid_index[point_predict_semantic_instance_index_bandwidth_1 == ins_index] = False
          vis_pointcloud.save_points((predict_semantic_offset_node_position_bandwidth_1)[valid_index],
              (point_predict_semantic_instance_index_bandwidth_1+1)[valid_index], pc_filename, depth=3)


          pc_filename = os.path.join(obj_dir, 'pc_L2_gt_seg_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_gt_part_index_2_value, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L2_pred_seg_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_predict_part_index_2_value, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L2_gt_ins_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_gt_instance_index_2_value+1, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L2_pred_ins_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_predict_semantic_instance_index_bandwidth_2+1, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L2_pred_ins_shifted_{:06d}_{:04d}.obj'.format(0, it))
          valid_index = np.ones(point_predict_semantic_instance_index_bandwidth_2.size, dtype=bool)
          for ins_index in np.unique(point_predict_semantic_instance_index_bandwidth_2):
            if np.sum(point_predict_semantic_instance_index_bandwidth_2 == ins_index) < 50:
              valid_index[point_predict_semantic_instance_index_bandwidth_2 == ins_index] = False
          vis_pointcloud.save_points((predict_semantic_offset_node_position_bandwidth_2)[valid_index],
              (point_predict_semantic_instance_index_bandwidth_2+1)[valid_index], pc_filename, depth=3)


          pc_filename = os.path.join(obj_dir, 'pc_L3_gt_seg_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_gt_part_index_3_value, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L3_pred_seg_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_predict_part_index_3_value, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L3_gt_ins_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_gt_instance_index_3_value+1, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L3_pred_ins_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_predict_semantic_instance_index_bandwidth_3+1, pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L3_pred_ins_shifted_{:06d}_{:04d}.obj'.format(0, it))
          valid_index = np.ones(point_predict_semantic_instance_index_bandwidth_3.size, dtype=bool)
          for ins_index in np.unique(point_predict_semantic_instance_index_bandwidth_3):
            if np.sum(point_predict_semantic_instance_index_bandwidth_3 == ins_index) < 50:
              valid_index[point_predict_semantic_instance_index_bandwidth_3 == ins_index] = False
          vis_pointcloud.save_points((predict_semantic_offset_node_position_bandwidth_3)[valid_index],
              (point_predict_semantic_instance_index_bandwidth_3+1)[valid_index], pc_filename, depth=3)

        n_point_count += n_shape_point

      all_point_gt_part_index_1 = all_point_gt_part_index_1[:n_point_count]
      all_point_predict_part_index_1 = all_point_predict_part_index_1[:n_point_count]
      all_point_predict_semantic_instance_index_bandwidth_1 = all_point_predict_semantic_instance_index_bandwidth_1[:n_point_count]
      all_point_predict_prob_entropy_1 = all_point_predict_prob_entropy_1[:n_point_count]
      all_point_gt_part_index_2 = all_point_gt_part_index_2[:n_point_count]
      all_point_predict_part_index_2 = all_point_predict_part_index_2[:n_point_count]
      all_point_predict_semantic_instance_index_bandwidth_2 = all_point_predict_semantic_instance_index_bandwidth_2[:n_point_count]
      all_point_predict_prob_entropy_2 = all_point_predict_prob_entropy_2[:n_point_count]
      all_point_gt_part_index_3 = all_point_gt_part_index_3[:n_point_count]
      all_point_predict_part_index_3 = all_point_predict_part_index_3[:n_point_count]
      all_point_predict_semantic_instance_index_bandwidth_3 = all_point_predict_semantic_instance_index_bandwidth_3[:n_point_count]
      all_point_predict_prob_entropy_3 = all_point_predict_prob_entropy_3[:n_point_count]
      all_point_gt_instance_index_1 = all_point_gt_instance_index_1[:n_point_count]
      all_point_gt_instance_index_2 = all_point_gt_instance_index_2[:n_point_count]
      all_point_gt_instance_index_3 = all_point_gt_instance_index_3[:n_point_count]
      avg_test_loss /= test_iter
      all_point_shape_index = all_point_shape_index[:n_point_count]
      avg_offset_loss_1 /= test_iter
      avg_sem_offset_loss_1 /= test_iter
      avg_seg_loss_1 /= test_iter
      avg_seg_accuracy_1 /= test_iter
      miou_v1_1, _ = compute_iou_v1(all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, n_part_1, delete_0=FLAGS.delete_0)
      miou_v2_1, _ = compute_iou_v2(all_point_predict_part_index_1, all_point_gt_part_index_1, n_part_1, delete_0=FLAGS.delete_0)
      miou_v3_1, _ = compute_iou_v3(all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, n_part_1, delete_0=FLAGS.delete_0)
      avg_offset_loss_2 /= test_iter
      avg_sem_offset_loss_2 /= test_iter
      avg_seg_loss_2 /= test_iter
      avg_seg_accuracy_2 /= test_iter
      miou_v1_2, _ = compute_iou_v1(all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, n_part_2, delete_0=FLAGS.delete_0)
      miou_v2_2, _ = compute_iou_v2(all_point_predict_part_index_2, all_point_gt_part_index_2, n_part_2, delete_0=FLAGS.delete_0)
      miou_v3_2, _ = compute_iou_v3(all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, n_part_2, delete_0=FLAGS.delete_0)
      avg_offset_loss_3 /= test_iter
      avg_sem_offset_loss_3 /= test_iter
      avg_seg_loss_3 /= test_iter
      avg_seg_accuracy_3 /= test_iter
      miou_v1_3, _ = compute_iou_v1(all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, n_part_3, delete_0=FLAGS.delete_0)
      miou_v2_3, _ = compute_iou_v2(all_point_predict_part_index_3, all_point_gt_part_index_3, n_part_3, delete_0=FLAGS.delete_0)
      miou_v3_3, _ = compute_iou_v3(all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, n_part_3, delete_0=FLAGS.delete_0)

      total_result_string = ''

      # level 1
      _, _, shape_mAP_1 = per_shape_mean_ap(all_point_predict_semantic_instance_index_bandwidth_1, all_point_gt_instance_index_1, all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, all_point_predict_prob_entropy_1, n_part_1, iou_threshold=0.5, delete_0=FLAGS.delete_0)
      _, _, mAP25_1, _ = per_part_mean_ap(all_point_predict_semantic_instance_index_bandwidth_1, all_point_gt_instance_index_1, all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, all_point_predict_prob_entropy_1, n_part_1, iou_threshold=0.25, delete_0=FLAGS.delete_0)
      _, _, mAP50_1, _ = per_part_mean_ap(all_point_predict_semantic_instance_index_bandwidth_1, all_point_gt_instance_index_1, all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, all_point_predict_prob_entropy_1, n_part_1, iou_threshold=0.5, delete_0=FLAGS.delete_0)
      _, _, mAP75_1, _ = per_part_mean_ap(all_point_predict_semantic_instance_index_bandwidth_1, all_point_gt_instance_index_1, all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, all_point_predict_prob_entropy_1, n_part_1, iou_threshold=0.75, delete_0=FLAGS.delete_0)

      result_string = 'Level1:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n seg accu    : {:6.4f}\n shape mAP   : {:5.2f}\n mAP25       : {:5.2f}\n mAP50       : {:5.2f}\n mAP75       : {:5.2f}\n seg loss    : {:6.4f}\n off loss    : {:6.4f}\n sem loss    : {:6.4f}\n'.format(miou_v1_1*100.0, miou_v2_1*100.0, miou_v3_1*100.0, avg_seg_accuracy_1, shape_mAP_1*100.0, mAP25_1*100.0, mAP50_1*100.0, mAP75_1*100.0, avg_seg_loss_1, avg_offset_loss_1, avg_sem_offset_loss_1)
      total_result_string += result_string

      # level 2
      _, _, shape_mAP_2 = per_shape_mean_ap(all_point_predict_semantic_instance_index_bandwidth_2, all_point_gt_instance_index_2, all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, all_point_predict_prob_entropy_2, n_part_2, iou_threshold=0.5, delete_0=FLAGS.delete_0)
      _, _, mAP25_2, _ = per_part_mean_ap(all_point_predict_semantic_instance_index_bandwidth_2, all_point_gt_instance_index_2, all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, all_point_predict_prob_entropy_2, n_part_2, iou_threshold=0.25, delete_0=FLAGS.delete_0)
      _, _, mAP50_2, _ = per_part_mean_ap(all_point_predict_semantic_instance_index_bandwidth_2, all_point_gt_instance_index_2, all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, all_point_predict_prob_entropy_2, n_part_2, iou_threshold=0.5, delete_0=FLAGS.delete_0)
      _, _, mAP75_2, _ = per_part_mean_ap(all_point_predict_semantic_instance_index_bandwidth_2, all_point_gt_instance_index_2, all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, all_point_predict_prob_entropy_2, n_part_2, iou_threshold=0.75, delete_0=FLAGS.delete_0)

      result_string = '\nLevel2:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n seg accu    : {:6.4f}\n shape mAP   : {:5.2f}\n mAP25       : {:5.2f}\n mAP50       : {:5.2f}\n mAP75       : {:5.2f}\n seg loss    : {:6.4f}\n off loss    : {:6.4f}\n sem loss    : {:6.4f}\n'.format(miou_v1_2*100.0, miou_v2_2*100.0, miou_v3_2*100.0, avg_seg_accuracy_2, shape_mAP_2*100.0, mAP25_2*100.0, mAP50_2*100.0, mAP75_2*100.0, avg_seg_loss_2, avg_offset_loss_2, avg_sem_offset_loss_2)
      total_result_string += result_string

      # level 3
      _, _, shape_mAP_3 = per_shape_mean_ap(all_point_predict_semantic_instance_index_bandwidth_3, all_point_gt_instance_index_3, all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, all_point_predict_prob_entropy_3, n_part_3, iou_threshold=0.5, delete_0=FLAGS.delete_0)
      _, _, mAP25_3, _ = per_part_mean_ap(all_point_predict_semantic_instance_index_bandwidth_3, all_point_gt_instance_index_3, all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, all_point_predict_prob_entropy_3, n_part_3, iou_threshold=0.25, delete_0=FLAGS.delete_0)
      _, _, mAP50_3, _ = per_part_mean_ap(all_point_predict_semantic_instance_index_bandwidth_3, all_point_gt_instance_index_3, all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, all_point_predict_prob_entropy_3, n_part_3, iou_threshold=0.5, delete_0=FLAGS.delete_0)
      _, _, mAP75_3, _ = per_part_mean_ap(all_point_predict_semantic_instance_index_bandwidth_3, all_point_gt_instance_index_3, all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, all_point_predict_prob_entropy_3, n_part_3, iou_threshold=0.75, delete_0=FLAGS.delete_0)

      result_string = '\nLevel3:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n seg accu    : {:6.4f}\n shape mAP   : {:5.2f}\n mAP25       : {:5.2f}\n mAP50       : {:5.2f}\n mAP75       : {:5.2f}\n seg loss    : {:6.4f}\n off loss    : {:6.4f}\n sem loss    : {:6.4f}\n'.format(miou_v1_3*100.0, miou_v2_3*100.0, miou_v3_3*100.0, avg_seg_accuracy_3, shape_mAP_3*100.0, mAP25_3*100.0, mAP50_3*100.0, mAP75_3*100.0, avg_seg_loss_3, avg_offset_loss_3, avg_sem_offset_loss_3)
      total_result_string += result_string

      print(total_result_string); sys.stdout.flush()
      with open(os.path.join(FLAGS.logdir, 'test_result_{}_total.txt'.format(test_iter)), 'w') as f:
        f.write(total_result_string)


if __name__ == '__main__':
  tf.app.run()
