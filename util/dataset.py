import os
import sys
import tensorflow as tf

assert(os.path.isdir('ocnn/tensorflow'))
sys.path.append('ocnn/tensorflow')

from libs import *
import numpy as np
from transform import get_transform_matrix, get_inverse_transform_matrix


def get_label_mapping(category, slevel=3, dlevel=2):
  assert(slevel > dlevel)
  label_mapping = {
      'Bag':
          [
            [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
            ],
            [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
            ]
          ],
      'Bed':
          [
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], tf.int32), # 15
              tf.constant([0,1,2,3,3,4,4,4,5,5, 5, 6, 7, 8, 9], tf.int32), # 10
              tf.constant([0,1,1,2,2,2,2,2,2,2, 2, 2, 2, 3, 3], tf.int32), # 4
            ],
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9], tf.int32), # 10
              tf.constant([0,1,1,2,2,2,2,2,3,3], tf.int32), # 4
            ]
          ],
      'Bottle':
          [
            [
              tf.constant([0,1,2,3,4,5,6,7,8], tf.int32), # 9
              tf.constant([0,1,0,2,3,0,0,4,5], tf.int32), # 6
              tf.constant([0,1,0,2,3,0,0,4,5], tf.int32), # 6
            ],
            [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
            ]
          ],
      'Bowl':
          [
            [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
            ],
            [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
            ]
          ],
      'Chair':
          [
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38], tf.int32), # 39
              tf.constant([0,1,2,3,3,3,4,5,6,6, 6, 7, 8, 9,10,11,12,13,13,13,14,15,15,16,17,18,19,20,21,22,23,24,25,25,26,26,27,28,29], tf.int32), # 30
              tf.constant([0,1,1,2,2,2,2,2,2,2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 0, 0, 0], tf.int32), # 6
            ],
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], tf.int32), # 30
              tf.constant([0,1,1,2,2,2,2,3,3,3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 0, 0, 0], tf.int32), # 6
            ]
          ],
      'Clock':
          [
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10], tf.int32), # 11
              tf.constant([0,1,1,2,2,3,3,4,5,5, 0], tf.int32), # 6
              tf.constant([0,1,1,2,2,3,3,4,5,5, 0], tf.int32), # 6
            ],
            [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
            ]
          ],
      'Dishwasher':
          [
            [
              tf.constant([0,1,2,3,4,5,6], tf.int32), # 7
              tf.constant([0,1,2,2,3,3,4], tf.int32), # 5
              tf.constant([0,1,1,1,2,2,2], tf.int32), # 3
            ],
            [
              tf.constant([0,1,2,3,4], tf.int32), # 5
              tf.constant([0,1,1,2,2], tf.int32), # 3
            ]
          ],
      'Display':
          [
            [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,2], tf.int32), # 3
              tf.constant([0,1,2,2], tf.int32), # 3
            ],
            [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
            ]
          ],
      'Door':
          [
            [
              tf.constant([0,1,2,3,4], tf.int32), # 5
              tf.constant([0,1,2,2,3], tf.int32), # 4
              tf.constant([0,1,2,2,2], tf.int32), # 3
            ],
            [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,2], tf.int32), # 3
            ]
          ],
      'Earphone':
          [
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9], tf.int32), # 10
              tf.constant([0,1,1,1,2,3,4,4,4,5], tf.int32), # 6
              tf.constant([0,1,1,1,2,3,4,4,4,5], tf.int32), # 6
            ],
            [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
            ]
          ],
      'Faucet':
          [
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11], tf.int32), # 12
              tf.constant([0,1,2,3,4,4,4,5,6,7, 7, 7], tf.int32), # 8
              tf.constant([0,1,2,3,4,4,4,5,6,7, 7, 7], tf.int32), # 8
            ],
            [
              tf.constant([0,1,2,3,4,5,6,7], tf.int32), # 8
              tf.constant([0,1,2,3,4,5,6,7], tf.int32), # 8
            ]
          ],
      'Hat':
          [
            [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
            ],
            [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
            ]
          ],
      'Keyboard':
          [
            [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
            ],
            [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
            ]
          ],
      'Knife':
          [
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9], tf.int32), # 10
              tf.constant([0,1,1,1,2,3,3,3,4,4], tf.int32), # 5
              tf.constant([0,1,1,1,2,3,3,3,4,4], tf.int32), # 5
            ],
            [
              tf.constant([0,1,2,3,4], tf.int32), # 5
              tf.constant([0,1,2,3,4], tf.int32), # 5
            ]
          ],
      'Lamp':
          [
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], tf.int32), # 51
              tf.constant([0,1,2,2,3,4,5,6,7,8, 8, 9,10,11,12,13,14,15,16,16,17,17,17,17,18,19,20,21,21,22,23,24,25,26,26,27,27,27,27,27,27], tf.int32), # 42
              tf.constant([0,1,1,1,2,3,4,5,6,6, 6, 7, 8, 9, 9, 9, 9,10,10,10,10,10,10,10,11,11,12,13,13,13,14,15,16,17,17,17,17,17,17,17,17], tf.int32), # 11
            ],
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27], tf.int32), # 42
              tf.constant([0,1,1,2,3,4,5,6,6,7, 8, 9, 9, 9, 9,10,10,10,11,11,12,13,13,14,15,16,17,17], tf.int32), # 11
            ]
          ],
      'Laptop':
          [
            [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
            ],
            [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
            ]
          ],
      'Microwave':
          [
            [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,2,3,4], tf.int32), # 5
              tf.constant([0,1,1,1,1,2], tf.int32), # 3
            ],
            [
              tf.constant([0,1,2,3,4], tf.int32), # 5
              tf.constant([0,1,1,1,2], tf.int32), # 3
            ]
          ],
      'Mug':
          [
            [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
            ],
            [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
            ]
          ],
      'Refrigerator':
          [
            [
              tf.constant([0,1,2,3,4,5,6], tf.int32), # 7
              tf.constant([0,1,2,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,1,1,1,2,2], tf.int32), # 3
            ],
            [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,1,1,2,2], tf.int32), # 3
            ]
          ],
      'Scissors':
          [
            [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
            ],
            [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
            ]
          ],
      'StorageFurniture':
          [
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], tf.int32), # 24
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,16,17,17,17,18,18,18], tf.int32), # 19
              tf.constant([0,1,2,3,3,3,3,3,3,3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6], tf.int32), # 7
            ],
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], tf.int32), # 19
              tf.constant([0,1,2,3,3,3,3,3,3,3, 3, 4, 4, 4, 4, 4, 5, 5, 6], tf.int32), # 7
            ]
          ],
      'Table':
          [
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50], tf.int32), # 51
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,13,13,14,15,16,17,18,19,20,21,22,23,24,24,25,26,27,28,29,30,31,31,32,32,32,32,32,33,34,35,36,37,37,38,39,40,41], tf.int32), # 42
              tf.constant([0,1,2,3,3,0,4,4,5,6, 7, 0, 8, 9, 9, 9, 9, 9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10], tf.int32), # 11
            ],
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41], tf.int32), # 42
              tf.constant([0,1,2,3,3,0,4,4,5,6, 7, 0, 8, 9, 9, 9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10], tf.int32), # 11
            ]
          ],
      'TrashCan':
          [
            [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10], tf.int32), # 11
              tf.constant([0,1,1,1,2,2,2,2,3,4, 4], tf.int32), # 5
              tf.constant([0,1,1,1,2,2,2,2,3,4, 4], tf.int32), # 5
            ],
            [
              tf.constant([0,1,2,3,4], tf.int32), # 5
              tf.constant([0,1,2,3,4], tf.int32), # 5
            ]
          ],
      'Vase':
          [
            [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,1,2,3,3], tf.int32), # 4
              tf.constant([0,1,1,2,3,3], tf.int32), # 4
            ],
            [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
            ]
          ],
  }
  return label_mapping[category][3-slevel][slevel-dlevel]


def compute_instance_center(pts, label):
  label.astype(np.int32)
  pts_instance_center = np.empty_like(pts)
  for label_id in np.unique(label):
    instance_mask = label==label_id
    instance_pts = pts[instance_mask]
    instance_center = np.mean(instance_pts, axis=0)
    pts_instance_center[instance_mask] = instance_center
  return pts_instance_center

def compute_semantic_center(instance_label, semantic_label, instance_center):
  instance_label = instance_label.astype(np.int32).flatten()
  semantic_label = semantic_label.astype(np.int32).flatten()
  pts_semantic_center = np.empty_like(instance_center)
  for slabel in np.unique(semantic_label):
    semantic_center = np.zeros([3], dtype=np.float32)
    instance_count = 0
    for ilabel in np.unique(instance_label[semantic_label==slabel]):
      semantic_center += instance_center[np.argwhere(instance_label==ilabel)[0]].flatten()
      instance_count += 1
    semantic_center /= instance_count
    pts_semantic_center[semantic_label==slabel] = semantic_center
  return pts_semantic_center

def float_to_bytes(data):
  return data.tobytes()

def apply_transform_to_points(transform_matrix, points):
  points = tf.pad(points, tf.constant([[0, 0], [0, 1]]), constant_values=1.0) # [n_point, 4]
  transformed_points = tf.matmul(transform_matrix, points, transpose_b=True) # [4, n_point]
  return tf.transpose(transformed_points[:3, :])

def split_semantic_instance_label(label):
  semantic_label = label % 100
  instance_label = label // 100
  return semantic_label.astype(np.float32), instance_label.astype(np.float32)

class PointsPreprocessor:
  def __init__(self, depth, test=False):
    self._depth = depth
    self._test = test

  def __call__(self, record):
    raw_points_bytes, label_1, label_2, label_3, points_flag, shape_index = self.parse_example(record)
    radius, center = bounding_sphere(raw_points_bytes)
    raw_points_bytes = normalize_points(raw_points_bytes, radius, center)

    # get semantic and instance label
    semantic_points_bytes_1, instance_points_bytes_1 = self.split_label(raw_points_bytes, label_1) # [], []
    semantic_points_bytes_2, instance_points_bytes_2 = self.split_label(raw_points_bytes, label_2) # [], []
    semantic_points_bytes_3, instance_points_bytes_3 = self.split_label(raw_points_bytes, label_3) # [], []

    # get instance and semantic center
    points_instance_center_1 = self.get_instance_center(instance_points_bytes_1) # [n_point, 3]
    points_semantic_center_1 = self.get_semantic_center(instance_points_bytes_1, semantic_points_bytes_1, points_instance_center_1) # [n_point, 3]
    points_instance_center_2 = self.get_instance_center(instance_points_bytes_2) # [n_point, 3]
    points_semantic_center_2 = self.get_semantic_center(instance_points_bytes_2, semantic_points_bytes_2, points_instance_center_2) # [n_point, 3]
    points_instance_center_3 = self.get_instance_center(instance_points_bytes_3) # [n_point, 3]
    points_semantic_center_3 = self.get_semantic_center(instance_points_bytes_3, semantic_points_bytes_3, points_instance_center_3) # [n_point, 3]

    if self._test:
      octree = points2octree(semantic_points_bytes_1, depth=self._depth, full_depth=2, node_dis=True)
      instance_center_bytes_1 = tf.py_func(float_to_bytes, [points_instance_center_1], Tout=tf.string)
      semantic_center_bytes_1 = tf.py_func(float_to_bytes, [points_semantic_center_1], Tout=tf.string)
      instance_center_bytes_2 = tf.py_func(float_to_bytes, [points_instance_center_2], Tout=tf.string)
      semantic_center_bytes_2 = tf.py_func(float_to_bytes, [points_semantic_center_2], Tout=tf.string)
      instance_center_bytes_3 = tf.py_func(float_to_bytes, [points_instance_center_3], Tout=tf.string)
      semantic_center_bytes_3 = tf.py_func(float_to_bytes, [points_semantic_center_3], Tout=tf.string)
      return [octree, semantic_points_bytes_1, instance_center_bytes_1, instance_points_bytes_1, semantic_center_bytes_1,
          semantic_points_bytes_2, instance_center_bytes_2, instance_points_bytes_2, semantic_center_bytes_2,
          semantic_points_bytes_3, instance_center_bytes_3, instance_points_bytes_3, semantic_center_bytes_3]
    else:
      # augment points
      transform_matrix, rotation_matrix = self.get_augment_matrix() # [4, 4], [4, 4]
      points_bytes_1 = self.augment_points(semantic_points_bytes_1, transform_matrix, rotation_matrix) # []
      points_bytes_2 = self.augment_points(semantic_points_bytes_2, transform_matrix, rotation_matrix) # []
      points_bytes_3 = self.augment_points(semantic_points_bytes_3, transform_matrix, rotation_matrix) # []
      # clip points
      inbox_points_bytes_1, _, inbox_instance_points_bytes_1, instance_center_bytes_1, semantic_center_bytes_1 = self.get_clip_pts(
          points_bytes_1, instance_points_bytes_1, points_instance_center_1, points_semantic_center_1, transform_matrix) # [], _, [], []
      inbox_points_bytes_2, _, inbox_instance_points_bytes_2, instance_center_bytes_2, semantic_center_bytes_2 = self.get_clip_pts(
          points_bytes_2, instance_points_bytes_2, points_instance_center_2, points_semantic_center_2, transform_matrix) # [], _, [], []
      inbox_points_bytes_3, _, inbox_instance_points_bytes_3, instance_center_bytes_3, semantic_center_bytes_3 = self.get_clip_pts(
          points_bytes_3, instance_points_bytes_3, points_instance_center_3, points_semantic_center_3, transform_matrix) # [], _, [], []
      # get octree
      octree = points2octree(inbox_points_bytes_1, depth=self._depth, full_depth=2, node_dis=True)
      return [octree, inbox_points_bytes_1, inbox_points_bytes_2, inbox_points_bytes_3, points_flag,
          inbox_instance_points_bytes_1, inbox_instance_points_bytes_2, inbox_instance_points_bytes_3,
          instance_center_bytes_1, semantic_center_bytes_1, instance_center_bytes_2, semantic_center_bytes_2,
          instance_center_bytes_3, semantic_center_bytes_3]


  def split_label(self, points_bytes, label):
    points_pts = points_property(points_bytes, property_name='xyz', channel=3) # [n_point, 3]
    points_normal = points_property(points_bytes, property_name='normal', channel=3) # [n_point, 3]
    points_semantic_label, points_instance_label = tf.py_func(split_semantic_instance_label, [label], Tout=[tf.float32, tf.float32]) # [10000], [10000]
    semantic_points_bytes = points_new(points_pts, points_normal, tf.zeros([0]), points_semantic_label)
    instance_points_bytes = points_new(points_pts, points_normal, tf.zeros([0]), points_instance_label)
    return semantic_points_bytes, instance_points_bytes

  def get_instance_center(self, points_bytes):
    points_pts = points_property(points_bytes, property_name='xyz', channel=3) # [n_point, 3]
    points_label = points_property(points_bytes, property_name='label', channel=1) # [n_point, 1]
    points_label = tf.reshape(points_label, [-1]) # [n_point]
    points_instance_center = tf.py_func(compute_instance_center, [points_pts, points_label], Tout=tf.float32) # [n_point, 3]
    return points_instance_center

  def get_semantic_center(self, instance_points_bytes, semantic_points_bytes, points_instance_center):
    points_pts = points_property(instance_points_bytes, property_name='xyz', channel=3) # [n_point, 3]
    points_instance_label = points_property(instance_points_bytes, property_name='label', channel=1) # [n_point, 1]
    points_semantic_label = points_property(semantic_points_bytes, property_name='label', channel=1) # [n_point, 1]
    points_semantic_center = tf.py_func(compute_semantic_center, [points_instance_label, points_semantic_label, points_instance_center], Tout=tf.float32) # [n_point, 3]
    return points_semantic_center

  def get_augment_matrix(self):
    rotation_angle = 10
    rnd = tf.random.uniform(shape=[3], minval=-rotation_angle, maxval=rotation_angle, dtype=tf.int32)
    angle = tf.cast(rnd, dtype=tf.float32) * 3.14159265 / 180.0
    scale = tf.random.uniform(shape=[3], minval=0.75, maxval=1.25, dtype=tf.float32)
    scale = tf.stack([scale[0]]*3)
    jitter = tf.random.uniform(shape=[3], minval=-0.125, maxval=0.125, dtype=tf.float32)
    transform_matrix, rotation_matrix = tf.py_func(get_transform_matrix, [angle, jitter, scale, True], Tout=[tf.float32, tf.float32]) # [4, 4], [4, 4]
    return transform_matrix, rotation_matrix

  def augment_points(self, points_bytes, transform_matrix, rotation_matrix):
    points_pts = points_property(points_bytes, property_name='xyz', channel=3) # [n_point, 3]
    points_normal = points_property(points_bytes, property_name='normal', channel=3) # [n_point, 3]
    points_label = points_property(points_bytes, property_name='label', channel=1) # [n_point, 1]
    transformed_points_pts = apply_transform_to_points(transform_matrix, points_pts) # [n_point, 3]
    transformed_points_normal = apply_transform_to_points(rotation_matrix, points_normal) # [n_point, 3]
    points_bytes = points_new(transformed_points_pts, transformed_points_normal, tf.zeros([0]), points_label)
    return points_bytes

  def get_clip_pts(self, points_bytes, instance_points_bytes, points_instance_center, points_semantic_center, transform_matrix):
    points_pts = points_property(points_bytes, property_name='xyz', channel=3) # [n_point, 3]
    points_normal = points_property(points_bytes, property_name='normal', channel=3) # [n_point, 3]
    points_label = points_property(points_bytes, property_name='label', channel=1) # [n_point, 1]
    points_instance_label = points_property(instance_points_bytes, property_name='label', channel=1) # [n_point, 1]
    inbox_mask = self.clip_pts(points_pts) # [n_point]
    points_pts = tf.boolean_mask(points_pts, inbox_mask) # [n_inbox_point, 3]
    points_normal = tf.boolean_mask(points_normal, inbox_mask) # [n_inbox_point, 3]
    points_label = tf.boolean_mask(points_label, inbox_mask) # [n_inbox_point, 1]
    points_instance_label = tf.boolean_mask(points_instance_label, inbox_mask) # [n_inbox_point, 1]
    points_bytes = points_new(points_pts, points_normal, tf.zeros([0]), points_label)
    instance_points_bytes = points_new(points_pts, points_normal, tf.zeros([0]), points_instance_label)
    points_instance_center = tf.boolean_mask(points_instance_center, inbox_mask) # [n_inbox_point, 3]
    points_instance_center = apply_transform_to_points(transform_matrix, points_instance_center) # [n_inbox_point, 3]
    points_instance_center_bytes = tf.py_func(float_to_bytes, [points_instance_center], Tout=tf.string)
    points_semantic_center = tf.boolean_mask(points_semantic_center, inbox_mask) # [n_inbox_point, 3]
    points_semantic_center = apply_transform_to_points(transform_matrix, points_semantic_center) # [n_inbox_point, 3]
    points_semantic_center_bytes = tf.py_func(float_to_bytes, [points_semantic_center], Tout=tf.string)
    return points_bytes, inbox_mask, instance_points_bytes, points_instance_center_bytes, points_semantic_center_bytes

  def clip_pts(self, pts):
    abs_pts = tf.abs(pts) # [n_point, 3]
    max_value = tf.math.reduce_max(abs_pts, axis=1) # [n_point]
    inbox_mask = tf.cast(max_value <= 1.0, dtype=tf.bool) # [n_point]
    return inbox_mask

  def parse_example(self, record):
    features = {'points_bytes': tf.io.FixedLenFeature([], tf.string),
                'label_1': tf.io.FixedLenFeature([10000], tf.int64),
                'label_2': tf.io.FixedLenFeature([10000], tf.int64),
                'label_3': tf.io.FixedLenFeature([10000], tf.int64),
                'points_flag': tf.io.FixedLenFeature([1], tf.int64),
                'shape_index': tf.io.FixedLenFeature([1], tf.int64)
                 }
    parsed = tf.io.parse_single_example(record, features)
    return [parsed['points_bytes'], parsed['label_1'], parsed['label_2'], parsed['label_3'],
      parsed['points_flag'], parsed['shape_index']]


def points_dataset(record_name, batch_size, depth=6, test=False):
  def merge_octrees_training(octrees, inbox_points_bytes_1, inbox_points_bytes_2, inbox_points_bytes_3, points_flag,
      inbox_instance_points_bytes_1, inbox_instance_points_bytes_2, inbox_instance_points_bytes_3,
      instance_center_bytes_1, semantic_center_bytes_1, instance_center_bytes_2,
      semantic_center_bytes_2, instance_center_bytes_3, semantic_center_bytes_3):
    octree = octree_batch(octrees)
    return [octree, inbox_points_bytes_1, inbox_points_bytes_2, inbox_points_bytes_3, points_flag,
        inbox_instance_points_bytes_1, inbox_instance_points_bytes_2, inbox_instance_points_bytes_3,
        instance_center_bytes_1, semantic_center_bytes_1, instance_center_bytes_2,
        semantic_center_bytes_2, instance_center_bytes_3, semantic_center_bytes_3]
  def merge_octrees_test(octrees, points_bytes_1, instance_center_bytes_1, instance_points_bytes_1, semantic_center_bytes_1,
      points_bytes_2, instance_center_bytes_2, instance_points_bytes_2, semantic_center_bytes_2,
      points_bytes_3, instance_center_bytes_3, instance_points_bytes_3, semantic_center_bytes_3):
    octree = octree_batch(octrees)
    return [octree, points_bytes_1, instance_center_bytes_1, instance_points_bytes_1, semantic_center_bytes_1,
        points_bytes_2, instance_center_bytes_2, instance_points_bytes_2, semantic_center_bytes_2,
        points_bytes_3, instance_center_bytes_3, instance_points_bytes_3, semantic_center_bytes_3]
  with tf.name_scope('points_dataset'):
    dataset = tf.data.TFRecordDataset([record_name]).repeat()
    if test is False:
      dataset = dataset.shuffle(100)
    return dataset.map(PointsPreprocessor(depth, test=test), num_parallel_calls=8).batch(batch_size) \
                  .map(merge_octrees_test if test else merge_octrees_training, num_parallel_calls=8) \
                  .prefetch(8).make_one_shot_iterator().get_next()
