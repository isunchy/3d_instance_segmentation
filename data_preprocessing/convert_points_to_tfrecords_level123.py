import os
import sys
import argparse
import tensorflow as tf
from tqdm import tqdm

sys.path.append('PointsData')
from Points import *


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def parse_points(filename):
  assert os.path.isfile(filename), filename
  p = Points()
  p.read_from_Relabeled_PartNet(filename)
  points_bytes = p.write_to_points_1()
  return points_bytes


def load_points(filename):
  assert os.path.isfile(filename), filename
  with open(filename, 'rb') as f:
    points_bytes = f.read()
  p = Points()
  p.read_points_1(filename)
  label = p.labels
  assert(label.size == 10000)
  return points_bytes, label


def get_data_label_pair(list_file):
  shape_id_list_1 = []
  shape_id_list_2 = []
  shape_id_list_3 = []
  shape_flag_list = []
  shape_index_list = []
  with open(list_file) as f:
    content = f.readlines()
  for line in content:
    shape_id_1, shape_id_2, shape_id_3, shape_flag, shape_index = line.split()
    shape_id_list_1.append(shape_id_1)
    shape_id_list_2.append(shape_id_2)
    shape_id_list_3.append(shape_id_3)
    shape_flag_list.append(int(shape_flag))
    shape_index_list.append(int(shape_index))
  return shape_id_list_1, shape_id_list_2, shape_id_list_3, shape_flag_list, shape_index_list


def write_data_to_tfrecords(list_file, records_name):
  assert 'level123' in list_file, list_file
  shape_id_list_1, shape_id_list_2, shape_id_list_3, shape_flag_list, shape_index_list = get_data_label_pair(list_file)
  n_shape = len(shape_id_list_1)
  writer = tf.io.TFRecordWriter(records_name)
  for i in tqdm(range(n_shape)):
    points_bytes_1, label_1 = load_points(shape_id_list_1[i])
    points_bytes_2, label_2 = load_points(shape_id_list_2[i])
    points_bytes_3, label_3 = load_points(shape_id_list_3[i])

    feature = {'points_bytes': _bytes_feature(points_bytes_1),
               'label_1': _int64_feature(label_1.astype(np.int64).tolist()),
               'label_2': _int64_feature(label_2.astype(np.int64).tolist()),
               'label_3': _int64_feature(label_3.astype(np.int64).tolist()),
               'points_flag': _int64_feature([shape_flag_list[i]]),
               'shape_index': _int64_feature([shape_index_list[i]])}

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Required Arguments
  parser.add_argument('--list_file',
                      type=str,
                      help='File containing the list of points data, and \
                            append the identity filename as the label',
                      required=True)
  parser.add_argument('--records_name',
                      type=str,
                      help='Name of tfrecords',
                      required=True)

  args = parser.parse_args()

  write_data_to_tfrecords(args.list_file,
                          args.records_name)