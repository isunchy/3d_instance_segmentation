import os
import sys
import numpy as np
import h5py
import json
from tqdm import tqdm

sys.path.append('PointsData')
from Points import *

pts_h5_folder = 'ins_seg_h5'
category_list = os.listdir(pts_h5_folder)
# print(category_list)
label_h5_folder = 'ins_seg_h5_for_detection'
save_points_folder = 'ins_seg_points'
if not os.path.isdir(save_points_folder): os.mkdir(save_points_folder)

def get_h5_file_list(folder, prefix):
  return [file for file in os.listdir(folder) if file.startswith(prefix) and file.endswith('h5')]

def get_instance_label(gt_mask, gt_valid):
  instance_label = -np.ones(gt_mask.shape[1], dtype=np.int32)
  for i, valid in enumerate(gt_valid):
    if valid: instance_label[gt_mask[i]] = i
  return instance_label

for category in category_list:
  print(category)
  for phase in ['train', 'val', 'test']:
    if phase == 'test':
      label_h5_folder = 'ins_seg_h5_gt'
    else:
      label_h5_folder = 'ins_seg_h5_for_detection'
    h5_file_list = get_h5_file_list(os.path.join(pts_h5_folder, category), phase)
    print(h5_file_list)
    with open(os.path.join('train_val_test_split', '{}.{}.json'.format(category, phase)), 'r') as f:
      shape_name_list = json.load(f)
    print(len(shape_name_list))
    shape_cum_index = 0
    for h5_file in h5_file_list:
      filename = os.path.join(pts_h5_folder, category, h5_file)
      f = h5py.File(filename, 'r')
      filename = os.path.join(label_h5_folder, '{}-1'.format(category), h5_file)
      f1 = h5py.File(filename, 'r')
      f2, f3 = None, None
      filename = os.path.join(label_h5_folder, '{}-2'.format(category), h5_file)
      if os.path.isfile(filename): f2 = h5py.File(filename, 'r')
      filename = os.path.join(label_h5_folder, '{}-3'.format(category), h5_file)
      if os.path.isfile(filename): f3 = h5py.File(filename, 'r')
      # print(f1, f2, f3)
      n_shape = f['pts'].shape[0]
      print(shape_cum_index, n_shape, h5_file)
      for i in tqdm(range(n_shape)):
        points = f['pts'][i]
        normals = f['nor'][i]
        shape_index = shape_cum_index+i
        model_id = shape_name_list[shape_index]['model_id']
        anno_id = shape_name_list[shape_index]['anno_id']
        for j, file in enumerate([f1, f2, f3]):
          if file is not None:
            instance_labels = get_instance_label(file['gt_mask'][i], file['gt_valid'][i] if phase != 'test' else file['gt_mask_valid'][i])
            if phase != 'test':
              semantic_labels = file['gt_label'][i]
            else:
              mask_label = file['gt_mask_label'][i]
              semantic_labels = mask_label[instance_labels]+1
              semantic_labels[instance_labels==-1] = 0
            p = Points()
            p.read_from_numpy(points, normals, semantic_labels, instance_labels=instance_labels)
            save_filename = os.path.join(save_points_folder, '{}_{:04d}_{}_{}_level{}_{}.points'.format(category, shape_index, model_id, anno_id, j+1, phase))
            # print(save_filename)
            points_bytes = p.write_to_points_1(save_filename)
      shape_cum_index += n_shape