from sklearn.cluster import MeanShift
import numpy as np


def semantic_mean_shift_cluster(point_semantic_label, pts, bandwidth=0.1):
  point_instance_label = np.zeros_like(point_semantic_label, dtype=np.int32)
  cum_instance_index = 0
  for label in np.unique(point_semantic_label):
    cur_pts_index = np.reshape(np.argwhere(point_semantic_label==label), [-1])
    if cur_pts_index.size > 1:
      cur_pts = pts[cur_pts_index]
      ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
      ms.fit(cur_pts)
      point_instance_label[cur_pts_index] = ms.labels_ + cum_instance_index
      cur_n_instance = ms.cluster_centers_.shape[0]
    else:
      point_instance_label[cur_pts_index] = cum_instance_index
      cur_n_instance = 1
    cum_instance_index += cur_n_instance
  return point_instance_label

