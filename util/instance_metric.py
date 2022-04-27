import os
import numpy as np
from scipy import stats


def compute_instance_score(point_predict_instance_label, point_predict_prob_entropy, min_points_per_cluster=10):
  n_instance = np.max(point_predict_instance_label)+1
  instance_score = np.zeros(n_instance, dtype=np.float32)
  instance_valid = np.zeros(n_instance, dtype=np.bool)
  for i in np.unique(point_predict_instance_label):
    instance_score[i] = 1-np.mean(point_predict_prob_entropy[point_predict_instance_label==i])
    instance_valid[i] = np.sum(point_predict_instance_label==i) > min_points_per_cluster
  return instance_score, instance_valid


def compute_ap(tp, fp, gt_npart, n_bins=100, plot_fn=None):
  assert len(tp) == len(fp)
  tp = np.cumsum(tp)
  fp = np.cumsum(fp)
  rec = tp/gt_npart
  prec = tp/(tp+fp)
  rec = np.insert(rec, 0, 0.0)
  prec = np.insert(prec, 0, 1.0)
  ap = 0.0
  delta = 1.0/n_bins
  out_rec = np.arange(0, 1+delta, delta)
  out_prec = np.zeros(n_bins+1, dtype=np.float32)
  for idx, t in enumerate(out_rec):
    prec1 = prec[rec>=t]
    if prec1.size == 0:
      p = 0.0
    else:
      p = np.max(prec1)
    out_prec[idx] = p
    ap = ap + p/(n_bins+1)
  if plot_fn is not None:
    import matplotlib.pyplot as plt
    base_folder = os.path.split(plot_fn)[0]
    if not os.path.isdir(base_folder): os.mkdir(base_folder)
    fig = plt.figure()
    plt.plot(out_rec, out_prec, 'b-')
    plt.title('PR-Curve (AP: {:4.2f}%)'.format(ap*100))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    fig.savefig(plot_fn)
    plt.close(fig)
  return ap


def per_shape_mean_ap(point_predict_instance_label, point_gt_instance_label, point_predict_semantic_label, point_gt_semantic_label, point_shape_index, point_predict_prob_entropy, n_part, iou_threshold=0.5, folder=None, delete_0=True, non_instance_semantic_label=0, min_points_per_cluster=10):
  n_shape = np.max(point_shape_index) + 1
  mean_aps = []
  shape_valids = []
  for i in range(n_shape):
    shape_point_index = point_shape_index == i
    shape_point_predict_instance_label = point_predict_instance_label[shape_point_index]
    shape_point_gt_instance_label = point_gt_instance_label[shape_point_index]
    shape_point_predict_semantic_label = point_predict_semantic_label[shape_point_index]
    shape_point_gt_semantic_label = point_gt_semantic_label[shape_point_index]
    shape_point_predict_prob_entropy = point_predict_prob_entropy[shape_point_index]
    if delete_0:
      non_neg_mask = shape_point_gt_semantic_label > non_instance_semantic_label
      if np.sum(non_neg_mask) == 0: 
        mean_aps.append(0.0)
        shape_valids.append(False)
        continue
      shape_point_predict_instance_label = shape_point_predict_instance_label[non_neg_mask]
      shape_point_gt_instance_label = shape_point_gt_instance_label[non_neg_mask]
      shape_point_predict_semantic_label = shape_point_predict_semantic_label[non_neg_mask]
      shape_point_gt_semantic_label = shape_point_gt_semantic_label[non_neg_mask]
      shape_point_predict_prob_entropy = shape_point_predict_prob_entropy[non_neg_mask]

    gt_ins_label = np.unique(shape_point_gt_instance_label)
    pred_n_ins = np.max(shape_point_predict_instance_label) + 1
    shape_instance_score, shape_instance_valid = compute_instance_score(shape_point_predict_instance_label, shape_point_predict_prob_entropy, min_points_per_cluster=min_points_per_cluster)

    true_pos_list = [[] for i in range(n_part)]
    false_pos_list = [[] for i in range(n_part)]
    gt_npart = np.zeros(n_part, dtype=np.int32)

    gt_mask_per_cat = [[] for i in range(n_part)]
    for j in gt_ins_label:
      sem_id = stats.mode(shape_point_gt_semantic_label[shape_point_gt_instance_label==j])[0][0]
      gt_mask_per_cat[sem_id].append(j)
      gt_npart[sem_id] += 1

    order = np.argsort(-shape_instance_score)
    gt_used = set()
    for j in range(pred_n_ins):
      idx = order[j]
      if shape_instance_valid[idx]:
        sem_id = stats.mode(shape_point_predict_semantic_label[shape_point_predict_instance_label==idx])[0][0]
        iou_max = 0.0; match_gt_id = -1
        for k in gt_mask_per_cat[sem_id]:
          if not(k in gt_used):
            predict_instance_index = shape_point_predict_instance_label==idx
            gt_instance_index = shape_point_gt_instance_label==k
            intersect = np.sum(predict_instance_index & gt_instance_index)
            union = np.sum(predict_instance_index | gt_instance_index)
            iou = intersect*1.0 / union
            if iou > iou_max:
              iou_max = iou
              match_gt_id = k
        if iou_max > iou_threshold:
          gt_used.add(match_gt_id)
          true_pos_list[sem_id].append(True)
          false_pos_list[sem_id].append(False)
        else:
          true_pos_list[sem_id].append(False)
          false_pos_list[sem_id].append(True)

    aps = np.zeros(n_part, dtype=np.float32)
    ap_valids = np.zeros(n_part, dtype=np.bool)
    start_j = 1 if delete_0 else 0
    for j in range(start_j, n_part):
      has_pred = len(true_pos_list[j]) > 0
      has_gt = gt_npart[j] > 0

      if has_pred and has_gt:
        cur_true_pos = np.array(true_pos_list[j], dtype=np.float32)
        cur_false_pos = np.array(false_pos_list[j], dtype=np.float32)
        aps[j] = compute_ap(cur_true_pos, cur_false_pos, gt_npart[j])
        ap_valids[j] = True
      elif has_pred and not has_gt:
        aps[j] = 0.0
        ap_valids[j] = True
      elif not has_pred and has_gt:
        aps[j] = 0.0
        ap_valids[j] = True
    if np.sum(ap_valids) > 0:
      mean_aps.append(np.sum(aps*ap_valids)/np.sum(ap_valids))
      shape_valids.append(True)
    else:
      mean_aps.append(0.0)
      shape_valids.append(False)

  mean_aps = np.array(mean_aps, dtype=np.float32)
  shape_valids = np.array(shape_valids, dtype=np.bool)
  mean_mean_ap = np.sum(mean_aps*shape_valids)/np.sum(shape_valids)
  return mean_aps, shape_valids, mean_mean_ap


def per_part_mean_ap(point_predict_instance_label, point_gt_instance_label, point_predict_semantic_label, point_gt_semantic_label, point_shape_index, point_predict_prob_entropy, n_part, iou_threshold=0.5, folder=None, delete_0=True, non_instance_semantic_label=0, min_points_per_cluster=10):
  n_shape = np.max(point_shape_index) + 1

  true_pos_list = [[] for i in range(n_part)]
  false_pos_list = [[] for i in range(n_part)]
  conf_score_list = [[] for i in range(n_part)]
  gt_npart = np.zeros(n_part, dtype=np.int32)

  for i in range(n_shape):
    shape_point_index = point_shape_index == i
    shape_point_predict_instance_label = point_predict_instance_label[shape_point_index]
    shape_point_gt_instance_label = point_gt_instance_label[shape_point_index]
    shape_point_predict_semantic_label = point_predict_semantic_label[shape_point_index]
    shape_point_gt_semantic_label = point_gt_semantic_label[shape_point_index]
    shape_point_predict_prob_entropy = point_predict_prob_entropy[shape_point_index]
    if delete_0:
      non_neg_mask = shape_point_gt_semantic_label > non_instance_semantic_label
      if np.sum(non_neg_mask) == 0: continue
      shape_point_predict_instance_label = shape_point_predict_instance_label[non_neg_mask]
      shape_point_gt_instance_label = shape_point_gt_instance_label[non_neg_mask]
      shape_point_predict_semantic_label = shape_point_predict_semantic_label[non_neg_mask]
      shape_point_gt_semantic_label = shape_point_gt_semantic_label[non_neg_mask]
      shape_point_predict_prob_entropy = shape_point_predict_prob_entropy[non_neg_mask]

    gt_ins_label = np.unique(shape_point_gt_instance_label)
    pred_n_ins = np.max(shape_point_predict_instance_label) + 1
    shape_instance_score, shape_instance_valid = compute_instance_score(shape_point_predict_instance_label, shape_point_predict_prob_entropy, min_points_per_cluster=min_points_per_cluster)

    gt_mask_per_cat = [[] for i in range(n_part)]
    for j in gt_ins_label:
      if j == -1: print('detect -1 instance label')
      sem_id = stats.mode(shape_point_gt_semantic_label[shape_point_gt_instance_label==j])[0][0]
      gt_mask_per_cat[sem_id].append(j)
      gt_npart[sem_id] += 1

    order = np.argsort(-shape_instance_score)
    gt_used = set()
    for j in range(pred_n_ins):
      idx = order[j]
      if shape_instance_valid[idx]:
        sem_id = stats.mode(shape_point_predict_semantic_label[shape_point_predict_instance_label==idx])[0][0]
        iou_max = 0.0; match_gt_id = -1
        for k in gt_mask_per_cat[sem_id]:
          if not(k in gt_used):
            predict_instance_index = shape_point_predict_instance_label==idx
            gt_instance_index = shape_point_gt_instance_label==k
            intersect = np.sum(predict_instance_index & gt_instance_index)
            union = np.sum(predict_instance_index | gt_instance_index)
            iou = intersect*1.0 / union
            if iou > iou_max:
              iou_max = iou
              match_gt_id = k
        if iou_max > iou_threshold:
          gt_used.add(match_gt_id)
          true_pos_list[sem_id].append(True)
          false_pos_list[sem_id].append(False)
          conf_score_list[sem_id].append(shape_instance_score[idx])
        else:
          true_pos_list[sem_id].append(False)
          false_pos_list[sem_id].append(True)
          conf_score_list[sem_id].append(shape_instance_score[idx])

  aps = np.zeros(n_part, dtype=np.float32)
  ap_valids = np.ones(n_part, dtype=np.bool)
  for i in range(n_part):
    if delete_0 and (i == 0):
      ap_valids[i] = False
      continue

    has_pred = len(true_pos_list[i]) > 0
    has_gt = gt_npart[i] > 0

    if not has_gt:
      ap_valids[i] = False
      continue
    if has_gt and not has_pred:
      continue

    cur_true_pos = np.array(true_pos_list[i], dtype=np.float32)
    cur_false_pos = np.array(false_pos_list[i], dtype=np.float32)
    cur_conf_score = np.array(conf_score_list[i], dtype=np.float32)

    order = np.argsort(-cur_conf_score)
    sorted_true_pos = cur_true_pos[order]
    sorted_false_pos = cur_false_pos[order]

    if folder is not None:
      filename = os.path.join(folder, 'img', 'part_{:04d}.png'.format(i))
      aps[i] = compute_ap(sorted_true_pos, sorted_false_pos, gt_npart[i], plot_fn=filename)
    else:
      aps[i] = compute_ap(sorted_true_pos, sorted_false_pos, gt_npart[i], plot_fn=None)

  mean_ap = np.sum(aps*ap_valids)/np.sum(ap_valids)
  return aps, ap_valids, mean_ap, gt_npart
