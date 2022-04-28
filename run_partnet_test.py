import os

from util.category_info import category_info


category_list = category_info.keys()


print(category_info)

print(category_list)


gpu_id = 2


for category in category_list:
  n_test = category_info[category]['#shape']['test']
  train_data = os.path.join('data', '{}_level123_test_{}.tfrecords'.format(category, n_test))
  test_data = os.path.join('data', '{}_level123_test_{}.tfrecords'.format(category, n_test))
  assert os.path.isfile(train_data)
  assert os.path.isfile(test_data)
  n_part_1 = category_info[category]['#part'][1]
  n_part_2 = category_info[category]['#part'][2]
  n_part_3 = category_info[category]['#part'][3]

  script = 'python 3DInsSegNet.py --logdir log/PartNet/{} --train_data {} --test_data {} --test_data_visual {} --train_batch_size 8 --test_batch_size 1 --max_iter 100000 --test_every_iter 5000 --test_iter {} --test_iter_visual 0  --gpu {} --n_part_1 {} --n_part_2 {} --n_part_3 {} --level_1_weight 1 --level_2_weight 1 --level_3_weight 1 --phase test --seg_loss_weight 1 --offset_weight 1 --sem_offset_weight 1 --learning_rate 0.1 --ckpt weight/{} --delete_0 --notest_visual --depth 6 --weight_decay 0.0001 --stop_gradient --category {}'.format(category, train_data, test_data, test_data, n_test, gpu_id, n_part_1, n_part_2, n_part_3, category, category)

  print(script)
  os.system(script)