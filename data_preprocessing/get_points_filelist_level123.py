import os
import argparse

def get_points_filelist(category, phase, f1, f2, f3):
  points_folder = 'ins_seg_points'
  content1 = open(f1).readlines()
  content2 = open(f2).readlines()
  content3 = open(f3).readlines()
  n_shape = len(content1)
  assert(n_shape==len(content2)==len(content3))
  filelist_folder = 'filelist_level123'
  if not os.path.isdir(filelist_folder): os.mkdir(filelist_folder)
  out_file = os.path.join(filelist_folder, '{}_level123_{}_{}.txt'.format(category, phase, n_shape))
  flag = 1
  with open(out_file, 'w') as f:
    for index in range(n_shape):
      f.write('{} {} {} {} {:04d}\n'.format(content1[index].split()[0], content2[index].split()[0], content3[index].split()[0], flag, index))
  print('write to {}'.format(out_file))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Required Arguments
  parser.add_argument('--category',
                      type=str,
                      help='Category',
                      default='Chair')
  parser.add_argument('--phase',
                      type=str,
                      help='train/val/test',
                      default='test')
  parser.add_argument('--f1',
                      type=str)
  parser.add_argument('--f2',
                      type=str)
  parser.add_argument('--f3',
                      type=str)

  args = parser.parse_args()

  get_points_filelist(args.category, args.phase, args.f1, args.f2, args.f3)