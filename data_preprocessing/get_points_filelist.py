import os
import argparse

def get_points_filelist(category, phase, level):
  points_folder = 'ins_seg_points'
  shape_list = sorted([file for file in os.listdir(points_folder) if file.startswith(category) and file.endswith('level{}_{}.points'.format(level, phase))])
  print(category, phase, level, len(shape_list))
  out_file = os.path.join('filelist', '{}_level{}_{}_{}.txt'.format(category, level, phase, len(shape_list)))
  flag = 1
  with open(out_file, 'w') as f:
    for index, shape in enumerate(shape_list):
      f.write('{} {} {:04d}\n'.format(os.path.join('ins_seg_points', shape), flag, index))
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
  parser.add_argument('--level',
                      type=str,
                      help='1/2/3',
                      default='3')

  args = parser.parse_args()

  get_points_filelist(args.category, args.phase, args.level)