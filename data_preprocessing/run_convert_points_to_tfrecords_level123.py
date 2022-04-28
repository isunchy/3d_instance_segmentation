import os
from tqdm import tqdm

if not os.path.isdir('tfrecords_level123'): os.mkdir('tfrecords_level123')

filelist_folder = 'filelist_level123'

filelist = os.listdir(filelist_folder)
print(len(filelist))
print(filelist)
for file in tqdm(filelist):
  category = file.split('_')[0]
  cmd = 'python convert_points_to_tfrecords_level123.py --list_file {} --records_name {}'.format(os.path.join(filelist_folder, file), os.path.join('tfrecords_level123', file.replace('.txt', '.tfrecords')))
  print(cmd)
  os.system(cmd)




