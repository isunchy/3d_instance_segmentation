import os

if not os.path.isdir('filelist'): os.mkdir('filelist')

category_level = {}

category_level_folder = 'after_merging_label_ids'

category_level_filelist = os.listdir(category_level_folder)
for file in category_level_filelist:
  if len(file.split('-'))==3:
    if file.split('-')[1] == 'level':
      category = file.split('-')[0]
      level = file[:-4].split('-')[-1]
      if category not in category_level:
        category_level[category] = [level]
      else:
        category_level[category].append(level)
print(category_level)
      
for k, v in category_level.items():
  print(k, v)
  for level in v:
    for phase in ['train', 'val', 'test']:
      cmd = 'python get_points_filelist.py --category {} --phase {} --level {}'.format(k, phase, level)
      print(cmd)
      os.system(cmd)




