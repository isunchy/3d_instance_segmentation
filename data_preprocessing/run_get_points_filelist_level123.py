import os

if not os.path.isdir('filelist_level123'): os.mkdir('filelist_level123')

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

filelist_list = os.listdir('filelist')

for category, v in category_level.items():
  print(category, v)
  for phase in ['train', 'val', 'test']:
    flist = []
    for level in range(1, 4):
      level = str(level)
      if level in v:
        filename = [file for file in filelist_list if file.startswith('{}_level{}_{}'.format(category, level, phase))][0]
        flist.append(filename)
      else:
        filename = [file for file in filelist_list if file.startswith('{}_level{}_{}'.format(category, str(int(level)-1), phase)) or file.startswith('{}_level{}_{}'.format(category, str(int(level)-2), phase))][0]
        flist.append(filename)
      print(category, level, phase, filename)
    print(phase, category, flist)
    cmd = 'python get_points_filelist_level123.py --category {} --phase {} --f1 {} --f2 {} --f3 {}'.format(category, phase, os.path.join('filelist', flist[0]), os.path.join('filelist', flist[1]), os.path.join('filelist', flist[2]))
    print(cmd)
    os.system(cmd)
