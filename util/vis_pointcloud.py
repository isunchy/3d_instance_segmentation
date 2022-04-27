import numpy as np
import os
import shutil

cube_vert_raw = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float)
cube_face = np.array([[1, 3, 7, 5], [1, 2, 4, 3], [3, 4, 8, 7], [5, 7, 8, 6], [1, 5, 6, 2], [2, 6, 8, 4]])
good_color = np.array([
[246,  83,  20],
[124, 187,   0],
[  0, 161, 241],
[255, 187,   0],
[ 11, 239, 239],
[247, 230,  49],
[255,  96, 165],
[178,  96, 255],
[242, 198,   4],
[252, 218, 123],
[ 77, 146,  33],
[161, 206, 107],
[ 41, 125, 198],
[126, 193, 221],
[198,  31,  40],
[252, 136, 123],
[  5, 112, 103],
[ 87, 193, 177],
[107,  53, 168],
[139, 117, 198],
[206,  37, 135],
[247, 155, 222],
[196,  98,  13],
[253, 184,  99],
[158,   1,  66],
[233,  93,  71],
[253, 190, 111],
[230, 245, 152],
[125, 203, 164],
[ 64, 117, 180],
[163,  79, 132]], np.float32)/256.


def generate_random_color_palette(n_color):
  np.random.seed(0)
  color = np.random.rand(n_color, 3)
  if n_color >=1:
    color[0] = np.array([1., 1., 1.])
    n_copy = min(31, n_color-1)
    color[1:1+n_copy] = good_color[:n_copy]
  return color


def generate_squential_color_palette():
  palette = np.array([
    [255,255,255],
    [253,212,158],
    [252,141,89],
    [215,48,31],
    [127,0,0]], dtype=float) / 255
  return palette


def save_material(palette, output_file):
  with open(output_file, 'w') as f:
    n_color = np.shape(palette)[0]
    for i in range(n_color):
      part_color = palette[i]
      f.write('newmtl m{}\nKd {} {} {}\nKa 0 0 0\n'.format(i,
          float(part_color[0]), float(part_color[1]), float(part_color[2])))


def copy_material(src_mtl_filename, des_mtl_filename):
  shutil.copyfile(src_mtl_filename, des_mtl_filename)


def save_points(position, part_ids, save_file, depth=5, refmtl_filename=None,
    squantial_color=False):
  n_color = np.max(part_ids) + 1
  mtl_filename = save_file.replace('.obj', '.mtl')
  if refmtl_filename is None:
    if squantial_color:
      color_palette = generate_squential_color_palette()
    else:
      color_palette = generate_random_color_palette(n_color)
    save_material(color_palette, mtl_filename)
  else:
    copy_material(refmtl_filename, mtl_filename)

  with open(save_file, 'w') as f:
    n_vert = np.shape(position)[0]
    assert(n_vert == np.shape(part_ids)[0])
    f.write('mtllib {}\n'.format(mtl_filename.split('/')[-1]))
    vert_offset = 0
    cube_vert = (cube_vert_raw-0.5) / (2**depth * 2) 
    for i in range(n_vert):
      part_id = part_ids[i]
      if squantial_color:
        if part_id > 4: part_id = 4
      f.write('usemtl m{}\n'.format(part_id))
      for j in range(8):
        x = position[i][0] + cube_vert[j][0]
        y = position[i][1] + cube_vert[j][1]
        z = position[i][2] + cube_vert[j][2]
        f.write("v {:6.4f} {:6.4f} {:6.4f}\n".format(x, y, z))
      faces = cube_face + vert_offset
      for j in range(6):
        f.write("f {} {} {} {}\n".format(faces[j][0], faces[j][1], faces[j][2], faces[j][3]))
      vert_offset += 8
