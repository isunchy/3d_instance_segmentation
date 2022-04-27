import os
import sys
import numpy as np


def fill_rotation_matrix(angle):
  cosx = np.cos(angle[0]); sinx = np.sin(angle[0])
  cosy = np.cos(angle[1]); siny = np.sin(angle[1])
  cosz = np.cos(angle[2]); sinz = np.sin(angle[2])
  rotx = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, cosx, sinx, 0.0], [0.0, -sinx, cosx, 0.0], [0.0, 0.0, 0.0, 1.0]])
  roty = np.array([[cosy, 0.0, -siny, 0.0], [0.0, 1.0, 0.0, 0.0], [siny, 0.0, cosy, 0.0], [0.0, 0.0, 0.0, 1.0]])
  rotz = np.array([[cosz, sinz, 0.0, 0.0], [-sinz, cosz, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
  return np.matmul(rotz, np.matmul(rotx, roty)).astype(np.float32)

def fill_translation_matrix(jitter):
  translation_matrix = np.array([[1.0, 0.0, 0.0, jitter[0]], [0.0, 1.0, 0.0, jitter[1]], [0.0, 0.0, 1.0, jitter[2]], [0.0, 0.0, 0.0, 1.0]])
  return translation_matrix.astype(np.float32)

def fill_scale_matrix(scale):
  scale_matrix = np.array([[scale[0], 0.0, 0.0, 0.0], [0.0, scale[1], 0.0, 0.0], [0.0, 0.0, scale[2], 0.0], [0.0, 0.0, 0.0, 1.0]])
  return scale_matrix.astype(np.float32)

def get_transform_matrix(angle, jitter, scale, return_rotation=False):
  rotation_matrix = fill_rotation_matrix(angle)
  translation_matrix = fill_translation_matrix(jitter)
  scale_matrix = fill_scale_matrix(scale)
  transform_matrix = np.matmul(scale_matrix, np.matmul(translation_matrix, rotation_matrix))
  if return_rotation is False:
    return transform_matrix
  else:
    return transform_matrix, rotation_matrix

def get_inverse_transform_matrix(angle, jitter, scale):
  rotation_matrix = fill_rotation_matrix(-angle)
  translation_matrix = fill_translation_matrix(-jitter)
  scale_matrix = fill_scale_matrix(1.0/scale)
  inverse_transform_matrix = np.matmul(rotation_matrix, np.matmul(translation_matrix, scale_matrix))
  return inverse_transform_matrix

if __name__ == '__main__':
  for i in range(5):
    angle = np.random.uniform(-5, 5, 3)
    angle = angle * 3.1415926 / 180.0
    jitter = np.random.uniform(-0.125, 0.125, 3)
    scale = np.random.uniform(0.75, 0.125, 1)
    scale = np.array([scale[0], scale[0], scale[0]])
    # for item in [angle, jitter, scale]:
    #   print(item, type(item))
    m = get_transform_matrix(angle, jitter, scale)
    # print(m)
    im = get_inverse_transform_matrix(angle, jitter, scale)
    # print(im)
    imm = np.matmul(im, m)
    print(imm)
    np.testing.assert_allclose(imm, np.eye(4), atol=1e-2)