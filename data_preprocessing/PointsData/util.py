import numpy as np
import openmesh
import sys
import os

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stdout = sys.__stderr__

def boundingBox(points):
    coord_min = np.min(points, axis=0)
    coord_max = np.max(points, axis=0)
    return coord_max, coord_min

def FreeFormDeformation(points):
    coord_max, coord_min = boundingbox(points)

def encode_int32(x):
    x=int(x)
    return x.to_bytes(length=4, byteorder='little', signed=True)

def IntList_to_Bytes(int_list):
    #list_bytes = struct.pack('i'*len(int_list), *int_list)
    x = np.array(int_list, dtype=np.int32)
    list_bytes = x.tobytes()
    return list_bytes

def DoubleList_to_Bytes(float_list):
    #list_bytes = struct.pack('d'*len(float_list), *float_list)
    x = np.array(float_list, dtype=np.float64)
    list_bytes = x.tobytes()
    return list_bytes

def Float32List_to_Bytes(float_list):
    #list_bytes = struct.pack('f'*len(float_list), *float_list)
    x = np.array(float_list, dtype=np.float32)
    list_bytes = x.tobytes()
    return list_bytes

def normalized(a, axis=-1, order=2):
    norm = np.atleast_1d(np.linalg.norm(a, order, axis))
    norm[norm==0] = 1
    return a / np.expand_dims(norm, axis)

def compose_transform_list(transform_list):
    if(len(transform_list)==1):
        return transform_list[0]
    transform = np.eye(4,4)
    for i in range(len(transform_list)):
        transform = np.matmul(transform_list[i], transform)
    return transform

def apply_transform_list(points, transform_list, is_vector=False):
    squeeze = False
    if points.ndim == 1:
        points = np.expand_dims(points, axis=0)
        squeeze = True
    if is_vector:
        points = np.insert(points, 3, 0, axis=-1)
    else:
        points = np.insert(points, 3, 1, axis=-1)
    transform = compose_transform_list(transform_list)
    transform = np.array(transform, dtype=np.float32)
    points = np.matmul(transform, points.transpose()).transpose()[:,0:3]
    if squeeze:
        points = np.squeeze(points, axis=0)
    return points

def Sheer_Matrx(center, sheer):
    T1 = Translation_Matrix(-center, 1)
    Sh = np.eye(4)
    Sh[0][1] = sheer[0]
    Sh[0][2] = sheer[1]
    Sh[1][0] = sheer[2]
    Sh[1][2] = sheer[3]
    Sh[2][0] = sheer[4]
    Sh[2][1] = sheer[5]
    T2 = Translation_Matrix(center, 1)
    return np.matmul(T2, np.matmul(Sh, T1))

def Scale_Matrix(center, scale):
    T1 = Translation_Matrix(-center, 1)
    Sc = np.diag([scale[0], scale[1], scale[2], 1.])
    T2 = Translation_Matrix(center, 1)
    return np.matmul(T2, np.matmul(Sc, T1))

def Translation_Matrix(direction, step):
    direction = np.array(direction, dtype=np.float32)
    v = direction * step
    transform = np.array([
        [1, 0, 0, v[0]],
        [0, 1, 0, v[1]],
        [0, 0, 1, v[2]],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    return transform

def Rotation_Matrix(center, direction, angle):
    T1 = Translation_Matrix(-center, 1)
    v = normalized(direction, order=2)[0]
    R = np.array([
        [np.cos(angle)+v[0]*v[0]*(1-np.cos(angle)), v[0]*v[1]*(1-np.cos(angle))-v[2]*np.sin(angle), v[0]*v[2]*(1-np.cos(angle))+v[1]*np.sin(angle), 0],
        [v[1]*v[0]*(1-np.cos(angle))+v[2]*np.sin(angle), np.cos(angle)+v[1]*v[1]*(1-np.cos(angle)), v[1]*v[2]*(1-np.cos(angle))-v[0]*np.sin(angle), 0],
        [v[2]*v[0]*(1-np.cos(angle))-v[1]*np.sin(angle), v[2]*v[1]*(1-np.cos(angle))+v[0]*np.sin(angle), np.cos(angle)+v[2]*v[2]*(1-np.cos(angle)), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    T2 = Translation_Matrix(center, 1)
    return np.matmul(T2, np.matmul(R, T1))

def Spiral_Matrix(center, direction, step, angle):
    return np.matmul(Translation_Matrix(direction, step), Spiral_Matrix(center, direction, angle))

def points_from_mesh(mesh_path, density=5000):
    mesh = openmesh.read_trimesh(mesh_path)
    points = mesh.points()
    mesh.update_vertex_normals()
    normals = mesh.vertex_normals()
    if mesh.n_vertices() == 0:
        return points, normals
    #help(mesh)
    for fh in mesh.faces():
        point_list = []
        face_normal = mesh.calc_face_normal(fh)
        for vh in mesh.fv(fh):
            point_list.append(mesh.point(vh))
        p, q, r = point_list
        area = np.linalg.norm(np.cross(p-q, p-r))/2
        sample_num = max(int(area * density), 1)
        bary_coord = np.random.uniform(0, 1, (sample_num, 3))
        bary_coord = normalized(bary_coord, order=1)
        vert_coord = np.array(point_list)
        sample_coord = np.matmul(bary_coord, vert_coord)
        points = np.concatenate([points, sample_coord], axis=0)
        if sample_num > 1:
            stacked_face_normal = np.stack([face_normal]*sample_num, axis=0)
        else:
            stacked_face_normal = np.reshape(face_normal, (1,3))
        normals = np.concatenate([normals, stacked_face_normal], axis=0)
    return points, normals
