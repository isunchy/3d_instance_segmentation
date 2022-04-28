import numpy as np
import os
from util import *
import json
import math
class Part():
    def __init__(self):
        self.__part_id = 0
        self.__parent_part_id = 0
        self.__motion_parent_id = -1
        self.__level_id = 0
        self.with_motion = 0
        self.motion_type = -1
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.direction = np.array([0, 0, 0], dtype=np.float32)
        self.int_attri_num = 5
        self.float_attri_num = 6
        self.motion_sons = []
        self.hrch_sons = []
        self.__transform_list = []
        self.__points_index = None
        
    def get_id(self):
        return self.__part_id

    def get_parent_id(self):
        return self.__parent_part_id

    def get_motion_parent_id(self):
        return self.__motion_parent_id

    def get_level_id(self):
        return self.__level_id

    def set_id(self, part_id, parent_part_id, motion_parent_id, level_id):
        self.__part_id = part_id
        self.__parent_part_id = parent_part_id
        self.__motion_parent_id = motion_parent_id
        self.__level_id = level_id

    def get_header_length(self):
        return self.int_attri_num * 4 + self.float_attri_num * 4

    def get_header(self):
        header = bytes()
        header += encode_int32(self.__part_id)
        header += encode_int32(self.__parent_part_id)
        header += encode_int32(self.with_motion)
        header += encode_int32(self.__motion_parent_id)
        header += encode_int32(self.motion_type)
        header += self.center.tobytes()
        header += self.direction.tobytes()
        return header
    def decode_header(self, header):
        int_attri = self.int_attri_num
        bytes_offset = 0
        self.__part_id, self.__parent_part_id, self.with_motion, self.__motion_parent_id, self.motion_type = np.frombuffer(header, dtype=np.int32, count=int_attri, offset=bytes_offset)
        bytes_offset += int_attri * 4
        self.center = np.frombuffer(header, dtype=np.float32, count=3, offset=bytes_offset)
        bytes_offset += 3 * 4
        self.direction = np.frombuffer(header, dtype=np.float32, count=3, offset=bytes_offset)
        return

    def build_points_index(self, part_id_list):
        self.__points_index = part_id_list == self.__part_id

    def pass_transform(self, transform):
        self.__transform_list.append(transform)
        for child_part in self.motion_sons:
            child_part.pass_transform(transform)

    def pass_global_transform(self, transform):
        self.center = apply_transform_list(self.center, [transform])
        self.direction = apply_transform_list(self.direction, [transform])
        for child_part in self.motion_sons:
            child_part.pass_global_transform(transform)

    def apply_transform(self, points, normals):
        part_points = points[self.__points_index]
        part_points = apply_transform_list(part_points, self.__transform_list)
        points[self.__points_index] = part_points
        part_normals = normals[self.__points_index]
        part_normals = apply_transform_list(part_normals, self.__transform_list)
        normals[self.__points_index] = part_normals
        self.__transform_list = []

    def generate_random_motion(self):
        step = np.random.uniform(0, 0.1)
        angle = np.random.uniform(-math.pi/2, math.pi/2)
        self.center = apply_transform_list(self.center, self.__transform_list)
        self.direction = apply_transform_list(self.direction, self.__transform_list, is_vector=True)
        if self.motion_type == -1:
            return
        elif self.motion_type == 0:#tranlation
            transform = Translation_Matrix(self.direction, step)
        elif self.motion_type == 1:#rotation
            transform = Rotation_Matrix(self.center, self.direction, angle)            
        elif self.motion_type == 2:#spiral
            transform = Spiral_Matrix(self.center, self.direction, step, angle)
        self.pass_transform(transform)
        return        

class Level():
    def __init__(self):
        self.part_num = 0
        self.part_list = []
        self.attr_num = 1
        self.parent_dict = {}
        self.part_index_dict = {}
        self.root_part = None

    def get_header_length(self):
        return self.attr_num * 4

    def get_header(self):
        header = bytes()
        header += encode_int32(self.part_num)
        return header

    def decode_header(self, header):
        self.part_num = np.frombuffer(header, dtype=np.int32)[0]
        for part_i in range(self.part_num):
            self.part_list.append(Part())
        return
    
    def build_index_dict(self):
        self.part_index_dict = {}
        for i in range(self.part_num):
            self.part_index_dict[self.part_list[i].get_id()] = i

    def build_parent_dict(self):
        self.parent_dict = {}
        for i in range(self.part_num):
            self.parent_dict[self.part_list[i].get_id()] = self.part_list[i].get_parent_id()

    def get_parent_id(self, part_id):
        if self.parent_dict.get(part_id)!=None:
            return self.parent_dict[part_id]
        else:
            return -1

    def get_part(self, part_id):
        if self.part_index_dict.get(part_id)!=None:
            return self.part_list[self.part_index_dict[part_id]]
        else:
            return None
    def build_motion_tree(self):
        for part in self.part_list:
            if part.get_motion_parent_id()==-1:
                self.root_part = part
            else:
                self.get_part(part.get_motion_parent_id()).motion_sons.append(part)
           
    def apply_transform(self, points, normals):
        for part in self.part_list:
            part.apply_transform(points, normals)

    def generate_random_motion(self):
        node_stack = [self.root_part]
        while len(node_stack)>0:
            curr_part = node_stack.pop()
            curr_part.generate_random_motion()
            for child_part in curr_part.motion_sons:
                node_stack.append(child_part)

    def pass_global_transform(self, transform_list):
        global_transform = compose_transform_list(transform_list)
        self.root_part.pass_global_transform(global_transform)
        return

class Points():
    def __init__(self):
        self.points = None
        self.normals = None
        self.part_ids = None
        self.labels = None
        self.features = None

        self.points_num = 0
        self.level_num = 0
        self.with_part = 0
        self.with_label = 0
        self.with_motion = 0
        self.feature_channel = 0
        self.attr_num = 6
        self.level_list = []
        self.center = None
        self.bBox = None
        self.radius = 0
        self.upright = np.array([0, 0, 1], dtype=np.float32)

    def calc_boundingBox(self):
        coord_max, coord_min = boundingBox(self.points)
        self.center = (coord_max + coord_min) / 2
        self.bBox = [coord_max, coord_min]
        self.radius = np.linalg.norm(coord_max-coord_min, ord=2) / 2

    def global_transform(self):
        scale = np.random.uniform(1, 1.1, (3))
        normal_transform_list = []
        global_transform_list = []
        global_transform_list.append(Scale_Matrix(self.center, scale))
        sheer = np.random.uniform(0, 0.1, (6))
        global_transform_list.append(Sheer_Matrx(self.center, sheer))
        #translation_direction = np.random.uniform(0, 0.5*self.radius, (3))
        #global_transform_list.append(Translation_Matrix(translation_direction, 1))
        angle = np.random.uniform(-math.pi, math.pi)
        global_transform_list.append(Rotation_Matrix(self.center, self.upright, angle))
        normal_transform_list.append(Rotation_Matrix(np.array([0., 0., 0.]), self.upright, angle))
        self.points = apply_transform_list(self.points, global_transform_list)
        self.normals = apply_transform_list(self.normals, normal_transform_list)
        if self.with_motion:
            self.level_list[0].pass_global_transform(global_transform_list) 
        return

    def get_part(self, level, part_id):
        return self.level_list[i].get_part(part_id)

    def get_part_id(self, index, level):
        part_id = -1
        lowest_part_id = self.part_ids[index]
        if level == self.level_num-1:
            part_id = lowest_part_id
        else:
            for i in range(self.level_num-1, level, -1):
                part_id = self.level_list[i].get_parent_id(part_id)
        return part_id

    def get_header_length(self):
        return self.attr_num * 4

    def build_check_tree(self):
        for level_i in self.level_list:
            level_i.build_index_dict()
            level_i.build_parent_dict()
            for part_i in level_i.part_list:
                part_i.build_points_index(self.part_ids)
        #check util last but one level
        for i in range(self.level_num-2):
            for part in self.level_list[i+1].part_list:
                parent_part = self.level_list[i].get_part(part.get_parent_id())
                if parent_part == None:
                    print("Error parent ID: %d of part %d in level %d"%(part.get_parent_id(), part.get_id(), i+1))
                else:
                    parent_part.hrch_sons.append(part)
        return

    def build_motion_tree(self):
        self.level_list[0].build_motion_tree()

    def apply_transform(self):
        self.level_list[0].apply_transform(self.points, self.normals)

    def generate_random_motion(self):
        self.level_list[0].generate_random_motion()

    def get_points_length(self):
        bytes_length = 0
        bytes_length += self.points_num * 3 * 4
        bytes_length += self.points_num * 3 * 4
        if self.with_part:
            bytes_length += self.points_num * 4
        if self.with_label:
            bytes_length += self.points_num * 4
        if self.feature_channel:
            bytes_length += self.points_num * self.feature_channel * 4
        return bytes_length

    def read_from_PartNet(self):
        return

    def read_from_numpy(self, points, normals, semantic_labels, instance_labels=None):
        points_num = points.shape[0]
        assert(points.shape == (points_num, 3))
        assert(normals.shape == (points_num, 3))
        assert(semantic_labels.shape == (points_num,))
        if instance_labels is not None: assert(instance_labels.shape == (points_num,))
        part_ids = semantic_labels
        if instance_labels is not None: part_ids = instance_labels*100 + semantic_labels
        Level0 = Level()
        part_id_list = np.unique(np.array(part_ids))
        Level0.part_num = len(part_id_list)
        for i in range(len(part_id_list)):
            new_part = Part()
            new_part.set_id(part_id_list[i], 0, -1, 0)
            Level0.part_list.append(new_part)
        self.points_num = points_num
        self.level_num = 1
        self.level_list.append(Level0)
        self.points = points
        self.normals = normals
        self.part_ids = semantic_labels
        self.with_part = 1
        self.with_motion = 0
        self.with_label = 1
        self.labels = part_ids.astype(np.float32)
        return

    def read_from_Relabeled_PartNet(self, filename):
        with open(filename) as f:
            content = f.readlines()
        points_num = len(content)
        points = np.zeros([points_num, 3], dtype=np.float32)
        normals = np.zeros([points_num, 3], dtype=np.float32)
        part_ids = np.zeros([points_num], dtype=np.int32)
        for i in range(points_num):
            x, y, z, nx, ny, nz, part_id = content[i].split(' ')
            points[i] = [x, y, z]
            normals[i] = [nx, ny, nz]
            part_ids[i] = int(float(part_id))
        part_ids[part_ids==36] = -1 # note the bad shape index
        valid_point_index = part_ids != -1
        points = points[valid_point_index]
        normals = normals[valid_point_index]
        part_ids = part_ids[valid_point_index]
        # new_points_num = part_ids.size
        select_index = np.linspace(0, part_ids.size-1, num=10000, dtype=int)
        points = points[select_index]
        normals = normals[select_index]
        part_ids = part_ids[select_index]
        new_points_num = part_ids.size
        Level0 = Level()
        part_id_list = np.unique(np.array(part_ids))
        Level0.part_num = len(part_id_list)
        for i in range(len(part_id_list)):
            new_part = Part()
            new_part.set_id(part_id_list[i], 0, -1, 0)
            Level0.part_list.append(new_part)
        self.points_num = new_points_num
        self.level_num = 1
        self.level_list.append(Level0)
        self.points = points
        self.normals = normals
        self.part_ids = part_ids
        self.with_part = 1
        self.with_motion = 0
        # for point integration
        self.with_label = 1
        self.labels = part_ids.astype(np.float32)
        # #####################
        return

    def read_from_ShapeNetCorev2(self, dataset_path, class_name, object_id):
        dir_path = os.path.join(dataset_path, class_name, object_id)
        with open(os.path.join(dir_path, 'models', 'model_normalized_deformation_mix.pts')) as f:
            content = f.readlines()
        points_num = len(content)
        points = np.zeros([points_num, 3], dtype=np.float32)
        normals = np.zeros([points_num, 3], dtype=np.float32)
        part_ids = np.zeros([points_num], dtype=np.int32)
        for i in range(points_num):
            x, y, z, nx, ny, nz, part_id, _, _, _ = content[i].split(' ')
            points[i] = [x, y, z]
            normals[i] = [nx, ny, nz]
            part_ids[i] = part_id
        Level0 = Level()
        part_id_list = np.unique(np.array(part_ids))
        Level0.part_num = len(part_id_list)
        for i in range(len(part_id_list)):
            new_part = Part()
            new_part.set_id(part_id_list[i], 0, -1, 0)
            Level0.part_list.append(new_part)
        self.points_num = points_num
        self.level_num = 1
        self.level_list.append(Level0)
        self.points = points
        self.normals = normals
        self.part_ids = part_ids
        self.with_part = 1
        self.with_motion = 0
        return

    def read_from_MotionDataset(self, dataset_path, class_name, object_id, density):
        dir_path = os.path.join(dataset_path, class_name, object_id)
        with open(os.path.join(dir_path, "motion_attributes.json"), "r") as fid:
            root_node = json.loads(fid.read())
            fid.close()
        node_stack = [(root_node, -1)]
        points = []
        normals = []
        part_ids = []
        part_id_count = 0
        Level0 = Level()
        motion_type_dict = {
            'none': -1,
            'translation': 0,
            'rotation': 1,
            'spiral': 2
        }
        while len(node_stack)>0:
            curr_node, motion_parent_id = node_stack.pop()
            new_part = Part()
            new_part.set_id(part_id_count, -1, motion_parent_id, 0)
            part_id_count += 1
            new_part.with_motion = 1
            new_part.motion_type = motion_type_dict[curr_node['motion_type']]
            new_part.center = np.array(curr_node['center'], dtype=np.float32)
            new_part.direction = np.array(curr_node['direction'], dtype=np.float32)
            if motion_parent_id!=-1:
                name = curr_node['dof_name']
            else:
                name = "none_motion"
            part_points, part_normals = points_from_mesh(os.path.join(dir_path, "part_objs", name+".obj"), density=density)
            part_points_num = part_points.shape[0]
            #print("part %d point num: %d" % (new_part.get_id(), part_points_num))
            self.points_num += part_points_num
            points.append(part_points)
            normals.append(part_normals)
            part_ids = part_ids + [new_part.get_id()] * part_points_num 
            Level0.part_num += 1
            Level0.part_list.append(new_part)
            for child in curr_node['children']:
                node_stack.append((child, new_part.get_id()))
        self.level_num = 1
        self.level_list.append(Level0)
        self.points = np.array(np.concatenate(points, axis=0), dtype=np.float32)
        self.normals = np.array(np.concatenate(normals, axis=0), dtype=np.float32)
        self.part_ids = np.array(part_ids, dtype=np.int32)
        self.with_part = 1
        self.with_motion = 1
        self.build_check_tree()
        self.build_motion_tree()
        self.calc_boundingBox()
        return

    def get_header(self):
        header = bytes()
        header += encode_int32(self.points_num)
        header += encode_int32(self.level_num)
        header += encode_int32(self.with_part)
        header += encode_int32(self.with_label)
        header += encode_int32(self.with_motion)
        header += encode_int32(self.feature_channel)
        return header

    def decode_header(self, header):
        self.points_num, self.level_num, self.with_part, self.with_label, self.with_motion, self.feature_channel = np.frombuffer(header, dtype=np.int32)
        for level_i in range(self.level_num):
            self.level_list.append(Level())
        return 

    def read_points_1(self, path):
        with open(path, "rb") as f:
            self.__init__()
            magic_str_ = f.read(16)
            self.points_num = np.frombuffer(f.read(4), dtype=np.int32)[0]
            int_flags_ = np.frombuffer(f.read(4), dtype=np.int32)[0]
            content_flags_ = [int_flags_%2, (int_flags_>>1)%2, (int_flags_>>2)%2, (int_flags_>>3)%2]
            channels_ = np.frombuffer(f.read(4 * 8), dtype=np.int32)
            ptr_dis_ = np.frombuffer(f.read(4 * 8), dtype=np.int32)

            self.points = np.frombuffer(f.read(4 * 3 * self.points_num), dtype=np.float32)
            self.points = np.reshape(self.points, (self.points_num, 3))
            self.normals = np.frombuffer(f.read(4 * 3 * self.points_num), dtype=np.float32)
            self.normals = np.reshape(self.normals, (self.points_num, 3))
            if content_flags_[2]:
                self.feature_channel = channels_[2]
                self.features = np.frombuffer(f.read(4 * self.feature_channel * self.points_num), dtype=np.float32)
                self.features = np.reshape(self.features, (self.points_num, self.feature_channel))
            if content_flags_[3]:
                self.with_label = 1
                # self.labels = np.frombuffer(f.read(4 * self.points_num), dtype=np.int32)
                self.labels = np.frombuffer(f.read(4 * self.points_num), dtype=np.float32)
            f.close()
        self.with_part = 0
        self.level_num = 1
        level0 = Level()
        level0.part_num = 1
        part0 = Part()
        level0.part_list.append(part0)
        self.level_list.append(level0)
        self.build_check_tree()
        self.calc_boundingBox()

    def write_to_points_1(self, path=None):
        magic_str_ = '_POINTS_1.0_\x00\x00\x00\x00'
        channels = [3, 3, self.feature_channel, self.with_label , 0, 0, 0, 0]
        point_bytes = self.points.tobytes()
        normal_bytes = self.normals.tobytes()
        # label_bytes = IntList_to_Bytes(self.labels if self.with_label else [])
        label_bytes = self.labels.tobytes() if self.with_label else []
        feature_bytes  = Float32List_to_Bytes(self.features if self.feature_channel>0 else [])

        ptr_dis_ = [0] * 8
        offset = len(magic_str_) + 4 + 4 + 4 * len(channels) + 4 * len(ptr_dis_)
        ptr_dis_[0] = offset
        offset += len(point_bytes)
        ptr_dis_[1] = offset
        offset += len(normal_bytes)
        ptr_dis_[2] = offset
        offset += len(feature_bytes)
        ptr_dis_[3] = offset
        offset += len(label_bytes)
        ptr_dis_[4] = offset

        content_flags_ = 1 + (1<<1) + (int(self.feature_channel>0)<<2) + (int(self.with_label)<<3)

        bytes_list = bytes()
        bytes_list += magic_str_.encode()
        bytes_list += IntList_to_Bytes(self.points_num)
        bytes_list += IntList_to_Bytes(content_flags_)
        bytes_list += IntList_to_Bytes(channels)
        bytes_list += IntList_to_Bytes(ptr_dis_)
        bytes_list += point_bytes
        bytes_list += normal_bytes
        bytes_list += feature_bytes
        bytes_list += label_bytes
        if path is not None:
            f = open(path, "wb")
            f.write(bytes_list)
            f.close()
            return
        else:
            return bytes_list


    def encode_points(self):
        bytes_list = bytes()
        bytes_list += self.points.tobytes()
        bytes_list += self.normals.tobytes()
        if self.with_part:
            bytes_list += self.part_ids.tobytes()
        if self.with_label:
            bytes_list += IntList_to_Bytes(self.labels)
        if self.feature_channel:
            bytes_list += Float32List_to_Bytes(self.features)
        return bytes_list

    def decode_points(self, bytes_list):
        bytes_offset = 0
        self.points = np.frombuffer(bytes_list, dtype=np.float32, count=self.points_num * 3, offset=bytes_offset)
        self.points = self.points.reshape((self.points_num, 3))
        bytes_offset += self.points_num * 3 * 4
        self.normals = np.frombuffer(bytes_list, dtype=np.float32, count=self.points_num * 3, offset=bytes_offset)
        self.normals = self.normals.reshape((self.points_num, 3))
        bytes_offset += self.points_num * 3 * 4
        if self.with_part:
            self.part_ids = np.frombuffer(bytes_list, dtype=np.int32, count=self.points_num, offset=bytes_offset)
            bytes_offset += self.points_num * 4
        if self.with_label:
            self.labels = np.frombuffer(bytes_list, dtype=np.int32, count=self.points_num, offset=bytes_offset)
            bytes_offset += self.points_num * 4
        if self.feature_channel:
            self.labels = np.frombuffer(bytes_list, dtype=np.float32, count=self.points_num * self.feature_channel, offset=bytes_offset)
            bytes_offset += self.points_num * self.feature_channel * 4
        return 

    def write_to_off(self, path):
        np.savetxt(path, self.points, fmt="%.6f", header="OFF\n%d 0 0"%self.points.shape[0], comments="")

    def write_to_points(self, path):
        bytes_array = bytes()
        #Header
        bytes_array += self.get_header()
        #LevelHeader
        for level_i in self.level_list:
            bytes_array += level_i.get_header()
        #PartHeader
        for level_i in self.level_list:
            for part_i in level_i.part_list:
                bytes_array += part_i.get_header()
        bytes_array += self.encode_points()
        with open(path, "wb") as f:
            f.write(bytes_array)
            f.close()
        return

    def read_from_points(self, path):
        self.__init__()
        with open(path, "rb") as f:
            #Decode Header
            self.decode_header(f.read(self.get_header_length()))
            #Decode LevelHeader
            for level_i in self.level_list:
                level_i.decode_header(f.read(level_i.get_header_length()))
            #Decode PartHeader
            for level_i in self.level_list:
                for part_i in level_i.part_list:
                    part_i.decode_header(f.read(part_i.get_header_length()))
            self.decode_points(f.read(self.get_points_length()))
            f.close()
        return

    def read_from_points_buffer(self, bytes_list):
        self.__init__()
        #Decode Header
        offset_begin = 0
        offset_end = offset_begin + self.get_header_length()
        self.decode_header(bytes_list[offset_begin: offset_end])
        
        #Decode LevelHeader
        for level_i in self.level_list:
            offset_begin = offset_end
            offset_end = offset_begin + level_i.get_header_length()
            level_i.decode_header(bytes_list[offset_begin: offset_end])
        #Decode PartHeader
        for level_i in self.level_list:
            for part_i in level_i.part_list:
                offset_begin = offset_end
                offset_end = offset_begin + part_i.get_header_length()
                part_i.decode_header(bytes_list[offset_begin: offset_end])
        offset_begin = offset_end        
        self.decode_points(bytes_list[offset_begin:-1])
        return