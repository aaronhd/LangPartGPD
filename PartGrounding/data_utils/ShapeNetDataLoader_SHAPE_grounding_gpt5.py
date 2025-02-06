# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizerFast  # BertTokenizer
import glob
import random
from scipy.spatial.transform import Rotation as R
import h5py
import copy

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    # print('normnalize point cloud!')
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[0])
    np.random.shuffle(idx)
    return batch_data[idx, :]


# 作用: 对每个点云进行随机缩放, 实现方法是乘积因子直接与点云数据相乘即可
def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    batch_data = np.expand_dims(batch_data, 0)
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)  # 0.8~1.25间的随机缩放
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]  # 每个点都进行缩放
    return batch_data[0]


# 作用: 对每个点云进行随机平移, 对点云中的每个点添加一个随机的移动距离
def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    batch_data = np.expand_dims(batch_data, 0)
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))  # 对每个batch的点云设置一个随机的移动偏差
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]  # 每个点都进行移动

    return batch_data[0]


def load_corpus(root_path, mode='full'):
    cache_dict = {}
    parts = set()
    lang_mode = {'known_all': 'part_A', 'object_unknown': 'part_B', 'part_unknown': 'part_C',
                 'part_unknown_object_known': 'part_D'}
    if mode == 'full':
        for sub_folder in ['part_A', 'part_B', 'part_C', 'part_D', 'part_E']:
            file_list = glob.glob(root_path + '/Lang_SHAPE_sentence/instructions/' + sub_folder + "/*.txt")
            for file in file_list:
                cat_part = file.split('/')[-1][:-4]
                # part = cat_part.split('_')[-1]
                with open(file, 'r') as f:
                    lang_data = f.readlines()
                    if sub_folder == 'part_A':
                        cache_dict[cat_part] = lang_data
                    else:
                        cache_dict[cat_part].extend(lang_data)

        # print(cache_dict['earphone_headband'][24618])
        # print(len(cache_dict['earphone_headband']))
    else:
        sub_folder = lang_mode[mode]
        file_list = glob.glob(root_path + '/Lang_SHAPE_sentence/instructions/' + sub_folder + "/*.txt")
        for file in file_list:
            cat_part = file.split('/')[-1][:-4]
            part = cat_part.split('_')[-1]
            parts.add(part)
            with open(file, 'r') as f:
                lang_data = f.readlines()
                cache_dict[cat_part] = lang_data
    return cache_dict


class PartNormalDataset(Dataset):
    def __init__(self, root='/media/aaronstc/hard_1/dataset/shapenet_mesh_tmp_dataset', npoints=2500, split='train',
                 train_mode='object-wise', class_choice=None, normal_channel=False, data_mode='full'):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.data_mode = data_mode
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.aux_info = []
        print('GPT dataloader Partial Dataset Loader with Rotate aug: gpt5')
        self.aux_info.append('GPT dataloader Partial Dataset Loader with Rotate aug: gpt5')
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print('----1----')
        # print(self.cat)
        self.cat = {k: v for k, v in self.cat.items()}
        self.cat2 = {v: k for k, v in self.cat.items()}

        # print('----2----')
        # print(self.cat)
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))
        # print(self.classes_original)
        # {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8,
        # 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}
        # corpus_path = self.root + '/split_corpus/' + self.data_mode + '/' + split + '_dict.json'
        corpus_path = self.root + '/LangSHAPE/corpus_data/' + self.data_mode + '/' + split + '_dict.json'

        with open(corpus_path, 'r') as f:
            data_str = f.read()
            self. lang_dict = json.loads(data_str)

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        if train_mode == 'part-wise':
            # train_mode_tmp = train_mode + '_20240513'
            train_mode_tmp = train_mode
        else:
            train_mode_tmp = train_mode

        print(train_mode_tmp, split)
        self.aux_info.append(train_mode_tmp)
        if split == 'trainval':
            with open(os.path.join(self.root, 'split_data', train_mode_tmp, split + '.txt')) as f:
                data = f.readlines()
                fns = [item.strip() for item in data]
        if split == 'train':
            with open(os.path.join(self.root, 'split_data', train_mode_tmp, split + '.txt')) as f:
                data = f.readlines()
                fns = [item.strip() for item in data]
        if split == 'val':
            with open(os.path.join(self.root, 'split_data', train_mode_tmp, split + '.txt')) as f:
                data = f.readlines()
                fns = [item.strip() for item in data]
        if split == 'test':
            with open(os.path.join(self.root, 'split_data', train_mode_tmp, split + '.txt')) as f:
                data = f.readlines()
                fns = [item.strip() for item in data]
        if split == 'trainvaltest':
            with open(os.path.join(self.root, 'split_data', train_mode_tmp, split + '.txt')) as f:
                data = f.readlines()
                fns = [item.strip() for item in data]
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        # print(self.classes)
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 60000
        # mapping from original label to share part label
        # 39-->0  39-->4
        self.shape_dict = {'0': 0, '5': 0, '11': 0, '21': 0, '35': 0, '37': 0, '39': 0, '41': 0, '1': 1, '2': 2, '3': 3,
                           '4': 4,
                           '23': 4, '33': 4, '36': 4, '6': 5, '7': 6, '8': 7, '9': 8, '12': 9, '13': 10, '31': 10,
                           '14': 11, '48': 11,
                           '15': 12, '16': 13, '17': 14, '18': 15, '19': 16, '20': 17, '22': 18, '24': 19, '26': 19,
                           '25': 20, '27': 21, '28': 22,
                           '29': 23, '30': 24, '10': 25, '32': 25, '44': 25, '34': 26, '38': 27, '40': 28, '42': 29,
                           '43': 30, '45': 31, '46': 32,
                           '47': 33, '49': 34}

        self.shape_to_label = {'body': 0, 'wing': 1, 'tail': 2, 'engine': 3, 'handle': 4, 'peak': 5, 'panel': 6,
                               'roof': 7, 'hood': 8, 'back': 9, 'seat': 10, 'leg': 11, 'arm': 12, 'earphone': 13,
                               'headband': 14, 'microphone': 15, 'head': 16, 'neck': 17, 'blade': 18, 'base': 19,
                               'shade': 20, 'tube': 21, 'keyboard': 22, 'screen': 23, 'tank': 24, 'wheel': 25,
                               'light': 26, 'barrel': 27, 'trigger': 28, 'fin': 29, 'nose': 30, 'deck': 31,
                               'bearing': 32, 'top': 33, 'drawer': 34}

        self.label_to_shape = {0: 'body', 1: 'wing', 2: 'tail', 3: 'engine', 4: 'handle', 5: 'peak', 6: 'panel',
                               7: 'roof',
                               8: 'hood', 9: 'back', 10: 'seat', 11: 'leg', 12: 'arm', 13: 'earphone', 14: 'headband',
                               15: 'microphone', 16: 'head', 17: 'neck', 18: 'blade', 19: 'base', 20: 'shade',
                               21: 'tube',
                               22: 'keyboard', 23: 'screen', 24: 'tank', 25: 'wheel', 26: 'light', 27: 'barrel',
                               28: 'trigger', 29: 'fin', 30: 'nose', 31: 'deck', 32: 'bearing', 33: 'top', 34: 'drawer'}

        # create training index
        self.datapath = []
        for fn in fns:
            cat_number = fn.split('/')[0]
            # print(self.cat2[cat_number], type(fn))
            cat = self.cat2[cat_number]
            file_name = fn.split('/')[-1]
            part = file_name.split('_')[-1]
            new_part = int(self.shape_dict[part])
            new_part_name = self.label_to_shape[new_part]
            cat_part = cat + '_' + new_part_name
            # print(cat_part)
            # print(self.lang_dict.keys())
            if cat_part.lower() in self.lang_dict.keys():
                self.datapath.append((self.cat2[cat_number], fn))
        # for item in self.cat:
        #     for fn in self.meta[item]:
        #         self.datapath.append((item, fn))

    def mapping_shape(self, point_seg, file_name):
        # print(point_seg)
        ground_part = int(file_name.split('_')[-1])
        for idx in range(len(point_seg)):
            # print('------1------', idx, point_seg[idx, ])
            # print(point_seg[idx], ground_part)
            if point_seg[idx] != ground_part:
                point_seg[idx] = 0
            else:
                point_seg[idx] = 1
            # point_seg[idx] = self.shape_dict[str(point_seg[idx, ])]
            # print('------2------', point_seg[idx])
        return point_seg

    def rot_augmentation(self, pc):
        degree = [0, 45, 90, 135, 180, 225, 270, 315]
        ind = np.random.choice(len(degree), size=3, replace=True)
        # print(ind)
        r_z = R.from_euler('z', degree[ind[0]], degrees=True)
        rot_z = r_z.as_matrix()
        r_y = R.from_euler('y', degree[ind[1]], degrees=True)
        rot_y = r_y.as_matrix()
        r_x = R.from_euler('x', degree[ind[2]], degrees=True)
        rot_x = r_x.as_matrix()
        tmp = pc[:, 0:3]
        tmp = np.dot(rot_z, tmp.T).T
        tmp = np.dot(rot_y, tmp.T).T
        tmp = np.dot(rot_x, tmp.T).T
        return tmp

    def get_lang_tokenizer(self, obj_cls):
        seg_classes = {'Earphone': [13, 14, 15], 'Motorbike': [25, 4, 26, 0, 24, 10], 'Rocket': [0, 29, 30],
                       'Car': [7, 8, 25, 0], 'Laptop': [22, 23], 'Cap': [5, 6], 'Skateboard': [25, 31, 32],
                       'Mug': [4, 0],
                       'Guitar': [16, 17, 0], 'Bag': [4, 0], 'Lamp': [19, 20, 21], 'Table': [33, 11, 34],
                       'Airplane': [0, 1, 2, 3], 'Pistol': [27, 4, 28], 'Chair': [9, 10, 11, 12], 'Knife': [18, 4]}

        label_to_shape = {0: 'body', 1: 'wing', 2: 'tail', 3: 'engine', 4: 'handle', 5: 'peak', 6: 'panel', 7: 'roof',
                          8: 'hood', 9: 'back', 10: 'seat', 11: 'leg', 12: 'arm', 13: 'earphone', 14: 'headband',
                          15: 'microphone', 16: 'head', 17: 'neck', 18: 'blade', 19: 'base', 20: 'shade', 21: 'tube',
                          22: 'keyboard', 23: 'screen', 24: 'tank', 25: 'wheel', 26: 'light', 27: 'barrel',
                          28: 'trigger', 29: 'fin', 30: 'nose', 31: 'deck', 32: 'bearing', 33: 'top', 34: 'drawer'}

        cls_label_to_cat = {0: 'Airplane', 1: 'Bag', 2: 'Cap', 3: 'Car', 4: 'Chair', 5: 'Earphone', 6: 'Guitar',
                            7: 'Knife', 8: 'Lamp', 9: 'Laptop', 10: 'Motorbike', 11: 'Mug', 12: 'Pistol', 13: 'Rocket',
                            14: 'Skateboard', 15: 'Table'}

        obj_name = cls_label_to_cat[obj_cls]
        shape_in_obj = []
        for seg_label in seg_classes[obj_name]:
            shape_in_obj.append(label_to_shape[seg_label])
        obj_sent = ' [SEP] '.join(shape_in_obj)
        encoded_input = self.tokenizer(obj_sent, return_tensors='pt')
        tokens_tensor = encoded_input['input_ids'].cuda()
        token_type_ids = encoded_input['token_type_ids'].cuda()
        attention_mask = encoded_input['attention_mask'].cuda()

        encoded_input = {'input_ids': tokens_tensor,
                         'token_type_ids': token_type_ids,
                         'attention_mask': attention_mask}
        print('<==========')
        print(obj_name)
        print(obj_sent)
        print(encoded_input)
        print('==========>')
        return encoded_input

    def __getitem__(self, index):
        # if index in self.cache:
        #     # point_set, cls, seg = self.cache[index]
        #     point_set, cls, seg, file_name = self.cache[index]
        # else:
        fn = self.datapath[index]
        cat = self.datapath[index][0]

        file_name = fn[1].split('/')[-1]
        part = file_name.split('_')[-1]
        new_part = int(self.shape_dict[part])
        new_part_name = self.label_to_shape[new_part]
        cat_part = cat + '_' + new_part_name

        cls = self.classes[cat]
        cls = np.array([cls]).astype(np.int32)

        # print(self.lang_dict.keys())
        lang_data = self.lang_dict[cat_part.lower()]
        sent = random.choice(lang_data).strip()


        h5_path = os.path.join(self.root, 'LangSHAPE/pd_grounding_data', fn[1], 'partial_pc_grasp.h5')

        # print('Lang_SHAPE_simple_h5_v2')
        places = ['0', '1', '2']
        # views = [['view_1', 'view_2'], ['view_2', 'view_3'], ['view_3', 'view_4'], ['view_1', 'view_4']]

        views = [['view_1'], ['view_2'], ['view_3'], ['view_4'], ['merge'], ['view_1', 'view_2'], ['view_2', 'view_3'],
                 ['view_3', 'view_4'], ['view_1', 'view_4']]

        # views = [['view_1'], ['view_2'], ['view_3'], ['view_4'], ['merge']]

        with h5py.File(h5_path, 'r') as f:
            # pc_data = f['collect_pc'][()]
            place = random.choice(places)
            view = random.choice(views)
            # print(place)
            # print(view)

            if len(view) == 1:
                # one view
                pc_data = f[place + '/' + view[0]][()]
            else:
                # two view merge
                pc_data1 = f[place + '/' + view[0]][()]
                pc_data2 = f[place + '/' + view[1]][()]
                pc_data = np.concatenate([pc_data1, pc_data2], axis=0)

        pc_data = shuffle_points(pc_data)
        # file_name = fn[1].split('/')[-1]
        if not self.normal_channel:
            point_set = pc_data[:, 0:3]
        else:
            point_set = pc_data[:, 0:6]
        seg = pc_data[:, -1].astype(np.int32)
        seg = self.mapping_shape(seg, file_name)
        # print(type(seg), seg.shape, len(seg), np.sum(seg))
        # print(seg)

        # if len(self.cache) < self.cache_size:
        #     # self.cache[index] = (point_set, cls, seg)
        #     self.cache[index] = (point_set, cls, seg, file_name)

        point_set = self.rot_augmentation(point_set[:, 0:3])
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set = jitter_point_cloud(point_set)
        # point_set = shift_point_cloud(point_set)
        # point_set = random_scale_point_cloud(point_set)

        if len(seg) < self.npoints:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
        else:
            choice = np.random.choice(len(seg), self.npoints, replace=False)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        # print(type(cls), cls[0])
        # word_id = self.get_lang_tokenizer(cls[0])
        # print(word_id)
        return point_set, cls, seg, sent, file_name
        # return point_set, cls, seg, file_name

    def __len__(self):
        return len(self.datapath)
