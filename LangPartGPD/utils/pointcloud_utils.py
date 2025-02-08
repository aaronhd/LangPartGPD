#!/usr/bin/env python
# encoding: utf-8

import numpy as np


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


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # print('random_point_dropout')
    batch_pc = np.expand_dims(batch_pc, 0)
    batch_pc = shuffle_points(batch_pc)
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc[0]


if __name__ == '__main__':
    pass
