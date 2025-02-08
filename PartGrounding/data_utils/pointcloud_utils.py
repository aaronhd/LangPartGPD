#!/usr/bin/env python
# encoding: utf-8


import numpy as np
from scipy.spatial.transform import Rotation as R


def pc_normalize(pc):
    # print('normnalize point cloud!')
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def rot_augmentation(pc):
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
