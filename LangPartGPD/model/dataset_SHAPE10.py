#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import pickle
import random
import sys

import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import json
from autolab_core import YamlConfig
from dexnet.grasping import (
    GpgGraspSampler,
)  # temporary way for show 3D gripper using mayavi
from dexnet.grasping import RobotGripper

sys.path.append("../")
from utils.Timeit import TimeIt
from utils.pointcloud_utils import *
import h5py
from tqdm import tqdm

# from dexnet.grasping import RobotGripper
# from autolab_core import YamlConfig

print("Dataloader SHAPE10 simple:v4 v5  merge grasping_label and pc")
home_dir = "/home/aaronsxxx/code/LangPartGPD"
gripper_path = "/dex-net/test/config_TASE.yaml"
yaml_config = YamlConfig(home_dir + gripper_path)
print("gripper_path", gripper_path)
gripper_name = "robotiq_85"
gripper = RobotGripper.load(gripper_name, home_dir + "/dex-net/data/grippers")
ags = GpgGraspSampler(gripper, yaml_config)

NORMAL_PC = False
if NORMAL_PC:
    print("normal pc used! ")


# mymodule.py
def get_module_path():
    return os.path.abspath(__file__)


def pc_normalize(pc):
    # print('normnalize point cloud!')
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class PointGraspOneViewDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        grasp_points_num,
        path,
        tag,
        split_mode="object-wise",
        split="train",
        attitude_used=True,
        vis=False,
    ):
        self.grasp_points_num = grasp_points_num
        self.path = path
        self.tag = tag
        self.attitude_used = attitude_used
        self.min_point_limit = 50
        self.split = split
        self.vis = vis
        self.normal_K = 10
        self.voxel_point_num = 50
        self.projection_margin = 1
        self.minimum_point_amount = 150
        self.grasp_file_list = []

        file_list = self.get_example_list(split=split, split_mode=split_mode)

        with TimeIt(" Generate index using simple version: "):
            for example in tqdm(file_list):
                # # load npy file

                grasp_example = os.path.join(self.path, "pd_grasping_data", example)
                # print(grasp_example)
                if not os.path.exists(grasp_example):
                    # print(grasp_example)
                    continue
                try:
                    grasp_path = glob.glob(
                        os.path.join(grasp_example, "grasping_list_*.npy")
                    )[0]
                except IndexError:
                    print("----", grasp_example)

                grasp_list = np.load(grasp_path)
                grasp_num = grasp_list.shape[0]

                for idx in range(int(grasp_num)):
                    example_descrip = [example, idx]
                    self.grasp_file_list.append(example_descrip)
        print(
            "%s, %s, example number: %d, grasp number: %s."
            % (split_mode, split, len(file_list), len(self.grasp_file_list))
        )

    def get_example_list(self, split, split_mode):
        with open(
            os.path.join(self.path, "split_data", split_mode, split + ".txt")
        ) as f:
            data = f.readlines()
            tmp = [item.strip() for item in data]
            # fns = [item for item in tmp if item not in invalid_grasping_list]
        return tmp

    def get_target_label(self, grasp_num, grasp, example_path):
        level_score, refine_score = grasp[-2:]

        if grasp_num in self.collision_dict[example_path]:
            label = 0
        else:
            if level_score <= 0.6:
                label = 1
            elif level_score <= 1.6 and refine_score > 0.8:
                label = 1
            else:
                label = 0

        return label

    def collect_pc(self, grasp, pc):
        """
        grasp:  (5, 3)
        configuration[0:3] = center  fingertip
        configuration[3:6] = axis
        configuration[6] = width
        configuration[7] = angle
        configuration[8] = jaw_width
        configuration[9] = min_width
        configuration[10] = friction
        configuration[11] = canny_quality
        """

        center = grasp[0]  # .reshape(3, 1)
        approach = grasp[1]  # .reshape(3, 1)
        binormal = grasp[2]  # .reshape(3, 1)
        minor_normal = grasp[3]  # .reshape(3, 1)
        grasp_bottom_center = grasp[4]  # .reshape(3, 1)
        gripper_in_world_mat = np.hstack(
            [approach.reshape(3, 1), binormal.reshape(3, 1), minor_normal.reshape(3, 1)]
        )
        world_in_gripper_mat = gripper_in_world_mat.T

        # pc_t/left_t/right_t is in local coordinate(with center as origin)
        # other(include pc) are in pc coordinate
        pc_t = (np.dot(world_in_gripper_mat, (pc - grasp_bottom_center).T)).T
        width = ags.gripper.hand_outer_diameter - 2 * ags.gripper.finger_width
        x_limit = ags.gripper.hand_depth + 0.01
        # z_limit = ags.gripper.hand_height * 0.5
        z_limit = width * 0.25

        y_limit = width * 0.5

        x1 = pc_t[:, 0] > 0
        x2 = pc_t[:, 0] < x_limit
        y1 = pc_t[:, 1] > -y_limit
        y2 = pc_t[:, 1] < y_limit
        z1 = pc_t[:, 2] > -z_limit
        z2 = pc_t[:, 2] < z_limit

        a = np.vstack([x1, x2, y1, y2, z1, z2])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]
        # print('collect point number: ', len(self.in_ind))
        # if len(self.in_ind) < self.min_point_limit:
        #     return None

        return pc_t[self.in_ind], pc[self.in_ind], approach, self.in_ind

    def collect_pc_pcl(self, grasp, pc):
        grasp_bottom_center = grasp[4]
        approach = grasp[1]
        binormal = grasp[2]
        minor_normal = grasp[3]

        approach = approach.reshape(3, 1)
        binormal = binormal.reshape(3, 1)
        minor_normal = minor_normal.reshape(3, 1)

        # grasp in object_frame representation, and then transpose(inv) to get object_frame in grasp_frame representation
        # matrix = np.hstack([approach, binormal, minor_normal]).T
        world_in_gripper_mat = np.hstack([approach, binormal, minor_normal]).T

        pc_t = (np.dot(world_in_gripper_mat, (pc - grasp_bottom_center).T)).T
        width = ags.gripper.hand_outer_diameter - 2 * ags.gripper.finger_width
        x_limit = ags.gripper.hand_depth + 0.01
        # z_limit = ags.gripper.hand_height * 0.5
        z_limit = width * 0.25
        y_limit = width * 0.5

        x1 = pc_t[:, 0] > 0
        x2 = pc_t[:, 0] < x_limit
        y1 = pc_t[:, 1] > -y_limit
        y2 = pc_t[:, 1] < y_limit
        z1 = pc_t[:, 2] > -z_limit
        z2 = pc_t[:, 2] < z_limit

        a = np.vstack([x1, x2, y1, y2, z1, z2])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]
        return pc_t[self.in_ind], pc[self.in_ind], approach, self.in_ind

    def __getitem__(self, index):
        # try:
        # obj_ind, grasp_ind = np.unravel_index(index, (len(self.object), self.grasp_amount_per_file))
        # print(self.grasp_file_list)

        example_descrip = self.grasp_file_list[index]

        # example_path, grasp_num = example_descrip.split('-')[0], example_descrip.split('-')[1]
        # grasping_pose_folder, grasping_label_folder, example_path, grasp_num = example_descrip[0], example_descrip[1]
        example_path, grasp_num = example_descrip

        h5_path = os.path.join(
            self.path, "pd_grasping_data", example_path, "pc_grasp.h5"
        )
        with h5py.File(h5_path, "r") as f:
            pc_data = f["collect_pc"][()]
            # grasp_list = f['grasp_list'][()]
        pc = pc_data[:, 0:3]

        grasp_example = os.path.join(self.path, "pd_grasping_data", example_path)
        grasp_path = glob.glob(os.path.join(grasp_example, "grasping_list_*.npy"))[0]
        grasp_list = np.load(grasp_path)
        grasp = grasp_list[grasp_num]

        grasp_example_label = os.path.join(self.path, "pd_grasping_data", example_path)
        grasp_label_path = glob.glob(
            os.path.join(grasp_example_label, "grasping_label_*.npy")
        )[0]

        grasp_label = np.load(grasp_label_path)
        label = grasp_label[grasp_num]

        # if grasping_pose_folder == "Lang_SHAPE_grasp_pclsample_v5":
        #     pc_in_grasp, pc_in_world, approach, in_hand_pc_index = self.collect_pc_pcl(grasp, pc)
        # else:
        pc_in_grasp, pc_in_world, approach, in_hand_pc_index = self.collect_pc(
            grasp, pc
        )
        assert len(in_hand_pc_index) == pc_in_grasp.shape[0]
        pc_in_grasp = random_point_dropout(pc_in_grasp)

        # print(pc_in_grasp.shape)
        # with TimeIt('Sample pc: '):
        if self.attitude_used:
            if len(pc_in_grasp) > self.grasp_points_num - 1:
                # print('attitude_used')
                grasp_pc = pc_in_grasp[
                    np.random.choice(
                        len(pc_in_grasp), size=self.grasp_points_num - 1, replace=False
                    )
                ].T
            else:
                try:
                    grasp_pc = pc_in_grasp[
                        np.random.choice(
                            len(pc_in_grasp),
                            size=self.grasp_points_num - 1,
                            replace=True,
                        )
                    ].T
                except ValueError as e:
                    print("poor pc example: ", example_descrip)

            try:
                if NORMAL_PC:
                    grasp_pc = pc_normalize(grasp_pc)  # aaron add 20241217
                input_pc = np.concatenate([grasp_pc, approach], axis=1)
            except ValueError as e:
                print(example_descrip, grasp_pc.shape, approach.shape)

        else:
            if len(pc_in_grasp) > self.grasp_points_num:
                grasp_pc = pc_in_grasp[
                    np.random.choice(
                        len(pc_in_grasp), size=self.grasp_points_num, replace=False
                    )
                ].T
            else:
                try:
                    grasp_pc = pc_in_grasp[
                        np.random.choice(
                            len(pc_in_grasp), size=self.grasp_points_num, replace=True
                        )
                    ].T
                except ValueError as e:
                    print("poor pc example: ", example_descrip)
            input_pc = grasp_pc

            if NORMAL_PC:
                input_pc = pc_normalize(input_pc)  # aaron add 20241217
        # # return set
        # example_descrip: [cat_sample_part, grasp num]
        if self.vis:
            return (
                pc_in_world,
                pc,
                in_hand_pc_index,
                label,
                grasp,
                input_pc,
                example_descrip,
            )

        #         example_path, grasp_num = example_descrip[0], example_descrip[1]
        if "train" in self.split:
            return input_pc, label
        if self.split == "test":
            return input_pc, label, example_descrip

    def __len__(self):
        return len(self.grasp_file_list)


def get_grasp_list(grasp, pcl_sampler=True):
    if pcl_sampler:
        print("pcl sampler")
        grasp_bottom_center = grasp[4]
        approach = grasp[1]
        binormal = grasp[2]
        minor_normal = grasp[3]

        approach = approach  # .reshape(3, 1)
        binormal = binormal  # .reshape(3, 1)
        minor_normal = minor_normal  # .reshape(3, 1)

        # grasp in object_frame representation, and then transpose(inv) to get object_frame in grasp_frame representation
        # matrix = np.hstack([approach, binormal, minor_normal]).T
        gripper_in_world_mat = np.hstack([approach, binormal, minor_normal])
        return [
            grasp_bottom_center,
            approach,
            binormal,
            minor_normal,
            grasp_bottom_center,
        ]

    else:
        print("mesh sampler", grasp.shape)
        center = grasp[0:3]
        axis = grasp[3:6]  # binormal
        width = grasp[6]
        angle = grasp[7]

        axis = axis / np.linalg.norm(axis)
        binormal = axis
        # cal approach
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        R1 = np.c_[[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]]
        axis_y = axis
        axis_x = np.array([axis_y[1], -axis_y[0], 0])
        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array([1, 0, 0])
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)
        R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
        approach = R2.dot(R1)[:, 0]
        approach = approach / np.linalg.norm(approach)
        minor_normal = np.cross(approach, axis)
        minor_normal = minor_normal / np.linalg.norm(minor_normal)

        approach = approach  # .reshape(3, 1)
        binormal = binormal  # .reshape(3, 1)
        minor_normal = minor_normal  # .reshape(3, 1)

        gripper_in_world_mat = np.hstack(
            [approach.reshape(3, 1), binormal.reshape(3, 1), minor_normal.reshape(3, 1)]
        )
        wolrd_in_gripper_mat = gripper_in_world_mat.T

        grasp_bottom_center = center + np.dot(
            gripper_in_world_mat, np.array([-0.125, 0, 0]).T
        )
        return [center, approach, binormal, minor_normal, grasp_bottom_center]


def test_dataset(vis=False):
    dataset_root = "/media/aaronsxxx/hard_1/dataset"
    grasp_points_num = 750
    # obj_points_num = 50000
    # pc_file_used_num = 20
    thresh_good = 0.6
    thresh_bad = 0.6

    input_size = 60
    input_chann = 12  # 12
    a = PointGraspOneViewDataset(
        grasp_points_num=grasp_points_num,
        path=dataset_root,
        tag="train",
        split_mode="object-wise",
        split="test",
        vis=vis,
        attitude_used=False,
    )
    while 1:
        index = random.randint(0, len(a.grasp_file_list))
        pc_in_world, pc, in_hand_pc_index, label, grasp, example_descrip = (
            a.__getitem__(index)
        )
        # pc_in_world, pc, in_hand_pc_index, label, grasp, example_descrip = a.__getitem__(123)
        print(label, example_descrip)
        # if '03797390' in example_descrip[0]:
        #     break

        print(example_descrip[1])
        if example_descrip[1] == "Lang_SHAPE_grasp_v5":
            show_mode = True
        else:
            show_mode = False
        if label:
            break

    if vis:
        from mayavi import mlab

        fig = mlab.figure(size=(600, 600))
        mlab.points3d(
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            color=(
                0.22,
                1,
                1,
            ),
            scale_factor=0.002,
        )
        mlab.points3d(
            pc_in_world[:, 0],
            pc_in_world[:, 1],
            pc_in_world[:, 2],
            color=(1, 0, 0),
            scale_factor=0.003,
        )

        # mlab.points3d(pc_in_grasp[:, 0], pc_in_grasp[:, 1], pc_in_grasp[:, 2], color=(1, 0, 0), scale_factor=0.003)
        # mlab.points3d(in_hand_pc_index[:, 0], in_hand_pc_index[:, 1], in_hand_pc_index[:, 2], color=(1, 0, 0), scale_factor=0.003)
        # print(get_grasp_list(grasp))
        # ags.show_all_grasps(pc, [get_grasp_list(grasp, show_mode)])
        ags.show_all_grasps(pc, [grasp])
        mlab.show()


if __name__ == "__main__":
    print(ags.gripper.hand_outer_diameter)
    print(ags.gripper.finger_width)
    print(ags.gripper.hand_depth)
    print(ags.gripper.hand_height)
    print(ags.gripper.hand_outer_diameter - 2 * ags.gripper.finger_width)
    # print(width)
    test_dataset(vis=True)

    # visualization using t-read_grasps_from_shapenet_draw_xxx_check_train_grasp.py
    # using show_dataloader_check()

    hand_outer_diameter = 0.191
    finger_width = 0.0255
    hand_depth = 0.125
    hand_height = 0.03
