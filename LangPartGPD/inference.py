import argparse
import time
from mayavi import mlab
import random
from tqdm import tqdm
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import datetime
from model.dataset_SHAPE10 import *
from model.pointnet import PointNetCls
from model.dataset_SHAPE10 import PointGraspOneViewDataset


torch.cuda.manual_seed(1)
np.random.seed(int(time.time()))


def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2**31 - 1))


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


grasp_points_num = 512
point_channel = 3


def langpartgpd_network(model_, local_pc):
    # local_pc = local_pc.T
    local_pc = local_pc[np.newaxis, ...]
    local_pc = torch.FloatTensor(local_pc)
    local_pc = local_pc.cuda()
    with torch.no_grad():
        output, _ = model_(local_pc)  # N*C
    # print('1',output)
    output = output.softmax(1)  # +torch.tensor([-0.2,0.9])
    # print('2',output, output.shape)
    # pred = output.data.max(1, keepdim=True)[1]
    score, pred = output.data.max(1, keepdim=True)
    # print('3', score, pred)
    output = output.cpu()
    return pred[0], output.data.numpy()


def show_grasping_part(
    in_hand_point, object_points, color_p=(0, 0, 1), color_f=(0.7, 0.7, 0.7)
):
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
    mlab.points3d(
        object_points[:, 0],
        object_points[:, 1],
        object_points[:, 2],
        color=color_f,
        scale_factor=0.011,
    )
    # color_p = (0, 0, 1) # (192/255.0,105/255.0,166/255.0)
    # pc_semantic_1 = pc_semantic[seg==1]  # 4  36
    mlab.points3d(
        in_hand_point[:, 0],
        in_hand_point[:, 1],
        in_hand_point[:, 2],
        color=color_p,
        scale_factor=0.011,
    )
    # mlab.show()


def main():
    grasp_points_num = 512
    point_channel = 3
    model_path = (
        "assets/learned_models/2025-02-04_23-32_fullv_xxx_part-wise_dict_best.pth"
    )
    model = PointNetCls(num_points=grasp_points_num, input_chann=point_channel, k=2)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    print("load is ok")

    dataset_root = "/media/aaronsxxx/hard_1/dataset/LangSHAPE"

    test_loader = PointGraspOneViewDataset(
        grasp_points_num=grasp_points_num,
        path=dataset_root,
        tag="test",
        split_mode="object-wise",
        split="test",
        vis=True,
        attitude_used=False,
    )
    idx = random.randint(0, test_loader.__len__())
    pc_in_world, pc, in_hand_pc_index, label, grasp, input_pc, example_descrip = (
        test_loader.__getitem__(index=idx)
    )
    # input_pc [3, N] in hand frame
    grasp_pred, grasp_score = langpartgpd_network(model, input_pc)
    grasp_pred = grasp_pred.cpu().numpy()[0]
    grasp_score = grasp_score[0]

    print("prediction: ", grasp_pred, grasp_score)
    print("example id: ", example_descrip)
    print("grasping label: ", label)

    show_grasping_part(pc_in_world, pc)
    if ags is not None:
        ags.show_all_grasps(pc, [grasp])
    mlab.show()


if __name__ == "__main__":
    main()
