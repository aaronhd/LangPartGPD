#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os, sys
import time
import pickle
import logging
from tqdm import tqdm
import shutil
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import datetime

from model.dataset_SHAPE10 import PointGraspOneViewDataset, get_module_path
from model.pointnet import PointNetCls, DualPointNetCls
# from pytorchtools import EarlyStopping


dataloader_script_path = get_module_path()

# script_name = sys.argv[0]
train_script_path = os.path.abspath(__file__)
# script_dir = os.path.dirname(script_path)


parser = argparse.ArgumentParser(description="LangPartetGPD_SHAPE")
parser.add_argument("--tag", type=str, default="fullv_zju")
parser.add_argument("--epoch", type=int, default=60)
parser.add_argument("--mode", choices=["train", "test"], required=True)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--attitude_used", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--load-model", type=str, default="")
parser.add_argument("--load-epoch", type=int, default=-1)
parser.add_argument(
    "--model-path",
    type=str,
    default="./assets/learned_models",
    help="pre-trained model path",
)
parser.add_argument(
    "--data-path",
    type=str,
    default="/media/aaronszju/hard_1/dataset/LangSHAPE",
    help="data path",
)
parser.add_argument("--log-interval", type=int, default=10)
parser.add_argument("--save-interval", type=int, default=1)
parser.add_argument(
    "--split_mode", type=str, default="part-wise", help="object-wise or part-wise"
)


def log_string(str):
    logger.info(str)
    print(str)


args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.attitude_used:
    args.tag += "_attitude_used"

if args.cuda:
    print("GPU used.")
    torch.cuda.manual_seed(1)

timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
conf_descrip = args.tag + "_" + args.split_mode
print("file time tag: ", timestr)
tb_dir = "./assets/logs"
log_path = os.path.join(tb_dir, timestr + "_" + conf_descrip)
os.makedirs(log_path, exist_ok=True)
writer = SummaryWriter(log_path)
# writer = SummaryWriter(os.path.join('./assets/log/', 'GPD_SHAPE_{}'.format(args.tag)))
np.random.seed(int(time.time()))

logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(
    "%s/%s.txt" % (os.path.join(tb_dir, timestr + "_" + conf_descrip), args.tag)
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
log_string(os.path.basename(__file__))
log_string("PARAMETER ...")
log_string(args)


def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2**31 - 1))


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


grasp_points_num = 512
point_channel = 3


def train(model, loader, epoch, maxepoch, optimizer):
    # scheduler.step()
    model.train()
    torch.set_grad_enabled(True)
    correct = 0
    dataset_size = 0
    # for batch_idx, (data, target) in enumerate(loader):
    loop = tqdm(enumerate(loader), total=len(loader), smoothing=0.9)
    for batch_idx, (data, target) in loop:
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        # print(data.shape)  # torch.Size([1, 3, 750])
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        acc = float(correct) / float(dataset_size)
        # if batch_idx % args.log_interval == 0:
        #     percentage = 100. * batch_idx * args.batch_size / len(loader.dataset)
        #     print(f'Train Epoch: {epoch} [{batch_idx * args.batch_size}/{len(loader.dataset)} ({percentage}%)]'
        #           f'\tLoss: {loss.item()}\t{args.tag}')
        #     writer.add_scalar('train_loss', loss.cpu().item(), batch_idx + epoch * len(loader))
        loop.set_description(f"Train Epoch [{epoch}/{maxepoch}]")
        loop.set_postfix(loss=loss.cpu().item(), acc=acc)
    return acc, loss.cpu().item()


def test(model, loader, epoch=None, maxepoch=None):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss = 0
    correct = 0
    dataset_size = 0
    # da = {}
    # db = {}
    res = []
    # for data, target, obj_name in loader:
    loop = tqdm(enumerate(loader), total=len(loader), smoothing=0.9)
    for batch_idx, (data, target, obj_name) in loop:
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output, _ = model(data)  # N*C
        test_loss += F.nll_loss(output, target, size_average=False).cpu().item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        for i, j, k in zip(
            obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()
        ):
            res.append((i, j[0], k))
        if maxepoch:
            loop.set_description(f"Test Epoch [{epoch}/{maxepoch}]")
        # loop.set_postfix(loss=loss.cpu().item(), acc=acc)
    test_loss /= len(loader.dataset)
    acc = float(correct) / float(dataset_size)
    return acc, test_loss


def main():
    train_loader = torch.utils.data.DataLoader(
        PointGraspOneViewDataset(
            grasp_points_num=grasp_points_num,
            path=args.data_path,
            tag="train",
            split_mode=args.split_mode,
            split="trainval",
            attitude_used=args.attitude_used,
        ),
        batch_size=args.batch_size,
        num_workers=64,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        PointGraspOneViewDataset(
            grasp_points_num=grasp_points_num,
            path=args.data_path,
            tag="test",
            split_mode=args.split_mode,
            split="test",
            attitude_used=args.attitude_used,
        ),
        batch_size=args.batch_size,
        num_workers=64,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate,
    )

    # shutil.copy("./train.py", log_path)
    shutil.copy(train_script_path, log_path)
    # shutil.copy("./model/dataset_SHAPE10.py", log_path)
    shutil.copy(dataloader_script_path, log_path)

    is_resume = 0
    if args.load_model and args.load_epoch != -1:
        is_resume = 1

    if is_resume or args.mode == "test":
        model = torch.load(args.load_model, map_location="cuda:{}".format(args.gpu))
        model.device_ids = [args.gpu]
        print("load model {}".format(args.load_model))
    else:
        model = PointNetCls(num_points=grasp_points_num, input_chann=point_channel, k=2)
    if args.cuda:
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            model = model.cuda()
        else:
            device_id = [0, 1, 2, 3]
            torch.cuda.set_device(device_id[0])
            model = nn.DataParallel(model, device_ids=device_id).cuda()

    if args.mode == "train":
        best_test_acc = 0
        # early_stopping = EarlyStopping(8, verbose=False)
        for epoch in range(is_resume * args.load_epoch, args.epoch):
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

            acc_train, loss_train = train(
                model, train_loader, epoch, args.epoch, optimizer
            )
            scheduler.step()

            writer.add_scalar("train/loss", loss_train, epoch)
            writer.add_scalar("train/acc", acc_train, epoch)
            log_string("Train accuracy is: %.5f, loss: %.5f" % (acc_train, loss_train))

            # print('Train done, acc={}'.format(acc_train))
            acc, loss = test(model, test_loader, epoch, args.epoch)
            # print('Test done, acc={}, loss={}'.format(acc, loss))
            log_string("Test accuracy is: %.5f, loss: %.5f" % (acc, loss))

            # writer.add_scalar('train_acc', acc_train, epoch)
            writer.add_scalar("test/loss", loss, epoch)
            writer.add_scalar("test/acc", acc, epoch)

            if acc > best_test_acc:
                best_test_acc = acc
            if acc >= best_test_acc:
                # logger.info("Save model...")
                # # savepath = str(checkpoints_dir) + '/best_model.pth'
                # savepath = os.path.join(
                #     args.model_path, timestr + "_" + conf_descrip + "_best.pth"
                # )
                # log_string("Saving at %s" % savepath)
                # torch.save(model, savepath)

                savepath2 = os.path.join(
                    args.model_path, timestr + "_" + conf_descrip + "_dict_best.pth"
                )
                log_string("Saving at %s" % savepath2)
                torch.save(model.state_dict(), savepath2)

                log_string("Saving model....")
            # early_stopping(loss, model)
            # if early_stopping.early_stop:
            #     print("early stop!")
            #     break

            # if epoch % args.save_interval == 0:
            #     path = os.path.join(args.model_path, timestr + '_' + args.tag + '_{}.model'.format(epoch))
            #     torch.save(model, path)
            #     print('Save model @ {}'.format(path))
    else:
        print("testing...")
        acc, loss = test(model, test_loader)
        print("Test done, acc={}, loss={}".format(acc, loss))


if __name__ == "__main__":
    main()
