"""
Author: Benny
Date: Nov 2019
"""
import argparse
import copy
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# from typing import TYPE_CHECKING
import torch
# import logging
import sys
# import importlib
# from tqdm import tqdm
import numpy as np
from visualizer.show3d_balls_SHAPE import showpoints

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from models.pointnet2_part_seg_ssglang_sent import get_model


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet_grounding')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_votes', type=int, default=5, help='aggregate segmentation scores with voting')
    parser.add_argument('--vis', action='store_true', default=False, help='visualization')
    return parser.parse_args()


# semantic dictionary
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

cls_classes = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7,
               'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14,
               'Table': 15}

cls_label_to_cat = {}
for cat in cls_classes.keys():
    label = cls_classes[cat]
    cls_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def get_lang_feat(sents, gpu_use=True):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    # print('<==========')
    # print(label.shape, target.shape)
    for i in range(len(sents)):
        grounding_sent = sents[i]
        encoded_dict = tokenizer(grounding_sent, return_tensors='pt', max_length=20, pad_to_max_length=True,
                                 truncation=True)

        # Add the encoded sentences to the list.
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict["token_type_ids"])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    if gpu_use:
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()
        attention_masks = attention_masks.cuda()

    encoded_input = {'input_ids': input_ids,
                     'token_type_ids': token_type_ids,
                     'attention_mask': attention_masks}
    return encoded_input


def grounding_network(model_, pc, sent):
    # print(pc.shape, type(pc))
    with torch.no_grad():
        batch_size = pc.shape[0]
        points = pc.float().cuda()
        # print('<-----', points.shape)

        points = points.transpose(2, 1)
        lang_encoded_input = get_lang_feat(sent, gpu_use=True)

        vote_pool = np.zeros([points.shape[0], points.shape[2], 2])
        num_votes = 5
        for _ in range(num_votes):
            seg_pred_once, _ = model_(xyz=points, encoded_text=lang_encoded_input)  # (1, 2048, 2)
            vote_pool += seg_pred_once.cpu().data.numpy()

        seg_pred = vote_pool / num_votes

    # print(seg_pred.shape)
    # seg_pred = seg_pred.cpu().data.numpy()
    if batch_size == 1:
        pred_val_logits = seg_pred[0]
        prediction = np.argmax(pred_val_logits, 1).reshape([-1, 1])
        # print('grounding_network: ', pred_val_logits.shape, prediction.shape)
        # print('----->')
        return seg_pred, prediction
    else:
        return seg_pred, None


def single_input(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def pc_normalize(pc):
        # print('normnalize point cloud!')
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    pc_path = '/home/aaronsxxx/project_repostories/TASE_pub/PartGrounding/test_data/all_pc_in_world.txt'

    points = np.loadtxt(pc_path)
    points = points[:, :3]

    if points.shape[0] > 2048:
        choice = np.random.choice(points.shape[0], 2048, replace=False)
    else:
        choice = np.random.choice(points.shape[0], 2048, replace=True)
    points = points[choice, :]

    points = points[:, :3]
    point_set = copy.deepcopy(points)

    points = pc_normalize(points)
    points = np.expand_dims(points, 0)
    points = torch.FloatTensor(points)

    sent = ["leg of chair."]

    # input(sent)

    # vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()
    vote_pool = np.zeros([points.size()[0], points.size()[1], num_part])
    for _ in range(args.num_votes):
        seg_pred, prediction = grounding_network(classifier, points, sent)
        vote_pool += seg_pred

    cur_pred_val = vote_pool / args.num_votes
    # # cur_pred_val = seg_pred.cpu().data.numpy()
    # cur_pred_val_logits = cur_pred_val
    # cur_pred_val = np.argmax(cur_pred_val_logits[:, :], 1)

    pred_val_logits = cur_pred_val[0]
    prediction = np.argmax(pred_val_logits, 1).reshape([-1, 1])

    output = np.concatenate([point_set, prediction], 1)
    np.savetxt('./output/prediction.txt', output)

    prediction = np.squeeze(prediction)
    print(prediction.shape)
    print(np.max(prediction))

    if args.vis:
        showpoints(point_set, part_gt=None, part_pred=prediction, waittime=0, showrot=True, magnifyBlue=0,
                   freezerot=False, background=(255, 255, 255), normalizecolor=True, ballradius=10)


if __name__ == '__main__':
    args = parse_args()
    num_classes = 16
    num_part = 2

    '''MODEL LOADING'''
    model_name = 'pointnet2_part_seg_ssglang_sent'
    print('===> model structure name: ', model_name)

    if 'sent' in model_name:
        from transformers import AutoTokenizer, AutoModel

        # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v1')
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    else:
        from transformers import BertModel, BertTokenizerFast  # BertTokenizer

        # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizerFast.from_pretrained(
            "/home/aaronskinova/Pointnet_Pointnet2_pytorch_kw20221016/local_bert/bert_tokenizer", local_files_only=True)

        # tokenizer.save_pretrained("/home/aaronskinova/Pointnet_Pointnet2_pytorch_kw20221016/pre_trained_models/bert_tokenizer")

    # MODEL = importlib.import_module(model_name)
    classifier = get_model(num_part, normal_channel=False).cuda()
    # print("Language grounding model: {}".format(str(experiment_dir) + '/checkpoints/best_model.pth'))
    model_path = '/home/aaronsxxx/project_repostories/TASE_pub/PartGrounding/log/part_seg/model/part_grounding_model.pth'
    checkpoint = torch.load(model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    # torch.set_grad_enabled(False)

    single_input(args)
