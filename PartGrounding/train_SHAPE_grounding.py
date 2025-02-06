"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader_SHAPE_grounding_gpt5 import PartNormalDataset
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
# from transformers import BertModel, BertTokenizerFast  # BertTokenizer
from transformers import AutoTokenizer, AutoModel

# from pytorchtools import EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v1')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch Size during training')
    parser.add_argument('--epoch', default=160, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--descrip', type=str, default='_', help='training description')
    parser.add_argument('--train_mode', type=str, default='object-wise', help='object-wise or part-wise model')
    parser.add_argument('--data_mode', type=str, default='full',
                        help='full, known_all, part_unknown, object_unknown, object_known_part_unknown, object_descrip')

    return parser.parse_args()


seg_classes = {'Earphone': [13, 14, 15], 'Motorbike': [25, 4, 26, 0, 24, 10], 'Rocket': [0, 29, 30],
               'Car': [7, 8, 25, 0], 'Laptop': [22, 23], 'Cap': [5, 6], 'Skateboard': [25, 31, 32], 'Mug': [4, 0],
               'Guitar': [16, 17, 0], 'Bag': [4, 0], 'Lamp': [19, 20, 21], 'Table': [33, 11, 34],
               'Airplane': [0, 1, 2, 3], 'Pistol': [27, 4, 28], 'Chair': [9, 10, 11, 12], 'Knife': [18, 4]}

# seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
# for cat in seg_classes.keys():
#     for label in seg_classes[cat]:
#         seg_label_to_cat[label] = cat

cls_classes = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7,
               'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14,
               'Table': 15}

cls_label_to_cat = {}
for cat in cls_classes.keys():
    label = cls_classes[cat]
    cls_label_to_cat[label] = cat

# mapping from original label to share part label
shape_mapping = {'0': 0, '5': 0, '11': 0, '21': 0, '35': 0, '37': 0, '39': 0, '41': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                 '23': 4, '33': 4, '36': 4, '6': 5, '7': 6, '8': 7, '9': 8, '12': 9, '13': 10, '31': 10, '14': 11,
                 '48': 11, '15': 12, '16': 13, '17': 14, '18': 15, '19': 16, '20': 17, '22': 18, '24': 19, '26': 19,
                 '25': 20,
                 '27': 21, '28': 22, '29': 23, '30': 24, '10': 25, '32': 25, '44': 25, '34': 26, '38': 27, '40': 28,
                 '42': 29, '43': 30,
                 '45': 31, '46': 32, '47': 33, '49': 34}

label_to_shape = {0: 'body', 1: 'wing', 2: 'tail', 3: 'engine', 4: 'handle', 5: 'peak', 6: 'panel', 7: 'roof',
                  8: 'hood', 9: 'back', 10: 'seat', 11: 'leg', 12: 'arm', 13: 'earphone', 14: 'headband',
                  15: 'microphone', 16: 'head', 17: 'neck', 18: 'blade', 19: 'base', 20: 'shade', 21: 'tube',
                  22: 'keyboard', 23: 'screen', 24: 'tank', 25: 'wheel', 26: 'light', 27: 'barrel', 28: 'trigger',
                  29: 'fin', 30: 'nose', 31: 'deck', 32: 'bearing', 33: 'top', 34: 'drawer'}

shape_to_label = {'body': 0, 'wing': 1, 'tail': 2, 'engine': 3, 'handle': 4, 'peak': 5, 'panel': 6,
                  'roof': 7, 'hood': 8, 'back': 9, 'seat': 10, 'leg': 11, 'arm': 12, 'earphone': 13,
                  'headband': 14, 'microphone': 15, 'head': 16, 'neck': 17, 'blade': 18, 'base': 19,
                  'shade': 20, 'tube': 21, 'keyboard': 22, 'screen': 23, 'tank': 24, 'wheel': 25,
                  'light': 26, 'barrel': 27, 'trigger': 28, 'fin': 29, 'nose': 30, 'deck': 31,
                  'bearing': 32, 'top': 33, 'drawer': 34}
except_part_50 = [3, 8, 9, 18, 29, 30, 40, 46, 49]  # 9 parts cannot be grasped, we drop in dataset.
except_part_35 = [shape_mapping[str(item)] for item in except_part_50]
print(except_part_35)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def get_lang_feat(sents, gpu_use=True):
    encoded_input = tokenizer(list(sents), padding=True, truncation=True, return_tensors='pt')
    # print(encoded_input.keys())
    if gpu_use:
        for key in encoded_input.keys():
            encoded_input[key] = encoded_input[key].cuda()
        # encoded_input['input_ids'] = encoded_input['input_ids'].cuda()
        # encoded_input['attention_mask'] = encoded_input['attention_mask'].cuda()
    return encoded_input


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    print('file timestr:', timestr)
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    tb_dir = exp_dir.joinpath('tensorboard')
    exp_dir = exp_dir.joinpath('model')
    exp_dir.mkdir(exist_ok=True)

    conf_descip = args.descrip + '_zju_' + args.train_mode + '_' + args.data_mode
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr + '_shape' + '_' + args.model.split('_')[-1] + conf_descip)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir + '_shape_' + '_' + args.model.split('_')[-1] + conf_descip)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    savepath = str(checkpoints_dir) + '/best_model.pth'
    # earlyStopping = EarlyStopping(patience=5, verbose=True, path=savepath)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    writer = SummaryWriter(os.path.join(tb_dir, timestr + '_' + args.model.split('_')[-1]) + conf_descip)

    root = '/media/aaronszju/hard_1/dataset'
    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, train_mode=args.train_mode, split='trainval',
                                      normal_channel=args.normal, data_mode=args.data_mode)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, train_mode=args.train_mode, split='test',
                                     normal_channel=args.normal, data_mode=args.data_mode)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)
    # for loader_info in TRAIN_DATASET.aux_info():
    log_string(TRAIN_DATASET.aux_info)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 16
    num_part = 2

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_part_avg_iou = 0
    best_inctance_avg_iou = 0

    select_acc = 0
    select_class_avg_iou = 0
    select_part_avg_iou = 0
    select_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        losses = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, label, target, sent, _) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader),
                                                        smoothing=0.9):
            optimizer.zero_grad()
            lang_encoded_input = get_lang_feat(sent)
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes), lang_encoded_input)
            seg_pred = seg_pred.contiguous().view(-1, num_part)

            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            losses.append(loss.cpu().detach().numpy() / float(args.batch_size))
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        # epoch_loss = np.mean(losses)
        epoch_loss = np.sum(losses)
        # earlyStopping(epoch_loss, classifier)
        # earlyStopping(-train_instance_acc, classifier)
        # writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/instance_acc', train_instance_acc, epoch)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
        log_string('Loss is: %.5f' % epoch_loss)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            part_shape_ious = {part: [] for part in shape_to_label.keys() if
                               shape_to_label[part] not in except_part_35}  # for part  35-9

            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            for batch_id, (points, label, target, sent, file_name) in tqdm(enumerate(testDataLoader),
                                                                           total=len(testDataLoader),
                                                                           smoothing=0.9):
                lang_encoded_input = get_lang_feat(sent)

                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points, to_categorical(label, num_classes), lang_encoded_input)
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    # cat = seg_label_to_cat[target[i, 0]]
                    # logits = cur_pred_val_logits[i, :, :]
                    # cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, :], 1)

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                # for l in range(num_part):
                for l in range(2):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    fn = file_name[i]
                    part_label = shape_mapping[fn.split('_')[-1]]  # 0 1 2...
                    part_name = label_to_shape[part_label]  # wing ...

                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    # cat = seg_label_to_cat[segl[0]]

                    label_ref = label[i, :].cpu().data.numpy()[0]
                    cat = cls_label_to_cat[label_ref]

                    # part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    part_ious = [0.0 for _ in range(2)]

                    # for l in seg_classes[cat]:
                    #     if (np.sum(segl == l) == 0) and (
                    #             np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    #         part_ious[l - seg_classes[cat][0]] = 1.0
                    #     else:
                    #         print('<--------------------')
                    #         print(len(segl), len(segp))
                    #         print(segl, segp, l)
                    #         print(np.sum((segl == l) & (segp == l)))
                    #         print(np.sum((segl == l) | (segp == l)))
                    #         print('-------------------->')
                    #         part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                    #             np.sum((segl == l) | (segp == l)))

                    # for l in seg_classes[cat]:
                    # for idx in range(len(seg_classes[cat])):
                    for l in range(2):
                        # l = seg_classes[cat][idx]
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            # part_ious[idx] = 1.0
                            part_ious[l] = 1.0

                        else:
                            # print('<--------------------')
                            # print(len(segl), len(segp))
                            # print(segl, segp, l)
                            # print(np.sum((segl == l) & (segp == l)))
                            # print(np.sum((segl == l) | (segp == l)))
                            # print('-------------------->')
                            # part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            #     np.sum((segl == l) | (segp == l)))
                            # part_ious[idx] = np.sum((segl == l) & (segp == l)) / float(
                            #     np.sum((segl == l) | (segp == l)))
                            part_ious[l] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))

                    # shape_ious[cat].append(np.mean(part_ious))
                    shape_ious[cat].append(part_ious[1])
                    part_shape_ious[part_name].append(part_ious[1])

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))

            all_shape_ious_part = []
            for part_name in part_shape_ious.keys():  # per part
                for iou in part_shape_ious[part_name]:
                    all_shape_ious_part.append(iou)
                part_shape_ious[part_name] = np.mean(part_shape_ious[part_name])
            mean_shape_ious_part = np.mean(list(part_shape_ious.values()))

            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            print('========================')
            test_metrics['part_avg_iou'] = mean_shape_ious_part  # miou per object cat
            for part_name in sorted(part_shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (part_name + ' ' * (14 - len(part_name)), part_shape_ious[part_name]))

            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f  Part avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'],
            test_metrics['inctance_avg_iou'], test_metrics['part_avg_iou']))
        # log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
        #     epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model_instance.pth'

            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'part_avg_iou': test_metrics['part_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving best instance iou model....')

        if (test_metrics['part_avg_iou'] >= best_part_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model_part.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'part_avg_iou': test_metrics['part_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving best part iou model....')

        # selection update: part-wise data:
        if args.train_mode == 'part-wise':
            log_string('best_part_avg_iou selected')
            if test_metrics['part_avg_iou'] > best_part_avg_iou:
                select_acc = test_metrics['accuracy']
                select_class_avg_iou = test_metrics['class_avg_iou']
                select_inctance_avg_iou = test_metrics['inctance_avg_iou']
                select_part_avg_iou = test_metrics['part_avg_iou']

        else:
            log_string('best_class_avg_iou selected')
            if test_metrics['class_avg_iou'] > best_class_avg_iou:
                select_acc = test_metrics['accuracy']
                select_class_avg_iou = test_metrics['class_avg_iou']
                select_inctance_avg_iou = test_metrics['inctance_avg_iou']
                select_part_avg_iou = test_metrics['part_avg_iou']

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        if test_metrics['part_avg_iou'] > best_part_avg_iou:
            best_part_avg_iou = test_metrics['part_avg_iou']

        log_string('==========>>  Model config: %s' % checkpoints_dir)
        log_string('Best/Select accuracy is: %.5f   %.5f' % (best_acc, select_acc))
        log_string('Best/Select class avg mIOU is: %.5f   %.5f' % (best_class_avg_iou, select_class_avg_iou))
        log_string('Best/Select inctance avg mIOU is: %.5f   %.5f' % (best_inctance_avg_iou, select_inctance_avg_iou))
        log_string('Best/Select part avg mIOU is: %.5f   %.5f' % (best_part_avg_iou, select_part_avg_iou))

        writer.add_scalar('eval/best_acc', best_acc, epoch)
        writer.add_scalar('eval/best_class_avg_iou', best_class_avg_iou, epoch)
        writer.add_scalar('eval/best_inctance_avg_iou', best_inctance_avg_iou, epoch)
        writer.add_scalar('eval/best_part_avg_iou', best_part_avg_iou, epoch)

        writer.add_scalar('select/select_acc', select_acc, epoch)
        writer.add_scalar('select/select_class_avg_iou', select_class_avg_iou, epoch)
        writer.add_scalar('select/select_inctance_avg_iou', select_inctance_avg_iou, epoch)
        writer.add_scalar('select/select_part_avg_iou', select_part_avg_iou, epoch)

        # if earlyStopping.early_stop:
        #     print('Early Stopping !')
        #     break

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
