import torch.nn as nn
import torch
import torch.nn.functional as F
# from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
# from transformers import BertModel #, BertTokenizerFast, BertTokenizer
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer, AutoModel


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6 + additional_channel,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        # self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 6 + additional_channel, mlp=[128, 128, 128])
        # self.conv1 = nn.Conv1d(128, 128, 1)
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        # lang_encoder_name = "bert-base-uncased"
        # lang_encoder_name = 'sentence-transformers/all-mpnet-base-v1'
        lang_encoder_name = 'sentence-transformers/all-MiniLM-L6-v2'
        print(lang_encoder_name)
        if 'all-mpnet-base-v1' in lang_encoder_name or 'bert' in lang_encoder_name:
            lang_dim = 768
        else:
            lang_dim = 384
        # self.textmodel = BertModel.from_pretrained("bert-base-uncased")
        # local bert
        # self.textmodel = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v1')

        self.textmodel = AutoModel.from_pretrained(lang_encoder_name)
        # self.textmodel.save_pretrained(
        #     '/home/aaronsamd37/reading_PJ/Pointnet_Pointnet2_pytorch_kw20221016_20241222/local_sent_bert/model')
        # self.textmodel = BertModel.from_pretrained("/home/aaronsamd37/reading_PJ/Pointnet_Pointnet2_pytorch_kw20221016_20241222/local_bert/bert-base-uncased_model")
        # self.textmodel.save_pretrained('/home/aaronsxxx/project_repostories/Pointnet_Pointnet2_pytorch_kw20221016/bert-base-uncased')
        self.mapping_lang = torch.nn.Sequential(
            nn.Linear(lang_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, xyz, cls_label=None, encoded_text=None):
        self.textmodel.eval()

        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        # print('<---1---')
        # print(l0_xyz.shape)  # [1, 3, 2048]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.shape, l1_points.shape)  # [1, 3, 512]) [1, 128, 512]
        # print('---end 1--->')
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # torch.Size([1, 3, 128]) torch.Size([1, 256, 128])
        # print('---end 2--->')
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # torch.Size([1, 3, 1]) torch.Size([1, 1024, 1])
        # print('---end 3--->')

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.shape)  # [1, 256, 128]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.shape)  # [1, 128, 512]
        # cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        # l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)
        # print(l0_xyz.shape, l0_points.shape, cls_label_one_hot.shape)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        # print(l0_points.shape)  # [1, 128, 2048]

        model_output = self.textmodel(**encoded_text)
        # last_hidden_state = self.textmodel(**encoded_text, output_hidden_states=False)[0]

        ## Sentence feature at the first position [cls]
        # print(last_hidden_state.shape)  torch.Size([24, 13, 768])
        # raw_flang = last_hidden_state[:, 0, :]
        raw_flang = mean_pooling(model_output, encoded_text['attention_mask'])


        # print(raw_flang.shape)  torch.Size([24, 768])
        # print('-----------------probe-------------')
        raw_flang = raw_flang.detach()
        flang = self.mapping_lang(raw_flang)
        flang = F.normalize(flang, p=2, dim=1)

        flang = flang.view(B, 128, 1).repeat([1, 1, 2048])
        # print(flang.shape)  # [24, 128, 2048]
        merge_feat = torch.cat([l0_points, flang], 1)  # [24, 256, 2048]

        # FC layers
        # feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = F.relu(self.bn1(self.conv1(merge_feat)))
        # print(feat.shape)  # [1, 128, 2048]
        x = self.drop1(feat)
        x = self.conv2(x)  # [1, 50, 2048]
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
