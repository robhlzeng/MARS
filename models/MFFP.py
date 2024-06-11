import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import drop_path, trunc_normal_
from functools import partial
from .build import MODELS
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule
from .model_utils import *

#### Backbone ####
class Residual(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs):
        identity = inputs
        out = F.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(identity)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, features_dim=1000):
        super(ResNet18, self).__init__()

        self.Layer1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.Layer2 = self._make_layer(64, 64, 2, stride=1)
        self.Layer3 = self._make_layer(64, 128, 2, stride=2)
        self.Layer4 = self._make_layer(128, 256, 2, stride=2)
        self.Layer5 = self._make_layer(256, 512, 2, stride=2)

        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.Fc = nn.Linear(512, features_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [Residual(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(Residual(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, imgs:torch.cuda.FloatTensor, end_imgs=None):
        if not end_imgs: end_imgs = {}
        features = self.Layer1(imgs)
        end_imgs['Layer1_features'] = features

        features = self.Layer2(features)
        end_imgs['Layer2_features'] = features

        features = self.Layer3(features)
        end_imgs['Layer3_features'] = features

        features = self.Layer4(features)
        end_imgs['Layer4_features'] = features

        features = self.Layer5(features)
        end_imgs['Layer5_features'] = features

        features = self.Avgpool(features)
        features = self.flatten(features)
        features = self.Fc(features)
        end_imgs['features'] = features
        return end_imgs


class Pointnet2(nn.Module):
    def __init__(self, config, input_feature_dim=0):
        super(Pointnet2, self).__init__()

        self.config = config
        self.point_set = self.config.point_set
        self.SA1 = PointnetSAModule(
            npoint=self.config.point_set[0],
            radius=0.1,
            nsample=64,
            mlp=[input_feature_dim, 32, 32, 64],
            use_xyz=True,
        )

        self.SA2 = PointnetSAModule(
            npoint=self.config.point_set[1],
            radius=0.2,
            nsample=32,
            mlp=[64, 64, 64, 128],
            use_xyz=True,
        )

        self.SA3 = PointnetSAModule(
            npoint=self.config.point_set[2],
            radius=0.4,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
        )

        self.SA4 = PointnetSAModule(
            npoint=self.config.point_set[3],
            radius=0.8,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
        )

        self.FP1 = PointnetFPModule(mlp=[256+256, 256, 256])
        self.FP2 = PointnetFPModule(mlp=[128+256, 128, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud:torch.cuda.FloatTensor, end_points=None):

        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        xyz, features = self.SA1(xyz, features)
        end_points['SA1_xyz'] = xyz
        end_points['SA1_features'] = features

        xyz, features = self.SA2(xyz, features)
        end_points['SA2_xyz'] = xyz
        end_points['SA2_features'] = features

        xyz, features = self.SA3(xyz, features)
        end_points['SA3_xyz'] = xyz
        end_points['SA3_features'] = features

        xyz, features = self.SA4(xyz, features)
        end_points['SA4_xyz'] = xyz
        end_points['SA4_features'] = features

        features = self.FP1(end_points['SA3_xyz'], end_points['SA4_xyz'], end_points['SA3_features'], end_points['SA4_features'])
        features = self.FP2(end_points['SA2_xyz'], end_points['SA3_xyz'], end_points['SA2_features'], features)
        end_points['FP2_features'] = features
        end_points['FP2_xyz'] = end_points['SA2_xyz']
        num_seed = end_points['FP2_xyz'].shape[1]
        return end_points
#### Backbone ####


#### MLDM ####
class MLDM_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_points):
        super(MLDM_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Point_Wise_Conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.Bn1 = nn.BatchNorm2d(self.out_channels)
        self.Global_Avg_Pooling = nn.AdaptiveAvgPool2d(1)
        self.Max_pooling = nn.MaxPool2d(kernel_size=(num_points, 1))

        self.Mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.BatchNorm1d(1),
        )

    def forward(self, img_features, point_features):
        img_features_layeri = self.Global_Avg_Pooling(F.relu(self.Bn1(self.Point_Wise_Conv1(img_features))))
        img_features_layeri = img_features_layeri.squeeze(-1).squeeze(-1)
        point_features = self.Max_pooling(point_features.transpose(2, 1)).squeeze(1)
        features_factor = img_features_layeri + point_features
        features_factor = self.Mlp(features_factor)
        return img_features_layeri, features_factor


class MLDM(nn.Module):
    def __init__(self, config):
        super(MLDM, self).__init__()
        self.vote_num = config.vote_num
        self.MLDM_block1 = MLDM_Block(64, 256, self.vote_num)
        self.MLDM_block2 = MLDM_Block(64, 256, self.vote_num)
        self.MLDM_block3 = MLDM_Block(128, 256, self.vote_num)
        self.MLDM_block4 = MLDM_Block(256, 256, self.vote_num)
        self.MLDM_block5 = MLDM_Block(512, 256, self.vote_num)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, end_imgs, end_points):
        F_layer1, ff_1 = self.MLDM_block1(end_imgs['Layer1_features'], end_points['FP2_features'])
        F_layer2, ff_2 = self.MLDM_block2(end_imgs['Layer2_features'], end_points['FP2_features'])
        F_layer3, ff_3 = self.MLDM_block3(end_imgs['Layer3_features'], end_points['FP2_features'])
        F_layer4, ff_4 = self.MLDM_block4(end_imgs['Layer4_features'], end_points['FP2_features'])
        F_layer5, ff_5 = self.MLDM_block5(end_imgs['Layer5_features'], end_points['FP2_features'])
        ff = F.softmax(torch.cat((ff_1, ff_2, ff_3, ff_4, ff_5), dim=-1), dim=1)
        ff_1 = ff[:, 0:1]
        ff_2 = ff[:, 1:2]
        ff_3 = ff[:, 2:3]
        ff_4 = ff[:, 3:4]
        ff_5 = ff[:, 4:5]
        F_layer = F_layer1 * ff_1 + F_layer2 * ff_2 + F_layer3 * ff_3 + F_layer4 * ff_4 + F_layer5 * ff_5
        F_layer = F_layer.unsqueeze(1).repeat(1, self.vote_num + 1, 1)
        F_point = end_points['FP2_features'].transpose(2, 1)
        F_point = torch.cat([self.cls_token.repeat(F_point.size(0), 1, 1), F_point], dim=1)
        F_fusion = torch.cat((F_layer, F_point), dim=-1)
        return F_fusion
#### MLDM ####


#### Share vote ####
class VotingModule(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim):
        super(VotingModule, self).__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim  # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3 + 3 + 1 + self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)

    def forward(self, seed_xyz, seed_features):
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed * self.vote_factor

        net = F.relu(self.bn1(self.conv1(seed_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, (3+out_dim)*vote_factor, num_seed)

        net = net.transpose(2, 1).view(batch_size, num_seed, self.vote_factor, 7 + self.out_dim)
        offset = net[:, :, :, 0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)

        vote_axis = net[:, :, :, 3:6]
        vote_axis = vote_axis.contiguous().view(batch_size, num_vote, 3)

        vote_state = net[:, :, :, 6:7]
        vote_state = vote_state.contiguous().view(batch_size, num_vote, 1)

        residual_features = net[:, :, :, 7:]  # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2, 1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)

        return vote_xyz, vote_axis, vote_state, vote_features
#### Share vote ####


#### Vision Transformer ####
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm =  norm_layer(embed_dim)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward_features(self, x):
        # x = self.patch_embed(x)
        B, _, _ = x.size()

        x = x
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        cls_representation = x[:, 0, :]
        return x[:, 1:, :], cls_representation
#### Vision Transformer ####



@MODELS.register_module()
class MFFP(nn.Module):
    def __init__(self, config):
        super(MFFP, self).__init__()
        self.config = config
        self.train_head = config.train_head
        self.joint_type = config.joint_type

        # transformer setting
        self.embed_dim = config.embed_dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.drop_path_rate = config.drop_path_rate

        self.img_backbone = ResNet18()
        self.point_backbone = Pointnet2(config)
        self.MLDM_fusion = MLDM(self.config)
        # self.feature_fusion_block = FFB(self.depths, self.embed_dim, self.num_heads, mlp_ratio=4.0)
        self.feature_fusion_block = VisionTransformer(embed_dim=self.embed_dim,
                                                      depth=self.depth,
                                                      num_heads=self.num_heads,
                                                      mlp_ratio=4.,
                                                      qkv_bias=True,
                                                      qk_scale=None,
                                                      drop_rate=0.,
                                                      attn_drop_rate=0.,
                                                      drop_path_rate=self.drop_path_rate,
                                                      norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                      init_values=0.,
                                                      )
        self.share_vote = VotingModule(1, 512)
        self.fc_type = nn.Linear(self.embed_dim, 1)

        self.BCE_LOSS = nn.BCEWithLogitsLoss()
        self.MAE_LOSS = nn.L1Loss()

    def compute_type_acc_loss(self, pred_type, type_truth):
        loss = self.BCE_LOSS(pred_type, type_truth.unsqueeze(-1).float())
        pred_type = torch.sigmoid(pred_type.squeeze(-1))
        pred_type = (pred_type > 0.5).long()
        correct_num = (pred_type == type_truth.long()).sum()
        accuracy = correct_num / type_truth.size(0)
        return accuracy, loss

    def compute_state_error_loss(self, pred_state, state_truth):
        aggregated_state = torch.mean(pred_state, dim=1)
        loss = self.MAE_LOSS(aggregated_state, state_truth.unsqueeze(-1))
        if self.joint_type == 'Revolute':
            state_error = torch.rad2deg(loss)
        else:
            state_error = loss
        return state_error, loss

    def compute_ori_error_loss(self, pred_ori, ori_truth):
        aggregated_ori = torch.mean(pred_ori, dim=1)
        aggregated_ori = F.normalize(aggregated_ori, p=2, dim=-1)
        cosine_similarity = F.cosine_similarity(aggregated_ori, ori_truth, dim=-1)
        error_radians = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))
        # error_radians = torch.acos(torch.clamp(torch.abs(cosine_similarity), -1.0, 1.0))
        error_degrees = torch.rad2deg(error_radians)
        return torch.mean(error_degrees, dim=0), torch.mean(error_degrees, dim=0)

    def compute_pos_error_loss(self, pred_pos, pos_truth, ori_truth):
        aggregated_pos = torch.mean(pred_pos, dim=1)
        axis_point_to_pred = aggregated_pos - pos_truth
        cross_product = torch.cross(axis_point_to_pred, ori_truth)
        distance = torch.norm(cross_product, dim=-1) / torch.norm(ori_truth, dim=-1)
        return torch.mean(distance, dim=0), torch.mean(distance, dim=0)


    def forward(self, img, point_cloud, label):
        outputs={}
        end_imgs = self.img_backbone(img)
        end_points = self.point_backbone(point_cloud)
        fusion_features = self.MLDM_fusion(end_imgs, end_points)
        fusion_features, cls_features = self.feature_fusion_block(fusion_features)
        vote_xyz, vote_axis, vote_state, vote_fetuares = self.share_vote(end_points['FP2_xyz'], fusion_features.transpose(2, 1))
        outputs['type'] = self.fc_type(cls_features)
        outputs['pos'] = vote_xyz
        outputs['ori'] = F.normalize(vote_axis, p=2, dim=-1)
        outputs['state'] = vote_state

        type_acc, type_loss = self.compute_type_acc_loss(outputs['type'], label['type'].cuda())
        state_error, state_loss = self.compute_state_error_loss(outputs['state'], label['state'].cuda())
        ori_error, ori_loss = self.compute_ori_error_loss(outputs['ori'], label['ori'].cuda())
        pos_error, pos_loss = self.compute_pos_error_loss( outputs['pos'], label['pos'].cuda(), label['ori'].cuda())
        loss = type_loss + state_loss + ori_loss + pos_loss
        pos_error = self.compute_pos_error(outputs['pos'], label['pos'], label['ori'])
        return loss, type_acc, state_error, ori_error, pos_error

