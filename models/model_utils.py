import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule


class Residual(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, map_shape, stride=1):
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

        self.Layer2 = self._make_layer(64, 64, 2, 56, stride=1)
        self.Layer3 = self._make_layer(64, 128, 2, 28, stride=2)
        self.Layer4 = self._make_layer(128, 256, 2, 14, stride=2)
        self.Layer5 = self._make_layer(256, 512, 2, 7, stride=2)

        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.Fc = nn.Linear(512, features_dim)

    def _make_layer(self, in_channels, out_channels, blocks, map_shape, stride):
        layers = [Residual(in_channels, out_channels, map_shape, stride)]
        for _ in range(1, blocks):
            layers.append(Residual(out_channels, out_channels, map_shape, 1))
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


class Pointnet2Backbone(nn.Module):
    def __init__(self, input_feature_dim=0):
        super(Pointnet2Backbone, self).__init__()

        self.SA1 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.1,
            nsample=64,
            mlp=[input_feature_dim, 32, 32, 64],
            use_xyz=True,
            normalize_xyz=True
        )

        self.SA2 = PointnetSAModuleVotes(
            npoint=128,
            radius=0.2,
            nsample=32,
            mlp=[64, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.SA3 = PointnetSAModuleVotes(
            npoint=64,
            radius=0.4,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.SA4 = PointnetSAModuleVotes(
            npoint=16,
            radius=0.8,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
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

        xyz, features, fps_inds = self.SA1(xyz, features)
        end_points['SA1_inds'] = fps_inds
        end_points['SA1_xyz'] = xyz
        end_points['SA1_features'] = features

        xyz, features, fps_inds = self.SA2(xyz, features)
        end_points['SA2_inds'] = fps_inds
        end_points['SA2_xyz'] = xyz
        end_points['SA2_features'] = features

        xyz, features, fps_inds = self.SA3(xyz, features)
        end_points['SA3_xyz'] = xyz
        end_points['SA3_features'] = features

        xyz, features, fps_inds = self.SA4(xyz, features)
        end_points['SA4_xyz'] = xyz
        end_points['SA4_features'] = features

        features = self.FP1(end_points['SA3_xyz'], end_points['SA4_xyz'], end_points['SA3_features'], end_points['SA4_features'])
        features = self.FP2(end_points['SA2_xyz'], end_points['SA3_xyz'], end_points['SA2_features'], features)
        end_points['FP2_features'] = features
        end_points['FP2_xyz'] = end_points['SA2_xyz']
        num_seed = end_points['FP2_xyz'].shape[1]
        end_points['FP2_inds'] = end_points['SA1_inds'][:,0:num_seed]
        return end_points
