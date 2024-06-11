import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from pointnet2_ops import pointnet2_utils
import pytorch_utils as pt_utils
from typing import List


class PointnetSAModuleVotes(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None, # for RBF pooling
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        if npoint is not None:
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)


    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        else:
            assert(inds.shape[1] == self.npoint)
        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped, inds
        ).transpose(1, 2).contiguous() if self.npoint is not None else None

        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)

        new_features = self.mlp_module(
            grouped_features
        )  # (B, mlp[-1], npoint, nsample)
        if self.pooling == 'max':
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'avg':
            new_features = F.avg_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'rbf':
            # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
            # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            rbf = torch.exp(-1 * grouped_xyz.pow(2).sum(1,keepdim=False) / (self.sigma**2) / 2) # (B, npoint, nsample)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(self.nsample) # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return new_xyz, new_features, inds, unique_cnt


def compute_mov_loss(pred_mov, mov_truth):
    Class_Loss = nn.BCEWithLogitsLoss()
    return Class_Loss(pred_mov, mov_truth.float())


def compute_score_loss(pred_score, score_truth):
    Class_Loss = nn.BCEWithLogitsLoss()
    return Class_Loss(pred_score, score_truth.float())


def compute_type_loss(pred_type, type_truth):
    Class_Loss = nn.BCEWithLogitsLoss()
    return Class_Loss(pred_type, type_truth.unsqueeze(-1).float())


def compute_per_point_state_loss(pred_state, state_truth):
    pred_stat_mean = torch.mean(pred_state, dim=1)
    MAE_loss = F.l1_loss(pred_stat_mean, state_truth.unsqueeze(-1))
    return MAE_loss


def compute_per_point_axis_loss(pred_axis, pred_points, axis_truth, pos_truth):
    aggregated_pos = torch.mean(pred_points, dim=1)
    axis_point_to_pred = aggregated_pos - pos_truth
    cross_product = torch.cross(axis_point_to_pred, axis_truth)
    distance_loss = torch.norm(cross_product, dim=-1) / torch.norm(axis_truth, dim=-1)
    distance_loss_mean = torch.mean(distance_loss)
    direction_loss = torch.norm(pred_axis - axis_truth.unsqueeze(1), dim=-1)
    direction_loss_mean = torch.mean(direction_loss, dim=1)
    return torch.mean(distance_loss_mean + direction_loss_mean, dim=0)


def get_loss(pred, label, train_head, weight_dict):
    loss = 0.
    if train_head == 'Mov':
        loss = compute_mov_loss(pred['mov'], label['mov'])
    elif train_head == 'Para':
        type_loss = compute_type_loss(pred['type'], label['type'])
        state_loss = compute_per_point_state_loss(pred['state'], label['state'])
        joint_axis_loss = compute_per_point_axis_loss(pred['ori'], pred['pos'], label['ori'], label['pos'], weight_dict)
        loss = type_loss * weight_dict['type'] + state_loss * weight_dict['state'] + joint_axis_loss
    elif train_head == 'Score':
        loss = compute_score_loss(pred['score'], label['score'])
    return loss

