import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import open3d as o3d
import torch.utils.data as data
from .build import DATASETS


@DATASETS.register_module()
class PartNet_Mobility(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.data_type = config.data_type
        self.head_type = config.head_type
        self.img_size = config.img_size
        self.point_size = 2048

        self.data_folders = os.path.join(self.data_root, self.data_type)
        categories = os.listdir(self.data_folders)
        print(categories)

        self.models = []
        self.samples = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.data_folders, c)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, self.subset + '.txt')
            with open(split_file, 'r') as f:
                models_c = [line.strip() for line in f.readlines() if line.strip()]

            self.samples += [
                {'category': c, 'sample': m.replace('.npz', '')}
                for m in models_c
            ]

    def random_sample(self, pc, num):
        permutation = np.arange(pc.shape[0])
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx = idx % len(self.samples)

        category = self.samples[idx]['category']
        sample = self.samples[idx]['sample']
        data_path = os.path.join(self.data_folders, category, sample + '.npz')
        try:
            with np.load(data_path) as data:
                img = data['img']
                pc = data['pc']
                type_label = data['type_label']
                pos_label = data['pos_label']
                axis_label = data['axis_label']
                state_label = data['state_label']
        except Exception as e:
            print(e)

        rgba = Image.fromarray(img)
        img = np.array(rgba.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
        img = torch.from_numpy(img).permute(2, 0, 1)

        points = self.random_sample(pc, self.point_size)
        points = torch.from_numpy(points).to(dtype=torch.float32)

        if self.head_type == 'Para':
            label = {'pos': torch.tensor(pos_label),
                     'ori': torch.tensor(axis_label),
                     'state': torch.tensor(state_label),
                     'type': torch.tensor(type_label)}
        else:
            raise ValueError('train_head is unknown')

        return img, points, label