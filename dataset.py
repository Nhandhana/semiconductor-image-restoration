import os
import numpy as np
import torch
from torch.utils.data import Dataset


class KLADataset(Dataset):

    def __init__(self, gt_dir, noisy_dir, filenames):

        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.filenames = filenames

    def __len__(self):

        return len(self.filenames)

    def __getitem__(self, index):

        filename = self.filenames[index]

        gt_path = os.path.join(self.gt_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)

        gt = np.load(gt_path)
        noisy = np.load(noisy_path)

        gt = torch.from_numpy(gt).float()
        noisy = torch.from_numpy(noisy).float()

        gt = gt.unsqueeze(0)
        noisy = noisy.unsqueeze(0)

        return noisy, gt