"""
MIT-BIH beats (N, S, V, F) with RR channels, exported by export_data.py.
Copied into ai8x-training/datasets/ by run_ai8x.sh.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import ai8x


class ECGRR(Dataset):
    """
    3 x 100 input: ECG window plus pre-RR and post-RR as constant channels, scaled to 0-1.
    """
    def __init__(self, root, split, transform=None, resample_seed=None):
        data = np.load(os.path.join(root, 'ECG_RR', 'ecg_rr.npz'))
        self.X = data[f'X_{split}']
        self.y = data[f'y_{split}']
        self.transform = transform

        if resample_seed is not None:
            # Same idea as WeightedRandomSampler in SmallCNN.py: weight 1/sqrt(count) per class
            counts = np.bincount(self.y)
            weights = 1.0 / np.sqrt(counts[self.y])
            rng = np.random.default_rng(resample_seed)
            idx = rng.choice(len(self.y), size=len(self.y), replace=True, p=weights / weights.sum())
            self.X, self.y = self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i])
        if self.transform:
            x = self.transform(x)
        return x, int(self.y[i])


def ecg_rr_get_datasets(data, load_train=True, load_test=True):
    """
    ai8x calls the second set "test". Here it is our validation data (DS2 is not used).
    """
    (data_dir, args) = data
    transform = transforms.Compose([ai8x.normalize(args=args)])

    train_dataset = ECGRR(data_dir, 'train', transform, resample_seed=0) if load_train else None
    test_dataset = ECGRR(data_dir, 'val', transform) if load_test else None
    return train_dataset, test_dataset


datasets = [
    {
        'name': 'ECG_RR',
        'input': (3, 100),
        'output': ('N', 'S', 'V', 'F'),
        'loader': ecg_rr_get_datasets,
    },
]
