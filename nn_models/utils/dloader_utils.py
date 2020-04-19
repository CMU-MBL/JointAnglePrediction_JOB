# -------------------------
#
# Classes for organizing different versions of datasets
#
# --------------------------

import numpy as np
import torch
import random
from math import ceil
from torch.utils import data


class SubjectDataset(data.Dataset):
    def __init__(self, sub_dict, seq_len, test=False):
        self.sub_dict = sub_dict
        self.num_sequences = len(self.sub_dict)
        self.seq_len = seq_len
        self.normalize = False
        self.augment = False
        self.test = test
        self.rnd_gen = np.random.RandomState(42)

        seq_len_list = []
        for xy in sub_dict.values():
            seq_len_list.append(xy[0].size()[1])
        self.min_seq_len = min(seq_len_list)
        self.max_seq_len = max(seq_len_list)

        if self.test:
            self.seq_len = self.min_seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        x_out_list = []
        y_out_list = []

        for idx in index.tolist():
            tmp = self.sub_dict[idx]
            # In case of a testing dataset, use right and left, and don't randomize start
            if self.test:
                start = 0
            else:
                start = self.rnd_gen.randint(0, tmp[0].size()[1] - self.seq_len)
            end = start + self.seq_len

            x_out_list.append(tmp[0][:, start:end, :])
            y_out_list.append(tmp[1][:, start:end, :])
        # Concatenate list into one tensor
        x_out = torch.cat(x_out_list, dim=0)
        y_out = torch.cat(y_out_list, dim=0)

        # Apply augmentation and normalization if desired
        if self.augment:
            x_out, y_out = self.augmentor.augment(x_out, y_out)
        if self.normalize:
            x_out, y_out = self.normalizer.normalize(x_out, y_out)

        return (x_out, y_out)

    def add_normalizer(self, normalizer):
        # Adds normalizer to dataset
        self.normalize = True
        self.normalizer = normalizer

    def add_augmentor(self, augmentor):
        # Adds augmentator to dataset
        self.augment = True
        self.augmentor = augmentor


class CustomDataLoader:
    def __init__(self, dset, batch_size=16, shuffle=True):
        self.dset = dset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_batches = ceil(len(dset)/batch_size)
        self.rnd_gen = np.random.RandomState(42)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.idx = 0

        # Get indices
        self.indices = np.arange(len(self.dset))
        if self.shuffle:
            self.rnd_gen.shuffle(self.indices)

        self.indices = torch.from_numpy(self.indices)

        # Split into batches
        self.batch_indices = torch.split(self.indices, self.batch_size, dim=0)

        return self

    def __next__(self):
        if self.idx < self.num_batches:
            x, y = self.dset[self.batch_indices[self.idx]]
            self.idx += 1
            return x, y
        else:
            raise StopIteration


if __name__ == "__main__":
    pass
