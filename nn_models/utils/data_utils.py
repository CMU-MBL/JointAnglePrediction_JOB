# -------------------------
#
# Classes for normalizing and augmenting datasets
#
# --------------------------

import numpy as np
import torch
import random
from math import pi


np.random.seed(42)
torch.random.manual_seed(42)
random.seed(42)


class Normalizer:
    def __init__(self, dset, norm_type='standardize', groups=False):
        self.params = {}
        self.norm_type = norm_type
        self.groups = groups
        if norm_type == 'standardize':
            self.calc_standardize_params(dset)

    def calc_standardize_params(self, dset):
        if dset is None:
            return
        # Get previous settings of the dataset
        prev_seq_len = dset.seq_len
        prev_test = dset.test
        # Change settings to maximum, uniformly weighted data
        dset.seq_len = dset.min_seq_len
        dset.test = True
        # Extract x,y pairs for all subjects
        x, y = dset[np.arange(len(dset))]
        # Reset previous settings of the dataset
        dset.seq_len = prev_seq_len
        dset.test = prev_test

        # Calculate normalization parameters
        x_mean = x.mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)
        x_sq_diff_sum = ((x-x_mean)**2).sum(dim=1, keepdim=True).sum(dim=0, keepdim=True)
        x_var = x_sq_diff_sum/(x.size(0)*x.size(1)-1)
        x_std = torch.sqrt(x_var)
        y_sq_diff_sum = ((y-y_mean)**2).sum(dim=1, keepdim=True).sum(dim=0, keepdim=True)
        y_var = y_sq_diff_sum/(y.size(0)*y.size(1)-1)
        y_std = torch.sqrt(y_var)
        if self.groups:
            # Repeat each 4th entry four times (i.e. normalize accs based on acc-norm etc)
            norm_cols = torch.arange(3, x.size()[2], 4)
            self.params['x_mean'] = x_mean
            self.params['x_std'] = x_std[:, :, norm_cols].view(
                1, -1).repeat(4, 1).transpose(0, 1).contiguous().view(1, 1, -1)
            self.params['y_mean'] = y_mean
            self.params['y_std'] = y_std
        else:
            self.params['x_mean'] = x_mean
            self.params['y_mean'] = y_mean
            self.params['x_std'] = x_std
            self.params['y_std'] = y_std

    def normalize(self, x, y):
        return ((x-self.params['x_mean'])/self.params['x_std'],
                (y-self.params['y_mean'])/self.params['y_std'])

    def revert(self, x, y):
        return (x*self.params['x_std']+self.params['x_mean'],
                y*self.params['y_std']+self.params['y_mean'])

    def save_norm_dict(self, pth):
        # Saves normalizer
        norm_dict = {'norm_type': self.norm_type, 'groups': self.groups, 'params': self.params}
        torch.save(norm_dict, pth+'norm_dict.pt')

    def load_norm_dict(self, pth, device):
        norm_dict = torch.load(pth+'norm_dict.pt', map_location=device)
        self.norm_type = norm_dict['norm_type']
        self.params = norm_dict['params']
        if 'groups' in norm_dict.keys():
            self.groups = norm_dict['groups']


class Augmentor:
    def __init__(self, rot_type, device):
        self.rotate = False
        self.noise = False
        self.rot_type = rot_type
        self.device = device
        self.rnd_gen = torch.Generator()
        self.rnd_gen.manual_seed(42)

    def augment(self, x, y):
        # Adds rotational and noise augmentation
        if self.rotate or self.noise:
            x_aug = x.clone()
            y_aug = y.clone()
        if self.rotate:
            x_aug = self.add_rot_noise(x_aug)
        if self.noise:
            x_aug, y_aug = self.add_white_noise(x_aug, y_aug)

        return x_aug, y_aug

    def add_rot_noise(self, x_aug):
        n_sensors = int(x_aug.size()[2]/8)
        for i in range(0, n_sensors):
            # Iterate over all sensors (i.e. groups of 8 input features)
            quats = self.draw_quaternions(x_aug).to(self.device)
            # Rotate accelerometer data of current sensor
            j = [i*8, i*8+3]  # Acceleration indices
            tmp_acc = 2*torch.cross(-1*quats[:, :, 1:], x_aug[:, :, j[0]:j[1]], dim=2)
            x_aug[:, :, j[0]:j[1]] = (x_aug[:, :, j[0]:j[1]] + quats[:, :, 0, None]*tmp_acc +
                                      torch.cross(-1*quats[:, :, 1:], tmp_acc, dim=2))
            # Rotate gyroscope data of current sensor
            j = [i*8+4, i*8+7]  # Gyroscope indices
            tmp_acc = 2*torch.cross(-1*quats[:, :, 1:], x_aug[:, :, j[0]:j[1]], dim=2)
            x_aug[:, :, j[0]:j[1]] = (x_aug[:, :, j[0]:j[1]] + quats[:, :, 0, None]*tmp_acc +
                                      torch.cross(-1*quats[:, :, 1:], tmp_acc, dim=2))

        return x_aug

    def add_white_noise(self, x_aug, y_aug):
        # Sample from zero-mean gaussians with x_std / y_std respectively
        x_noise = torch.randn(x_aug.size(), generator=self.rnd_gen).to(self.device)*self.x_std
        x_aug += x_noise

        y_noise = torch.randn(y_aug.size(), generator=self.rnd_gen).to(self.device)*self.y_std
        y_aug += y_noise

        return x_aug, y_aug

    def set_rot_noise(self, rot_spread=0.05*pi):
        # Sets rotational noise in radians
        self.rot_spread = rot_spread
        self.rotate = True

    def set_white_noise(self, x_std, y_std):
        # Sets white noise for input and output
        self.x_std = x_std
        self.y_std = y_std
        self.noise = True

    def draw_quaternions(self, x_aug):
        n_samples = x_aug.size()[0]
        n_timesteps = x_aug.size()[1]
        rnd_shape = (n_samples, 1, 1)
        # Draw random quaternion axis
        theta = 2*pi*torch.rand(rnd_shape, generator=self.rnd_gen)
        z = 2*torch.rand(rnd_shape, generator=self.rnd_gen) - 1
        # Draw random rotation angle
        if self.rot_type == 'normal':
            # Normal sampling
            phi = torch.normal(0, torch.ones(rnd_shape), generator=self.rnd_gen) * self.rot_spread
        elif self.rot_type == 'uniform':
            # Uniform sampling
            phi = self.rot_spread*torch.rand(rnd_shape, generator=self.rnd_gen)

        # Create scalar and vector part of the quaternion
        tmp_x = torch.sqrt(1-z**2)*torch.cos(theta)
        tmp_y = torch.sqrt(1-z**2)*torch.sin(theta)
        tmp_w = torch.cos(phi)
        vec = torch.sin(phi)*torch.cat((tmp_x, tmp_y, z), dim=2)
        # Concatenate them to full quaternion
        quats = torch.cat((tmp_w, vec), dim=2).expand(-1, n_timesteps, -1)

        return quats
