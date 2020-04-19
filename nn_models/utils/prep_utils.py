# -------------------------
#
# Functions to prepare for model training
#
# --------------------------

import numpy as np
import torch
import h5py
from math import ceil
from numpy.random import RandomState


def get_sub_dict(h5path, subject_ids, inp_fields, outp_fields, prediction, device):
    # Create dictionary of subject data

    if not isinstance(inp_fields, list):
        inp_fields = [inp_fields]
    if not isinstance(outp_fields, list):
        outp_fields = [outp_fields]

    sub_dict = {}
    with h5py.File(h5path, 'r') as fh:
        for eid, sid in enumerate(subject_ids):
            # Gather inputs from subject data
            y_ori_list, x_list = [], []
            for inp in inp_fields:
                # Checks if segments are labelled as left/right
                if inp[0] != 'r' and inp[0] != 'l' and not (inp == 'pelvis'):
                    # Extract right and left data and append norm as fourth column
                    # Right accelerations
                    right_acc = fh['s' + str(sid) + '/r' + inp + '/acc'][:, :]
                    tmp = np.linalg.norm(right_acc, axis=1, keepdims=True)
                    right_acc = np.concatenate((right_acc, tmp), axis=1)
                    # Right angular velocities
                    right_gyr = fh['s' + str(sid) + '/r' + inp + '/gyr'][:, :]
                    tmp = np.linalg.norm(right_gyr, axis=1, keepdims=True)
                    right_gyr = np.concatenate((right_gyr, tmp), axis=1)
                    # Left accelerations
                    left_acc = fh['s' + str(sid) + '/l' + inp + '/acc'][:, :]
                    tmp = np.linalg.norm(left_acc, axis=1, keepdims=True)
                    left_acc = np.concatenate((left_acc, tmp), axis=1)
                    # Left angular velocities
                    left_gyr = fh['s' + str(sid) + '/l' + inp + '/gyr'][:, :]
                    tmp = np.linalg.norm(left_gyr, axis=1, keepdims=True)
                    left_gyr = np.concatenate((left_gyr, tmp), axis=1)
                    # Right orientations
                    right_ori = fh['s' + str(sid) + '/r' + inp + '/ori'][:, :]
                    # Left orientations
                    left_ori = fh['s' + str(sid) + '/l' + inp + '/ori'][:, :]

                    # Stack right and left data
                    right_tmp = np.concatenate((right_acc, right_gyr), axis=1)
                    left_tmp = np.concatenate((left_acc, left_gyr), axis=1)

                    x_list.append(np.stack((right_tmp, left_tmp), axis=0))
                    # Add orientation ground truth
                    y_ori_list.append(np.stack((right_ori, left_ori), axis=0))
                else:
                    # Accelerations
                    acc = fh['s' + str(sid) + '/' + inp + '/acc'][:, :]
                    tmp = np.linalg.norm(acc, axis=1, keepdims=True)
                    acc = np.concatenate((acc, tmp), axis=1)
                    # Angular velocities
                    gyr = fh['s' + str(sid) + '/' + inp + '/gyr'][:, :]
                    tmp = np.linalg.norm(gyr, axis=1, keepdims=True)
                    gyr = np.concatenate((gyr, tmp), axis=1)
                    # Orientations
                    ori = fh['s' + str(sid) + '/' + inp + '/ori'][:, :]
                    # Concatenate accelerations and angular velocity columns
                    tmp = np.concatenate((acc, gyr), axis=1)
                    if inp == 'pelvis' and ('thigh' in inp_fields or
                                            'shank' in inp_fields or
                                            'foot' in inp_fields):
                        tmp = np.stack((tmp, tmp), axis=0)
                        ori = np.stack((ori, ori), axis=0)
                        x_list.append(tmp)
                        # Add orientation ground truth
                        y_ori_list.append(ori)
                    else:
                        x_list.append(tmp[None, :, :])
                        # Add orientation ground truth
                        y_ori_list.append(ori[None, :, :])
            x = np.concatenate(x_list, axis=2)

            y_angle_list = []
            for outp in outp_fields:
                if outp[0] != 'r' and outp[0] != 'l':
                    right_angle = fh['s' + str(sid) + '/r' + outp + '/angle'][:, :]
                    left_angle = fh['s' + str(sid) + '/l' + outp + '/angle'][:, :]
                    tmp = np.stack((right_angle, left_angle), axis=0)
                    y_angle_list.append(tmp)
                else:
                    angle = fh['s' + str(sid) + '/' + outp + '/angle'][:, :]
                    y_angle_list.append(angle[None, :, :])

            y_ori = np.concatenate(y_ori_list, axis=2)
            y_angle = np.concatenate(y_angle_list, axis=2)
            
            if prediction == 'angle':
                y = y_angle.copy()
            elif prediction == 'orientation':
                y = y_ori.copy()
            
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            sub_dict[eid] = [x, y]

    return sub_dict


def get_subject_split(h5path, split):
    # Splits dataset into train, validation, and test datasets

    # Extracts list of subjects who passed the checks
    sub_list = []
    with h5py.File(h5path, 'r') as fh:
        subs = list(fh.keys())
        subs.sort(key=lambda x: int(x[1:]))
        for sub in subs:
            if fh[sub].attrs['checks_passed']:
                sub_list.append(sub)
    # Create separate random number generator stream for subject split
    split_rnd_gen = RandomState(42)

    # Split up ids of valid subject into training, validation and testing
    ids = [int(x[1:]) for x in sub_list]
    split_rnd_gen.shuffle(ids)
    split_ids = [ceil(split[0]*len(ids)), ceil((split[0]+split[1])*len(ids))]
    train_ids = ids[0:split_ids[0]]
    val_ids = ids[split_ids[0]:split_ids[1]]
    test_ids = ids[split_ids[1]:]

    return train_ids, val_ids, test_ids


def get_device(show_bar=True):
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if show_bar:
        print('Using device:', device)
        print()
        # Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')

    return device


def get_cpu():
    # Check for cpu availability
    device = torch.device('cpu')

    return device


def parse_scheduler(general_spec):
    scheduler_spec = {}
    scheduler_type = general_spec['scheduler']
    scheduler_spec['type'] = scheduler_type

    if scheduler_type == 'ExponentialLR':
        arg_dict = {'gamma': general_spec['lr_decay']}
        scheduler_spec['args'] = arg_dict

    return scheduler_spec


def parse_stages(general_spec):
    seq_lens = general_spec['seq_len']
    batch_sizes = general_spec['batch_size']
    num_iters = general_spec['num_iter']

    # Makes sure amount of stage specs is consistent
    assert (len(seq_lens) == len(batch_sizes) and
            len(seq_lens) == len(num_iters)), 'Numbers of stages is inconsistent'

    num_stages = len(seq_lens)

    # Creates stage spec dictionary for each stage
    stage_spec = {}
    for i in range(0, num_stages):
        stage_spec[i] = {'seq_len': seq_lens[i], 'batch_size': batch_sizes[i],
                         'num_iter': num_iters[i], 'disprog': not general_spec['prog']}

    return stage_spec
