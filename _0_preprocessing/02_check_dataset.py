# -------------------------
#
# Checks simulated data and adds whether checks passed to process data
#
# --------------------------

import h5py
import numpy as np
import matplotlib.pyplot as plt  # noqa
from utils import check_utils

# TODO: Input path to processed data
h5path = 'Data/2_Processed/'

# Checks both walking and running data
activity_list = ['walking', 'running']

for act in activity_list:

    # Extracts specfic h5 file
    h5file = '{}{}_data.h5'.format(h5path, act)

    # Refer to Calgary_issue_report.pdf for examples of checks
    with h5py.File(h5file, 'r+') as fh:

        # List of segments and joints being analyzed
        seg_list = ['pelvis', 'rthigh', 'lthigh', 'rshank', 'lshank', 'rfoot', 'lfoot']
        joint_list = ['rhip', 'lhip', 'rknee', 'lknee', 'rankle', 'lankle']

        subs = list(fh.keys())
        subs.sort(key=lambda x: int(x[1:]))

        joint_error_list = []
        subj_error_list = []
        for sub in subs:
            checks_passed = True

            # Check all joint angles
            for joint in joint_list:
                angle_array = fh[sub + '/' + joint + '/angle'][:, :]
                test_passed_angle, angle_list, locs = check_utils.check_angles(angle_array[200:-200, :])
                if not test_passed_angle:
                    subj_error_list.append(sub)
                    joint_error_list.append([sub, joint, angle_list])
                    checks_passed = False

            # Check all accelerations
            for seg in seg_list:
                acc_array = fh[sub + '/' + seg + '/acc'][:, :]
                test_passed_acc, axis_list, locs = check_utils.check_acc(acc_array[200:-200, :])
                if not test_passed_acc:
                    subj_error_list.append(sub)
                    joint_error_list.append([sub, joint, axis_list])
                    checks_passed = False

            # Check all angular velocities
            for seg in seg_list:
                gyr_array = fh[sub + '/' + seg + '/gyr'][:, :]
                test_passed_gyr, axis_list, locs = check_utils.check_gyr(gyr_array[200:-200, :])
                if not test_passed_gyr:
                    subj_error_list.append(sub)
                    joint_error_list.append([sub, joint, axis_list])
                    checks_passed = False

            fh[sub].attrs['checks_passed'] = checks_passed
        
        # Remove remaining outliers from the joint angles
        rom_dict = {}
        valid_subs = []
        # Initialize lists
        for joint in joint_list:
            rom_dict[joint] = []
        # Iterate over all subjects and joints to get range of motion array
        for sub in subs:
            # If subject is invalid, skip
            if not fh[sub].attrs['checks_passed']:
                continue
            valid_subs.append(sub)
            # Otherwise iterate over all joints and add rom to dictionary list
            for joint in joint_list:
                tmp = fh[sub + '/' + joint + '/angle'][:, :]
                tmp_max = np.max(tmp, axis=0, keepdims=True)
                tmp_min = np.min(tmp, axis=0, keepdims=True)
                rom_dict[joint].append(tmp_max - tmp_min)
        rom_array_list = []
        # Create array over all 6 joints
        for joint in joint_list:
            rom_array_list.append(np.concatenate(rom_dict[joint], axis=0))
        rom_array = np.stack(rom_array_list, axis=2)

        # Find outliers
        n = rom_array.shape[0]
        percentile = 1/100
        n_drop = int(n*percentile/2)
        drop_list = []
        for i in range(0, rom_array.shape[-1]):
            for j in range(0, rom_array.shape[-2]):
                sorted_indices = rom_array[:, j, i].argsort()
                drop_list += list(sorted_indices[:n_drop])
                drop_list += list(sorted_indices[-n_drop:])
        # Get array of all subjects that have to be dropped
        drop_array = np.array(list(set(drop_list)))

        # Iterate over array of previously valid subjects to be dropped
        for drop_id in drop_array:
            fh[valid_subs[drop_id]].attrs['checks_passed'] = False

        valid_subs = 0
        invalid_subs = 0
        total_subs = len(subs)
        for sub in subs:
            if fh[sub].attrs['checks_passed']:
                valid_subs += 1
            else:
                invalid_subs += 1

        print('Total subjects: {0} of which {1} are valid and {2} are invalid'.format(
            total_subs, valid_subs, invalid_subs))
