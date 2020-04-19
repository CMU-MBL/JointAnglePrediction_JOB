# -------------------------
#
# Functions to visualize hfile structure and subject characteristics
#
# --------------------------

import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils.preproc_utils import create_h5_file, get_group_ids


def mean_std(data):
    return np.mean(data), np.std(data)  # Returns mean and std of data


def viz_h5_data(h5path):
    '''
    Function that opens .h5 file, needs to be edited to visualize different paths
    '''

    with h5py.File(h5path, 'r') as fh:
        subs = list(fh.keys())
        print(len(subs))
        print(list(fh[subs[1]].items()))
        print(list(fh[subs[1] + '/meta']))
        print(fh[subs[1] + '/meta'][0])  # Looks at meta data for subject
        print()


def subj_char(h5path):
    '''
    Prints average height, weight, and speed for all subjects male and female after checking data
    '''

    with h5py.File(h5path, 'r+') as fh:
        subs = list(fh.keys())

        valid_subs = 0
        invalid_subs = 0
        total_subs = len(subs)

        count_m = 0
        count_f = 0
        height_m = []
        height_f = []
        weight_m = []
        weight_f = []
        speed_m = []
        speed_f = []

        for sub in subs:
            if fh[sub].attrs['checks_passed']:
                valid_subs += 1

                meta = fh[sub+'/meta']
                if meta[0] == 0.0:
                    count_m += 1
                    if meta[1] > 0:
                        height_m.append(meta[1])
                    if meta[2] > 0:
                        weight_m.append(meta[2])
                    if meta[3] > 0:
                        speed_m.append(meta[3])
                elif meta[0] == 1.0:
                    count_f += 1
                    if meta[1] > 0:
                        height_f.append(meta[1])
                    if meta[2] > 0:
                        weight_f.append(meta[2])
                    if meta[3] > 0:
                        speed_f.append(meta[3])
            else:
                invalid_subs += 1

        height_both = height_m + height_f
        weight_both = weight_m + weight_f
        speed_both = speed_m + speed_f

        print('Males: {0} | Females: {1}'.format(count_m, count_f))

        hm_mean, hm_std = mean_std(height_m)
        wm_mean, wm_std = mean_std(weight_m)
        sm_mean, sm_std = mean_std(speed_m)

        hf_mean, hf_std = mean_std(height_f)
        wf_mean, wf_std = mean_std(weight_f)
        sf_mean, sf_std = mean_std(speed_f)

        hb_mean, hb_std = mean_std(height_both)
        wb_mean, wb_std = mean_std(weight_both)
        sb_mean, sb_std = mean_std(speed_both)

        print()
        print('Males:')
        print('Height: {0} +/- {1} cm'.format(hm_mean, hm_std))
        print('Weight: {0} +/- {1} kg'.format(wm_mean, wm_std))
        print('Speed:  {0} +/- {1} m/s'.format(sm_mean, sm_std))

        print()
        print('Females:')
        print('Height: {0} +/- {1} cm'.format(hf_mean, hf_std))
        print('Weight: {0} +/- {1} kg'.format(wf_mean, wf_std))
        print('Speed:  {0} +/- {1} m/s'.format(sf_mean, sf_std))

        print()
        print('All Valid:')
        print('Height: {0} +/- {1} cm'.format(hb_mean, hb_std))
        print('Weight: {0} +/- {1} kg'.format(wb_mean, wb_std))
        print('Speed:  {0} +/- {1} m/s'.format(sb_mean, sb_std))

        print()
        print('Total subjects: {0} of which {1} are valid and {2} are invalid'.format(
            total_subs, valid_subs, invalid_subs))


if __name__ == '__main__':

    # Input path to desired .h5 files
    h5walk = 'Data/1_Extracted/walking_meta.h5'
    h5run = 'Data/1_Extracted/running_meta.h5'

    viz_h5_data(h5walk)
    viz_h5_data(h5run)