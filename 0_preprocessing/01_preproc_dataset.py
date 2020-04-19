# -------------------------
#
# Functions to visualize sequence lengths of walking/running data
#  and creates proccessed h5 file with largest 3 groups
#
# --------------------------

import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils.preproc_utils import create_h5_file, get_group_ids


def viz_seq_lens(h5path):
    '''
    Displays the sequence lengths and the groups with the largest sequences
    '''

    with h5py.File(h5path, 'r') as fh:  # Extracts sequence lengths from h5file
        subs = list(fh.keys())
        seqlen_dict = {}
        for sub in subs:
            seqlen = fh[sub + '/markers/R_foot_1'].shape[0]
            if seqlen in seqlen_dict:
                seqlen_dict[seqlen] += 1
            else:
                seqlen_dict[seqlen] = 1

        frequent_seqlens = {key: val for key, val in seqlen_dict.items() if val > 0.1*len(subs)}
        rel_share = sum(frequent_seqlens.values())/len(subs)*100
        print('Initial number of subjects: {}'.format(sum(seqlen_dict.values())))
        print('Number of different sequence lengths: {}'.format(len(seqlen_dict)))
        print('{0} groups contain {1:3.1f}% of the data.'.format(len(frequent_seqlens), rel_share))
        print('Major groups: {}'.format(frequent_seqlens))
        print('Total number of subjects: {}'.format(sum(frequent_seqlens.values())))
        data = np.array([[key, val] for key, val in seqlen_dict.items()])
        plt.scatter(data[:, 0], data[:, 1])
        plt.show()


if __name__ == '__main__':

    # Processes both walking and running data
    activity_list = ['walking', 'running']

    # TODO: Input path to extracted .h5 walking file
    h5path = 'Data/1_Extracted/'
    # TODO: Input path to created folder for the processed data
    nh5path = 'Data/2_Processed/'

    for act in activity_list:

        # Extracts specfic h5 file
        h5file = '{}{}_meta.h5'.format(h5path, act)
        # Creates new file name
        nfile = '{}_data.h5'.format(act)

        # Visualizes sequence lengths
        viz_seq_lens(h5file)

        # Groups subjects into datasets of set sizes
        sub_ids = get_group_ids(h5file)

        # Creates processed .h5 file
        create_h5_file(h5file, nh5path, sub_ids, nfile)
