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

        angle_grads, acc_grads, gyr_grads = dict(), dict(), dict()
        for sub in subs:
            for joint in joint_list:
                angle_array = fh[sub + '/' + joint + '/angle'][:, :]
                angle_grad = np.abs(np.gradient(angle_array)[0])
                try:
                    angle_grads[joint] = np.concatenate((angle_grads[joint], angle_grad.max(0)[None]))
                except:
                    angle_grads[joint] = angle_grad.max(0)[None]
                
            # Check all accelerations
            for seg in seg_list:
                acc_array = fh[sub + '/' + seg + '/acc'][:, :]
                gyr_array = fh[sub + '/' + seg + '/gyr'][:, :]
                acc_grad = np.abs(np.gradient(acc_array)[0])
                gyr_grad = np.abs(np.gradient(gyr_array)[0])
                try:
                    acc_grads[seg] = np.concatenate((acc_grads[seg], acc_grad.max(0)[None]))
                    gyr_grads[seg] = np.concatenate((gyr_grads[seg], gyr_grad.max(0)[None]))
                except:
                    acc_grads[seg] = acc_grad.max(0)[None]
                    gyr_grads[seg] = gyr_grad.max(0)[None]

        import pdb; pdb.set_trace()

        a = 5