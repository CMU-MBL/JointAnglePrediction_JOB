import sys; sys.path.append('./')
from utils.optimization_utils import *

import numpy as np
from tqdm import tqdm, trange

import argparse
import os
import os.path as osp


def run_optimization(oris, gyrs, path, joint, result_file,
                     optim_method='BFGS', maxiter=200, gtol=1e-5, ftol=1e-6,
                     num_frames=0, resume_optim=False, save_optim_ori=False,
                     **kwargs):

    # If you want to run only part of your data, check this value to be <num of target frames>
    num_frames = oris.shape[1] if num_frames == 0 else num_frames
    oris, gyrs = oris[:, :num_frames], gyrs[:, :num_frames]

    result_file_path = osp.join(path, result_file)
    # If you want to save optimized orientation as well, set save_optim_ori as True
    resume_optim = resume_optim and osp.exists(result_file_path)

    new_oris = np.zeros((oris.shape[0], 2, oris.shape[1], 3, 3))
    
    if resume_optim:
        output = np.load(result_file_path)
        if save_optim_ori:
            new_oris = np.load(os.path.join(path, 'optim_ori.npy'))

    for subj in tqdm(range(gyrs.shape[0]), leave=True):
        # If you want to resume your optimization in case that you have previously run some subjects,
        # set resume_optim as True
        if resume_optim and subj < output.shape[0]:
            continue
        segment_list = ['Segment 1', 'Segment 2']

        for ori, gyr, seg in zip(np.split(oris[subj][None], 2, -1),
                                 np.split(gyrs[subj], 2, -1),
                                 segment_list):            

            with trange(gyrs.shape[1]-1, desc=seg, leave=False) as t:
                for frame in t:
                    label_gyr = gyr[frame+1]

                    ori[0, frame] = normalize_rotmat(ori[0, frame].copy())
                    prev_ori = ori[0, frame]
                    curr_ori = ori[0, frame+1]

                    result = optim_single_frame(prev_ori, curr_ori, label_gyr,
                                                optim_method, maxiter, gtol, ftol)

                    if result.x.shape[0] == 9:
                        rot = normalize_rotmat(result.x.reshape(3, 3))
                    else:
                        rot6 = result.x.reshape(3, 2)
                        rot = rot6_to_matrix(rot6)

                    ori[0, frame+1] = ori[0, frame] @ rot

                    msg = "Error (1e-4): %.3f"%(result.fun*1e4)
                    t.set_postfix_str(msg, refresh=True)

            # Optimize both segments
            new_oris[subj, segment_list.index(seg)] = ori

        # Calculate joint angle from optimized orientation
        leg = 'Right' if subj % 2 == 0 else 'Left'

        # Align sensor coordinate system with segment coordinate system
        opt_ori1 = new_oris[subj, 0]
        opt_ori2 = new_oris[subj, 1]
        ori_diff = np.transpose(opt_ori1, (0, 2, 1)) @ opt_ori2

        # ori_diff = np.transpose(new_oris[subj, 0], (0, 2, 1)) @ new_oris[subj, 1]
        optim_angle = compute_angle_from_matrix(ori_diff, joint, leg)

        if subj == 0:
            output = optim_angle[None]
        else:
            output = np.concatenate((output, optim_angle[None]), axis=0)

        # Save results
        np.save(result_file_path, output)
        if save_optim_ori:
            np.save(os.path.join(path, 'optim_ori.npy'), new_oris)


if __name__ == '__main__':

    ori_path = 'Data/5_Optimization/NN_Prediction'
    result_path = 'Data/5_Optimization/Results'
    result_file = 'optim_angle.npy'
    
    activity_list = ['Walking', 'Running']
    joint_list = ['Hip', 'Knee', 'Ankle']

    for activity in activity_list:
        for joint in joint_list:
            output_path = osp.join(result_path, activity, joint, 'predictions')
            if not osp.exists(output_path):
                os.makedirs(output_path)

            # Load initial orientation and input gyroscope data
            ori_path_ = osp.join(ori_path, activity, joint, 'bidir_lstm_70_70_70')
            pred_ori_file = osp.join(ori_path_, 'predictions', 'y_pred_test.npy')
            inpt_gyr_file = osp.join(ori_path_, 'predictions', 'x_test.npy')

            pred_ori = np.load(pred_ori_file)
            inpt_gyr = np.load(inpt_gyr_file)[:, :, [4, 5, 6, 12, 13, 14]]

            print("Starting optimization | %s %s set | Result saved in %s"%(activity, joint, osp.join(output_path, result_file)))

            # Running script
            run_optimization(pred_ori, inpt_gyr, output_path, joint, result_file=result_file)
