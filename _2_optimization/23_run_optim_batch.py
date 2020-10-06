from utils.optimization_utils_batch import *

import numpy as np
from tqdm import tqdm, trange

import argparse
import time
import os
import os.path as osp


segment_dict = {'Hip': ['pelvis', 'thigh'], 
                'Knee': ['thigh', 'shank'], 
                'Ankle': ['shank', 'foot']}


def run_optimization(oris, gyrs, path, joint,
                     result_file='optim_angle.npy',
                     optim_method='BFGS',
                     maxiter=20, gtol=1e-5, ftol=1e-6,
                     num_frames=0,
                     save_optim_ori=False,
                     **kwargs):

    num_frames = oris.shape[1] if num_frames == 0 else num_frames
    oris, gyrs = oris[:, :num_frames], gyrs[:, :num_frames]

    new_oris = np.zeros((oris.shape[0], 2, oris.shape[1], 3, 3))

    result_file_path = osp.join(path, result_file)

    segment_list = ['Segment 1', 'Segment 2']
    num_subjs = oris.shape[0]

    for ori, gyr, seg in zip(np.split(oris, 2, -1),
                                np.split(gyrs, 2, -1),
                                segment_list):            
        with trange(gyrs.shape[1]-1, desc=seg, leave=False) as t:
            for frame in t:
                label_gyr = gyr[:, frame+1]

                ori[:, frame] = normalize_rotmat(ori[:, frame].copy())
                prev_ori = ori[:, frame]
                curr_ori = ori[:, frame+1]

                t_st = time.time()
                result = optim_single_frame(prev_ori, curr_ori, label_gyr, optim_method, 
                                            maxiter, gtol*num_subjs, ftol*num_subjs)
                t_en = time.time()
                
                rot6 = result.x.reshape(-1, 3, 2)
                rot = rot6_to_matrix(rot6)

                ori[:, frame+1] = ori[:, frame] @ rot
                msg = "Error (1e-4): %.3f (%.1f sec)"%(result.fun*1e4, t_en-t_st)
                t.set_postfix_str(msg, refresh=True)

        # Optimize both segments
        new_oris[:, segment_list.index(seg)] = ori

    # Calculate joint angle from optimized orientation
    leg = 'Right' if subj % 2 == 0 else 'Left'
    # Align sensor coordinate system with segment coordinate system
    
    opt_ori1 = new_oris[:, 0]
    opt_ori2 = new_oris[:, 1]
    import pdb; pdb.set_trace()
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
    pred_ori_path = 'Data/5_Optimization/NN_Prediction_'
    result_path = 'Data/5_Optimization/Results'
    result_file = 'optim_angle.npy'
    
    activity_list = ['Walking', 'Running']
    joint_list = ['Hip', 'Knee', 'Ankle']
    # activity_list = ['Walking']
    # joint_list = ['Knee']
    
    for activity in activity_list:
        for joint in joint_list:
            output_path = osp.join(result_path, activity, joint, 'predictions_torch')
            if not osp.exists(output_path):
                os.makedirs(output_path)

            # Load initial orientation and input gyroscope data
            pred_ori_path_ = osp.join(pred_ori_path, activity, joint, 'bidir_lstm_70_70_70')
            pred_ori_file = osp.join(pred_ori_path_, 'predictions', 'y_pred_test.npy')
            inpt_gyr_file = osp.join(pred_ori_path_, 'predictions', 'x_test.npy')

            pred_ori = np.load(pred_ori_file)
            inpt_gyr = np.load(inpt_gyr_file)[:, :, [4, 5, 6, 12, 13, 14]]

            print("Starting optimization | %s %s set | Result saved in %s"%(activity, joint, osp.join(output_path, result_file)))

            # Running script
            run_optimization(pred_ori, inpt_gyr, output_path, joint, result_file=result_file)