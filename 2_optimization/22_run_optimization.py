from utils.optimization_utils import *

import numpy as np
from tqdm import tqdm, trange

import argparse
import os
import os.path as osp


def optimize_entire_frames(oris, gyrs, path,
                           result_file='optim_angle.npy',
                           optim_method='BFGS',
                           maxiter=200,
                           gtol=1e-5,
                           ftol=1e-6,
                           num_frames=0,
                           resume_optim=True,
                           save_optim_ori=False,
                           **kwargs):
    
    num_frames = oris.shape[1] if num_frames == 0 else num_frames
    oris, gyrs = oris[:, :num_frames], gyrs[:, :num_frames]

    new_oris = oris.copy()
    result_file_path = osp.join(path, result_file)
    resume_optim = resume_optim and osp.exists(result_file_path)
    
    if resume_optim:
        output = np.load(result_file_path)
        if save_optim_ori:
            new_oris = np.load(os.path.join(path, 'optim_ori.npy'))

    for subj in tqdm(range(gyrs.shape[0]), leave=True):
        if resume_optim and subj < output.shape[0]:
            continue
        segment_list = ['Segment 1', 'Segment 2']

        for ori, gyr, seg in zip(np.split(oris[subj], 2, -1), 
                                  np.split(gyrs[subj], 2, -1),
                                  segment_list):            
        
            with trange(gyrs.shape[1]-1, desc=seg, leave=False) as t:
                for frame in t:
                    prev_ori = ori[frame]
                    curr_ori = ori[frame+1]
                    label_gyr = gyr[frame+1]

                    result = optim_single_frame(prev_ori, 
                                                curr_ori, 
                                                label_gyr,
                                                optim_method,
                                                maxiter,
                                                gtol,
                                                ftol)

                    ori[frame+1] = result.x
                    msg = "Error (1e-6): %.3f"%(result.fun*1e6)
                    t.set_postfix_str(msg, refresh=True)

            # Optimize both segments
            if seg == 'Right segment':
                new_oris[subj, :, :4] = ori.copy()
            else:
                new_oris[subj, :, 4:] = ori.copy()


        # Calculate joint angle from optimized orientation
        optim_angle = oris_to_angle(new_oris[subj])
        if subj == 0:
            output = optim_angle[None]
        else:
            output = np.concatenate((output, optim_angle[None]), axis=0)

        # Save results
        np.save(result_file_path, output)
        if save_optim_ori:
            np.save(os.path.join(path, 'optim_ori.npy'), new_oris)


if __name__ == '__main__':

    # Define optimization arguments
    parser = argparse.ArgumentParser(description='Top-down optimization')
    
    parser.add_argument('--pred-ori-path', type=str, 
                        default='Data/5_Optimization/NN_Prediction',
                        help='orientation prediction path of the best model')
    
    parser.add_argument('--result_path', type=str, default='Data/5_Optimization/Results',
                        help='The path to save the optimization result')

    parser.add_argument('--result_file', type=str, default='optim_angle.npy',
                        help='The name of optimization result file')

    parser.add_argument('--activity', default=['Walking', 'Running'], nargs='*', type=str,
                        help='Activity to optimize either one of Walking or Running')

    parser.add_argument('--joint', default=['Hip', 'Knee', 'Ankle'], nargs='*', type=str,
                        help='Joint to optimize either one of Hip or Knee and Ankle')

    parser.add_argument('--optim-method', default='BFGS', type=str,
                        help='Optimization solver method')

    parser.add_argument('--maxiter', type=int, default=500,
                        help='The maximum limitation of iterations')
    
    parser.add_argument('--gtol', type=float, default=1e-5,
                        help='The tolerance threshold for the gradient')
    
    parser.add_argument('--ftol', type=float, default=1e-6,
                        help='The tolerance threshold for the objective function')

    parser.add_argument('--num-frames', type=int, default=0,
                        help='Number of frames to optimize')
    
    parser.add_argument('--resume-optim', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Choose whether to resume the optimization')

    parser.add_argument('--save-optim_ori', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Choose whether to save the optimized orientation')

    args = parser.parse_args()
    
    pred_ori_path = args.pred_ori_path
    result_path = args.result_path
    result_file = args.result_file
    
    activity_lists = args.activity
    joint_lists = args.joint
    
    optim_method = args.optim_method
    maxiter = args.maxiter
    gtol = args.gtol
    ftol = args.ftol

    num_frames = args.num_frames
    resume_optim = args.resume_optim
    save_optim_ori = args.save_optim_ori

    for activity in activity_lists:
        for joint in joint_lists:
            output_path = osp.join(result_path, activity, joint)
            if not osp.exists(output_path):
                os.makedirs(output_path)
            
            # Load initial orientation and input gyroscope data
            pred_ori_path_ = osp.join(pred_ori_path, activity, joint, 'bidir_lstm_70_70_70')
            pred_ori_file = osp.join(pred_ori_path_, 'predictions', 'y_pred_test.npy')
            inpt_gyr_file = osp.join(pred_ori_path_, 'predictions', 'x_test.npy')

            pred_ori = np.load(pred_ori_file)
            inpt_gyr = np.load(inpt_gyr_file)[:, :, [4, 5, 6, 12, 13, 14]]

            print("Starting optimization | %s %s set"%(activity, joint))
            
            # Running script
            optimize_entire_frames(pred_ori, 
                                   inpt_gyr, 
                                   output_path,
                                   result_file = result_file,
                                   optim_method=optim_method,
                                   maxiter=maxiter,
                                   gtol=gtol, 
                                   ftol=ftol,
                                   num_frames=num_frames,
                                   resume_optim=resume_optim,
                                   save_optim_ori=save_optim_ori)