import sys; sys.path.append('./_2_optimization/')

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import pickle
from utils.optimization_utils import *

import os
from os.path import join


def compute_angle_from_ori(gt_ori, joint):
    num_subj, num_frame = gt_ori.shape[:2]
    gt_beta = np.zeros((num_subj, num_frame, 3))

    for subj in range(num_subj):
        leg = 'Right' if subj % 2 ==0 else 'Left'
        
        if gt_ori.shape[-1] == 8:
            gt_beta[subj] = compute_angle_from_quats(gt_ori[subj], joint, leg)
        elif gt_ori.shape[-1] == 6:
            ori1 = gt_ori[subj, :, :, :3]
            ori2 = gt_ori[subj, :, :, 3:]
            diff = np.transpose(ori1 ,(0, 2, 1)) @ ori2
            gt_beta[subj] = compute_angle_from_matrix(diff, joint, leg)

    return gt_beta


def calculate_std_ratio(gt_angle, gt_ori, joint):
    """
    gt : Ground truth angle
    beta : angle calculated from two adjacent sensor orientations
    """

    gt_beta = compute_angle_from_ori(gt_ori, joint)
    gt_angle = gt_angle - gt_angle.mean(1)[:, None]
    gt_beta = gt_beta - gt_beta.mean(1)[:, None]

    std_gt_angle = gt_angle.std(1).mean(0)
    std_gt_beta = gt_beta.std(1).mean(0)
    
    std_ratio = std_gt_angle / std_gt_beta

    return std_ratio


def calculate_weight(gt_angle, alpha, gt_ori, std_ratio, joint):
    """
    gt_angle : Ground truth angle
    alpha : angle prediction from Neural Net
    beta : angle calculated from two adjacent sensor orientations
    """

    gt_beta = compute_angle_from_ori(gt_ori, joint)
    gt_beta = (gt_beta - gt_beta.mean(1)[:, None]) * std_ratio
    gt_beta = gt_beta + alpha.mean(1)[:, None]
    
    rmse = np.zeros((101, 3))
    for i in range(101):
        w = i/100
        theta = w * alpha + (1 - w) * gt_beta
        rmse[i] = np.sqrt(((gt_angle - theta)**2).mean(1)).mean(0)
    
    weight = np.argmin(rmse, axis=0)/100
    
    return weight


if __name__ == "__main__":
    activity_list = ['Walking', 'Running']
    joint_list = ['Hip', 'Knee', 'Ankle']

    output_file = 'Data/5_Optimization/parameters_rotmat.pkl'
    angle_path = 'Data/4_Best_Hyperopt/'
    ori_path = 'Data/5_Optimization/NN_Prediction'

    output = dict()

    for act in activity_list:
        for joint in joint_list:
            # From angle model path, load ground truth angle and prediction
            angle_path_ = join(angle_path, act, joint)
            _, angle_model, _ = next(os.walk(angle_path_))
            angle_path_ = join(angle_path_, angle_model[0], 'predictions')
            gt_angle = np.load(join(angle_path_, 'y_val.npy'))
            alpha = np.load(join(angle_path_, 'y_pred_val.npy'))

            # From orientation model path, load ground truth orientation
            ori_path_ = join(ori_path, act, joint)
            _, ori_model, _ = next(os.walk(ori_path_))
            ori_path_ = join(ori_path_, ori_model[0], 'predictions')
            gt_ori = np.load(join(ori_path_, 'y_val.npy'))

            std_ratio = calculate_std_ratio(gt_angle, gt_ori, joint)
            weight = calculate_weight(gt_angle, alpha, gt_ori, std_ratio, joint)

            output['%s_%s_std'%(joint, act)] = std_ratio
            output['%s_%s_weight'%(joint, act)] = weight
    
    with open(output_file, 'wb') as fopen:
        pickle.dump(output, fopen)
