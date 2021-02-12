from utils.optimization_utils import *
import sys; sys.path.append('./')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

import os
from os.path import join


def show_results(rmse_nn, std_nn, rmse_opt, std_opt, act, joint, calib=False):
    def print_result(rmse, std):
        rmse_flex, rmse_add, rmse_rot = rmse
        std_flex, std_add, std_rot = std

        print('%.2f +- %.2f (Flexion),  %.2f +- %.2f (Adduction),   %.2f +- %.2f (Rotation)'\
            %(rmse_flex, std_flex, rmse_add, std_add, rmse_rot, std_rot))

    calib_ = '(Uncalibrated)' if calib is False else 'Calibrated'

    print('%s NN results %s %s: '%(calib_, joint, act))
    print_result(rmse_nn, std_nn)

    print('%s OPT results %s %s: '%(calib_, joint, act))
    print_result(rmse_opt, std_opt)

    print('\n')


def calculate_results(gt_angle, alpha, beta, std_ratio, weight, act, joint, calib=False):
    
    if beta.shape[0] < gt_angle.shape[0]:
        gt_angle = gt_angle[:beta.shape[0], :beta.shape[1]]
        alpha = alpha[:beta.shape[0], :beta.shape[1]]

    else:
        gt_angle = gt_angle[:, :beta.shape[1]]
        alpha = alpha[:, :beta.shape[1]]

    if calib:
        gt_angle = gt_angle - gt_angle.mean(1)[:, None]
        alpha = alpha - alpha.mean(1)[:, None]
    
    org_beta = (beta - beta.mean(1)[:, None]).copy()

    beta = (beta - beta.mean(1)[:, None]) * std_ratio + alpha.mean(1)[:, None]
    theta = weight * alpha + (1 - weight) * beta
    
    rmse_nn = np.sqrt(((gt_angle - alpha)**2).mean(1)).mean(0)
    std_nn = np.sqrt(((gt_angle - alpha)**2).mean(1)).std(0)
    rmse_opt = np.sqrt(((gt_angle - theta)**2).mean(1)).mean(0)
    std_opt = np.sqrt(((gt_angle - theta)**2).mean(1)).std(0)

    show_results(rmse_nn, std_nn, rmse_opt, std_opt, act, joint, calib)


if __name__ == '__main__':

    angle_path = 'Data/4_Best_Hyperopt/'
    opt_path = 'Data/5_Optimization/Results'
    opt_exp = 'predictions'

    params_file = 'Data/5_Optimization/parameters.pkl'

    with open(params_file, 'rb') as fopen:
        params = pickle.load(fopen)
    
    activity_list = ['Walking', 'Running']
    joint_list = ['Knee', 'Hip', 'Ankle']

    for act in activity_list:
        for joint in joint_list:
            # From angle model path, load ground truth angle and prediction
            angle_path_ = join(angle_path, act, joint)
            _, angle_model, _ = next(os.walk(angle_path_))
            angle_path_ = join(angle_path_, angle_model[0], 'predictions')
            gt_angle = np.load(join(angle_path_, 'y_test.npy'))
            alpha = np.load(join(angle_path_, 'y_pred_test.npy'))

            # From orientation model path, load ground truth orientation
            opt_path_ = join(opt_path, act, joint, opt_exp)
            opt_oris = np.load(join(opt_path_, 'optim_ori.npy'))
            beta = np.load(join(opt_path_, 'optim_angle.npy'))
            
            std_ratio = params['%s_%s_std'%(joint, act)]
            weight = params['%s_%s_weight'%(joint, act)]

            calculate_results(gt_angle, alpha, beta, std_ratio, weight, act, joint)
            calculate_results(gt_angle, alpha, beta, std_ratio, weight, act, joint, calib=True)