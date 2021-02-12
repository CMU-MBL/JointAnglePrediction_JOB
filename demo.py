from nn_models.models.pure_conv import CustomConv1D
from nn_models.models.pure_lstm import CustomLSTM, rot6_to_rotmat
from _2_optimization.utils.optimization_utils import *

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

import pickle
import argparse
import os
from os import path as osp


def butter_low(data, order=4, fc=5, fs=100):
    """
    Zero-lag butterworth filter for column data (i.e. padding occurs along axis 0).
    The defaults are set to be reasonable for standard optoelectronic data.
    """
        
    # Filter design
    b, a = butter(order, 2*fc/fs, 'low')
    # Make sure the padding is neither overkill nor larger than sequence length permits
    padlen = min(int(0.5*data.shape[0]), 200)
    # Zero-phase filtering with symmetric padding at beginning and end
    filt_data = filtfilt(b, a, data, padlen=padlen, axis=1)
    return filt_data


def save_to_csv(path, nn, opt, calib_nn, calib_opt):
    """Creates csv file for results"""
    import pandas as pd

    if path is not None:
        cols = ['RMSE Type', 'Flexion', 'Adduction', 'Rotation']
        df_one = pd.DataFrame(columns=cols)
        df_one.loc[0] = ['NN'] + nn.tolist()
        df_one.loc[1] = ['NN + Opt'] + opt.tolist()
        df_one.loc[2] = ['Calibrated NN'] + calib_nn.to_list()
        df_one.loc[3] = ['Calibrated NN + Opt'] + calib_opt.to_list()
        df_one.to_csv(path+'/RMSE_loss.csv', index=False)

    else:
        pass


def evaluate_result(nn_result, combined_result, gt_angle, result_fldr=None, calib=False):
    """Calculate RMSE result (Neural network, Optimization combined model) by comparing with anatomical markers"""

    def print_result(string, array):
        flex, add, rot = array
        print('%s %.2f (Flexion),  %.2f (Adduction),  %.2f (Rotation)'%(string, flex, add, rot))

    if result_fldr is not None:
        # Save result
        if not osp.exists(result_fldr):
            import os; os.makedirs(result_fldr)
        
        calib_name = "calib_" if calib else ""
        np.save(osp.join(result_fldr, calib_name + "nn_result.npy"), nn_result)
        np.save(osp.join(result_fldr, calib_name + "combined_result.npy"), combined_result)

    if gt_angle is not None:
        gt_angle = gt_angle - gt_angle.mean(axis=1)[:, None, :] if calib else gt_angle
        nn_result = nn_result - nn_result.mean(axis=1)[:, None, :] if calib else nn_result
        combined_result = combined_result - combined_result.mean(axis=1)[:, None, :] if calib else combined_result
        
        rmse_nn_result = np.sqrt(((nn_result - gt_angle)**2).mean(axis=1)).mean(axis=0)
        rmse_opt_result = np.sqrt(((combined_result - gt_angle)**2).mean(axis=1)).mean(axis=0)

    else:
        rmse_nn_result = np.nan
        rmse_opt_result = np.nan
    
    # Print on terminal
    calib_print = '(Calibrated)' if calib else '(Uncalibrated)'
    print_result('Neural Network %s  :'%calib_print, rmse_nn_result)
    print_result('Optimization %s    :'%calib_print, rmse_opt_result)


def run_demo(inpt_data, gyro_data, 
             angle_norm_dict, ori_norm_dict, 
             angle_model, ori_model, 
             weight, std_ratio, result_fldr, 
             joint='Knee', leg='Left', gt_angle=None,
             **kwargs):
    
    # if the beginning part of data is not clean, select some specific sequence to estimate
    start, end = 0, -1

    inpt_data = inpt_data[:1, start:end]
    gyro_data = gyro_data[:1, start:end]
    
    if gt_angle is not None:
        gt_angle = gt_angle[:1, start:end]  
    
    # Neural Network Prediction
    with torch.no_grad():
        # normalize input data
        inpt_data_angle = (inpt_data - angle_norm_dict['x_mean']) / angle_norm_dict['x_std']
        inpt_data_ori = (inpt_data - ori_norm_dict['x_mean']) / ori_norm_dict['x_std']
        
        # Predict angle
        angle_model.eval()
        alpha = angle_model(inpt_data_angle)

        # Predict orientation
        ori_model.eval()
        ori_pred = ori_model(inpt_data_ori)
        ori_pred = rot6_to_rotmat(ori_pred)

        # Un-normalize output prediction
        alpha = alpha * angle_norm_dict['y_std'] + angle_norm_dict['y_mean']

        alpha = alpha.detach().cpu().double().numpy()
        ori_pred = ori_pred.detach().cpu().double().numpy()
        
    # Get beta from optimization
    beta = optimization_demo(ori_pred, gyro_data, joint=joint, leg=leg)

    # Get theta from alpha and beta
    beta = (beta - beta.mean(axis=1)[:, None]) * std_ratio + alpha.mean(axis=1)[:, None]
    theta = weight * alpha[:, start:end] + (1 - weight) * beta

    rmse_nn, rmse_opt = evaluate_result(alpha[:, start:end], theta, gt_angle[:, start:end], 
                                        result_fldr=result_fldr, calib=False)
    print('\n\n')
    rmse_nn_calib, rmse_opt_calib = evaluate_result(alpha[:, start:end], theta, gt_angle[:, start:end], 
                                                   result_fldr=result_fldr, calib=True)

    save_to_csv(result_fldr, rmse_nn, rmse_opt, rmse_nn_calib, rmse_opt_calib)
        
        

def load_custom_data(path, is_imu_data=True):
    """Load IMU data from path.
    Assuming data type as numpy array or torch tensor, other format has not been implemented yet.
    The size of data is Subjects X Frames X Dimension and dimension of the data can be
    three (X, Y, Z) or four (X, Y, Z, norm)."""
    
    if path[-3:] == "npy":
        _data = np.load(path)
        _data = torch.from_numpy(_data)
    elif path[-3:] == "pkl":
        with open(path, "rb") as fopen:
            _data = pickle.load(fopen)
            if isinstance(_data, np.ndarray):
                _data = torch.from_numpy(_data)
            else:
                err_msg = "Data type {} is not supported".format(type(_data))
                assert isinstance(_data, torch.Tensor), err_msg
    else:
        err_msg = "Input file format {} is not supported".format(path[-3:])
        NotImplementedError, err_msg

    # size of imu data is batch (subjects) X length X dimension
    if len(_data.shape) == 2:
        _data = _data[None]
    
    if not is_imu_data:
        if isinstance(_data, torch.Tensor):
            _data = _data.double().numpy()
        return _data
    
    sz_b, sz_l, sz_d = _data.shape
    assert sz_d in [3, 4], "Dimension of imu data should be 3 or 4"

    if sz_d == 3:
        norm = torch.norm(_data, p='fro', dim=-1, keepdim=True)
        _data = torch.cat([_data, norm], dim=-1)
    
    return _data


def prepare_data(root_path, leg, device, dtype):
    
    seg1_accel_path = osp.join(root_path, '%s_seg1_acc.npy'%leg)
    seg2_accel_path = osp.join(root_path, '%s_seg2_acc.npy'%leg)
    seg1_gyro_path = osp.join(root_path, '%s_seg1_gyr.npy'%leg)
    seg2_gyro_path = osp.join(root_path, '%s_seg2_gyr.npy'%leg)
    gt_angle_path = osp.join(root_path, '%s_mocap_ang.npy'%leg)

    # Load custom data
    seg1_accel = load_custom_data(seg1_accel_path)
    seg2_accel = load_custom_data(seg2_accel_path)
    seg1_gyro = load_custom_data(seg1_gyro_path)
    seg2_gyro = load_custom_data(seg2_gyro_path)

    if gt_angle_path is not "":
        gt_angle = load_custom_data(gt_angle_path, is_imu_data=False)
        
        # Smooth Ground-truth values
        b, a = butter(4, 2*5/100, 'low')
        padlen = min(int(0.5*gt_angle.shape[1]), 200)
        gt_angle = filtfilt(b, a, gt_angle, padlen=padlen, axis=1)
    
    else:
        gt_angle = None

    inpt_data = torch.cat([seg1_accel, seg1_gyro, seg2_accel, seg2_gyro], dim=-1)
    inpt_data = inpt_data.to(device=device, dtype=dtype)

    inpt_gyro = torch.cat([seg1_gyro[:, :, :-1], seg2_gyro[:, :, :-1]], dim=-1)
    inpt_gyro = inpt_gyro.double().numpy()

    return inpt_data, inpt_gyro, gt_angle


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Demo code arguments')
    
    parser.add_argument('--joint', choices=["Knee", "Hip", "Ankle"],
                        type=str, help="The type of joint")

    parser.add_argument('--activity', choices=["Walking", "Running"], 
                        type=str, help="The type of activity")

    parser.add_argument('--root-path', type=str, 
                        help="custom data root path")

    parser.add_argument('--angle-model-fldr', type=str, 
                        default="",
                        help="model folder of angle prediction")
    
    parser.add_argument('--ori-model-fldr', type=str, 
                        default="",
                        help="model folder of orientation prediction")

    parser.add_argument('--result-fldr', type=str, 
                        default="",
                        help="folder to save result files")    

    parser.add_argument('--use-cuda', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='cuda configuration')

    args = parser.parse_args()

    dtype = torch.float
    device = 'cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu'
    
    result_fldr = args.result_fldr
    joint = args.joint
    activity = args.activity
    root_path = osp.join(args.root_path, joint)
    leg = 'Left'    # Select the direction of your target leg
    
    inpt_data, inpt_gyro, gt_angle = prepare_data(root_path, leg, device, dtype)
    
    angle_model_fldr = osp.join(args.angle_model_fldr, activity, joint)
    ori_model_fldr = osp.join(args.ori_model_fldr, activity, joint)


    # Load prediction model
    for model_fldr in [angle_model_fldr, ori_model_fldr]:
        _, model, _ = next(os.walk(model_fldr))
        model_fldr_ = osp.join(model_fldr, model[0])
        with open(osp.join(model_fldr_, "model_kwargs.pkl"), "rb") as fopen:
            model_kwargs = pickle.load(fopen)
        model = globals()['CustomConv1D'](**model_kwargs) if model_kwargs["model_type"] == "CustomConv1D" \
                                                          else globals()['CustomLSTM'](**model_kwargs)
        state_dict = torch.load(osp.join(model_fldr_, "model.pt"))
        model.load_state_dict(state_dict)
        model.to(device=device, dtype=dtype)

        if model_fldr == angle_model_fldr:
            angle_model = model
            angle_norm_dict = torch.load(osp.join(model_fldr_, "norm_dict.pt"))['params']

        else:
            ori_model = model
            ori_norm_dict = torch.load(osp.join(model_fldr_, "norm_dict.pt"))['params']        

    # Get optimization parameters (weight, std ratio)
    with open('Data/5_Optimization/parameters.pkl', 'rb') as fopen:
        params = pickle.load(fopen)
    std_ratio = params['%s_%s_std'%(joint, activity)]
    weight = params['%s_%s_weight'%(joint, activity)]

    run_demo(inpt_data, inpt_gyro, angle_norm_dict, 
             ori_norm_dict, angle_model, 
             ori_model, weight, std_ratio, result_fldr, 
             joint=joint, leg=leg, gt_angle=gt_angle)
