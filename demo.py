from nn_models.models.pure_conv import CustomConv1D
from nn_models.models.pure_lstm import CustomLSTM

import sys; sys.path.append('2_optimization')
from utils.optimization_utils import optimization_demo

import numpy as np
import pandas as pd
import torch

import pickle
import argparse
from os import path as osp


def run_demo(inpt_data, gyro_data, gt_data, angle_model, ori_model, alpha, result_fldr):
    with torch.no_grad():
        # Predict angle
        angle_model.eval()
        nn_result = model(inpt_data)

        # Predict orientation
        ori_model.eval()
        ori_pred = model(inpt_data)

        nn_result = nn_result.detach().cpu().double().numpy()
        ori_pred = ori_pred.detach().cpu().double().numpy()

    # Optimize orientation
    opt_result = optimization_demo(ori_pred, gyro_data)

    combined_result = alpha * nn_result + (1 - alpha) * opt_result

    calib_nn_result = nn_result - nn_result.mean(axis=1)[:, None, :]
    calib_opt_result = opt_result - opt_result.mean(axis=1)[:, None, :]
    calib_combined_result = alpha * calib_nn_result + (1 - alpha) * calib_opt_result

    if not osp.exists(result_fldr):
        import os; os.makedirs(result_fldr)

    np.save(osp.join(result_fldr, "nn_result.npy"), nn_result)
    np.save(osp.join(result_fldr, "nn_result.npy"), combined_result)
    np.save(osp.join(result_fldr, "calib_nn_result.npy"), calib_nn_result)
    np.save(osp.join(result_fldr, "calib_combined_result.npy"), calib_combined_result)

    mse_nn_result = np.sqrt(((nn_result - gt_data)***2).mean(axis=1)).mean(axis=0)
    mse_opt_result = np.sqrt(((combined_result - gt_data)***2).mean(axis=1)).mean(axis=0)
    mse_calib_nn_result = np.sqrt(((nn_result - gt_data)***2).mean(axis=1)).mean(axis=0)
    mse_calib_opt_result =np.sqrt(((calib_combined_result - gt_data)***2).mean(axis=1)).mean(axis=0)

    #TODO: Save csv file using pandas


def load_custom_data(path, is_gt_data=False):
    """Load IMU data from path.
    Assuming data type as numpy array or torch tensor, other format has not been implemented yet.
    The size of data is Subjects X Frames X Dimension and dimension of the data can be
    three (X, Y, Z) or four (X, Y, Z, norm).
    """
    
    if path[-3:] == "npy":
        _data = np.load(path)
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
    
    if is_gt_data:
        return _data.double().numpy()

    sz_b, sz_l, sz_d = _data.size()
    assert sz_d in [3, 4], "Dimension of imu data should be 3 or 4"
    
    if sz_d == 3:
        norm = torch.norm(_data, p='fro', dim=-1, keepdim=True)
        _data = torch.cat([_data, norm], dim=-1)

    return _data


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Demo code arguments')
    
    parser.add_argument('--joint', choices=["knee", "hip", "ankle"], 
                        type=str, help="The type of joint")

    parser.add_argument('--activity', choices=["walking", "running"], 
                        type=str, help="The type of activity")

    parser.add_argument('--seg1-accel-path', type=str, 
                        default="",
                        help="custom data (segment 1 acceleration) path")
    
    parser.add_argument('--seg2-accel-path', type=str, 
                        default="",
                        help="custom data (segment 2 acceleration) path")
    
    parser.add_argument('--seg1-gyro-path', type=str, 
                        default="",
                        help="custom data (segment 1 gyroscope) path")
    
    parser.add_argument('--seg2-gyro-path', type=str, 
                        default="",
                        help="custom data (segment 2 gyroscope) path")

    parser.add_argument('--gt-angle-path', type=str, 
                        default="",
                        help="custom data (ground-truth angle) path")

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
    
    joint = args.joint
    activity = args.activity
    seg1_accel_path = args.seg1_accel_path
    seg2_accel_path = args.seg2_accel_path
    seg1_gyro_path = args.seg1_gyro_path
    seg2_gyro_path = args.seg2_gyro_path
    gt_angle_path = args.gt_angle_path

    angle_model_fldr = args.angle_model_fldr
    ori_model_fldr = args.ori_model_fldr

    result_fldr = args.result_fldr

    # Load custom data
    seg1_accel = load_custom_data(seg1_accel_path)
    seg2_accel = load_custom_data(seg2_accel_path)
    seg1_gyro = load_custom_data(seg1_gyro_path)
    seg2_gyro = load_custom_data(seg2_gyro_path)
    gt_angle = load_custom_data(gt_angle_path, is_gt_data=True)

    inpt_data = torch.cat([seg1_accel, seg1_ori, seg2_accel, seg2_ori], dim=-1)
    inpt_data.to(device=device, dtype=dtype)

    inpt_gyro = torch.cat([seg1_gyro, seg2_gyro], dim=-1)
    inpt_gyro = inpt_gyro.double().numpy()

    # Load prediction model
    for model_fldr in [angle_model_fldr, ori_model_fldr]:
        with open(osp.join(model_fldr, "model_kwargs.pkl"), "rb") as fopen:
            model_kwargs = pickle.load(fopen)
        model = CustomConv1D(model_kwargs) if model_kwargs["model_type"] == "CustomConv1D" \
                                           else CustomLSTM(model_kwargs)
        state_dict = torch.load(osp.join(model_fldr, "model.pt"))
        model.load_state_dict(state_dict)
        model.to(device=device, dtype=dtype)
        
        if model_fldr == angle_model_fldr:
            angle_model = model
        else:
            ori_model = model

    # Get weight prior alpha
    if activity == "walking":
        if joint == "knee":
            alpha = np.array([0.33, 0.96, 0.94])
        elif joint == "hip":
            alpha = np.array([0.26, 0.95, 0.88])
        elif joint == "ankle":
            alpha = np.array([0.44, 0.94, 0.68])

    else:
        if joint == "knee":
            alpha = np.array([0.36, 0.98, 0.97])
        elif joint == "hip":
            alpha = np.array([0.20, 0.95, 0.97])
        elif joint == "ankle":
            alpha = np.array([0.38, 0.98, 0.62])

    run_demo(inpt_data, inpt_gyro, gt_angle, angle_model, ori_model, alpha, result_fldr)