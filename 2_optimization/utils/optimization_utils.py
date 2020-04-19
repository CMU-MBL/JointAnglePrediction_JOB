# -------------------------
#
# Functions for use in hyperopt orientation optimization including
# establishing search space and comparing new IMU data
#
# --------------------------

import numpy as np
import scipy.optimize as so

import quaternion


def quaternion_to_euler(quat):
    # 
    # Convert to rotation matrix
    # 
    
    rm = quaternion.as_rotation_matrix(quat)
    sy = np.sqrt(rm[:, 0, 0] * rm[:, 0, 0] + rm[:, 1, 0] * rm[:, 1, 0])
    singular = sy < 1e-6
    if not singular.any():
        x = np.arctan2(rm[:, 2, 1], rm[:, 2, 2])
        y = np.arctan2(-rm[:, 2, 0], sy)
        z = np.arctan2(rm[:, 1, 0], rm[:, 0, 0])
    else:
        x = np.arctan2(-rm[:, 1, 2], rm[:, 1, 1])
        y = np.arctan2(-rm[:, 2, 0], sy)
        z = 0

    return np.array([x, y, z])


def euler_to_quaternion(roll, pitch, yaw):
    # 
    # Converts euler angle to quaternion and returns as quaternion object
    #

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return np.quaternion(qw, qx, qy, qz)


def ori_to_gyr(inpt_ori, fr=200):
    #
    # Converts quaternion orientation to gyroscope data
    #

    inpt_ori = quaternion.as_quat_array(inpt_ori)
    
    orientation = quaternion.as_rotation_matrix(inpt_ori).transpose((0, 2, 1))
    quats = quaternion.from_rotation_matrix(orientation)
    tmp = 2*fr*quaternion.as_float_array(np.log(quats.conj()[:-1]*quats[1:]))
    gyr = tmp[:, 1:]

    return gyr


def oris_to_angle(oris):
    # 
    # Converts two quaternion orientations to angle vector between them
    #

    quat1 = quaternion.as_quat_array(oris[:, :4])
    quat2 = quaternion.as_quat_array(oris[:, 4:])
    quat_between_segs = quat1.conj() * quat2
    pred_angle = quaternion_to_euler(quat_between_segs)
    
    return rad_to_deg(pred_angle).T


def rad_to_deg(angle):
    return angle * 180 / np.pi


def flip(angle, joint):
    # 
    # Flip angle direction for hip and ankle in specific axes
    # 

    flip = np.array([(-1)**i for i in range(angle.shape[0])])[None]
    tmp = angle.copy()
    if joint == 'Hip':
        tmp[:, :, 2] = angle[:, :, 2] * flip.T
    elif joint == 'Ankle':
        tmp[:, :, 2] = angle[:, :, 2] * ((-1)*flip).T
    angle = tmp.copy()
    return angle


class Objective_Function():
    def __init__(self, prev_ori, label_gyr):
        self.prev_ori = prev_ori
        self.label_gyr = label_gyr

    def __call__(self, curr_ori):
        return self.objective_function(curr_ori)

    def objective_function(self, curr_ori):
        
        oris = np.concatenate((self.prev_ori[None], curr_ori[None]), axis=0)
        pred_gyr = ori_to_gyr(oris)
        loss = np.sqrt(((self.label_gyr - pred_gyr)**2).mean())

        return loss


def build_objective_function(prev_ori, label_gyr):
    return Objective_Function(prev_ori, label_gyr)


def optim_single_frame(prev_ori, curr_ori, label_gyr, optim_method, maxiter, gtol, ftol):

    obj_fcn = build_objective_function(prev_ori, label_gyr)

    options = {'maxiter':maxiter, 'gtol':gtol}
    res = so.minimize(obj_fcn, curr_ori, 
                      method=optim_method, 
                      tol=ftol,
                      options=options)
    
    return res