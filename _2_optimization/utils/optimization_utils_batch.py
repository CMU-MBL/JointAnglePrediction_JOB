# -------------------------
#
# Functions for use in hyperopt orientation optimization including
# establishing search space and comparing new IMU data
#
# --------------------------

import numpy as np
import scipy.optimize as so
from scipy.spatial.transform import Rotation as R
import quaternion as Q

from tqdm import tqdm, trange


def normalize_rotmat(rot):
    norm = np.linalg.norm(rot, axis=-2)
    norm_rot = rot / norm[:, :, None]

    return norm_rot


def rot6_to_matrix(rot6):
    """
    Input: Rot 6 vector (B,3,2)
    Output: Rot matrix (B,3,3)
    """

    a1 = rot6[:, :, 0]
    a2 = rot6[:, :, 1]

    b1 = a1 / np.linalg.norm(a1, axis=-1)[:, None]
    b2 = a2 - np.einsum('bi,bi->b', b1, a2)[:, None] * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1)[:, None]
    b3 = np.cross(b1, b2)

    matrix = np.stack((b1, b2, b3), axis=-1)

    return matrix


def compute_angle_from_matrix(rmat, joint, leg, data='synthetic'):
    """Convert rotation matrix to joint angle
    Input:  rmat (Frames, 3, 3)
            joint (Knee / Hip / Ankle)
            leg (Left / Right)
    Output: Joint angle (Frames, 3)
    """
    
    _r = R.from_matrix(rmat)
    angle = _r.as_rotvec() * 180 / np.pi
    angle[:, [0, 1, 2]] = angle[:, [2, 0, 1]]

    flx_fct, add_fct, rot_fct = [-1, 1, 1]

    if joint == 'Hip':
        flx_fct = -1
        if leg == 'Right':
            add_fct, rot_fct = [-1, -1]
        elif leg == 'Left':
            add_fct, rot_fct = [1, 1]
        if data == 'mc10':
            add_fct = add_fct * (-1)

    if joint == 'Knee':
        flx_fct = -1
        if leg == 'Right':
            add_fct, rot_fct = [-1, -1]
        elif leg == 'Left':
            add_fct, rot_fct = [1, 1]
        if data == 'mc10':
            add_fct = add_fct * (-1)

    if joint == 'Ankle':
        flx_fct = -1
        if leg == 'Right':
            add_fct, rot_fct = [-1, -1]
        elif leg == 'Left':
            add_fct, rot_fct = [1, 1]

    flip = np.array([flx_fct, add_fct, rot_fct])
    angle = angle * flip

    return angle


class Objective_Function():
    def __init__(self, prev_ori, curr_ori, label_gyr, fr=200):
        self.prev_ori = prev_ori
        self.curr_ori = curr_ori
        self.label_gyr = label_gyr
        self.fr = fr

    def __call__(self, rot):
        
        return self.objective_function(rot)

    def objective_function(self, ori):
        
        rot6 = ori.reshape(-1, 3, 2)
        rot = rot6_to_matrix(rot6)

        rot_R = R.from_matrix(rot)
        pred_gyr = rot_R.as_rotvec() * self.fr
        loss = np.sqrt(((self.label_gyr - pred_gyr)**2).sum(-1).mean(0))
        loss_ = np.sqrt(((self.label_gyr - pred_gyr)**2).sum(-1))
        # import pdb; pdb.set_trace()
        return loss


def build_objective_function(prev_ori, curr_ori, label_gyr):
    return Objective_Function(prev_ori, curr_ori, label_gyr)


def optim_single_frame(prev_ori, curr_ori, label_gyr, optim_method, maxiter, gtol, ftol):
    obj_fcn = build_objective_function(prev_ori, curr_ori, label_gyr)
    
    start_point = np.array([[1., 0., 0.], [0., 1., 0.]]).T
    start_points = np.repeat(start_point[None], label_gyr.shape[0], axis=0)
    
    options = {'maxiter':maxiter, 'gtol':gtol}
    res = so.minimize(obj_fcn, start_points, method=optim_method, tol=ftol, options=options)
    
    return res


def optimization_demo(oris, gyrs,
                      result_file='optim_angle.npy',
                      optim_method='BFGS',
                      maxiter=50, gtol=1e-5, ftol=1e-6,
                      joint='Knee', leg='angle', **kwargs):
    
    new_oris = np.zeros((oris.shape[0], 2, oris.shape[1], 3, 3))

    for subj in tqdm(range(gyrs.shape[0]), leave=True):
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
                    
                    msg = "Error (1e-6): %.3f"%(result.fun*1e6)
                    t.set_postfix_str(msg, refresh=True)
            
            new_oris[subj, segment_list.index(seg)] = ori

        # Calculate joint angle from optimized orientation
        opt_ori1 = new_oris[subj, 0]
        opt_ori2 = new_oris[subj, 1]
        ori_diff = np.transpose(opt_ori1, (0, 2, 1)) @ opt_ori2
        
        optim_angle = compute_angle_from_matrix(ori_diff, joint, leg, data='mc10')
        
        if subj == 0:
            output = optim_angle[None]
        else:
            output = np.concatenate((output, optim_angle[None]), axis=0)

        print(" \n\n==> Optimization completed\n")
        return output


def compute_angle_from_quats(quats, joint, leg):
    """Convert rotation matrix to joint angle
    Input:  quats (Frames, 8)
            joint (Knee / Hip / Ankle)
            leg (Left / Right)
    Output: Joint angle (Frames, 3)
    """

    quat1 = Q.as_quat_array(quats[:, :4])
    quat2 = Q.as_quat_array(quats[:, 4:])
    diff = quat1.conj() * quat2
    rmat = Q.as_rotation_matrix(diff)

    _r = R.from_matrix(rmat)
    angle = _r.as_rotvec() * 180 / np.pi
    angle[:, [0, 1, 2]] = angle[:, [2, 0, 1]]

    flx_fct, add_fct, rot_fct = [-1, 1, 1]
    
    if leg == 'Right':
        add_fct, rot_fct = [-1, -1]
    elif leg == 'Left':
        add_fct, rot_fct = [1, 1]

    flip = np.array([flx_fct, add_fct, rot_fct])
    angle = angle * flip
    
    return angle


def align_to_segment_cs(orientation, seg, leg):
    if seg == 'pelvis':
        return orientation
    
    seg_ = leg[0].lower() + seg
    target_ori = get_target_orientation('pelvis')
    curr_ori = get_target_orientation(seg_)
    transform = curr_ori.T @ target_ori

    transformed_ori = orientation @ transform

    return transformed_ori