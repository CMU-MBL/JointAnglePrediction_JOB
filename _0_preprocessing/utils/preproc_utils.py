# -------------------------
#
# Functions to preprocess 3D motion capture data into inertial data
#
# --------------------------

from biomech_model.cs import CoordinateSystem

import h5py
import numpy as np
import quaternion
from scipy.signal import find_peaks
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def get_group_ids(h5path):
    # Returns subject ids belonging to top 3 most prevalent groups

    with h5py.File(h5path, 'r') as fh:
        subs = list(fh.keys())
        seqlen_dict = {}
        # First iteration: Get subjects that belong to one of the three major groups
        for sub in subs:
            seqlen = fh[sub + '/markers/R_foot_1'].shape[0]
            if seqlen in seqlen_dict:
                seqlen_dict[seqlen] += 1
            else:
                seqlen_dict[seqlen] = 1
        frequent_seqlens = {key: val for key, val in seqlen_dict.items() if val > 0.1*len(subs)}

        # Second iteration: get subject ids that have one of these sequence lengths
        sub_ids = []
        for sub in subs:
            seqlen = fh[sub + '/markers/R_foot_1'].shape[0]
            if seqlen in frequent_seqlens.keys():
                sub_ids.append(int(sub[1:]))
        sub_ids.sort()

    return sub_ids


def get_meta_data(fh, sub):
    # Gets subject meta data
    meta_array_full = fh[sub+'/meta'][0]

    # Sex (0 = Male, 1 = Female), Height (cm), Weight (kg), Speed (m/s)
    meta_array = meta_array_full[1:]
    return meta_array


def get_h5_data(fh, sub):
    # Used to map lab segment names to calgary segment names
    seg_map = {'pelvis': 'pelvis',
               'R_thigh': 'rthigh', 'R_shank': 'rshank', 'R_foot': 'rfoot',
               'L_thigh': 'lthigh', 'L_shank': 'lshank', 'L_foot': 'lfoot'}
    angle_map = {'pelvis': 'pelvis',
                 'R_hip': 'rhip', 'R_knee': 'rknee', 'R_ankle': 'rankle', 'R_foot': 'rfoot',
                 'L_hip': 'lhip', 'L_knee': 'lknee', 'L_ankle': 'lankle', 'L_foot': 'lfoot'}

    markers = {}
    angles = {}

    # Replaces name conventions with new naming convention
    for marker in fh[sub+'/markers/'].keys():
        new_key = marker.replace(marker[:-2], seg_map[marker[:-2]])
        markers[new_key] = fh[sub+'/markers/'+marker][:, :]

    for angle in fh[sub+'/angles/'].keys():
        new_key = angle.replace(angle, angle_map[angle])
        tmp = fh[sub+'/angles/'+angle][:, :]
        if angle[0].lower() == 'l':
            signs = np.array([1, -1, -1])
        else:
            signs = np.array([1, 1, 1])
        angles[new_key] = signs*tmp[:, [2, 0, 1]]  # Reorder to [flex|add|rot]

    return markers, angles


def get_cs_dict(markers, seg_list):
    # Return dictionary of created coordinate systems

    cs_dict = {}
    # Get right and left foot touchdown events
    revs, levs = get_events(markers)

    # Predefine "ideal" sensor orientation during foot touchdown

    # Iterate over requested segments
    for seg in seg_list:
        # Assert that markers 1 to 3 are unique
        msg = 'The marker set contains duplicates!'
        assert ((markers[seg + '_1'][0, :] != markers[seg + '_2'][0, :]).all() and
                (markers[seg + '_1'][0, :] != markers[seg + '_3'][0, :]).all() and
                (markers[seg + '_2'][0, :] != markers[seg + '_3'][0, :]).all()), msg

        # # Get orthonormal e1 to e3 axes
        # # IMPORTANT: These do not necessarily correlate to x, y, z in any form
        e1 = markers[seg + '_2'] - markers[seg + '_1']
        tmp = markers[seg + '_3'] - markers[seg + '_1']
        e2 = np.cross(tmp, e1, axis=1)
        e3 = np.cross(e1, e2, axis=1)
        
        # Normalize axis vectors
        e1 = e1/(np.linalg.norm(e1, axis=1)[:, None])
        e2 = e2/(np.linalg.norm(e2, axis=1)[:, None])
        e3 = e3/(np.linalg.norm(e3, axis=1)[:, None])

        # Construct coordinate system array
        E = np.stack((e1, e2, e3), axis=2)

        # Get cs with consistent foot touchdown orientation
        # x => Forward, Y => Up, Z => Right
        target_orientation = np.array([[0, 0, 1],
                                       [0, 1, 0],
                                       [-1, 0, 0]])

        if seg[0].lower() == 'l':
            corr_orientation = get_corrected_orientation(E, target_orientation, levs)
        else:
            corr_orientation = get_corrected_orientation(E, target_orientation, revs)

        # Origin of cs = average of markers 1-3
        origin = (markers[seg + '_1'] + markers[seg + '_2'] + markers[seg + '_3'])/3

        cs_dict[seg] = CoordinateSystem(origin, corr_orientation)

    return cs_dict


def get_events(markers):
    # Right and left events => Minima in y-coordinate = Foot touchdown

    revs, _ = find_peaks(-markers['rfoot_1'][:, 1], distance=100, prominence=50)
    levs, _ = find_peaks(-markers['lfoot_1'][:, 1], distance=100, prominence=50)
    return revs, levs


def get_corrected_orientation(E, target_orientation, evs):
    # Correct cs orientation so that it aligns with standard defined as foot touchdown

    E_quats = quaternion.from_rotation_matrix(E)
    target_quats = np.repeat(quaternion.from_rotation_matrix(target_orientation), len(evs))
    # Get relative rotation between marker-cs and ideal-cs during touchdowns
    rel_quats = E_quats[evs].conj()*target_quats
    rel_quat_array = quaternion.as_float_array(rel_quats)
    # Find the average relative orientation between the ideal cs and all touchdown cs
    eigvals, eigvecs = np.linalg.eig(rel_quat_array.T@rel_quat_array)
    av_rel_quat_array = eigvecs[:, np.argmax(eigvals)]
    # Check sign of the averaged quaternion for consistency
    av_signs = np.sign(av_rel_quat_array)
    ar_signs = np.sign(rel_quat_array)
    if (av_signs != ar_signs).all():
        av_rel_quat_array = -1*av_rel_quat_array
    elif (av_signs == ar_signs).all():
        pass
    else:
        pass

    av_rel_quats = np.repeat(quaternion.as_quat_array(av_rel_quat_array), len(E_quats))
    corrected_quats = E_quats*av_rel_quats
    corrected_orientation = quaternion.as_rotation_matrix(corrected_quats)

    return corrected_orientation


def simulate_inertial_data(cs_dict, freq, sub):
    
    for cs in cs_dict.values():
        cs.calc_acc(fr=freq)
        cs.calc_ang_vel(fr=freq)
        cs.repair_ang_vel(fr=freq)

def create_h5_file(h5path, nh5path, sub_ids, fname):
    # Creates processed h5 file in new directory with simulated imu results

    # Write new directory if it does not exist
    if not os.path.exists(nh5path):
        os.makedirs(nh5path)

    # Iterates through all subjects in sub_ids
    sub_list = ['/s'+str(sid) for sid in sub_ids]
    for i, sub in enumerate(tqdm(sub_list)):
        with h5py.File(h5path, 'r') as fh:
            markers, angles = get_h5_data(fh, sub)  # Extract data from h5
            meta = get_meta_data(fh, sub)

        seg_list = ['pelvis', 'rthigh', 'lthigh', 'rshank', 'lshank', 'rfoot', 'lfoot']
        joint_list = ['rhip', 'lhip', 'rknee', 'lknee', 'rankle', 'lankle']

        # Generate cs dictionary and simualte inertial data with 200 Hz output frequency
        cs_dict = get_cs_dict(markers, seg_list)
            
        simulate_inertial_data(cs_dict, 200, sub)

        # Begin writing new h5 file with simulated inertial data
        with h5py.File(nh5path+fname, 'a') as nfh:
            for seg in seg_list:
                # Write acc-data to h5 file
                dat = cs_dict[seg].acc[200:-200, :]
                nfh.create_dataset(sub + '/' + seg + '/acc', data=dat,
                                chunks=(200, dat.shape[1]),
                                maxshape=(None, None), dtype='f8')

                # Write gyr-data to h5 file
                dat = cs_dict[seg].gyr[200:-200, :]
                nfh.create_dataset(sub + '/' + seg + '/gyr', data=dat,
                                chunks=(200, dat.shape[1]),
                                maxshape=(None, None), dtype='f8')

                # Write ori-data to h5 file
                # dat = cs_dict[seg].orientation[200:-200, :].reshape(-1, 9)
                dat = cs_dict[seg].orientation[200:-200, :]
                nfh.create_dataset(sub + '/' + seg + '/rmat', data=dat,
                                chunks=(200, *dat.shape[1:]),
                                maxshape=(None, None, None), dtype='f8')

            for joint in joint_list:
                # Write joint-data to h5 file
                dat = angles[joint][200:-200, :]
                nfh.create_dataset(sub + '/' + joint + '/angle', data=dat,
                                chunks=(200, dat.shape[1]),
                                maxshape=(None, None), dtype='f8')

            # Add meta data to each subject
            nfh.create_dataset(sub + '/meta', data=meta)
            # Boolean to indicate whether the data passed all checks or not
            nfh.attrs['checks_passed'] = False
