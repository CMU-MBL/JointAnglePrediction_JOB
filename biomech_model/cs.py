# -------------------------
#
# Defines CoordinateSystem class and associated functions
#
# --------------------------

import numpy as np
import quaternion
from scipy.signal import butter, filtfilt


def butter_low(data, order=4, fc=5, fs=100):
    '''
    Zero-lag butterworth filter for column data (i.e. padding occurs along axis 0).
    The defaults are set to be reasonable for standard optoelectronic data.
    '''
    # Filter design
    b, a = butter(order, 2*fc/fs, 'low')
    # Make sure the padding is neither overkill nor larger than sequence length permits
    padlen = min(int(0.5*data.shape[0]), 200)
    # Zero-phase filtering with symmetric padding at beginning and end
    filt_data = filtfilt(b, a, data, padlen=padlen, axis=0)
    return filt_data


def rotate_vec(quat_mat, vec):
    t = 2*np.cross(quat_mat[:, 1:], vec, axis=1)
    new_vec = vec + quat_mat[:, 0, None]*t + np.cross(quat_mat[:, 1:], t, axis=1)

    return new_vec


class CoordinateSystem:
    def __init__(self, origin, orientation):
        self.c = {'x': (1, 0, 0, 1), 'y': (0, 1, 0, 1), 'z': (0, 0, 1, 1)}
        self.lw = {'x': 5, 'y': 5, 'z': 5}
        self.ax_len = 1

        self.origin = origin
        self.orientation = orientation
        self.calc_axes()

    def calc_axes(self):
        # Generates 3D cs axes from origin and orientation inputs
        self.axes = {}
        self.axes['x'] = np.stack(
            [self.origin, self.origin
             + self.ax_len*self.orientation[:, :, 0]], axis=2)
        self.axes['y'] = np.stack(
            [self.origin, self.origin
             + self.ax_len*self.orientation[:, :, 1]], axis=2)
        self.axes['z'] = np.stack(
            [self.origin, self.origin
             + self.ax_len*self.orientation[:, :, 2]], axis=2)

    def calc_acc(self, fr=100, grav_axis=1):
        # Calculate linear acceleration values from 3D positions

        self.acc = np.zeros(self.origin.shape)
        vel = fr*np.gradient(self.origin/1000, axis=0)
        self.acc = -fr*np.gradient(vel, axis=0)  # Sensor actually measures inertia => -1
        self.acc[:, grav_axis] -= 9.81
        quats = quaternion.from_rotation_matrix(self.orientation.transpose((0, 2, 1)))
        self.acc = rotate_vec(quaternion.as_float_array(quats.conj()), self.acc)

    def calc_ang_vel(self, fr=100):
        # Calculate angular velocity values from 3D positions

        self.gyr = np.zeros(self.origin.shape)
        quats = quaternion.from_rotation_matrix(self.orientation)
        tmp = 2*fr*quaternion.as_float_array(np.log(quats.conj()[:-1]*quats[1:]))
        self.gyr[0, :] = tmp[0, 1:]
        self.gyr[1:, :] = tmp[:, 1:]

    def calc_ori(self, fr=100):
        # Calculate orientation values from 3D positions
        
        quats = quaternion.from_rotation_matrix(self.orientation.transpose((0, 2, 1)))
        self.ori = quaternion.as_float_array(quats)

    def repair_ang_vel(self, fr=100):
        # Repair outliers in angular velocity generation

        diff = np.diff(self.gyr, axis=0)
        diff = np.concatenate((diff[None, 0, :], diff))
        outlier_idx = np.unique(np.argwhere(abs(diff) > 10)[:, 0])  # Find rows with outliers
        for idx in outlier_idx:
            start = max((idx-5), 0)
            end = min((idx+6), (self.gyr.shape[0]-1))
            self.gyr[idx, :] = np.median(self.gyr[start:end, :], axis=0)  # Rough fix
        self.gyr = butter_low(self.gyr, fs=fr, fc=8)
