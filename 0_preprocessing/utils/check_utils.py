# -------------------------
#
# Functions to check angles and calculated inertial data
#
# --------------------------

import numpy as np


def check_angles(angle_array):
    # Checks ground truth angles

    # Initialize Outputs
    test_passed = True
    angle_list = []

    grad_thresh = np.array([5, 3, 3])

    # Calculate gradient
    grad = np.gradient(angle_array, axis=0)

    # Find changepoints
    cp_arr = np.abs(grad) > grad_thresh

    if cp_arr[:, 0].any():
        test_passed = False
        angle_list.append('Flex')

    if cp_arr[:, 1].any():
        test_passed = False
        angle_list.append('Add')

    if cp_arr[:, 2].any():
        test_passed = False
        angle_list.append('Rot')

    return test_passed, angle_list, np.stack(np.nonzero(cp_arr))


def check_acc(acc_array):
    # Checks generated accelerometer data

    # Initialize Outputs
    test_passed = True
    axis_list = []

    # Thresholds for each axis
    grad_threshold = np.array([20, 15, 15])

    # Calculate gradient
    grad = np.gradient(acc_array, axis=0)

    bool_array = np.abs(grad) > grad_threshold

    if (bool_array[:, 0]).any():
        test_passed = False
        axis_list.append('X')

    if (bool_array[:, 1]).any():
        test_passed = False
        axis_list.append('Y')

    if (bool_array[:, 2]).any():
        test_passed = False
        axis_list.append('Z')

    return test_passed, axis_list, np.stack(np.nonzero(bool_array))


def check_gyr(gyr_array):
    # Checks generated gyroscope data

    # Initialize Outputs
    test_passed = True
    axis_list = []

    # Thresholds for each axis
    gyr_threshold = np.array([12.5, 12.5, 15])

    bool_array = np.abs(gyr_array) > gyr_threshold

    if (bool_array[:, 0]).any():
        test_passed = False
        axis_list.append('X')

    if (bool_array[:, 1]).any():
        test_passed = False
        axis_list.append('Y')

    if (bool_array[:, 2]).any():
        test_passed = False
        axis_list.append('Z')

    return test_passed, axis_list, np.stack(np.nonzero(bool_array))
