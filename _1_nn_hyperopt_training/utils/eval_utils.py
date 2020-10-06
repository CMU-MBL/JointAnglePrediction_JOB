# -------------------------
#
# Functions to evaluate training results
#
# --------------------------


import numpy as np
import pandas as pd


def create_hyperopt_df(col_names, col_angles, angle_metrics):
    # Creates dataframe compiling the results from hyperopt experiments

    columns = []
    for col in col_names:
        for angle in col_angles:
            if angle == 'Score':
                columns.append((col, angle, 'Avg. RMSE'))
            else:
                for metric in angle_metrics:
                    columns.append((col, angle, metric))
    df = pd.DataFrame(columns=columns)
    df.columns = pd.MultiIndex.from_tuples(columns)
    
    return df


def add_hyperopt_summary(df, rmse, model_name, col_names, col_angles, angle_metrics):
    # Adds hyperopt results to dataframe for each metric

    for i, col in enumerate(col_names):
        for j, angle in enumerate(col_angles):
            col_idx = i*3 + j if len(col_angles) == 3 else i*3 + j - 1
            if angle == 'Score':
                col_tuple = (col, angle, 'Avg. RMSE')
                df_value = np.mean(np.mean(rmse[:, :], axis=0))
                df.loc[model_name, col_tuple] = df_value
            else:
                for k, metric in enumerate(angle_metrics):
                    col_tuple = (col, angle, metric)
                    if metric == 'Mean':
                        df_value = np.mean(rmse[:, col_idx])
                    elif metric == 'Median':
                        df_value = np.median(rmse[:, col_idx])
                    elif metric == 'Std':
                        df_value = np.std(rmse[:, col_idx])
                    df.loc[model_name, col_tuple] = df_value

    return df


def load_predictions(result_path):
    # Load inputs and outputs for validaiton and test

    pred_path = result_path + '/predictions'

    x_val = np.load(pred_path+'/x_val.npy')
    y_val = np.load(pred_path+'/y_val.npy')
    y_pred_val = np.load(pred_path+'/y_pred_val.npy')
    x_test = np.load(pred_path+'/x_test.npy')
    y_test = np.load(pred_path+'/y_test.npy')
    y_pred_test = np.load(pred_path+'/y_pred_test.npy')

    return x_val, y_val, y_pred_val, x_test, y_test, y_pred_test


def calc_rmses(y, y_pred):
    # Calculate rmse value between predicted and true results

    sq_diff = (y - y_pred)**2
    rmse = np.sqrt(np.mean(sq_diff, axis=1))

    return rmse

# Functions for checking specific model progressions


def load_lr_progression(result_path):
    # Load learning rate over iterations

    log_path = result_path + '/logs'
    tmp_array = np.genfromtxt(log_path+'/lr_log.csv', delimiter=',')

    return tmp_array[:, 0], tmp_array[:, 1:]


def load_rmse_progression(result_path):
    # Load rmse over iterations

    log_path = result_path + '/logs'
    tmp_array = np.genfromtxt(log_path+'/rmse_log.csv', delimiter=',')

    return tmp_array[:, 0], tmp_array[:, 1:]


def load_losses(result_path):
    # Load losses over iterations

    log_path = result_path + '/logs'
    tmp_array = np.genfromtxt(log_path+'/loss_log.csv', delimiter=',')

    return tmp_array[:, 0], tmp_array[:, 1], tmp_array[:, 2]


def get_metrics(y):
    # Get min, max, range of motion, mean z for output

    ang_min = np.min(y, axis=1)
    ang_max = np.max(y, axis=1)
    rom = ang_max - ang_min

    sample_mean = np.mean(y, axis=1)
    angle_mean = np.mean(sample_mean, axis=0)
    angle_mean_std = np.std(sample_mean, axis=0)
    mean_z_val = (sample_mean-angle_mean)/angle_mean_std

    sample_std = np.std(y, axis=1)
    std_std = np.std(sample_std, axis=0)
    mean_std = np.mean(sample_std, axis=0)
    std_z_val = (sample_std - mean_std)/std_std

    return ang_min, ang_max, rom, mean_z_val, std_z_val
