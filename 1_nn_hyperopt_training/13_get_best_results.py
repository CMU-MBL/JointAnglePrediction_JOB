# -------------------------
#
# Retrieves best model from all trained hyperopt models
#
# --------------------------

import numpy as np
import os
import shutil
from utils import eval_utils
from glob import glob
import pandas as pd


if __name__ == '__main__':

    base_path = 'Data/3_Hyperopt_Results'
    new_path = 'Data/4_Best_Hyperopt'

    act_list = ['Walking', 'Running']
    sub_folders = ['hip', 'knee', 'ankle']

    for tfldr in act_list:
        task_path = os.path.join(base_path, tfldr)
        for sfldr in sub_folders:
            # Initialize RMSE list
            rmse_list = []
            # Get current path
            result_path = os.path.join(task_path, sfldr)
            # Create folder for best model if it doesn't exist already
            if not os.path.exists(os.path.join(new_path, tfldr, sfldr)):
                os.makedirs(os.path.join(new_path, tfldr, sfldr))
            # Get all model paths for this folder
            model_paths = glob(os.path.join(result_path, '*'))
            model_paths.sort()
            # Get RMSEs for all models
            for model in model_paths:
                (x_val, y_val, y_pred_val,
                 x_test, y_test, y_pred_test) = eval_utils.load_predictions(model)
                val_rmse = eval_utils.calc_rmses(y_val, y_pred_val)
                # test_rmse = eval_utils.calc_rmses(y_test, y_pred_test)
                rmse_list.append(np.mean(val_rmse, axis=0, keepdims=True))
            # Create RMSE array
            rmses = np.concatenate(rmse_list, axis=0)
            score = np.mean(rmses, axis=1, keepdims=True)
            # Get index of best models
            best_idx = np.argmin(score)
            best_model_name = model_paths[best_idx].split('\\')[-1]
            dest = os.path.join(new_path, tfldr, sfldr, best_model_name)
            shutil.copytree(model_paths[best_idx], dest)

    #
    # Creates spreadsheet summary
    #

    col_keys = {'hip': ['hip'],
                'knee': ['knee'],
                'ankle': ['ankle']}
    col_angles = ['Score', 'Flex', 'Add', 'Rot']
    angle_metrics = ['Mean', 'Median', 'Std']
    model_strings = ['conv*', 'lstm*']

    for act in act_list:
        result_path = new_path+'/'+act+'/'

        writer = pd.ExcelWriter(result_path + '_model_summaries.xlsx', engine='xlsxwriter')

        for fldr in sub_folders:
            curr_num_rows = 0
            for i, model_str in enumerate(model_strings):
                # Creates dataframe of validation and test data
                val_df = eval_utils.create_hyperopt_df(
                    col_keys[fldr], col_angles, angle_metrics)
                test_df = eval_utils.create_hyperopt_df(
                    col_keys[fldr], col_angles, angle_metrics)

                pth = result_path + fldr + '/'
                # Summarize all models
                model_paths = glob(pth + model_str)
                model_paths.sort()
                for model in model_paths:
                    model_name = model.replace('\\', '/').split('/')[-1]

                    x_val, y_val, y_pred_val, x_test, y_test, y_pred_test = eval_utils.load_predictions(
                        model)
                    val_rmse = eval_utils.calc_rmses(y_val, y_pred_val)
                    test_rmse = eval_utils.calc_rmses(y_test, y_pred_test)

                    # Adds validation RMSE data to spreadsheet
                    val_df = eval_utils.add_hyperopt_summary(
                        val_df, val_rmse, model_name, col_keys[fldr], col_angles, angle_metrics)
                    # Adds test RMSE data to spreadsheet
                    test_df = eval_utils.add_hyperopt_summary(
                        test_df, test_rmse, model_name, col_keys[fldr], col_angles, angle_metrics)

                # Write to excel (validation on left, test on right)
                val_df.to_excel(writer, sheet_name=fldr, startrow=curr_num_rows, startcol=0)
                test_df.to_excel(writer, sheet_name=fldr, startrow=curr_num_rows,
                                 startcol=val_df.shape[1] + 5)
                # Update start index for the rows
                curr_num_rows += val_df.shape[0] + 4 + 5

        writer.save()

        print('Best models extracted for {}!'.format(act))
