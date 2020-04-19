# -------------------------
#
# Compiles trained models by hyperoptimization into excel spreadsheet
#
# --------------------------

import pandas as pd
from utils import eval_utils
from glob import glob


if __name__ == '__main__':

    res_path = 'Data/3_Hyperopt_Results/'
    sub_folders = ['hip', 'knee', 'ankle']

    col_keys = {'hip': ['hip'],
                'knee': ['knee'],
                'ankle': ['ankle']}
    col_angles = ['Score', 'Flex', 'Add', 'Rot']
    angle_metrics = ['Mean', 'Median', 'Std']
    model_strings = ['conv*', 'lstm*']

    act_list = ['Walking', 'Running']

    for act in act_list:
        result_path = res_path+act+'/'

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

                    # Loads predictions and calculates RMSE from model results
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

        print('Model summary for {} complete!'.format(act))
