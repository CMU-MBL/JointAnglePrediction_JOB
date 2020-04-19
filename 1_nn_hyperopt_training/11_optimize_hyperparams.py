# -------------------------
#
# Train kinematic estimation neural network using hyperparameter optimization
#
# --------------------------

from utils.hyperopt_utils import run_model

if __name__ == '__main__':

    # Walking pathways
    walking_data_path = 'Data/2_Processed/walking_data.h5'
    walking_result_path = 'Data/3_Hyperopt_Results/Walking/'

    # Running pathways
    running_data_path = 'Data/2_Processed/running_data.h5'
    running_result_path = 'Data/3_Hyperopt_Results/Running/'

    # Run through convolution models
    conv_model = 'CustomConv1D'
    num_eval = 1000

    # Run through for each joint
    # Could take days to run, can comment out certain joints nad run in chunks
    print('Walking convolution begin...')
    walking_conv_hip = run_model(
        walking_data_path, walking_result_path, num_eval, conv_model, ['hip'])
    walking_conv_knee = run_model(
        walking_data_path, walking_result_path, num_eval, conv_model, ['knee'])
    walking_conv_ankle = run_model(
        walking_data_path, walking_result_path, num_eval, conv_model, ['ankle'])
    print('Walking convolution complete!\n')

    print('Running convolution begin...')
    running_conv_hip = run_model(
        running_data_path, running_result_path, num_eval, conv_model, ['hip'])
    running_conv_knee = run_model(
        running_data_path, running_result_path, num_eval, conv_model, ['knee'])
    running_conv_ankle = run_model(
        running_data_path, running_result_path, num_eval, conv_model, ['ankle'])
    print('Running convolution complete!\n')

    # Run through LSTM models
    lstm_model = 'CustomLSTM'
    num_eval = 300

    # Run through each joint
    print('Walking LSTM begin...')
    walking_lstm_hip = run_model(
        walking_data_path, walking_result_path, num_eval, lstm_model, ['hip'])
    walking_lstm_knee = run_model(
        walking_data_path, walking_result_path, num_eval, lstm_model, ['knee'])
    walking_lstm_ankle = run_model(
        walking_data_path, walking_result_path, num_eval, lstm_model, ['ankle'])
    print('Walking LSTM complete!\n')

    print('Running LSTM begin...')
    running_lstm_hip = run_model(
        running_data_path, running_result_path, num_eval, lstm_model, ['hip'])
    running_lstm_knee = run_model(
        running_data_path, running_result_path, num_eval, lstm_model, ['knee'])
    running_lstm_ankle = run_model(
        running_data_path, running_result_path, num_eval, lstm_model, ['ankle'])
    print('Running LSTM complete!\n')
