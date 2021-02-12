import os.path as osp
from copy import copy

import sys; sys.path.append('./')
from nn_models.train_model import main


def run_prediction(general_spec, model_spec):
    try:
        main(copy(general_spec), copy(model_spec))
    except AssertionError:
        print('Skipping ' + general_spec['name'] +
              ' because the experiment folder exists already!')


def get_default_model(data_path, result_path):
    # 
    # Predict orientation value for initialization
    # Default model hyper parameters are set as below
    # 
    
    # General specification
    general_spec = {'data_path': data_path, 'result_path': result_path,
                    'model_type': 'CustomLSTM', 'name': 'bidir_lstm_70_70_70',
                    # Logging specification
                    'log_metrics': ['loss', 'lr', 'rmse'], 'log_freq': 100, 'check_freq': 1000,
                    # Optimizer and learning rate schedule
                    'optim': 'Adam', 'lr': 0.0015, 'loss': 'MSELoss', 'scheduler': 'ExponentialLR',
                    'lr_decay': 0.9997,  'prog': True,
                    # Data augmentation
                    'aug': True, 'rot_type': 'normal', 'rot_spread': 0.075,  # 0.5,
                    'x_noise': 0.15, 'y_noise': 0,
                    # Training schedule
                    'seq_len': [200, 400], 'num_iter': [5000, 5000],
                    'batch_size': [8, 16]}
    
    # Model parameters
    model_spec = {'inp_size': [16],
        'prediction': 'angle',
        'outp_size': [3],
        'layers':[70, 70, 70],
        'bidir': True,
        'dropout':[0, 0, 0]}

    general_spec.update({'model_type': 'CustomLSTM'})
    
    return general_spec, model_spec


if __name__ == '__main__':
    # Input/output combinations to iterate over
    inp_lists = [['pelvis', 'thigh'],
                ['thigh', 'shank'],
                ['shank', 'foot']]
    outp_lists = [['hip'],
                ['knee'],
                ['ankle']]
    folder_names = ['Hip',
                    'Knee',
                    'Ankle']

    for activity in ['Walking', 'Running']:
        data_file = 'walking_data.h5' if activity == 'Walking' else 'running_data.h5'
        data_path = osp.join('Data', '2_Processed', data_file)
        base_result_path = osp.join('Data/5_Optimization/NN_Prediction_angle', activity)

        for (inp, outp, fldr) in zip(inp_lists, outp_lists, folder_names):
            result_path = osp.join(base_result_path, fldr) + '/'
            general_spec, model_spec = get_default_model(data_path, result_path)
            general_spec['inp'] = inp
            general_spec['outp'] = outp

            run_prediction(general_spec, model_spec)
