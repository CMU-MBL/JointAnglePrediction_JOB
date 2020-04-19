# -------------------------
#
# Functions for use in hyperopt parameter optimization including
# establishing search space, assigning model params, and training/scoring functions
#
# --------------------------

import sys
from utils.eval_utils import load_predictions, calc_rmses
sys.path.append('./')  # noqa
from nn_models import train_model
from copy import copy
import numpy as np
from collections import defaultdict
import pandas as pd
from functools import partial
import torch

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


def search_space(model):
    # Creates a dictionary of the parameters and the spaces to search over

    space = {}

    hidden_layers = [10, 20, 30, 40, 50, 60, 70]

    # Creates parameter space for general parameters
    space = {
        'lr': hp.uniform('lr', 0.00001, 0.01),  # Uniform learning rate
        'lr_decay': hp.uniform('lr_decay', 0.9000, 0.9997),  # Unifrom learning rate decay
        'iter': hp.choice('iter', [500, 1000, 5000]),  # Selects batch iteration
        'dropout': hp.uniform('dropout', 0.0, 0.3),  # Uniform dropout value
        'num_layers': hp.choice('num_layers', [1, 2, 3]),  # Selects number of hidden layers
        'size_layers': hp.choice('size_layers', hidden_layers),  # Selects size of hidden layers
    }

    # Creates parameter space for specific models
    if model == 'CustomConv1D':
        space['window'] = hp.choice('window', [11, 21, 31, 41, 51])  # Select window size
    elif model == 'CustomLSTM':
        space['bidir'] = hp.choice('bidir', [False, True])  # Selects directionality

    return space


def assign_params(params, model, joint, data_path, result_path):
    # Assigns the parameters to the general and model specifications

    # General specification
    general_spec = {
        'model_type': model,
        'data_path': data_path, 'result_path': result_path+joint+'/',
        # Model input and output specification
        'inp': ['pelvis', 'thigh', 'shank', 'foot'], 'outp': ['hip', 'knee', 'ankle'],
        # Logging specification
        'log_metrics': ['loss', 'lr', 'rmse'], 'log_freq': 100, 'check_freq': 1000,
        # Optimizer and learning rate schedule
        'optim': 'Adam', 'lr': params['lr'], 'loss': 'MSELoss', 'scheduler': 'ExponentialLR',
        'lr_decay': params['lr_decay'],  'prog': True,
        # Data augmentation
        'aug': True, 'rot_type': 'normal', 'rot_spread': 0.075,
        'x_noise': 0.15, 'y_noise': 0,
        # Training schedule
        'seq_len': [200, 400], 'num_iter': 2*[params['iter']],
        'batch_size': [8, 16]
    }

    # Assigns inputs and outputs according to joint analyzed
    if joint == 'hip':
        general_spec['inp'] = ['pelvis', 'thigh']
    elif joint == 'knee':
        general_spec['inp'] = ['thigh', 'shank']
    elif joint == 'ankle':
        general_spec['inp'] = ['shank', 'foot']

    general_spec['outp'] = [joint]

    # Model parameters and assigns name string
    if model == 'CustomLSTM':
        model_spec = {
            'inp_size': [8*len(general_spec['inp'])],
            'outp_size': [3*len(general_spec['outp'])],
            'layers': params['num_layers']*[params['size_layers']],
            'dropout': params['num_layers']*[params['dropout']],
            'bidir': params['bidir']
        }
        if params['bidir'] is True:
            tmp_name = 'lstm_bidir'
        else:
            tmp_name = 'lstm_unidir'
        mdl_name = '{}_layers{}_size{}_drop{}_lr{}_decay{}_iter{}'.format(
            tmp_name, params['num_layers'], params['size_layers'],
            params['dropout'], params['lr'],
            params['lr_decay'], params['iter']
        )
    elif model == 'CustomConv1D':
        model_spec = {
            'inp_size': [8*len(general_spec['inp'])],
            'outp_size': [3*len(general_spec['outp'])],
            'conv_layers': params['num_layers']*[params['size_layers']],
            'window': params['window'], 'groups': params['num_layers']*[1],
            'conv_activation': params['num_layers']*['ReLU'],
            'conv_dropout': params['num_layers']*[params['dropout']], 'conv_batchnorm': True,
            'lin_layers': [30],
            'lin_activation': ['Sigmoid'],
            'lin_dropout': []
        }
        model_spec['lin_layers'] = [max(20, 10*len(general_spec['outp']))]
        tmp_name = 'conv1d'
        mdl_name = '{}_layers{}_size{}_win{}_drop{}_lr{}_decay{}_iter{}'.format(
            tmp_name, params['num_layers'], params['size_layers'],
            params['window'], params['dropout'], params['lr'],
            params['lr_decay'], params['iter']
        )

        # Input, output and linear size
        model_spec['inp_size'] = [8*len(general_spec['inp'])]
        model_spec['outp_size'] = [3*len(general_spec['outp'])]

    general_spec['name'] = mdl_name
    model_spec['prediction'] = 'angle'

    return general_spec, model_spec


def obj_fcn(params, model, joint, data_path, result_path):
    # Returns score from training model - averaging RMSE values for all degrees of freedom

    # Assigns parameters to variables
    general_spec, model_spec = assign_params(params, model, joint, data_path, result_path)

    # Train model with current parameters
    train_model.main(copy(general_spec), copy(model_spec), show_bar=False)

    torch.cuda.empty_cache()

    # Calculate average RMSE from validation data by loading predictions
    model_path = general_spec['result_path'] + general_spec['name']
    (x_val, y_val, y_pred_val, x_test, y_test, y_pred_test) = load_predictions(model_path)

    val_rmse = calc_rmses(y_val, y_pred_val)

    # Score = mean RMSE from all degrees of freedom
    score = np.mean(np.mean(val_rmse, axis=0))

    return {'loss': score, 'status': STATUS_OK, 'params': params}


def run_model(data_path, result_path, num_eval, model, joint_list):
    # Runs the hyperoptimization

    # Sets random seed
    seed = 100

    # Creates trials objects for each joint
    trials_hip = Trials()
    trials_knee = Trials()
    trials_ankle = Trials()

    best_models = []

    # Optimizes parameters for each joint
    for joint in joint_list:

        if joint == 'hip':
            trials = trials_hip
        elif joint == 'knee':
            trials = trials_knee
        elif joint == 'ankle':
            trials = trials_ankle

        # Creates search space
        space = search_space(model)

        # Performs hyperopimization to minimize the hyperparameters
        best_model = fmin(
            fn=partial(obj_fcn, model=model, joint=joint,
                       data_path=data_path, result_path=result_path),
            space=space, algo=tpe.suggest, trials=trials,
            max_evals=num_eval, rstate=np.random.RandomState(seed),
            show_progressbar=True)

        print('Minimum RMSE score attained for {}:     {:.4f}'.format(
            joint, trials.best_trial['result']['loss']))

        # Create dataframe of all parameters for each trial
        params_list = [p['params'] for p in trials.results]
        param_dict = defaultdict(list)
        for k, v in ((k, v) for d in params_list for k, v in d.items()):
            param_dict[k].append(v)
        param_df = pd.DataFrame.from_dict(param_dict)

        # Add loss data to dataframe
        loss_dict = {'loss': [x['loss'] for x in trials.results]}
        loss_df = pd.DataFrame.from_dict(loss_dict)

        result_df = pd.concat([loss_df, param_df], axis=1)

        # Export loss data to csv file
        file_path = result_path+joint+'_loss_results_'+model+'_.csv'
        result_df.to_csv(file_path, index=True)
        print('File saved: '+joint+'_loss_results_'+model+'_.csv')
        print()

        if joint == 'hip':
            trials_hip = trials
        elif joint == 'knee':
            trials_knee = trials
        elif joint == 'ankle':
            trials_ankle = trials

        # Returns best model for each joint
        best_models.append(best_model)

    return best_models


if __name__ == '__main__':
    pass
