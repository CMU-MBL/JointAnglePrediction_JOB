import argparse

def get_train_configuration(parser):
    # Define training configuration from 
    parser.add_argument('--activity', type=str, default='Walking', help='Walking or Running')
    parser.add_argument('--meta_arch', type=str, default='conv', help='conv or lstm')
    parser.add_argument('--prediction', type=str, default='angle', help='What to predict')
    args = parser.parse_args()

    params = dict()
    params['activity'] = args.activity
    params['meta_arch'] = args.meta_arch
    params['prediction'] = args.prediction

    return params


def get_model_spec(params):
    assert params['activity'] in ['Walking', 'Running']
    assert params['meta_arch'] in ['conv', 'lstm']

    result_path = 'Dataset/3_Results/final/'
    result_path += params['prediction'] + '/'
        
    if params['activity'] == 'Walking':
        data_path = 'Data/2_Processed/walking_data.h5'
        result_path += 'Walking/'
    else:
        data_path = 'Data/2_Processed/running_data.h5'
        result_path += 'Running/'
    
    # General specification
    general_spec = {'data_path': data_path, 'result_path': result_path,
                    # Logging specification
                    'log_metrics': ['loss', 'lr', 'rmse'], 'log_freq': 100, 'check_freq': 1000,
                    # Optimizer and learning rate schedule
                    'optim': 'Adam', 'lr': 0.0015, 'loss': 'MSELoss', 'scheduler': 'ExponentialLR',
                    'lr_decay': 0.9997,  'prog': True,
                    # Data augmentation
                    'aug': True, 'rot_type': 'normal', 'rot_spread': 0.075,
                    'x_noise': 0.15, 'y_noise': 0, 
                    'acc_bias': 0.0, 'gyro_bias': 0.0,
                    # Training schedule
                    'seq_len': [200, 400], 'num_iter': [5000, 5000],
                    'batch_size': [8, 16]}

    # Model parameters
    model_spec = {'inp_size': [16],
        'prediction': params['prediction']}

    if params['meta_arch'] == 'conv':
        general_spec.update({'model_type': 'CustomConv1D'})
        model_spec.update(
            {'window': 41, 
            'conv_dropout': [0, 0, 0], 
            'conv_batchnorm': True,
            'lin_activation': ['Sigmoid'],
            'lin_dropout': []})

    else:
        general_spec.update({'model_type': 'CustomLSTM'})
    
    if model_spec['prediction'] == 'angle':
        model_spec['outp_size'] = [3]
    elif model_spec['prediction'] == 'orientation':
        model_spec['outp_size'] = [12]
    
    return general_spec, model_spec


def get_layers_list(params):
    if params['meta_arch'] == 'conv':
        layers_list = [
            # One-layer models
            [20], [30], [40], [50], [60], [70], [80],
            # Two-layer models
            [20, 20], [30, 30], [40, 40], [50, 50], [60, 60], [70, 70],
            # Three-layer models
            [20, 20, 20], [30, 30, 30], [40, 40, 40], [50, 50, 50],
            [60, 60, 60], [70, 70, 70],
            # Shrinking models
            [40, 30, 20], [50, 40, 30], [60, 50, 40], [70, 60, 50], [70, 50, 30]]

    else:
        layers_list = [
            # One-layer models
            [20], [30], [40], [50], [60], [70],
            # Two-layer models
            [20, 20], [30, 30], [40, 40], [50, 50], [60, 60], [70, 70],
            # Three-layer models
            [20, 20, 20], [30, 30, 30], [40, 40, 40], [50, 50, 50],
            [60, 60, 60], [70, 70, 70]]

    return layers_list