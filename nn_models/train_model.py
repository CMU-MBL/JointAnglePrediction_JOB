# -------------------------
#
# Initiates deep neural network model training given general specifications and model specifications
#  using custom 2 stage scheme
#
# --------------------------

import numpy as np
import torch
import pickle
from math import pi
from torch.nn import *  # noqa
from torch.optim import *  # noqa
from torch.optim.lr_scheduler import *  # noqa
from nn_models.utils import prep_utils, data_utils, dloader_utils  # noqa
from nn_models.utils import log_utils, train_utils  # noqa
from nn_models.models.pure_conv import *  # noqa
from nn_models.models.pure_lstm import *  # noqa


def main(general_spec, model_spec, show_bar=True):
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parse arguments of the training script
    scheduler_spec = prep_utils.parse_scheduler(general_spec)
    model_type = general_spec['model_type']
    stage_spec = prep_utils.parse_stages(general_spec)
    h5path = general_spec['data_path']

    # Create logger and specify what to log
    log_spec = [general_spec['result_path'], general_spec['name'], general_spec['log_metrics'],
                general_spec['log_freq'], general_spec['check_freq']]
    logger = log_utils.TrainLogger(*log_spec)

    # Get device and parse model, optimizer etc
    device = prep_utils.get_device(show_bar)
    model = globals()[model_type](**model_spec).to(device)
    optimizer = globals()[general_spec['optim']](model.parameters(), lr=general_spec['lr'])
    scheduler = globals()[scheduler_spec['type']](optimizer, **scheduler_spec['args'])
    criterion = globals()[general_spec['loss']]()

    if show_bar:
        # Print model info
        print(repr(model))
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total number of trainable parameters: {}'.format(n_params))

    # Add model scheduler and loss to the logger
    logger.set_logging_objects(model, scheduler, criterion)
    log_utils.write_info(general_spec['result_path'], general_spec['name'],
                         general_spec, stage_spec, model, scheduler)

    # Pickle dictionary with the model type and parameters
    model_kwargs_dict = model_spec
    model_kwargs_dict['model_type'] = model_type
    with open(general_spec['result_path'] + general_spec['name'] + '/model_kwargs.pkl', 'wb') as f:
        pickle.dump(model_kwargs_dict, f)

    # Split up subjects into 80% training, 10% validation, 10% test
    train_ids, val_ids, test_ids = prep_utils.get_subject_split(h5path, [0.8, 0.1, 0.1])

    # Load data
    train_dict = prep_utils.get_sub_dict(h5path, train_ids, general_spec['inp'],
                                         general_spec['outp'], model_spec['prediction'], device)
    val_dict = prep_utils.get_sub_dict(h5path, val_ids, general_spec['inp'],
                                       general_spec['outp'], model_spec['prediction'], device)
    test_dict = prep_utils.get_sub_dict(h5path, test_ids, general_spec['inp'],
                                        general_spec['outp'], model_spec['prediction'], device)

    # Create datasets
    train_dset = dloader_utils.SubjectDataset(train_dict, None, test=False)
    val_dset = dloader_utils.SubjectDataset(val_dict, None, test=True)
    test_dset = dloader_utils.SubjectDataset(test_dict, None, test=True)

    # Get and save normalizer parameters
    normalizer = data_utils.Normalizer(train_dset)
    normalizer.save_norm_dict(logger.exp_path+'/')

    # Add normalizer to datasets
    train_dset.add_normalizer(normalizer)
    val_dset.add_normalizer(normalizer)
    test_dset.add_normalizer(normalizer)

    # Set up data augmentor by adding rotational and white noise
    augmentor = data_utils.Augmentor(general_spec['rot_type'], device)
    augmentor.set_rot_noise(rot_spread=general_spec['rot_spread']*pi)
    augmentor.set_white_noise(normalizer.params['x_std']*general_spec['x_noise'],
                              normalizer.params['y_std']*general_spec['y_noise'])

    # Add augmentor to training dataset
    if general_spec['aug']:
        train_dset.add_augmentor(augmentor)

    # Add normalizer to validation dataset (-> Normalize) and logger (-> De-Normalize)
    val_dset.add_normalizer(normalizer)
    logger.set_normalizer(normalizer)

    # Create validation and test dataloaders
    val_dloader = dloader_utils.CustomDataLoader(val_dset, batch_size=len(val_dset), shuffle=False)
    test_dloader = dloader_utils.CustomDataLoader(test_dset, batch_size=len(test_dset),
                                                  shuffle=False)

    # Iterate through training stages
    for i in range(0, len(stage_spec)):

        # Shows progress bar for training over iterations
        if show_bar:
            print('Starting stage {0}/{1}'.format(str(i+1), len(stage_spec)))

        # Set specifications for particular stage
        stage = stage_spec[i]

        # Set sequence length of the training dataloader
        train_dset.seq_len = stage['seq_len']

        # Get train dataloader
        train_dloader = dloader_utils.CustomDataLoader(train_dset,
                                                       batch_size=stage['batch_size'], shuffle=True)

        # Run current training stage
        model, scheduler, logger = train_utils.run_stage(model, scheduler, criterion,
                                                         train_dloader, val_dloader, test_dloader,
                                                         logger, stage, show_bar)

        if show_bar:
            print('Stage finished! \n')
    logger.stop_logging()


if __name__ == '__main__':
    pass
