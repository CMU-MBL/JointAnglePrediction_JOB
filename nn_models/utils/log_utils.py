# -------------------------
#
# Functions to log training progress
#
# --------------------------

import json
import json2html
import logging
import numpy as np
import os
import torch
from collections import OrderedDict

formatter = logging.Formatter('%(message)s')


class TrainLogger():
    def __init__(self, result_path, exp_name, log_metrics, log_freq, check_freq):
        # Assign logging parameters and objects
        self.acc_time = 0
        self.idx = 0
        self.exp_name = exp_name
        self.log_freq = log_freq
        self.check_freq = check_freq

        # Specify folder structure and create it
        self.exp_path = result_path + exp_name

        self.log_path = self.exp_path + '/logs'
        self.check_path = self.exp_path + '/checkpoints'
        self.pred_path = self.exp_path + '/predictions'
        self.create_exp_folder(self.exp_name)
        # Create loggers for requested metrics
        if 'loss' in log_metrics:
            self.log_loss = True
            self.loss_logger = self.create_logger('loss')
        if 'lr' in log_metrics:
            self.log_lr = True
            self.lr_logger = self.create_logger('lr')
        if 'rmse' in log_metrics:
            self.log_rmse = True
            self.rmse_logger = self.create_logger('rmse')

    def create_logger(self, metric, level=logging.INFO):
        # Infer logfile from name
        log_file = self.log_path + '/' + metric + '_log.csv'
        # Define filehandler for the logfile
        handler = logging.FileHandler(log_file, mode='w')
        handler.setFormatter(formatter)
        # Set up logger
        logger = logging.getLogger(self.exp_name + '_' + metric)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

    def set_logging_objects(self, model, scheduler, criterion):
        self.model = model
        self.scheduler = scheduler
        self.criterion = criterion

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def step(self, loss, val_loss, x_val, y_val_pred, y_val):
        # Log errors along training iterations

        # Check if data needs to be logged currently
        if (self.idx % self.log_freq) == 0:
            loss = loss.clone().detach().cpu().item()
            val_loss = val_loss.clone().detach().cpu().item()
            # Log train and val loss
            if self.log_loss:
                log_list = [self.idx, loss, val_loss]
                log_str = ','.join(map(str, log_list))
                self.loss_logger.info(log_str)

            # Log learning rates of optimizer
            if self.log_lr:
                log_list = [self.idx] + [pg['lr'] for pg in
                                         self.scheduler.optimizer.param_groups]
                log_str = ','.join(map(str, log_list))
                self.lr_logger.info(log_str)

            # Log feature-wise and overall RMSE
            if self.log_rmse:
                _, y_val = self.normalizer.revert(x_val, y_val)
                _, y_val_pred = self.normalizer.revert(x_val, y_val_pred)
                sample_sq_err = (y_val_pred - y_val)**2
                sample_sq_err = sample_sq_err.detach().cpu().numpy()
                sample_rmse = np.sqrt(sample_sq_err)
                feature_rmse = np.mean(sample_rmse, axis=(0, 1))
                av_rmse = np.mean(feature_rmse)
                log_list = ([self.idx] + [av_rmse] +
                            feature_rmse.tolist())
                log_str = ','.join(map(str, log_list))
                self.rmse_logger.info(log_str)

        # Create checkpoint
        if (self.idx % self.check_freq) == 0:
            check_name = self.check_path+'/i'+str(self.idx)+'.pt'
            check_dict = {'model': self.model.state_dict(),
                          'scheduler': self.scheduler.state_dict(),
                          'optimizer': self.scheduler.optimizer.state_dict()}
            torch.save(check_dict, check_name)

        self.idx += 1

    def log_end_of_stage(self, x_val, y_val, y_pred_val, x_test, y_test, y_pred_test):
        # Save final model after training stage
        torch.save(self.model.state_dict(), self.exp_path+'/model.pt')
        # Rescale data to physical scale
        # True validation data
        _, tmp_val = self.normalizer.revert(x_val, y_val)
        tmp_val = tmp_val.detach().cpu().numpy()
        # True testing data
        _, tmp_test = self.normalizer.revert(x_test, y_test)
        tmp_test = tmp_test.detach().cpu().numpy()
        # Predicted validation data
        tmp_x_val, tmp_pred_val = self.normalizer.revert(x_val, y_pred_val)
        tmp_x_val = tmp_x_val.detach().cpu().numpy()
        tmp_pred_val = tmp_pred_val.detach().cpu().numpy()
        # Predicted testing data
        tmp_x_test, tmp_pred_test = self.normalizer.revert(x_test, y_pred_test)
        tmp_x_test = tmp_x_test.detach().cpu().numpy()
        tmp_pred_test = tmp_pred_test.detach().cpu().numpy()

        # Save numpy arrays
        np.save(self.pred_path+'/x_val.npy', tmp_x_val)
        np.save(self.pred_path+'/y_val.npy', tmp_val)
        np.save(self.pred_path+'/y_pred_val.npy', tmp_pred_val)
        np.save(self.pred_path+'/x_test.npy', tmp_x_test)
        np.save(self.pred_path+'/y_test.npy', tmp_test)
        np.save(self.pred_path+'/y_pred_test.npy', tmp_pred_test)

    def create_exp_folder(self, exp_name):
        msg = 'Error! Experiment folder exists already!'
        assert not os.path.exists(self.exp_path), msg

        os.makedirs(self.exp_path)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if not os.path.exists(self.check_path):
            os.makedirs(self.check_path)

        if not os.path.exists(self.pred_path):
            os.makedirs(self.pred_path)

    def stop_logging(self):
        logging.shutdown()


def write_info(result_path, exp_name, general_spec, stage_spec, model, scheduler):
    # Writes model information to result path

    info_path = result_path + '/' + exp_name + '/info'

    exp_info = OrderedDict()
    exp_info['Experiment name'] = exp_name
    tmp = vars(model)['_modules']
    tmp = OrderedDict((key, repr(tmp[key])) for key in tmp)
    exp_info['Model '] = tmp
    tmp = vars(scheduler.optimizer)['defaults']
    tmp = OrderedDict((key, tmp[key]) for key in tmp.keys())
    tmp['Type '] = general_spec['optim']
    tmp.move_to_end('Type ', last=False)
    exp_info['Optimizer '] = tmp
    tmp = OrderedDict()
    tmp['Type '] = general_spec['scheduler']
    tmp['Gamma '] = general_spec['lr_decay']
    exp_info['LR schedule '] = tmp

    train_info = OrderedDict()
    for i in range(0, len(stage_spec)):
        tmp = OrderedDict()
        tmp['Number of iterations '] = stage_spec[i]['num_iter']
        tmp['Batch size '] = stage_spec[i]['batch_size']
        tmp['Sequence length '] = stage_spec[i]['seq_len']
        train_info['Stage '+str(i)] = tmp
        del(tmp)
    exp_info['Training information '] = train_info
    with open(info_path+'.json', 'w') as f:
        json.dump(exp_info, f, indent=4)  # sort_keys=True,

    formatted_table = json2html.json2html.convert(json=exp_info)
    with open(info_path+'.html', "w") as f:
        f.write(formatted_table)
        f.close()
