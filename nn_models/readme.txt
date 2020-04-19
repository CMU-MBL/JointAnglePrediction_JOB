#  JointAnglePrediction.nn_models

Folders - 
    models\ - contains custom model frameworks
        pure_conv.py - Create custom 1-Dimensional Convolution Model
        pure_lstm.py - Create custom Long Short-term Memory model
    utils\ - utilities for training models
        data_utils.py - Classes for normalizing and augmenting datasets
        dloader_utils.py - Classes for organizing different versions of datasets
        log_utils.py - Functions to log training progress
        prep_utils.py - Functions to prepare for model training
        train_utils.py - Functions to run model training

Files - 
    train_model.py - Initiates deep neural network model training given general specifications and model specifications using custom 2 stage scheme