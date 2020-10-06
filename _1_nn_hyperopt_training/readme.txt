#  JointAnglePrediction.1_nn_hyperopt_training

Pathway to run neural network training - 
    *** Make sure to change import / output / input pathways before running ***
    11_optimize_hyperparams.py
    12_summarize_results.py
    13_get_best_results.py
    
Folders -
    utils\ - utilities for hyperparameter optimization and evaluating results
        hyperopt_utils.py - Functions for use in hyperopt parameter optimization
        eval_utils.py - Functions to evaluate training results

Files - 
    11_optimize_hyperparams.py - Train kinematic estimation neural network using hyperparameter optimization
    12_summarize_results.py - Compiles trained models by hyperoptimization into excel spreadsheet
    13_get_best_results.py - Retrieves best model from all trained hyperopt models