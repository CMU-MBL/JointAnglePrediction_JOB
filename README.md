# JointAnglePrediction_JOB

## Pathway to run framework - 
    00_matlab_to_python\
    0_preprocessing\
    1_nn_hyperopt_training\

## Folders (each folder contains its own readme.txt file) -
    00_matlab_to_python\ - Scripts for converting .mat file in .h5 file
    0_preprocessing\ - Scripts for preprocessing dataset for deep learning
    1_nn_hyperopt_training\ - Scripts for training neural network and optimizaing hyperparameters using hyperopt
    biomech_model\ - Scripts for simulating biomechanical features
    nn_models\ - Scripts for creating and training neural network models
    
 # Installation:
 ## 1. Download source code via git clone
 ```bash
  git clone https://github.com/CMU-MBL/JointAnglePrediction_JOB.git
  ```

  ## 2. Set up data folder structure
  ```bash
  cd JointAnglePrediction_JOB && mkdir Data
  mv <your extracted data folder> ./Data/
  ```
  Thus, the we hope the folder structure as:
  ```
  JointAnglePrediction_JOB -- Data                    -- 1_Extracted      -- walking/running_meta.h5
                           -- 0_preprocessing
                           -- 1_nn_hyperopt_training
                           ...
  ```
  
  ## 3. Install requirement libraries
  ```bash
  pip install -r requirements.txt
  ```
  
  # How to run:
  ## 1. Data preprocessing
  ```bash
  # Purpose of the file
  python 0_preprocessing/00_hfile_check.py
  ```
  
  ```bash
  # Purpose of the file
  python 0_preprocessing/01_preproc_dataset.py
  ```
  
  ```bash
  # Check unusual features in dataset. 
  # Refer to Calgary_issue_report.pdf for examples of checks
  python 0_preprocessing/02_check_dataset.py
  ```
  
  ## 2. Get best neural network model
  ```bash
  # Run hyper-parameters optimization script
  python 1_nn_hyperopt_training/11_optimize_hyperparams.py
  ```
  
  ```bash
  # Summarize hyper-parameters optimization results
  python 1_nn_hyperopt_training/12_summarize_results.py
  ```
  
  ```bash
  # Compare the performance and get the best performing model configuration
  python 1_nn_hyperopt_training/13_get_best_results.py
  ```
  
  ## 3. Run optimization
  ```bash
  # Predict orientation value from neural network
  python 2_optimization/21_predict_orientation.py
  ```
  
  ```bash
  # Run top-down optimization which minimizes reconstruction error of angular velocity data
  python 2_optimization/22_run_optimization.py
  ```
