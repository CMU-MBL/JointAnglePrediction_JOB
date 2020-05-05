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
 # Clone the repository
  git clone https://github.com/CMU-MBL/JointAnglePrediction_JOB.git
  ```

  ## 2. Set up data folder structure
  ```bash
  cd JointAnglePrediction_JOB && mkdir Data
  mv <your extracted data folder> ./Data/
  ```
  ```
  # Desired data folder structure
  JointAnglePrediction_JOB | Data  | 1_Extracted        | walking/running_meta.h5
                                   | 2_Processed
                                   | 3_Hyperopt_Results
                                   | 4_Best_Results
                                   | 5_Optimization
  ```
  
  ## 3. Install requirement libraries
  ```bash
  pip install -r requirements.txt
  ```
  
  # How to run (Demo version):
  This demo version of the code allows you to run the framework using your own IMU data. The following steps are required.
  ## 1. Preparation
  IMU data (acceleration and angular velocity) from two segments and ground truth joint angle (optional), as well as the trained model of angle and orientation are needed. In the model folder, both ```model.pt``` and ```model_kwargs.pkl``` should exist.
  
  ## 2. Run the code
  ```bash
  python demo.py --joint <the type of joint ('knee', 'hip', 'ankle') \
                 --activity <the type of activity ('walking', 'running') \
                 --seg1-accel-path <path of segment 1 acceleration data> \
                 --seg2-accel-path <path of segment 2 acceleration data> \
                 --seg1-gyro-path <path of segment 1 angular velocity data> \
                 --seg2-gyro-path <path of segment 2 angular velocity data> \
                 --angle-model-fldr <folder path of angle prediction model> \
                 --ori-model-fldr <folder path of orientation prediction model> \
                 --result-fldr <folder to save the result files> \
                 --use-cuda <cuda configuration (True, False)> \
                 --gt-angle-path <path of ground truth angle data (optional)>
  ```
  
  # How to run (Entire Framework):
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
