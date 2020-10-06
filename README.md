# JointAnglePrediction_JOB
​
 # Installation:
 ## 1. Download source code via git clone
 ```bash
 # Clone the repository
  git clone https://github.com/CMU-MBL/JointAnglePrediction_JOB.git
  ```
​
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
  # Given marker cluster data, this piece of code will create coordinate systems and generate simulated inertial data.
  python 0_preprocessing/01_preproc_dataset.py
  ```
  
  ```bash
  # Given processed dataset of inertial data, this script checks for unusual features in dataset and excludes those subjects. 
  # Refer to Calgary_issue_report.pdf for examples of checks
  python 0_preprocessing/02_check_dataset.py
  ```
  
  ## 2. Get best neural network model
  ```bash
  # Given processed and checked dataset, this script trains both CNNs and LSTMs utilizing hyperparameter optimization to predict joint kinematics.
  # Hyperopt sweeps over given sets of parameters, and each evaluation tries a different combination of those parameters.
  python 1_nn_hyperopt_training/11_optimize_hyperparams.py
  ```
  
  ```bash
  # This script compiles the model results from optimizing the hyperparameters and outputs an Excel file to compare the different performances.
  python 1_nn_hyperopt_training/12_summarize_results.py
  ```
  
  ```bash
  # This script compares the performances of the model results and saves the best performing model configuration in a separate directory for use in the framework.
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
