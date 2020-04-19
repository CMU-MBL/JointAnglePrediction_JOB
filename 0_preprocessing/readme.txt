#  JointAnglePrediction.0_preprocessing

Pathway to run preprocessing - 
    *** Make sure to change import / output / input pathways before running ***
    01_preproc_dataset.py
    02_check_dataset.py
    
Folders -
    utils\ - utilities for preprocessing and checking dataset
        preproc_utils.py - Functions to preprocess 3D motion capture data into inertial data
        check_utils.py - Functions to check angles and calculated inertial data

Files - 
    00_hfile_check.py - Functions to visualize hfile structure and subject characteristics (not necessary for preprocessing)
    01_preproc_dataset.py - Functions to visualize sequence lengths of data and creates proccessed h5 file
    02_check_dataset.py - Checks simulated data and adds whether checks passed to process data
    Calgary_issue_report.pdf - Documents typical issues experienced in simulated inertial data
