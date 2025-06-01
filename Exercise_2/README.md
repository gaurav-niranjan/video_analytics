# Video Analytics Assignment 2
## Gaurav Niranjan, Matr. no: 6599177

The assignment has been divided into 2 python folder:

    1. TSN: Task 1 of the assignment with TSN on RGB and Optical flow files
    2. 3DResNet: Task 2 with implemntation of 3D equivalent model of the 2D ResNet

data.zip is to be extracted within this directory and all data files should be inside the data folder.
First, extract optical flow files. Then run TSN/extract_frames.py to extract all frames from all videos.



- Exercie_2
    - TSN
        - runs -> Contains tensorboard records and saved models used in late fusion
        - dataset.py
        - extract_frames.py
        - late_fusion.py -> Late fusion on trained models
        - main_flow.py  -> TSN on Optical Flow
        - main.py -> TSN on RGB frames
        - model.py
    - 3DResNet
        - dataset.py
        - inflation.py
        - main.py
        - model.py
    - data
        - frames
        - mini_UCF
        - mini_UCF_flow
        - classes.txt
        - train.txt
        - validation.txt