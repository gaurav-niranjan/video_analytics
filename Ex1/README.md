# Video Analytics Assignment 1
## Gaurav Niranjan, Matr. no: 6599177

The assignment has been divided into 4 python files:
    1. dense_points.py: Samples dense points in the video frames and tracks the trajectory of those points. 
                        Finally returns the 30-dim descriptors of every in the video
    2. histogram_descriptors.py: Creates a 32x32x15 block following each trajectory's head. Then divides each of
                        these blocks into 2x2x3 tubes. For each tube, HoG, HoF, MBHx and MBHy features are created,
                        concatenated with the 30-dim trajectory descriptor and returned. 
    3. gmm_pca.py: Applies PCA on the features and reduces the feature dimensions to 64. We then fit a GMM
                        for all these 64-dim features. 5 Gaussian are used, as it was quick to fit and 
                        also gives good results. Fischer vectors are calculated for each video using the 64-dim 
                        features and the fitted GMM.
    4. main.py: Driver of all the code. Applies a linear SVM on the fischer vectors. Each fischer vector is of shape:
                        2x5x64 = 640

Simply run main.py Ensure there is a folder datasets in the same directory as the files.

- Ex1
    - dense_points.py
    - histogram_descriptors.py
    - gmm_pca.py
    - main.py
    - datasets
        - UCF-3
            - TrampolineJumping
            - UnevenBars
            - VolleyballSpiking
            - test.txt
            - train.txt