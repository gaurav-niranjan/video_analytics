import numpy as np
from itertools import cycle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path
from tqdm import tqdm
import os

from histogram_descriptors import compute_desc426
from gmm_pca import fit_pca, fit_gmm



CACHE_FV = Path("fv_cache")      # one .npy per video
CACHE_FV.mkdir(exist_ok=True)

def read_split(txt):
    vids, labels = [], []
    for line in Path(txt).read_text().splitlines():
        fname, lab = line.strip().split()
        vids.append(os.path.join(os.getcwd(),f'datasets/UCF-3/{fname}'))
        labels.append(int(lab))
    return vids, np.array(labels, np.int32)

def fv_for_video(vid):
    fv_file = CACHE_FV / (Path(vid).stem + ".npy")
    if fv_file.exists():
        return np.load(fv_file)
    # else compute from scratch
    desc426 = compute_desc426(vid)
    fv = video_fv(desc426, pca, gmm)           # returns (FVdim,)
    np.save(fv_file, fv)
    return fv

def main():


    print("Loading train / test file lists …")
    dataset_dir = os.path.join(os.getcwd(), 'datasets/UCF-3')
    print(dataset_dir)
    train_vids, y_train = read_split(os.path.join(dataset_dir, 'train.txt'))
    test_vids,  y_test  = read_split(os.path.join(dataset_dir, 'test.txt'))

    print("Gathering 426-D descriptors to fit PCA …")
    all_desc = []
    for vid in tqdm(train_vids):
        desc426 = compute_desc426(vid)         # Task-2 output
        all_desc.append(desc426)
    all_desc = np.vstack(all_desc)                 # shape (M, 426)

    print("Fitting 64-D PCA on", all_desc.shape[0], "train descriptors …")
    pca = fit_pca(all_desc, n_comp=64)
    desc64 = pca.transform(all_desc)

    print("Fitting GMM …")
    gmm = fit_gmm(desc64, n_comp=5)

    print("Building Fisher vectors …")
    X_train = np.vstack([fv_for_video(v) for v in tqdm(train_vids)])
    X_test  = np.vstack([fv_for_video(v) for v in tqdm(test_vids)])

    print("Feature dims:", X_train.shape, X_test.shape)

    clf = LinearSVC(C=1.0, dual=False)             # dual=False for high-dim data
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc*100:.2f}%")
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)


if __name__ == '__main__':
    main()


