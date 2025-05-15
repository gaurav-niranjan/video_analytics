import numpy as np
import cv2
from itertools import cycle
from collections import deque
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from pathlib import Path
from tqdm import tqdm
import os


def fit_pca(desc426_stack, n_comp=64):
    pca = PCA(n_components=n_comp, svd_solver='randomized', whiten=True)
    pca.fit(desc426_stack)              # train on all 426-D descriptors
    return pca

def project_pca(pca, desc426):
    return pca.transform(desc426).astype(np.float32)   # (N, 64)


def fit_gmm(desc64_stack, n_comp=5):
    gmm = GaussianMixture(n_components=n_comp,
                          covariance_type='diag',
                          max_iter=100, verbose=1)
    gmm.fit(desc64_stack)
    return gmm

def fisher_vector(desc64, gmm):
    """
    desc64 : (N_i , 64) array of one video (after pca)
    Returns : (2*K*64,) FV with signed square-root + L2 norm
    """
    Q = gmm.predict_proba(desc64)          # (N_i, K) posteriors
    N = desc64.shape[0]

    # Compute first-order (means)
    diff = desc64[:, None, :] - gmm.means_  # (N_i,K,64)
    sigma = np.sqrt(gmm.covariances_)       # diag σ
    diff /= sigma                           # normalize
    S1 = (Q[..., None] * diff).sum(axis=0) / np.sqrt(gmm.weights_)[:, None]

    # Second-order (variances)
    diff2 = (diff**2 - 1)
    S2 = (Q[..., None] * diff2).sum(axis=0) / np.sqrt(2*gmm.weights_)[:, None]

    fv = np.hstack((S1.flatten(), S2.flatten())).astype(np.float32)

    # power- & L2-norm
    fv = np.sign(fv) * np.sqrt(np.abs(fv))
    fv /= (np.linalg.norm(fv) + 1e-8)
    return fv

def video_fv(descriptors426, pca, gmm):
    desc64 = project_pca(pca, descriptors426)   # (N,64)
    return fisher_vector(desc64, gmm)           # (2*K*64,)

def train_svm(X_train, y_train, C=1.0):
    clf = LinearSVC(C=C)
    clf.fit(X_train, y_train)
    return clf

def evaluate(clf, X_test, y_test):
    pred = clf.predict(X_test)
    acc  = accuracy_score(y_test, pred)
    return acc
