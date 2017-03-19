import cv2
import scipy
import numpy as np
import os

def main():
    home = os.environ.get("HOME")

    # input/output paths
    fpath_inp = os.path.join(home, "datasets/cifar10/cifar-10.npz")
    fpath_out = os.path.join(home, "datasets/cifar10/zca_matrix.npz") # Z; ZCA whitening matrix saved

    # Pre setting
    X = np.load(fpath_inp)["train_x"]  # n x d matrix of cifar 10 training samples
    N = X.shape[0]
    D = X.shape[1]
    X = X / 255.
    x_mean = np.mean(X, axis=0)
    X -= x_mean
    
    # Eigen value decomposition
    C = np.dot(X.T, X) / N
    U, lam, V = np.linalg.svd(C)
    
    # ZCA Whitening
    eps = 1e-12
    sqlam = np.sqrt(lam + eps)
    Uzca = np.dot(U/sqlam[np.newais, :], U.T)
    Z = np.dot(X, Uzca.T)

    # Save
    np.save(fpath_out, {"Z": Z})


if __name__ == '__main__':
    main()
