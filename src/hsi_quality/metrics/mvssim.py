import numpy as np
import xarray as xr
from tqdm import tqdm

def calculate_mvssim_score(X: xr.DataArray, Y: xr.DataArray, size: int = 11):
    C1 = 1e-8
    C2 = 1e-8
    C3 = 1e-8

    alpha = 1
    beta = 1
    gamma = 1

    Q = X.shape[2]
    N = size * size

    X_N = X.reshape(-1, Q)
    Y_N = Y.reshape(-1, Q)

    # Calculate the local sample means with 2D convolution
    mu_X = np.mean(X_N, axis=0)
    mu_Y = np.mean(Y_N, axis=0)

    mu_X_sq = mu_X ** 2
    mu_Y_sq = mu_Y ** 2
    mu_X_mu_Y = mu_X * mu_Y

    # Calculate the local sample covariance and cross-covariance
    Sigma_X = np.cov(X_N, rowvar=False)
    Sigma_Y = np.cov(Y_N, rowvar=False)
    Sigma_XY = np.cov(X_N, Y_N, rowvar=False)[:Q, Q:]

    sigma_X = np.diag(Sigma_X)
    sigma_Y = np.diag(Sigma_Y)
    sigma_XY = np.diag(Sigma_XY)

    # Calculate singular values
    lambda_q = np.linalg.svdvals(Sigma_X)
    lambda_s = np.sum(lambda_q)

    d_q = np.linalg.svdvals(Sigma_Y)
    d_s = np.sum(d_q)

    # Luminance similarity between X and Y
    l = (2 * np.sum(mu_X_mu_Y, axis=0) + C1) / (np.sum(mu_X_sq, axis=0) + np.sum(mu_Y_sq, axis=0) + C1)

    # Contrast similarity between X and Y
    c = (2 * np.sqrt(lambda_s) * np.sqrt(d_s) + C2) / (lambda_s + d_s + C2)

    # Spatial structural similarity between X and Y
    s = np.mean((sigma_XY + C3) / (np.sqrt(sigma_X * sigma_Y) + C3))

    mvssim_score = l**alpha * c**beta * s**gamma

    return mvssim_score

def calculate_mvssim(X: xr.DataArray, Y: xr.DataArray, size: int = 11):
    X = X.values
    Y = Y.values

    H, W, Q = X.shape
    
    output_h = H - size + 1
    output_w = W - size + 1

    output = np.zeros((output_h, output_w))
    
    for i in tqdm(range(output_h)):
        for j in range(output_w):
            X_patch = X[i:i+size, j:j+size, :]
            Y_patch = Y[i:i+size, j:j+size, :]

            mvssim_score = calculate_mvssim_score(X_patch, Y_patch, size)
            output[i, j] = mvssim_score

    return output