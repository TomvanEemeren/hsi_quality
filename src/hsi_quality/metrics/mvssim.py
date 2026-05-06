import numpy as np
from scipy.ndimage import uniform_filter

def calculate_mvssim(X, Y, size=11, alpha=1, beta=1, gamma=1):
    C1 = 0.01**2
    C2 = 0.03**2
    C3 = C2 / 2

    # Convert to numpy arrays of shape (Height, Width, Channels) = (H, W, Q)
    X = np.asarray(X.values, dtype=np.float32)
    Y = np.asarray(Y.values, dtype=np.float32)

    # Number of pixels in the window
    N = size * size

    # Compute the local sample means by sliding a window with zero padding
    muX = uniform_filter(X, size=(size, size, 1), mode="constant", cval=0.0)
    muY = uniform_filter(Y, size=(size, size, 1), mode="constant", cval=0.0)

    muX_sq = muX ** 2
    muY_sq = muY ** 2
    muX_muY = muX * muY

    # Calculate diagonal elements of sample covariance matrix
    varX = uniform_filter(X * X, size=(size, size, 1), mode="constant", cval=0.0) - muX_sq
    varY = uniform_filter(Y * Y, size=(size, size, 1), mode="constant", cval=0.0) - muY_sq
    covXY = uniform_filter(X * Y, size=(size, size, 1), mode="constant", cval=0.0) - muX_muY

    # Numerical instability can make some variances negative
    varX = np.maximum(varX, 0)
    varY = np.maximum(varY, 0)

    # Multiply by factor to get sample covariance
    factor = N / (N - 1)
    varX *= factor
    varY *= factor
    covXY *= factor

    # Nuclear norm can be approximated by trace
    lambda_s = np.sum(varX, axis=2)  # sum over channels
    d_s = np.sum(varY, axis=2)

    # Luminance similarity between X and Y
    l = (2 * np.sum(muX_muY, axis=2) + C1) / (np.sum(muX_sq, axis=2) + np.sum(muY_sq, axis=2) + C1)

    # Contrast similarity between X and Y
    c = (2 * np.sqrt(lambda_s * d_s) + C2) / (lambda_s + d_s + C2)

    # Spatial structure similarity between X and Y
    s = ((covXY + C3) / (np.sqrt(varX * varY) + C3)).mean(axis=2)

    # Calculate the MvSSIM index per pixel
    mvssim_values = (l ** alpha) * (c ** beta) * (s ** gamma)

    # Average over all MvSSIM values
    mvssim_score = np.mean(mvssim_values)

    return mvssim_score