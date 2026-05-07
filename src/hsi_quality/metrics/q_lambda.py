import numpy as np
from scipy.ndimage import convolve1d

from .metric import Metric


class QLambda(Metric):
    def __init__(self):
        super().__init__(name="Q-Lambda")

    def calculate(self, X, Y, size=11):
        # Convert to numpy arrays of shape (Height, Width, Channels) = (H, W, Q)
        X = np.asarray(X.values, dtype=np.float32)
        Y = np.asarray(Y.values, dtype=np.float32)

        H, W, Q = X.shape
        N = H * W

        # Flatten the spatial dimensions
        X = X.reshape(N, Q)
        Y = Y.reshape(N, Q)

        # Create uniform window
        kernel = np.ones(size) / size

        # Calculate local means
        muX = convolve1d(X, kernel, axis=1, mode="constant", cval=0.0)
        muY = convolve1d(Y, kernel, axis=1, mode="constant", cval=0.0)

        muX_sq = muX ** 2
        muY_sq = muY ** 2

        # Calculate variances and covariance
        varX = convolve1d(X * X, kernel, axis=1, mode="constant", cval=0.0) - muX_sq
        varY = convolve1d(Y * Y, kernel, axis=1, mode="constant", cval=0.0) - muY_sq
        covXY = convolve1d(X * Y, kernel, axis=1, mode="constant", cval=0.0) - muX * muY

        # Numerical instability can make some variances negative
        varX = np.maximum(varX, 0)
        varY = np.maximum(varY, 0)

        # Calculate SSIM across spectral dimension
        ssim_values = (4 * covXY * muX * muY) / ((varX + varY) * (muX_sq + muY_sq) + 1e-8)

        # Calculate SSIM score per pixel
        ssim_score = np.mean(ssim_values, axis=1)

        # Take the minimum over all pixels
        q_lambda = np.min(ssim_score)

        return q_lambda
    