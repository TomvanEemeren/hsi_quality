import numpy as np
from scipy.ndimage import convolve1d

from .metric import FullReferenceMetric


class SSIMLambda(FullReferenceMetric):
    def __init__(self, params: dict = None):
        super().__init__(name="SSIMLambda", params=params)
        
        self.size = self.params.get("size", 11)
        self.alpha = self.params.get("alpha", 1)
        self.beta = self.params.get("beta", 1)
        self.gamma = self.params.get("gamma", 1)
        self.C1 = self.params.get("C1", 0.01**2)
        self.C2 = self.params.get("C2", 0.03**2)
        self.C3 = self.params.get("C3", 0.03**2 / 2)

    def calculate(self, X, Y):
        # Convert to numpy arrays of shape (Height, Width, Channels) = (H, W, Q)
        X = np.asarray(X.values, dtype=np.float32)
        Y = np.asarray(Y.values, dtype=np.float32)

        H, W, Q = X.shape
        N = H * W

        # Flatten the spatial dimensions
        X = X.reshape(N, Q)
        Y = Y.reshape(N, Q)

        # Create uniform window
        kernel = np.ones(self.size) / self.size

        # Calculate local means
        muX = convolve1d(X, kernel, axis=1, mode="constant", cval=0.0)
        muY = convolve1d(Y, kernel, axis=1, mode="constant", cval=0.0)

        muX_sq = muX ** 2
        muY_sq = muY ** 2
        muX_muY = muX * muY

        # Calculate variances and covariance
        varX = convolve1d(X * X, kernel, axis=1, mode="constant", cval=0.0) - muX_sq
        varY = convolve1d(Y * Y, kernel, axis=1, mode="constant", cval=0.0) - muY_sq
        covXY = convolve1d(X * Y, kernel, axis=1, mode="constant", cval=0.0) - muX * muY

        # Numerical instability can make some variances negative
        varX = np.maximum(varX, 0)
        varY = np.maximum(varY, 0)

        # Multiply by factor to get sample covariance
        factor = N / (N - 1)
        varX *= factor
        varY *= factor
        covXY *= factor

        # Luminance
        l = (2 * muX_muY + self.C1) / (muX_sq + muY_sq + self.C1)
        
        # Contrast
        c = (2 * np.sqrt(varX) * np.sqrt(varY) + self.C2) / (varX + varY + self.C2)

        # Structure
        s = (covXY + self.C3) / (np.sqrt(varX) * np.sqrt(varY) + self.C3)

        # Calculate the SSIM index
        ssim_values = (l ** self.alpha) * (c ** self.beta) * (s ** self.gamma)

        # Calculate SSIM score per pixel
        ssim_scores = np.mean(ssim_values, axis=1)

        # Take the average over all pixels
        ssim_lambda = np.mean(ssim_scores)

        return ssim_lambda, None
    