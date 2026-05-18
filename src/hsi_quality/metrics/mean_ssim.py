import numpy as np
from scipy.ndimage import uniform_filter

from .metric import FullReferenceMetric


class MeanSSIM(FullReferenceMetric):
    def __init__(self, params: dict = None):
        super().__init__(name="MeanSSIM", params=params)
        
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

        # Number of pixels in the window
        N = self.size * self.size

        # Compute the local means by sliding a window per channel
        muX = uniform_filter(X, size=(self.size, self.size, 1), mode="constant", cval=0.0)
        muY = uniform_filter(Y, size=(self.size, self.size, 1), mode="constant", cval=0.0)

        muX_sq = muX ** 2
        muY_sq = muY ** 2
        muX_muY = muX * muY

        # Compute the variances and covariance per channel
        varX = uniform_filter(X * X, size=(self.size, self.size, 1), mode="constant", cval=0.0) - muX_sq
        varY = uniform_filter(Y * Y, size=(self.size, self.size, 1), mode="constant", cval=0.0) - muY_sq
        covXY = uniform_filter(X * Y, size=(self.size, self.size, 1), mode="constant", cval=0.0) - muX_muY

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

        # Calculate the SSIM index per channel
        ssim = (l ** self.alpha) * (c ** self.beta) * (s ** self.gamma)

        # Average over channels
        mean_ssim = np.mean(ssim, axis=2)

        # Average over all mean SSIM values
        mean_ssim_score = np.mean(mean_ssim)

        return mean_ssim_score, None 