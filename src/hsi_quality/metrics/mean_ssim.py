import numpy as np
from scipy.ndimage import uniform_filter

from .metric import Metric


class MeanSSIM(Metric):
    def __init__(self):
        super().__init__(name="MeanSSIM")
        self.C1 = 0.01**2
        self.C2 = 0.03**2
        self.C3 = self.C2 / 2

    def calculate(self, X, Y, size=11, alpha=1, beta=1, gamma=1):
        # Convert to numpy arrays of shape (Height, Width, Channels) = (H, W, Q)
        X = np.asarray(X.values, dtype=np.float32)
        Y = np.asarray(Y.values, dtype=np.float32)

        # Number of pixels in the window
        N = size * size

        # Compute the local means by sliding a window per channel
        muX = uniform_filter(X, size=(size, size, 1), mode="constant", cval=0.0)
        muY = uniform_filter(Y, size=(size, size, 1), mode="constant", cval=0.0)

        muX_sq = muX ** 2
        muY_sq = muY ** 2
        muX_muY = muX * muY

        # Compute the variances and covariance per channel
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

        # Luminance
        l = (2 * muX_muY + self.C1) / (muX_sq + muY_sq + self.C1)
        
        # Contrast
        c = (2 * np.sqrt(varX) * np.sqrt(varY) + self.C2) / (varX + varY + self.C2)

        # Structure
        s = (covXY + self.C3) / (np.sqrt(varX) * np.sqrt(varY) + self.C3)

        # Calculate the SSIM index per channel
        ssim = (l ** alpha) * (c ** beta) * (s ** gamma)

        # Average over channels
        mean_ssim = np.mean(ssim, axis=2)

        # Average over all mean SSIM values
        mean_ssim_score = np.mean(mean_ssim)

        return mean_ssim_score