import torch
import torch.nn.functional as F

def calculate_mean_ssim(X, Y, size=11, alpha=1, beta=1, gamma=1):
    C1 = 0.01**2
    C2 = 0.03**2
    C3 = C2 / 2

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Convert to torch tensors
    X = torch.from_numpy(X.values).float().to(device)
    Y = torch.from_numpy(Y.values).float().to(device)

    # Reshape to (1, Q, H, W)
    X = X.permute(2, 0, 1).unsqueeze(0)
    Y = Y.permute(2, 0, 1).unsqueeze(0)

    Q = X.shape[1]

    # Create uniform window
    window = torch.ones((Q, 1, size, size)) / (size * size)
    window = window.to(device)

    # Compute the local means by sliding a window per band
    muX = F.conv2d(X, window, padding="same", groups=Q)
    muY = F.conv2d(Y, window, padding="same", groups=Q)

    muX_sq = muX ** 2
    muY_sq = muY ** 2
    muX_muY = muX * muY

    # Compute the variances and covariance per band
    sigmaX = F.conv2d(X * X, window, padding="same", groups=Q) - muX_sq
    sigmaY = F.conv2d(Y * Y, window, padding="same", groups=Q) - muY_sq
    sigmaXY = F.conv2d(X * Y, window, padding="same", groups=Q) - muX_muY

    # Numerical instability makes some variances negative
    sigmaX = torch.clamp(sigmaX, min=0.0)
    sigmaY = torch.clamp(sigmaY, min=0.0)

    # Luminance
    l = (2 * muX_muY + C1) / (muX_sq + muY_sq + C1)
    
    # Contrast
    c = (2 * torch.sqrt(sigmaX * sigmaY) + C2) / (sigmaX + sigmaY + C2)

    # Structure
    s = (sigmaXY + C3) / (torch.sqrt(sigmaX * sigmaY) + C3)

    # Calculate the SSIM index per band
    ssim = (l ** alpha) * (c ** beta) * (s ** gamma)

    # Average over bands
    mean_ssim = ssim.mean(dim=1)

    # Average over all mean SSIM values
    mean_ssim_score = mean_ssim.mean()

    return mean_ssim_score.cpu().numpy()