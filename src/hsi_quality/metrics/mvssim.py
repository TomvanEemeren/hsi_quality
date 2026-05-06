import torch
import torch.nn.functional as F

def calculate_mvssim(X, Y, size=11, alpha=1, beta=1, gamma=1):
    C1 = 0.01**2
    C2 = 0.03**2
    C3 = C2 / 2

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Convert to torch tensors
    X = torch.from_numpy(X.values).float().to(device)
    Y = torch.from_numpy(Y.values).float().to(device)

    # Reshape to (batch, channels, height, width) = (1, Q, H, W)
    X = X.permute(2, 0, 1).unsqueeze(0)
    Y = Y.permute(2, 0, 1).unsqueeze(0)

    Q = X.shape[1]
    N = size * size

    # Create uniform window
    window = torch.ones((Q, 1, size, size)) / (size * size)
    window = window.to(device)

    # Compute the local sample means by sliding a window
    muX = F.conv2d(X, window, padding="same", groups=Q)
    muY = F.conv2d(Y, window, padding="same", groups=Q)

    muX_sq = muX ** 2
    muY_sq = muY ** 2
    muX_muY = muX * muY

    # Calculate diagonal elements of sample covariance matrix
    varX = F.conv2d(X * X, window, padding="same", groups=Q) - muX_sq
    varY = F.conv2d(Y * Y, window, padding="same", groups=Q) - muY_sq
    covXY = F.conv2d(X * Y, window, padding="same", groups=Q) - muX_muY

    varX = torch.clamp(varX, min=0)
    varY = torch.clamp(varY, min=0)

    varX = varX * (N / (N - 1))
    varY = varY * (N / (N - 1))
    covXY = covXY * (N / (N - 1))

    # Nuclear norm can be approximated by trace
    lambda_s = varX.sum(dim=1)
    d_s = varY.sum(dim=1)

    # Luminance similarity between X and Y
    l = (2 * muX_muY.sum(dim=1) + C1) / (muX_sq.sum(dim=1) + muY_sq.sum(dim=1) + C1)

    # Contrast similarity between X and Y
    c = (2 * torch.sqrt(lambda_s * d_s) + C2) / (lambda_s + d_s + C2)

    # Spatial structure similarity between X and Y
    s = ((covXY + C3) / (torch.sqrt(varX * varY) + C3)).mean(dim=1)

    mvssim_values = (l ** alpha) * (c ** beta) * (s ** gamma)

    mvssim_score = mvssim_values.mean()

    return mvssim_score.cpu().numpy()