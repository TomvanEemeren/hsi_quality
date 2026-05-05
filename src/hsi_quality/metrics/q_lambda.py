import torch
import torch.nn.functional as F

def calculate_q_lambda(X, Y, size=11):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.from_numpy(X.values).float().to(device)
    Y = torch.from_numpy(Y.values).float().to(device)

    H, W, Q = X.shape
    N = H * W

    # Input of shape (mini_batch, channels, length) = (N, 1, Q)
    X = X.reshape(N, Q).unsqueeze(1)  
    Y = Y.reshape(N, Q).unsqueeze(1)
    
    kernel = torch.ones((1, 1, size), device=device) / size

    muX = F.conv1d(X, kernel, padding="same")
    muY = F.conv1d(Y, kernel, padding="same")

    muX_sq = muX ** 2
    muY_sq = muY ** 2

    varX = F.conv1d(X * X, kernel, padding="same") - muX_sq
    varY = F.conv1d(Y * Y, kernel, padding="same") - muY_sq
    covXY = F.conv1d(X * Y, kernel, padding="same") - muX * muY

    varX = torch.clamp(varX, min=0.0)
    varY = torch.clamp(varY, min=0.0)

    ssim = (4 * covXY * muX * muY) / ((varX + varY) * (muX_sq + muY_sq) + 1e-8)

    ssim_score = ssim.mean(dim=-1).squeeze(1)
    ssim_score = ssim_score.reshape(H, W)

    q_lambda = ssim_score.min()

    return q_lambda.cpu().numpy()
    