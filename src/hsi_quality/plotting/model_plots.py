import numpy as np
import xarray as xr  
from pathlib import Path
from scipy.stats import norm, gaussian_kde
from matplotlib import pyplot as plt  

from hsi_quality import MODELS_DIR
from hsi_quality.analysis import Prior, Model


def plot_model(model: Model, X: np.ndarray, y: np.ndarray, save: bool = False):
    name = model.get_name()

    x_plot = np.linspace(0, X.max() + 2, 200)
    y_mean, y_lower, y_upper = model.predict(x_plot)

    mm = 1/25.4
    fig, ax = plt.subplots(figsize=(74*mm, 50*mm), constrained_layout=True)
    ax.scatter(X, y, color="tab:blue", label="Scores")
    ax.plot(x_plot, y_mean, color="black", label="Posterior mean")
    ax.fill_between(x_plot, y_lower, y_upper, color="tab:orange", alpha=0.3, label="95% Confidence interval")

    if name == "GRD":
        cosine = 1 / np.cos(np.radians(x_plot))**2
        ax.plot(x_plot, cosine, color="tab:red", linestyle="--", label=r"$1 / \cos^2(\theta)$")
        ax.set_ylabel("Normalized GRD")
    elif name == "SSIMLambda":
        ax.set_ylabel("DSSIM")
    else:
        ax.set_ylabel("Score")
        
    ax.set_xlabel(r"Off-nadir angle, $\theta$, (deg)")
    ax.set_xlim(x_plot.min(), x_plot.max())
    ax.grid(True)
    ax.set_axisbelow(True) 
    ax.legend(loc="upper left")

    if save:
        base_dir = Path(MODELS_DIR)
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{name}_fit"
        fig.savefig(path.with_suffix(".pdf"), dpi=300)
        fig.savefig(path.with_suffix(".png"), dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_prior(prior: Prior, feature_names: list = None, n: int = 1000, save: bool = False):
    n_features = len(prior.mu)

    if feature_names is None:
        feature_names = [f"$w{i}$" for i in range(n_features)]

    mm = 1/25.4
    fig, ax = plt.subplots(figsize=(74*mm, 40*mm), constrained_layout=True)

    for mu, sigma, name in zip(prior.mu, prior.sigma, feature_names):
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, n)
        y = norm.pdf(x, mu, sigma)
        ax.plot(x, y, label=name)
    ax.set_xlabel("Weight, $w$")
    ax.set_ylabel("PDF")
    ax.legend()
    ax.grid(True)

    if save:
        base_dir = Path(MODELS_DIR)
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / "prior"
        fig.savefig(path.with_suffix(".pdf"), dpi=300)
        fig.savefig(path.with_suffix(".png"), dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_posterior(idata: xr.DataTree, save: bool = False):
    order = idata.attrs.get("order")
    name = idata.attrs.get("name")

    # (chains, draws, features)
    samples = idata.posterior["weights"].values

    # (n_samples, n_features)
    posterior_weights = samples.reshape(-1, samples.shape[-1])

    mm = 1/25.4
    fig, ax = plt.subplots(figsize=(74*mm, 40*mm), constrained_layout=True)

    for i in range(order + 1):
        x = posterior_weights[:, i]
        kde = gaussian_kde(x)

        xs = np.linspace(x.min(), x.max(), 200)
        ys = kde(xs)

        ax.plot(xs, ys / ys.max(), label=f"$w{i}$")

    ax.set_xlabel("Weight, $w$")
    ax.set_ylabel("Normalized PDF")
    ax.legend()
    ax.grid(True)

    if save:
        base_dir = Path(MODELS_DIR)
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{name}_posterior"
        fig.savefig(path.with_suffix(".pdf"), dpi=300)
        fig.savefig(path.with_suffix(".png"), dpi=300)
        plt.close(fig)
    else:
        plt.show()  
