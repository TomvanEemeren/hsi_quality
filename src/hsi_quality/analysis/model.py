import numpy as np
import pymc as pm
import arviz as az
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

from hsi_quality import MODELS_DIR


class Model:
    def __init__(self, seed: int = None, order: int = 1):
        self.seed = seed
        self.order = order
        self.idata = None
        self.name = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = X.reshape(-1, 1)
        Z = np.hstack([X**i for i in range(self.order + 1)])

        coords = {
            "trial": np.arange(len(y)),
            "features": ["bias"] + [f"x^{i}" for i in range(1, self.order + 1)]
        }
        
        with pm.Model(coords=coords) as model:
            X = pm.Data("X", Z, dims=["trial", "features"])

            # Model parameters
            weights = pm.Normal("weights", dims="features")
            sigma = pm.HalfNormal("sigma")

            # Linear model
            mu = X @ weights

            # Likelihood
            likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="trial")

            # Inference on observed data
            idata = pm.sample(random_seed=self.seed, quiet=True)

        with model:
            pm.compute_log_likelihood(idata, progressbar=False)

        idata.attrs["order"] = self.order

        self.idata = idata
        return idata
    
    def save(self, name: str):
        base_dir = Path(MODELS_DIR)
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / name
        self.name = name
        self.idata.to_netcdf(path.with_suffix(".nc"))

    def load(self, name: str):
        base_dir = Path(MODELS_DIR)
        path = base_dir / name
        self.idata = az.from_netcdf(path.with_suffix(".nc"))
        self.order = self.idata.attrs.get("order", 1)

    def plot(self, X: np.ndarray, y: np.ndarray, save: bool = False):
        x_plot = np.linspace(0, X.max() + 2, 200)
        Z_plot = np.vstack([x_plot**k for k in range(self.order + 1)]).T

        # (chains, draws, features)
        samples = self.idata.posterior["weights"].values

        # (n_samples, n_features)
        posterior_weights = samples.reshape(-1, samples.shape[-1])

        y_pred = Z_plot @ posterior_weights.T
        y_mean = y_pred.mean(axis=1)
        y_lower = np.percentile(y_pred, 2.5, axis=1)
        y_upper = np.percentile(y_pred, 97.5, axis=1)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X, y, color="tab:blue", label="Data")
        ax.plot(x_plot, y_mean, color="black", label="Posterior mean")
        ax.fill_between(x_plot, y_lower, y_upper, color="tab:orange", alpha=0.3, label="95% Confidence interval")
        ax.set_xlabel("Off-nadir angle (degrees)")
        ax.set_ylabel("Normalized GRD")
        ax.set_xlim(x_plot.min(), x_plot.max())
        ax.grid(True)
        ax.set_axisbelow(True) 
        ax.legend(loc="upper left")

        if save:
            base_dir = Path(MODELS_DIR)
            base_dir.mkdir(parents=True, exist_ok=True)
            if self.name:
                path = base_dir / self.name
            else:
                path = base_dir / f"model_order_{self.order}"
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

def compare_models(idata1: xr.DataTree, idata2: xr.DataTree):
    loo1 = az.loo(idata1)
    loo2 = az.loo(idata2)

    df_comp_loo = az.compare({"model1": loo1, "model2": loo2})

    return df_comp_loo