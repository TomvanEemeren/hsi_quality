import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
from dataclasses import dataclass

from hsi_quality import MODELS_DIR


@dataclass
class Prior:
    mu: float | np.ndarray
    sigma: float | np.ndarray

    def create(self, name: str, dims=None):
        return pm.Normal(name, mu=self.mu, sigma=self.sigma, dims=dims)


class Model:
    def __init__(self, seed: int = None, order: int = 1):
        self.seed = seed
        self.order = order
        self.idata = None
        self.name = None

    def fit(self, X: np.ndarray, y: np.ndarray, prior: Prior, metric: str = None):
        X = X.reshape(-1, 1)
        Z = np.hstack([X**i for i in range(self.order + 1)])

        coords = {
            "trial": np.arange(len(y)),
            "features": ["bias"] + [f"x^{i}" for i in range(1, self.order + 1)]
        }
        
        with pm.Model(coords=coords) as model:
            X = pm.Data("X", Z, dims=["trial", "features"])

            # Model parameters
            weights = prior.create("weights", dims="features")
            sigma = pm.HalfNormal("sigma", mu=0, sigma=2.0)

            # Linear model
            mu = X @ weights

            # Likelihood
            likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="trial")

            # Inference on observed data
            idata = pm.sample(random_seed=self.seed, quiet=True)

        with model:
            pm.compute_log_likelihood(idata, progressbar=False)

        if metric:
            self.name = metric
        else:
            self.name = f"order_{self.order}"

        idata.attrs["order"] = self.order
        idata.attrs["name"] = self.name
        self.idata = idata
        return idata
    
    def predict(self, X: np.ndarray):
        Z_plot = np.vstack([X**k for k in range(self.order + 1)]).T

        # (chains, draws, features)
        samples = self.idata.posterior["weights"].values

        # (n_samples, n_features)
        posterior_weights = samples.reshape(-1, samples.shape[-1])

        y_pred = Z_plot @ posterior_weights.T
        y_mean = y_pred.mean(axis=1)
        y_lower = np.percentile(y_pred, 2.5, axis=1)
        y_upper = np.percentile(y_pred, 97.5, axis=1)

        return y_mean, y_lower, y_upper

    def save(self):
        name = self.idata.attrs.get("name", f"order_{self.order}")
        base_dir = Path(MODELS_DIR)
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / name
        self.idata.to_netcdf(path.with_suffix(".nc"))

    def load(self, name: str):
        base_dir = Path(MODELS_DIR)
        path = base_dir / name
        self.idata = az.from_netcdf(path.with_suffix(".nc"))

        self.order = self.idata.attrs.get("order")
    
    def get_name(self):
        return self.name

def compare_orders(X: np.ndarray, y: np.ndarray, orders: list[int], priors: list[Prior], visualize: bool = False):
    loos = {}
    for order, prior in zip(orders, priors):
        model = Model(seed=42, order=order)
        idata = model.fit(X, y, prior=prior)
        loo = az.loo(idata)
        loos[f"order_{order}"] = loo

    df_comp_loo = az.compare(loos)

    if visualize:
        az.plot_compare(df_comp_loo)

    return df_comp_loo