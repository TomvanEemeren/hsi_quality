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
    noise_sigma: float = 1.0

    def create(self, name: str, dims=None):
        return pm.Normal(name, mu=self.mu, sigma=self.sigma, dims=dims)


@dataclass
class Standardization:
    mean: float
    std: float

    def transform(self, values: np.ndarray):
        scale = self.std if self.std != 0 else 1.0
        return (values - self.mean) / scale

    def inverse_transform(self, values: np.ndarray):
        scale = self.std if self.std != 0 else 1.0
        return values * scale + self.mean


class Model:
    def __init__(self, seed: int = None, order: int = 1):
        self.seed = seed
        self.order = order
        self.idata = None
        self.name = None
        self.x_scaler: Standardization | None = None
        self.y_scaler: Standardization | None = None

    @staticmethod
    def _make_scaler(values: np.ndarray):
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std == 0:
            std = 1.0
        return Standardization(mean=mean, std=std)

    def fit(self, X: np.ndarray, y: np.ndarray, prior: Prior, metric: str = None):
        X = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        self.x_scaler = self._make_scaler(X)
        self.y_scaler = self._make_scaler(y)

        X_norm = self.x_scaler.transform(X)
        y_norm = self.y_scaler.transform(y)

        Z = np.hstack([X_norm[:, None] ** i for i in range(self.order + 1)])

        coords = {
            "trial": np.arange(len(y_norm)),
            "features": ["bias"] + [f"x^{i}" for i in range(1, self.order + 1)]
        }
        
        with pm.Model(coords=coords) as model:
            X = pm.Data("X", Z, dims=["trial", "features"])

            # Model parameters
            weights = prior.create("weights", dims="features")
            sigma = pm.HalfNormal("sigma", sigma=prior.noise_sigma)

            # Linear model
            mu = X @ weights

            # Likelihood
            likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y_norm, dims="trial")

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
        idata.attrs["x_mean"] = self.x_scaler.mean
        idata.attrs["x_std"] = self.x_scaler.std
        idata.attrs["y_mean"] = self.y_scaler.mean
        idata.attrs["y_std"] = self.y_scaler.std
        self.idata = idata
        return idata
    
    def predict(self, X: np.ndarray):
        X = np.asarray(X, dtype=float).reshape(-1)

        if self.x_scaler is None:
            self.x_scaler = Standardization(
                mean=float(self.idata.attrs.get("x_mean", 0.0)),
                std=float(self.idata.attrs.get("x_std", 1.0)),
            )

        if self.y_scaler is None:
            self.y_scaler = Standardization(
                mean=float(self.idata.attrs.get("y_mean", 0.0)),
                std=float(self.idata.attrs.get("y_std", 1.0)),
            )

        X_norm = self.x_scaler.transform(X)
        Z_plot = np.vstack([X_norm**k for k in range(self.order + 1)]).T

        # (chains, draws, features)
        samples = self.idata.posterior["weights"].values

        # (n_samples, n_features)
        posterior_weights = samples.reshape(-1, samples.shape[-1])

        y_pred = Z_plot @ posterior_weights.T
        y_pred = self.y_scaler.inverse_transform(y_pred)
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
        self.x_scaler = Standardization(
            mean=float(self.idata.attrs.get("x_mean", 0.0)),
            std=float(self.idata.attrs.get("x_std", 1.0)),
        )
        self.y_scaler = Standardization(
            mean=float(self.idata.attrs.get("y_mean", 0.0)),
            std=float(self.idata.attrs.get("y_std", 1.0)),
        )
    
    def get_name(self):
        return self.name

def compare_orders(X: np.ndarray, y: np.ndarray, orders: list[int], visualize: bool = False):
    loos = {}
    for order in orders:
        prior = Prior(
            mu=np.zeros(order + 1),
            sigma=np.ones(order + 1),
            noise_sigma=1.0
        )

        model = Model(seed=42, order=order)
        idata = model.fit(X, y, prior=prior)
        loo = az.loo(idata)
        loos[f"order_{order}"] = loo

    df_comp_loo = az.compare(loos)

    if visualize:
        az.plot_compare(df_comp_loo)

    return df_comp_loo