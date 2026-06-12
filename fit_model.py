import os
import sys

path = os.path.abspath(os.path.join(os.getcwd(),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.getcwd(),"src","hypso"))
sys.path.append(path)

import numpy as np
from hsi_quality.analysis import Model, Prior, load_combined_scores
from hsi_quality.plotting import plot_model, plot_posterior, plot_prior

def main():
    prior = Prior(
        mu=np.array([0.0, 0.0, 0.0]),
        sigma=np.array([1.0, 0.5, 0.2])
    )

    plot_prior(prior, save=True)

    scores = load_combined_scores("SSIMLambda")

    X = scores["off_nadir"].to_numpy(dtype=float)
    y = scores["norm_score"].to_numpy(dtype=float)
    
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    model = Model(seed=42, order=2)
    idata = model.fit(X, y, prior=prior, metric="SSIMLambda")
    plot_model(idata, X, y, save=True)

    plot_posterior(idata, save=True)

if __name__ == "__main__":
    main()