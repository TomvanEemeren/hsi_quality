import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.getcwd(),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.getcwd(),"src","hypso"))
sys.path.append(path)

import numpy as np
from hsi_quality.analysis import Model, Prior, load_combined_scores
from hsi_quality.plotting import plot_model, plot_posterior, plot_prior

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save the model.")
    parser.add_argument("--order", type=int, default=2, help="Order for the model.")
    parser.add_argument("--metric", type=str, default="SSIMLambda", help="Name of dataset.")
    
    args = parser.parse_args()

    prior = Prior(
        mu=np.zeros(args.order + 1),
        sigma=np.ones(args.order + 1),
        noise_sigma=1.0
    )
    plot_prior(prior, save=True)

    scores = load_combined_scores(args.metric)

    X = scores["off_nadir"].to_numpy(dtype=float)
    y = scores["norm_score"].to_numpy(dtype=float)

    model = Model(seed=42, order=args.order)
    idata = model.fit(X, y, prior=prior, metric=args.metric)
    plot_model(model, X, y, save=True)
    plot_posterior(idata, save=True)

    if args.save:
        model.save()

if __name__ == "__main__":
    main()