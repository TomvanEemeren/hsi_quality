import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.getcwd(),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.getcwd(),"src","hypso"))
sys.path.append(path)

from hsi_quality.analysis import Model
from hsi_quality.analysis import aggregate_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locations", type=str, nargs="+", default="all")
    parser.add_argument("--metric", type=str, default="GRD")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--name", type=str, default="regression_model.nc")

    args = parser.parse_args()

    aggregated_scores = aggregate_scores(args.locations, args.metric)

    X = aggregated_scores["off_nadir"].to_numpy(dtype=float)
    y = aggregated_scores["norm_score"].to_numpy(dtype=float)
    
    model = Model(seed=42, order=args.order)
    model.fit(X, y)
    model.save(args.name)
    model.plot(X, y, save=True)

if __name__ == "__main__":
    main()