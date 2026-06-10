import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from hsi_quality import RESULTS_DIR


def plot_outliers(scores: pd.DataFrame, target: str, save: bool = False):
    scores = scores[scores["location"] == target].copy()
    scores["z_score"] = (scores["score"] - scores["score"].mean()) / scores["score"].std()

    mm = 1/25.4
    fig, ax = plt.subplots(figsize=(74*mm, 40*mm))
    
    ax.hist(scores["z_score"], bins=20, edgecolor="black")
    ax.set_xlabel("Z-Score")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.75)

    if save:
        base_dir = Path(RESULTS_DIR) / "plots"
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{target}_outliers"
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, dpi=300)
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_combined_scores(scores: pd.DataFrame, save: bool = False):
    metric = scores["metric"].iloc[0]

    name = ""
    if metric in ["SSIMLambda", "MeanSSIM", "MvSSIM"]:
        name = "DSSIM"
    elif metric == "GRD":
        name = "Normalized GRD"

    mm = 1/25.4
    fig, ax = plt.subplots(figsize=(74*mm, 40*mm), constrained_layout=True)
    ax.scatter(scores["off_nadir"], scores["norm_score"])
    ax.set_xlabel(r"Off-nadir angle, $\theta$, (deg)")
    ax.set_ylabel(name)
    ax.set_xticks(np.arange(0, 70, 10))
    ax.grid(True, alpha=0.3)

    if save:
        base_dir = Path(RESULTS_DIR) / "plots"
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{metric}_combined"
        fig.savefig(path.with_suffix(".pdf"), dpi=300)
        fig.savefig(path.with_suffix(".png"), dpi=300)
        plt.close(fig)
    else:
        plt.show()