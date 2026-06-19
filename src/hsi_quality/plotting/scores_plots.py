import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from hsi_quality import RESULTS_DIR
from hsi_quality.analysis import Resampler
from hsi_quality.data import Dataset


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


def plot_ssim_images(scores: pd.DataFrame, dataset: Dataset, resampler: Resampler, capture_names: list, band: int = 29, save: bool = False):
    mm = 1/25.4
    fig, ax = plt.subplots(2,3, constrained_layout=True, sharex=True, sharey=True, figsize=(120*mm, 80*mm))
    for i, capture_name in enumerate(capture_names):
        angle = scores[scores["capture_name"] == capture_name]["off_nadir"].iloc[0]
        score = scores[scores["capture_name"] == capture_name]["norm_score"].iloc[0]

        satobj = dataset.get_capture(capture_name)
        cube = satobj.l1d_cube

        cube, _ = resampler.resample_capture(satobj, cube)

        image = cube.values[:, :, band]
        rotated_img = np.rot90(image, k=1)

        ax[i//3, i%3].imshow(rotated_img, cmap="gray")
        ax[i//3, i%3].axis("off")
        ax[i//3, i%3].set_title(f"Angle: {angle:.2f}, Score: {score:.3f}")

    if save:
        base_dir = Path(RESULTS_DIR) / "plots"
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / f"dssim_images"
        fig.savefig(out_path.with_suffix(".pdf"), dpi=300)
        fig.savefig(out_path.with_suffix(".png"), dpi=300)
        plt.close(fig)
    else:
        plt.show()

def plot_image_histograms(scores: pd.DataFrame, dataset: Dataset, resampler: Resampler, capture_names: list, band: int = 29, save: bool = False):
    mm = 1/25.4
    fig, ax = plt.subplots(2,3, constrained_layout=True, sharex=True, sharey=True, figsize=(148*mm, 80*mm))
    for i, capture_name in enumerate(capture_names):
        angle = scores[scores["capture_name"] == capture_name]["off_nadir"].iloc[0]
        score = scores[scores["capture_name"] == capture_name]["norm_score"].iloc[0]

        satobj = dataset.get_capture(capture_name)
        cube = satobj.l1d_cube
        cube, _ = resampler.resample_capture(satobj, cube)

        image = cube.values[:, :, band]

        ax[i//3, i%3].hist(image.ravel(), bins=50, alpha=0.85, rasterized=True)
        ax[i//3, i%3].set_title(f"Angle: {angle:.2f}, Score: {score:.3f}")
        ax[i//3, i%3].set_xlabel("Pixel value")
        ax[i//3, i%3].tick_params(axis='x', labelbottom=True)
        if i % 3 == 0:
             ax[i//3, i%3].set_ylabel("Count")

    if save:
        base_dir = Path(RESULTS_DIR) / "plots"
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / f"image_histograms"
        fig.savefig(out_path.with_suffix(".pdf"), dpi=300)
        fig.savefig(out_path.with_suffix(".png"), dpi=300)
        plt.close(fig)
    else:
        plt.show()