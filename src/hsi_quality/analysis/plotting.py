import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from hsi_quality.metrics import Metric
from hsi_quality.data import Dataset
from .resample import Resampler
from hsi_quality.utils import normalize_cube, clip_cube

from hypso import Hypso2

ROOT_DIR = Path(__file__).resolve().parents[3]
PLOTS_DIR = ROOT_DIR / "plots"


def plot_metric(dataset: Dataset, metric: Metric, resampler: Resampler, save: bool = False):
    target = dataset["location_description"].unique()[0]
    reference = None

    dataset = dataset.sort(by="off_nadir")

    scores = {}
    reference = None
    for idx, (satobj, metadata) in enumerate(tqdm(dataset, desc=f"Calculating {metric}", leave=False)):
        angle = metadata["off_nadir"]
        resampled_cube, cloud_mask = resampler.resample_capture(satobj)

        cloud_coverage = np.mean(cloud_mask == 2) * 100
        if cloud_coverage > 5:
            continue

        if reference is None:
            reference = resampled_cube
        
        score = metric.calculate(reference, resampled_cube)

        scores[angle] = score

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(scores.keys()), list(scores.values()), marker="o")
    ax.set_xlabel("Off-Nadir Angle (degrees)")
    ax.set_ylabel(f"{metric}")
    ax.grid(True)
    ax.set_ylim(0, 1)
    if save:
        base_dir = Path(PLOTS_DIR) / target
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{metric}.png"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
    else:
        plt.show()

def plot_blurred(satobj: Hypso2, metric: Metric, resampler: Resampler, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube
    resampled_cube, _ = resampler.resample_capture(satobj, cube)

    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    for idx, sigma in enumerate([0.0, 1.0, 2.0, 3.0, 4.0]):
        blurred_cube = np.empty_like(resampled_cube)
        if sigma == 0:
            blurred_cube = resampled_cube
        else:
            blurred_cube = gaussian_filter(resampled_cube.values, sigma=(sigma, sigma, 0))
            blurred_cube = resampled_cube.copy(data=blurred_cube)

        score = metric.calculate(resampled_cube, blurred_cube)

        img = blurred_cube.values[:, :, band]
        rotated_img = np.rot90(img, k=1)

        ax[idx].imshow(rotated_img)
        ax[idx].axis("off")
        ax[idx].set_title(f"Sigma: {sigma}, Score: {score:.4f}")
    if save:
        base_dir = Path(PLOTS_DIR)
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{metric}_blurred.png"
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()  