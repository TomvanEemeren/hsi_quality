import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from hsi_quality.metrics import Metric
from hsi_quality.data import Dataset
from .resample import Resampler

from hypso import Hypso2

ROOT_DIR = Path(__file__).resolve().parents[3]
PLOTS_DIR = ROOT_DIR / "plots"

# Parameters for plotting
rcParams = {
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 7,
    'legend.title_fontsize': 7,
    'figure.titlesize': 7,
    # 'lines.linewidth': 0.8,
    'lines.markersize': 3,
}

def set_plotting_style():
    plt.rcParams.update(rcParams)

def plot_metric(dataset: Dataset, metric: Metric, resampler: Resampler, save: bool = False):
    target = dataset["location_description"].unique()[0]
    
    scores = calculate_scores(dataset, metric, resampler)

    x = np.array(list(scores.keys()))[1:]
    y = np.array(list(scores.values()))[1:]

    coefficients = np.polyfit(x, y, 1)
    p = np.poly1d(coefficients)

    xp = np.linspace(0, 30, 100)

    fig, ax = plt.subplots(figsize=(2.5, 2))
    ax.scatter(x, y, label="Data Points")
    ax.plot(xp, p(xp), label="Regression line", color="red")
    ax.set_xlabel("Off-Nadir Angle (degrees)")
    ax.set_ylabel(f"{metric}")
    ax.grid(True)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    if save:
        base_dir = Path(PLOTS_DIR) / target
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{metric}"
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def calculate_scores(dataset: Dataset, metric: Metric, resampler: Resampler):
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

    return scores

def plot_blurred(satobj: Hypso2, metric: Metric, resampler: Resampler, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube
    resampled_cube, _ = resampler.resample_capture(satobj, cube)

    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    for idx, sigma in enumerate([0.0, 1.0, 2.0, 3.0, 4.0]):
        blurred_cube = np.empty_like(resampled_cube)
        if sigma == 0:
            blurred_cube = resampled_cube
        else:
            blurred_values = gaussian_filter(resampled_cube.values, sigma=(sigma, sigma, 0))
            blurred_cube = resampled_cube.copy(data=blurred_values)

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