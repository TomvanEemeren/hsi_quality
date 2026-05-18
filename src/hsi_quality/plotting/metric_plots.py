import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from hsi_quality.metrics import Metric
from hsi_quality.data import Dataset, Resampler

ROOT_DIR = Path(__file__).resolve().parents[3]
PLOTS_DIR = ROOT_DIR / "plots"


def plot_metric(dataset: Dataset, metric: Metric, resampler: Resampler, save: bool = False):
    target = dataset["location_description"].unique()[0]
    
    scores = calculate_scores(dataset, metric, resampler)

    x = np.array(list(scores.keys()))
    y = np.array(list(scores.values()))

    fig, ax = plt.subplots(figsize=(2.5, 2))
    ax.scatter(x, y, label="Data Points")
    ax.set_xlabel("Off-Nadir Angle (degrees)")
    ax.set_ylabel(f"{metric}")
    ax.grid(True)
    ax.legend(loc="lower right")
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
    dataset = dataset.sort(by="off_nadir")

    scores = {}
    if metric.name in ["GRD"]:
        for idx, (satobj, metadata) in enumerate(tqdm(dataset, desc=f"Calculating {metric}", leave=False)):
            angle = metadata["off_nadir"]
            cube = satobj.l1d_cube.values
            cloud_mask = satobj.cloud_mask

            cloud_coverage = np.mean(cloud_mask == 2) * 100
            if cloud_coverage > 5:
                continue
            
            score, info = metric.calculate(cube, cloud_mask, metadata)

            scores[angle] = score

    elif metric.name in ["MvSSIM", "MeanSSIM", "QLambda"]:
        reference = None
        for idx, (satobj, metadata) in enumerate(tqdm(dataset, desc=f"Calculating {metric}", leave=False)):
            angle = metadata["off_nadir"]
            resampled_cube, cloud_mask = resampler.resample_capture(satobj)

            cloud_coverage = np.mean(cloud_mask == 2) * 100
            if cloud_coverage > 5:
                continue

            if reference is None:
                reference = resampled_cube
            
            score, info = metric.calculate(reference, resampled_cube)

            scores[angle] = score

    return scores