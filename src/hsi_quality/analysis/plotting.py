import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from hsi_quality.metrics import Metric
from hsi_quality.data import Dataset
from .resample import Resampler

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