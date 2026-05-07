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
    for idx, (satobj, metadata) in enumerate(dataset):
        angle = metadata["off_nadir"]
        resampled_cube = resampler.resample_capture(satobj)
        
        if idx == 0:
            reference = resampled_cube
        
        score = metric.calculate(reference, resampled_cube)

        scores[angle] = score

    plt.figure(figsize=(10, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.xlabel("Off-nadir angle (degrees)")
    plt.ylabel(f"{metric}")
    plt.grid()
    if save:
        base_dir = Path(PLOTS_DIR) / target
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()