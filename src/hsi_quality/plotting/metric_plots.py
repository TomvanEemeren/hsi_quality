from pathlib import Path
from matplotlib import pyplot as plt

from hsi_quality import RESULTS_DIR


def plot_metric(scores, save: bool = False):
    metric_name = scores["metric"].iloc[0]
    target = scores["location"].iloc[0]
    x = scores["off_nadir"].values
    y = scores["score"].values

    fig, ax = plt.subplots(figsize=(2.5, 2))
    ax.scatter(x, y, label="Data Points")
    ax.set_xlabel("Off-Nadir Angle (degrees)")
    ax.set_ylabel(f"{metric_name}")
    ax.grid(True)
    ax.legend()
    if save:
        base_dir = Path(RESULTS_DIR) / target
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / metric_name
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()