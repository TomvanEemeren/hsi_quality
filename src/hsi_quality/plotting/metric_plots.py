import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from hsi_quality import RESULTS_DIR
from hsi_quality.analysis import Resampler
from hsi_quality.metrics import Metric

from hypso import Hypso2

def plot_metric(scores, save: bool = False):
    metric_name = scores["metric"].iloc[0]
    target = scores["location"].iloc[0]
    x = scores["off_nadir"].values
    y = scores["score"].values

    plot_name = metric_name
    if metric_name == "MeanSSIM":
        plot_name = "MSSIM"
    elif metric_name == "SSIMLambda":
        plot_name = r"SSIM-$\lambda$"

    mm = 1/25.4
    fig, ax = plt.subplots(figsize=(40*mm, 30*mm))
    ax.scatter(x, y)
    ax.set_xlabel("Off-nadir angle (deg)")
    ax.set_ylabel(f"{plot_name}")
    ax.grid(True)
    if save:
        base_dir = Path(RESULTS_DIR) / target
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / metric_name
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, dpi=300)
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
    else:
        plt.show()

def plot_blurred(satobj: Hypso2, metric: Metric, resampler: Resampler, band: int = 40, save: bool = False):
        cube = satobj.l1d_cube
        resampled_cube, _ = resampler.resample_capture(satobj, cube)

        mm = 1/25.4
        fig, ax = plt.subplots(1, 5, figsize=(149*mm, 30*mm))
        for idx, sigma in enumerate([0.0, 1.0, 2.0, 3.0, 4.0]):
            blurred_cube = np.empty_like(resampled_cube)
            if sigma == 0:
                blurred_cube = resampled_cube
            else:
                blurred_values = gaussian_filter(resampled_cube.values, sigma=(sigma, sigma, 0))
                blurred_cube = resampled_cube.copy(data=blurred_values)

            score, _ = metric.calculate(resampled_cube, blurred_cube)

            img = blurred_cube.values[:, :, band]
            rotated_img = np.rot90(img, k=1)

            ax[idx].imshow(rotated_img, cmap="gray")
            ax[idx].axis("off")
            ax[idx].set_title(rf"$\sigma$: {sigma}" + f"\n Score: {score:.4f}")
        if save:
            target = satobj.capture_target
            base_dir = Path(RESULTS_DIR) / target
            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = base_dir / f"{metric}_blurred.png"
            fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, dpi=300)
            fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close(fig)
        else:
            plt.show() 