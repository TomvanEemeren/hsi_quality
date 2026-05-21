import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from hsi_quality.analysis import Resampler
from hsi_quality import RESULTS_DIR

from hypso import Hypso2


class Metric:
    def __init__(self, name: str = None, params: dict = None):
        self.name = name
        self.params = params if params is not None else {}

    def __str__(self):
        return f"{self.name}"
    
    def calculate(self):
        raise NotImplementedError("Subclasses must implement the calculate method.")
    
    def plot(self):
        raise NotImplementedError("Subclasses must implement the plot method.")


class FullReferenceMetric(Metric):
    def __init__(self, name: str = None, params: dict = None):
        super().__init__(name=name, params=params)

    def calculate(self, X, Y):
        raise NotImplementedError("Subclasses must implement the calculate method for full reference metrics.")
    
    def plot(self, satobj: Hypso2, metric: Metric, resampler: Resampler, band: int = 40, save: bool = False):
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
            base_dir = Path(RESULTS_DIR)
            base_dir.mkdir(parents=True, exist_ok=True)
            output_path = base_dir / f"{metric}_blurred.png"
            fig.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
        else:
            plt.show() 