from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from hsi_quality.data import Dataset
from hsi_quality.analysis import Resampler
from hsi_quality import RESULTS_DIR


def plot_spectrum(dataset: Dataset, resampler: Resampler, x: int, y: int, save: bool = False):
    target = dataset["location_description"].unique()[0]
    cmap = plt.get_cmap("viridis")

    dataset = dataset.sort(by="off_nadir")
    max_off_nadir = dataset["off_nadir"].max()

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, (satobj, metadata) in enumerate(tqdm(dataset, desc="Plotting spectrum")):
        off_nadir = metadata["off_nadir"]

        resampled_cube, _ = resampler.resample_capture(satobj)

        spectrum = resampled_cube[x, y, :]
        bands = satobj.wavelengths

        ax.plot(bands, spectrum, label=f"off_nadir: {off_nadir}", color=cmap(off_nadir / max_off_nadir))

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save:
        base_dir = Path(RESULTS_DIR) / target
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"spectrum.png"
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
    else:
        plt.show()