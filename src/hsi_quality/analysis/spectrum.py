from matplotlib import pyplot as plt

from hsi_quality.data import Dataset
from .resample import Resampler


def plot_spectrum(dataset: Dataset, resampler: Resampler, x: int, y: int):
    cmap = plt.get_cmap("viridis")

    dataset = dataset.sort(by="off_nadir")
    max_off_nadir = dataset["off_nadir"].max()

    _, ax = plt.subplots(figsize=(10, 6))
    for _, (satobj, metadata) in enumerate(dataset):
        off_nadir = metadata["off_nadir"]

        resampled_cube = resampler.resample_capture(satobj)
        spectrum = resampled_cube[x, y, :]
        bands = satobj.wavelengths

        ax.plot(bands, spectrum, label=f"off_nadir: {off_nadir}", color=cmap(off_nadir / max_off_nadir))

        
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
