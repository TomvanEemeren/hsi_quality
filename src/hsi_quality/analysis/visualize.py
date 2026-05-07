import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from hsi_quality.data import Dataset

from hypso import Hypso2
from hypso.spectral_analysis import get_closest_wavelength_index

ROOT_DIR = Path(__file__).resolve().parents[3]
PLOTS_DIR = ROOT_DIR / "plots"


def plot_band(dataset: Dataset, band: int, save: bool = False):
    target = dataset["location_description"].unique()[0]

    dataset = dataset.sort(by="timestamp_acquired")

    for idx in range(len(dataset)):
        satobj, _ = dataset[idx]

        image = get_band_image(satobj, band)

        fig, ax = plt.subplots()
        ax.imshow(image, aspect=1/8)
        ax.axis("off")
        if save:
            capture_name = satobj.capture_name
            base_dir = Path(PLOTS_DIR) / target / f"band_{band}"
            base_dir.mkdir(parents=True, exist_ok=True)
            output_path = base_dir / f"{capture_name}.png"

            fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
        else:
            fig.show()

        plt.close(fig)

def get_band_image(satobj: Hypso2, band: int):
    cube = satobj.l1d_cube.values
    img = cube[:, :, band]
    rotated_img = np.rot90(img, k=1)

    return rotated_img

def plot_rgb(dataset: Dataset, save: bool = False):
    target = dataset["location_description"].unique()[0]

    dataset = dataset.sort(by="timestamp_acquired")

    for idx in range(len(dataset)):
        satobj, _ = dataset[idx]

        rgb_image = get_rgb_image(satobj)

        fig, ax = plt.subplots()
        ax.imshow(rgb_image, aspect=1/8)
        ax.axis("off")
        if save:
            capture_name = satobj.capture_name
            base_dir = Path(PLOTS_DIR) / target / "rgb"
            base_dir.mkdir(parents=True, exist_ok=True)
            output_path = base_dir / f"{capture_name}.png"

            fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
        else:
            fig.show()

        plt.close(fig)

def get_rgb_image(satobj: Hypso2):
    cube = satobj.l1d_cube

    # Get band index of wavelength
    red_wl = 630
    green_wl = 550
    blue_wl = 480

    # Get nearest band indices for RGB wavelengths
    r_idx = get_closest_wavelength_index(satobj, red_wl)
    g_idx = get_closest_wavelength_index(satobj, green_wl)
    b_idx = get_closest_wavelength_index(satobj, blue_wl)

    # Stack selected bands into an RGB image
    img = np.stack(
        [
            cube.isel(band=r_idx).values,
            cube.isel(band=g_idx).values,
            cube.isel(band=b_idx).values,
        ],
        axis=-1,
    )

    # Per-channel normalization for display
    p2 = np.percentile(img, 2, axis=(0, 1))
    p98 = np.percentile(img, 98, axis=(0, 1))
    img_norm = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)

    # Rotate the image for better visualization
    rotated_img = np.rot90(img_norm, k=1)

    return rotated_img