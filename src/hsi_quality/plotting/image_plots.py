import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from hsi_quality.data import Dataset, Resampler
from hsi_quality.utils import normalize_cube

from hypso import Hypso2
from hypso.spectral_analysis import get_closest_wavelength_index

ROOT_DIR = Path(__file__).resolve().parents[3]
PLOTS_DIR = ROOT_DIR / "plots"


def plot_full_images(dataset: Dataset, band: int = None, save: bool = False, mode: str = "rgb"):
    target = dataset["location_description"].unique()[0]

    for idx in tqdm(range(len(dataset)), desc="Plotting images", leave=False):
        satobj, _ = dataset[idx]
        cube = normalize_cube(satobj.l1d_cube, method="percentile")

        if mode == "rgb":
            image = get_rgb_image(satobj, cube)
        elif mode == "band" and band is not None:
            image = get_band_image(cube, band)
        else:
            raise ValueError("Invalid mode. Use 'rgb' or 'band'.")

        fig, ax = plt.subplots()
        ax.imshow(image, aspect=1/8)
        ax.axis("off")
        if save:
            capture_name = satobj.capture_name
            if mode == "rgb":
                base_dir = Path(PLOTS_DIR) / target / "rgb"
            elif mode == "band" and band is not None:
                base_dir = Path(PLOTS_DIR) / target / f"band_{band}"
            base_dir.mkdir(parents=True, exist_ok=True)
            output_path = base_dir / f"{capture_name}.png"

            fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close(fig)
        else:
            plt.show()

def plot_resampled_images(dataset: Dataset, resampler: Resampler, band: int = None, save: bool = False, mode: str = "rgb"):
    target = dataset["location_description"].unique()[0]

    for idx in tqdm(range(len(dataset)), desc="Plotting resampled images", leave=False):
        satobj, _ = dataset[idx]
        cube = normalize_cube(satobj.l1d_cube, method="percentile")
        cube, _ = resampler.resample_capture(satobj, cube)

        if mode == "rgb":
            image = get_rgb_image(satobj, cube)
        elif mode == "band" and band is not None:
            image = get_band_image(cube, band)
        else:
            raise ValueError("Invalid mode. Use 'rgb' or 'band'.")

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")
        if save:
            capture_name = satobj.capture_name
            base_dir = Path(PLOTS_DIR) / target / "resampled"
            base_dir.mkdir(parents=True, exist_ok=True)
            output_path = base_dir / f"{capture_name}.png"
            fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close(fig)
        else:
            plt.show()

def get_band_image(cube: xr.DataArray, band: int):
    img = cube.values[:, :, band]
    rotated_img = np.rot90(img, k=1)

    return rotated_img

def get_rgb_image(satobj: Hypso2, cube: xr.DataArray):
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

    # Rotate the image for better visualization
    rotated_img = np.rot90(img, k=1)

    return rotated_img

def plot_cloud_images(dataset: Dataset, resampler: Resampler = None, save: bool = False):
    target = dataset["location_description"].unique()[0]

    classes = np.array([0, 1, 2])
    cmap_base = plt.get_cmap("tab10")
    colors = [cmap_base(i) for i in range(len(classes))]
    cmap = ListedColormap(colors)
    vmin, vmax = classes.min(), classes.max()

    for idx in tqdm(range(len(dataset)), desc="Plotting cloud masks", leave=False):
        satobj, _ = dataset[idx]
        cloud_mask, aspect = None, None

        if resampler is not None:
            _, cloud_mask = resampler.resample_capture(satobj)
        else:
            cloud_mask = satobj.cloud_mask.values
            aspect = 1/8

        rotated_mask = np.rot90(cloud_mask, k=1)

        fig, ax = plt.subplots()
        ax.imshow(rotated_mask, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
        ax.axis("off")
        if save:
            capture_name = satobj.capture_name
            base_dir = Path(PLOTS_DIR) / target / "cloud_masks"
            base_dir.mkdir(parents=True, exist_ok=True)
            output_path = base_dir / f"{capture_name}.png"
            fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close(fig)
        else:
            plt.show()