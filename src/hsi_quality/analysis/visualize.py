import os
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from hypso import Hypso2
from hypso.spectral_analysis import get_closest_wavelength_index

ROOT_DIR = Path(__file__).resolve().parents[3]
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")


def plot_band(satobj: Hypso2, band: int):
    cube = satobj.l1d_cube.values
    img = cube[:, :, band]

    rotated_img = np.rot90(img, k=1)

    _, ax = plt.subplots()
    ax.imshow(rotated_img, aspect=1/8)
    ax.axis("off")
    plt.show()

    return rotated_img

def plot_rgb(satobj: Hypso2, save: bool = False, verbose: bool = False) -> np.ndarray:
    """
    Visualize the hyperspectral datacube as an RGB image using the RGB wavelengths.
    The hyperspectral image has an aspect ratio of 1:8.
    """      
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
    rgb = np.stack(
        [
            cube.isel(band=r_idx).values,
            cube.isel(band=g_idx).values,
            cube.isel(band=b_idx).values,
        ],
        axis=-1,
    )

    # Per-channel normalization for display
    p2 = np.percentile(rgb, 2, axis=(0, 1))
    p98 = np.percentile(rgb, 98, axis=(0, 1))
    rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

    # Rotate the image for better visualization
    rotated_rgb = np.rot90(rgb_norm, k=1)

    # Plot the RGB image
    fig, ax = plt.subplots()
    ax.imshow(rotated_rgb, aspect=1/8)
    ax.axis("off")
    
    if save:
        name = satobj.capture_name
        target = name.split("_")[0]

        base_dir = Path(PLOTS_DIR) / target / "rgb"
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{name}.png"

        # Save the RGB image
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
        if verbose:
            print(f"Saved RGB image to {output_path}")
    else:
        if verbose:
            name = satobj.capture_name
            print(f"RGB image for {name}")

        # Display the RGB image
        plt.show()
    
    plt.close(fig)

    return rotated_rgb