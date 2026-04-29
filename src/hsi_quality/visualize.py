import os
import numpy as np
import xarray as xr
from pathlib import Path
from matplotlib import pyplot as plt

from hypso import Hypso2
from hypso.spectral_analysis import get_closest_wavelength_index

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

def plot_rgb(satobj_h2: Hypso2, save: bool = False, verbose: bool = False) -> np.ndarray:
    """
    Visualize the hyperspectral datacube as an RGB image using the RGB wavelengths.
    The hyperspectral image has an aspect ratio of 1:8.

    Args:
        satobj_h2 (Hypso2): The Hypso2 satellite object.
        save (bool): Whether to save the RGB image. Defaults to False.
        verbose (bool): Whether to print verbose output. Defaults to False.
    Returns:
        np.ndarray: The RGB image as a NumPy array.
    """      
    cube = satobj_h2.l1d_cube

    # Get wavelengths of capture
    satobj_h2.wavelengths

    # Get band index of wavelength
    red_wl = 630
    green_wl = 550
    blue_wl = 480

    # Get nearest band indices for RGB wavelengths
    r_idx = get_closest_wavelength_index(satobj_h2, red_wl)
    g_idx = get_closest_wavelength_index(satobj_h2, green_wl)
    b_idx = get_closest_wavelength_index(satobj_h2, blue_wl)

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
        name = satobj_h2.l1d_name
        target = name.split("_")[0]
        os.makedirs(os.path.join(DATA_DIR, target, "images"), exist_ok=True)
        output_path = os.path.join(DATA_DIR, target, "images", name + ".png")

        # Save the RGB image
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
        if verbose:
            print(f"Saved RGB image to {output_path}")
    else:
        if verbose:
            name = satobj_h2.l1d_name
            print(f"RGB image for {name}")

        # Display the RGB image
        plt.show()
    
    plt.close(fig)

    return rotated_rgb