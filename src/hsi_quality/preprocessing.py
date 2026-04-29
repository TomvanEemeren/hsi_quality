import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from hypso import Hypso2
from hypso.write import write_l1d_nc_file

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

class Pipeline:
    def __init__(self, metadata: pd.DataFrame, full: bool = True):
        self.metadata = metadata
        self.full = full

    def run(self, target: str, nc_file: str):
        satobj = self.load_capture(target, nc_file)
        satobj.generate_l1b_cube(coeff_type="moved")
        satobj.generate_l1c_cube()
        satobj.generate_l1d_cube(use_direct_georef=True)

        if self.full:
            flipped_cube = self.flip_hyperspectral_image(satobj)
            rainbow_error = has_rainbow_error(flipped_cube)
            smear_error = has_smear_error(flipped_cube)
            satobj.l1d_cube = flipped_cube

        self.store_capture(satobj, target)

    def load_capture(self, target: str, nc_file: str):
        # Path to Hypso-2 capture
        path = os.path.join(DATA_DIR,target,"raw",nc_file)

        # Load Hypso-2 capture
        satobj = Hypso2(path=path, verbose=False)

        return satobj

    def flip_hyperspectral_image(satobj: Hypso2):
        l1d_cube = satobj.l1d_cube

        # Flip the image to always have same orientation
        if satobj.longitudes_direct[0][0] > satobj.longitudes_direct[0][-1]:
            flipped_cube = l1d_cube[:, ::-1, :]
            return flipped_cube

        # No flipping needed
        return l1d_cube
    
    def store_capture(self, satobj: Hypso2, target: str):
        name = satobj.capture_name

        if self.full:
            dir_name = "corrected"
        else:
            dir_name = "reflectance"

        # Check if reflectance directory exists
        os.makedirs(os.path.join(DATA_DIR,target,dir_name), exist_ok=True)

        # Save the reflectance data
        nc_file = name + "-l1d.nc"
        l1d_path = os.path.join(DATA_DIR,target,dir_name,nc_file)
        write_l1d_nc_file(satobj=satobj, l1d_path=l1d_path, overwrite=True)

def preprocess_data(target: str, full: bool = True):
    """
    Preprocess multiple hyperspectral images for a specific target location.
    The data is loaded from the raw directory and saved to the reflectance directory.

    Args:
        target (str): The target location for which to preprocess the data.
    """

    # Path to Hypso-2 captures
    raw_dir = os.path.join(DATA_DIR, target, "raw")
    nc_files = os.listdir(raw_dir)
    metadata = pd.read_csv(os.path.join(DATA_DIR, target, "metadata.csv"))

    pipeline = Pipeline(metadata=metadata, full=full)

    # Iterate through Hypso-2 captures
    for nc_file in nc_files:
        # Preprocess the hyperspectral image
        pipeline.run(target=target, nc_file=nc_file)

def crop_hyperspectral_image(satobj_h2: Hypso2, x1: int, x2: int, y1: int, y2: int) -> xr.DataArray:
    l1d_cube = satobj_h2.l1d_cube

    # Crop the image to the specified pixel coordinates
    cropped_cube = l1d_cube[y1:y2, x1:x2, :]

    return cropped_cube

def has_rainbow_error(cube: xr.DataArray) -> bool:
    pixel_var = np.var(cube.values, axis=2)

    median_var = np.median(pixel_var)
    mad = np.median(np.abs(pixel_var - median_var))

    z = (pixel_var - median_var) / (mad + 1e-8)

    return np.any(z > 30)

def has_smear_error(cube: xr.DataArray) -> bool:
    min_run = 5

    # difference between neighboring rows
    vert_diff = np.diff(cube, axis=0)  # shape (H-1, W, C)

    # magnitude of vertical change per column
    col_score = np.mean(np.abs(vert_diff), axis=(0, 2))  # (W,)

    median = np.median(col_score)
    mad = np.median(np.abs(col_score - median))

    z = (col_score - median) / (mad + 1e-8)

    # smear → unusually LOW variation
    anomalous = z < -4

    run = 0
    for a in anomalous:
        run = run + 1 if a else 0
        if run >= min_run:
            return True

    return False