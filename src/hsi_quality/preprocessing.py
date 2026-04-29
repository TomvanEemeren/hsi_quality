import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from hypso import Hypso2
from hypso.write import write_l1d_nc_file
from hsi_quality.utils import convert_timestamp

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

class Pipeline:
    def __init__(self, full: bool = True):
        self.full = full

    def run(self, row: pd.Series, nc_file: str, areas: np.ndarray) -> bool:
        target = row["location_description"]
        aoi = row["area"]
        smear_error = False
        rainbow_error = False
        has_error = False

        satobj = self.load_capture(target, nc_file)
        satobj.generate_l1b_cube(coeff_type="moved")
        satobj.generate_l1c_cube()
        satobj.generate_l1d_cube(use_direct_georef=True)

        if self.full:
            flipped_cube = self.flip_hyperspectral_image(satobj)
            satobj.l1d_cube = flipped_cube

            # Check for errors in the capture
            outlier = self.detect_outliers(aoi, areas)
            rainbow_error = self.has_rainbow_error(flipped_cube)
            smear_error = self.has_smear_error(flipped_cube)

            has_error = outlier or \
                        rainbow_error or \
                        smear_error or \
                        row["overexposed_samples_percentage"] > 5 or \
                        row["star_tracker_blinded_percentage"] > 90

        if not has_error:
            capture_name = nc_file.replace("-l1a.nc", "-l1d.nc")
            self.store_capture(satobj, target, capture_name)

            return True
        
        return False 

    def load_capture(self, target: str, nc_file: str):
        # Path to Hypso-2 capture
        path = os.path.join(DATA_DIR,target,"raw",nc_file)

        # Load Hypso-2 capture
        satobj = Hypso2(path=path, verbose=False)

        return satobj

    def flip_hyperspectral_image(self, satobj: Hypso2):
        l1d_cube = satobj.l1d_cube

        # Flip the image to always have same orientation
        if satobj.longitudes_direct[0][0] > satobj.longitudes_direct[0][-1]:
            flipped_cube = l1d_cube[:, ::-1, :]
            return flipped_cube

        # No flipping needed
        return l1d_cube
    
    def has_rainbow_error(self, cube: xr.DataArray) -> bool:
        pixel_var = np.var(cube.values, axis=2)

        median_var = np.median(pixel_var)
        mad = np.median(np.abs(pixel_var - median_var))

        z = (pixel_var - median_var) / (mad + 1e-8)

        return np.any(z > 30)

    def has_smear_error(self, cube: xr.DataArray) -> bool:
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
    
    def detect_outliers(self, aoi: float, areas: np.ndarray) -> bool:
        mean = np.mean(areas)
        std = np.std(areas)
        outlier = abs(aoi - mean) > 2 * std
        return outlier

    def store_capture(self, satobj: Hypso2, target: str, capture_name: str):
        if self.full:
            dir_name = "cleaned"
        else:
            dir_name = "reflectance"

        # Check if reflectance directory exists
        os.makedirs(os.path.join(DATA_DIR,target,dir_name), exist_ok=True)

        # Save the reflectance data
        nc_file = capture_name + "-l1d.nc"
        l1d_path = os.path.join(DATA_DIR,target,dir_name,nc_file)
        write_l1d_nc_file(satobj=satobj, l1d_path=l1d_path, overwrite=True)

def preprocess_data(target: str, full: bool = True):
    """
    Preprocess multiple hyperspectral images for a specific target location.

    Args:
        target (str): The target location for which to preprocess the data.
        full (bool): Whether to run the full pipeline.
    """
    metadata = pd.read_csv(os.path.join(DATA_DIR, target, "metadata.csv"))

    areas = metadata["area"].to_numpy()

    pipeline = Pipeline(full=full)

    # Iterate through Hypso-2 captures
    for idx in range(len(metadata)):
        row = metadata.iloc[idx]
        timestamp = convert_timestamp(row["timestamp_acquired_string"])
        target = row["location_description"]

        nc_file = f"{row['location_description']}_{timestamp}-l1a.nc"

        # Preprocess the hyperspectral image
        if not pipeline.run(row=row, nc_file=nc_file, areas=areas):
            print(f"Capture {nc_file} has errors and will be removed from the dataset.")
            metadata.drop(idx, inplace=True)

    # Save the updated metadata
    metadata.to_csv(os.path.join(DATA_DIR, target, "metadata_cleaned.csv"), index=False)

def crop_hyperspectral_image(satobj_h2: Hypso2, x1: int, x2: int, y1: int, y2: int) -> xr.DataArray:
    l1d_cube = satobj_h2.l1d_cube

    # Crop the image to the specified pixel coordinates
    cropped_cube = l1d_cube[y1:y2, x1:x2, :]

    return cropped_cube