import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from hypso import Hypso2
from hsi_quality.data import Dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

class Pipeline:
    def __init__(self, full: bool = True):
        self.full = full

    def run(self, satobj: Hypso2, metadata: pd.Series):
        # Find the row of the capture in metadata
        row = metadata[np.isclose(metadata["timestamp_acquired"], satobj.unixtime)].iloc[0]

        areas = metadata["area"].to_numpy()
        aoi = row["area"]
        smear_error = False
        rainbow_error = False
        has_error = False

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
        
        return satobj, has_error

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

def preprocess_data(target: str, dir: str = "processed", full: bool = True):
    """
    Preprocess multiple hyperspectral images for a specific target location.

    Args:
        target (str): The target location for which to preprocess the data.
        dir (str): The directory where the processed data will be stored.
        full (bool): Whether to run the full pipeline.
    """
    metadata = pd.read_csv(os.path.join(DATA_DIR, target, "metadata.csv"))

    raw_dataset = Dataset(metadata, data_dir="raw", level="l1a")

    pipeline = Pipeline(full=full)

    # Iterate through Hypso-2 captures
    clean_rows = []
    for idx in range(len(raw_dataset)):
        satobj = raw_dataset[idx]

        satobj, has_error = pipeline.run(satobj, metadata)

        if not has_error:
            raw_dataset.store_capture(satobj, target, satobj.capture_name, dir=dir, level="l1d")
            clean_rows.append(metadata.iloc[idx])

    clean_metadata = pd.DataFrame(clean_rows)
    clean_metadata.to_csv(os.path.join(DATA_DIR, target, dir, "clean_metadata.csv"), index=False)

def crop_hyperspectral_image(satobj_h2: Hypso2, x1: int, x2: int, y1: int, y2: int) -> xr.DataArray:
    l1d_cube = satobj_h2.l1d_cube

    # Crop the image to the specified pixel coordinates
    cropped_cube = l1d_cube[y1:y2, x1:x2, :]

    return cropped_cube