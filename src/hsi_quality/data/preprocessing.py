import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from hypso import Hypso2
from hsi_quality.data import RawDataset
from .store_data import store_capture

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

class Pipeline:
    def __init__(self, full: bool = True):
        self.full = full

    def run(self, satobj: Hypso2, row: pd.Series, areas: np.ndarray):
        aoi = row["area"]
        smear_error = False
        rainbow_error = False
        has_error = False

        satobj.generate_l1b_cube(coeff_type="moved")
        satobj.generate_l1c_cube()
        satobj.generate_l1d_cube(use_direct_georef=True)

        if self.full:
            cube, latitudes, longitudes = self.flip_hyperspectral_image(satobj)
            satobj.l1d_cube = cube
            satobj.latitudes = latitudes
            satobj.longitudes = longitudes

            # Check for errors in the capture
            outlier = self.detect_outliers(aoi, areas)
            rainbow_error = self.has_rainbow_error(cube)
            smear_error = self.has_smear_error(cube)

            has_error = outlier or \
                        rainbow_error or \
                        smear_error or \
                        row["overexposed_samples_percentage"] > 5 or \
                        row["star_tracker_blinded_percentage"] > 90
        
        return satobj, has_error

    def flip_hyperspectral_image(self, satobj: Hypso2):
        l1d_cube = satobj.l1d_cube
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes

        # Flip the image to always have same orientation
        if satobj.longitudes[0][0] > satobj.longitudes[0][-1]:
            flipped_latitudes = latitudes[:, ::-1]
            flipped_longitudes = longitudes[:, ::-1]
            flipped_cube = l1d_cube[:, ::-1, :]
            return flipped_cube, flipped_latitudes, flipped_longitudes

        # No flipping needed
        return l1d_cube, latitudes, longitudes
    
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

    raw_dataset = RawDataset(metadata)

    pipeline = Pipeline(full=full)

    areas = metadata["area"].to_numpy()

    # Iterate through Hypso-2 captures
    clean_rows = []
    for _, (satobj, row) in enumerate(raw_dataset):

        satobj, has_error = pipeline.run(satobj, row, areas)

        if not has_error:
            store_capture(satobj, row, dir)
            clean_rows.append(row)

    clean_metadata = pd.DataFrame(clean_rows)
    clean_metadata.to_csv(os.path.join(DATA_DIR, target, dir, "clean_metadata.csv"), index=False)