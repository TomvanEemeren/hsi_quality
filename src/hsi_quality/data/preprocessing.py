import numpy as np
import pandas as pd
import logging

from hypso import Hypso2

logger = logging.getLogger("data_logger")

class Pipeline:
    def __init__(self, config: dict, full: bool = True):
        self.full = full
        self.cfg = config

    def run(self, satobj: Hypso2, metadata: pd.Series, areas: np.ndarray):
        has_error = False

        # Smile and keystone correction
        satobj.generate_l1b_cube(coeff_type="moved")

        # Radiometric calibration
        satobj.generate_l1c_cube()

        # ToA reflectance calculation
        satobj.generate_l1d_cube(use_direct_georef=True)

        if self.full:
            # Flipping
            satobj = self._flip_hyperspectral_image(satobj)

            # Anomaly detection
            has_error = self._detect_anomalies(satobj, metadata, areas)
            if has_error:
                logger.warning(f"Detected anomaly in {satobj.capture_name}")

        return satobj, has_error

    def _flip_hyperspectral_image(self, satobj: Hypso2):
        l1d_cube = satobj.l1d_cube
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes
        labels = satobj.cloud_mask

        # Flip the image to always have same orientation
        if satobj.longitudes[0][0] > satobj.longitudes[0][-1]:
            satobj.l1d_cube = l1d_cube[:, ::-1, :]
            satobj.latitudes = latitudes[:, ::-1]
            satobj.longitudes = longitudes[:, ::-1]
            satobj.cloud_mask = labels[:, ::-1]

            return satobj

        # No flipping needed
        return satobj

    def _detect_anomalies(self, satobj: Hypso2, metadata: pd.Series, areas: np.ndarray) -> bool:
        has_error = False
        cube = satobj.l1d_cube.values

        # Sun glint causes pixels to saturate
        sun_glint = metadata["overexposed_samples_percentage"] > self.cfg["overexposed_samples_threshold"]

        # Loss of orientation causes target misalignment
        star_tracker_blinded = metadata["star_tracker_blinded_percentage"] > self.cfg["star_tracker_blinded_threshold"]

        # Some off-nadir angles are unrealistic
        invalid_angle = metadata["off_nadir"] > self.cfg["off_nadir_threshold"] 

        # Sometimes the satellite captures a different area than the other captures in the same location
        invalid_area = self._compare_area(metadata, areas)

        # Downlink errors cause strange patterns in the image
        downlink_error = self._has_downlink_error(cube)

        has_error = sun_glint or star_tracker_blinded or invalid_angle or invalid_area or downlink_error

        return has_error
    
    def _compare_area(self, metadata: pd.Series, areas: np.ndarray) -> bool:
        aoi = metadata["area"]

        mean = np.mean(areas)
        std = np.std(areas)

        return abs(aoi - mean) > 2 * std

    def _has_downlink_error(self, cube: np.ndarray) -> bool:
        pixel_var = np.var(cube, axis=2)

        median_var = np.median(pixel_var)
        mad = np.median(np.abs(pixel_var - median_var))

        z = (pixel_var - median_var) / (mad + 1e-8)

        # Unusually high variance
        if np.any(z > 30):
            return True

        # Difference between neighboring rows
        vert_diff = np.diff(cube, axis=0)  # shape (H, W, C)

        # Magnitude of vertical change per column
        col_score = np.mean(np.abs(vert_diff), axis=(0, 2))  # (W,)

        median = np.median(col_score)
        mad = np.median(np.abs(col_score - median))

        z = (col_score - median) / (mad + 1e-8)

        # Unusually low variance
        anomalous = z < -4

        min_run = 5
        run = 0
        for a in anomalous:
            run = run + 1 if a else 0
            if run >= min_run:
                return True

        return False
