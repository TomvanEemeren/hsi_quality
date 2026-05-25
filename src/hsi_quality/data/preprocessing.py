import logging
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

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
        name = satobj.capture_name
        print(f"Checking {name} for anomalies...")

        # Sun glint causes pixels to saturate
        sun_glint = metadata["overexposed_samples_percentage"] > self.cfg["overexposed_samples_threshold"]
        if sun_glint:
            logger.warning(f"Sun glint in {name}")

        # Loss of orientation causes target misalignment
        star_tracker_blinded = metadata["star_tracker_blinded_percentage"] > self.cfg["star_tracker_blinded_threshold"]
        if star_tracker_blinded:
            logger.warning(f"Star tracker blinded in {name}")

        # Some off-nadir angles are unrealistic
        invalid_angle = metadata["off_nadir"] > self.cfg["off_nadir_threshold"] 
        if invalid_angle:
            logger.warning(f"Invalid off-nadir angle in {name}")

        # Sometimes the satellite captures a different area than the other captures in the same location
        invalid_area = self._compare_target(satobj, metadata)
        if invalid_area:
            logger.warning(f"Invalid target in {name}")

        # Downlink errors cause strange patterns in the image
        downlink_error = self._has_downlink_error(cube)
        if downlink_error:
            logger.warning(f"Downlink error in {name}")

        has_error = sun_glint or star_tracker_blinded or invalid_angle or invalid_area or downlink_error

        return has_error
    
    def _compare_target(self, satobj: Hypso2, metadata: pd.Series) -> bool:
        target_longitude = metadata["target_longitude"]
        target_latitude = metadata["target_latitude"]

        if np.isnan(target_longitude) or np.isnan(target_latitude):
            return False
        
        actual_longitude = np.mean(satobj.longitudes)
        actual_latitude = np.mean(satobj.latitudes)

        deviation = np.sqrt((target_longitude - actual_longitude) ** 2 + (target_latitude - actual_latitude) ** 2)

        print(f"Target deviation: {deviation:.4f} degrees")

        return deviation > self.cfg["target_deviation_threshold"]

    def _has_downlink_error(self, cube: np.ndarray) -> bool:
        cube = cube[:,:,10:]
        smooth = savgol_filter(cube, 11, 2, axis=2)

        residual = cube - smooth

        signal_power = np.mean(smooth**2, axis=2)

        noise_power = np.mean(residual**2, axis=2)

        normalized_scores = noise_power / (signal_power + 1e-8)

        image_score = np.percentile(normalized_scores, 99)

        return image_score > self.cfg["downlink_error_threshold"]
