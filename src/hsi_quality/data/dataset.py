import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from hypso import Hypso2
from hsi_quality.utils import convert_timestamp

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")


@dataclass
class Capture:
    cube: np.ndarray
    capture_name: str
    longitudes: np.ndarray
    latitudes: np.ndarray
    off_nadir: float
    wavelengths: np.ndarray


class Dataset:
    def __init__(self, dataframe: pd.DataFrame, data_dir: str, level: str):
        self.df = dataframe
        self.data_dir = data_dir
        self.level = level

    def __len__(self):
        return len(self.df)


class RawDataset(Dataset):
    def __init__(self, dataframe, data_dir: str = "raw"):
        super().__init__(dataframe, data_dir, level="l1a")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get metadata for the capture
        target = row["location_description"]
        timestamp = row["timestamp_acquired_string"]
        timestamp = convert_timestamp(timestamp)

        # Create the path to the netcdf file
        capture_name = f"{target}_{timestamp}"

        # Load the capture
        satobj = self._load_capture(target, capture_name)

        return (satobj, row)

    def _load_capture(self, target: str, capture_name: str):
        # Load the data and store it in a Hypso2 object
        path = f"datasets/{target}/{self.data_dir}/{capture_name}-{self.level}.nc"
        satobj = Hypso2(path=path, verbose=False)

        # Load the latitudes obtained from indirect georeferencing
        path = f"datasets/{target}/latitudes_indirect/{capture_name}.dat"
        latitudes = np.fromfile(path, dtype=np.float32)
        satobj.latitudes = latitudes.reshape(satobj.spatial_dimensions)

        # Load the longitudes obtained from indirect georeferencing
        path = f"datasets/{target}/longitudes_indirect/{capture_name}.dat"
        longitudes = np.fromfile(path, dtype=np.float32)
        satobj.longitudes = longitudes.reshape(satobj.spatial_dimensions)

        return satobj


class ProcessedDataset(Dataset):
    def __init__(self, dataframe, data_dir: str = "processed"):
        super().__init__(dataframe, data_dir, level="l1d")

    def get_capture(self, target: str, timestamp: str):
        # Create the path to the netcdf file
        capture_name = f"{target}_{timestamp}"

        # Load the capture
        capture = self._load_capture(target, capture_name)

        return capture

    def filter(self, func):
        mask = self.df.apply(func, axis=1)
        filtered_df = self.df.loc[mask]
        return ProcessedDataset(filtered_df, data_dir=self.data_dir)

    def sort(self, by: str):
        sorted_df = self.df.sort_values(by=by)
        return ProcessedDataset(sorted_df, data_dir=self.data_dir)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.df[key].reset_index(drop=True)
        
        elif isinstance(key, int):
            row = self.df.iloc[key]

            # Get metadata for the capture
            target = row["location_description"]
            timestamp = row["timestamp_acquired_string"]
            timestamp = convert_timestamp(timestamp)

            # Create the path to the netcdf file
            capture_name = f"{target}_{timestamp}"

            # Load the capture
            capture = self._load_capture(target, capture_name)

            return capture

    def _load_capture(self, target: str, capture_name: str):
        # Load the data and store it in a Hypso2 object
        path = f"datasets/{target}/{self.data_dir}/{capture_name}-{self.level}.npz"
        data = np.load(path, allow_pickle=True)

        capture = Capture(
            cube=data["cube"],
            capture_name=data["capture_name"].item(),
            longitudes=data["longitudes"],
            latitudes=data["latitudes"],
            off_nadir=data["off_nadir"].item(),
            wavelengths=data["wavelengths"]
        )

        return capture