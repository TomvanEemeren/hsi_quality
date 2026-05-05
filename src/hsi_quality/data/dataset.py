import os
from matplotlib import path
import pandas as pd
from pathlib import Path

from hypso import Hypso2
from hypso.write import write_l1d_nc_file
from hsi_quality.utils import convert_timestamp

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")


class Dataset:
    def __init__(self, dataframe: pd.DataFrame, data_dir: str, level: str):
        self.df = dataframe
        self.data_dir = data_dir
        self.level = level

    def get_capture(self, target: str, timestamp: str):
        # Create the path to the netcdf file
        capture_name = f"{target}_{timestamp}"

        # Load the capture
        satobj = self._load_capture(target, capture_name)

        return satobj
    
    def _load_capture(self, target: str, capture_name: str):
        # Load the data and store it in a Hypso2 object
        path = f"datasets/{target}/{self.data_dir}/{capture_name}-{self.level}.nc"
        satobj = Hypso2(path=path, verbose=False)
        return satobj
    
    def __len__(self):
        return len(self.df)

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
            satobj = self._load_capture(target, capture_name)

            return satobj


class RawDataset(Dataset):
    def __init__(self, dataframe, data_dir: str = "raw"):
        super().__init__(dataframe, data_dir, level="l1a")

    def _load_capture(self, target: str, capture_name: str):
        # Load the data and store it in a Hypso2 object
        path = f"datasets/{target}/{self.data_dir}/{capture_name}-{self.level}.nc"
        satobj = Hypso2(path=path, verbose=False)

        path = f"datasets/{target}/latitudes_indirect/{capture_name}.dat"
        satobj.latitudes = pd.read_csv(path, header=None).to_numpy()

        path = f"datasets/{target}/longitudes_indirect/{capture_name}.dat"
        satobj.longitudes = pd.read_csv(path, header=None).to_numpy()

        return satobj


class ProcessedDataset(Dataset):
    def __init__(self, dataframe, data_dir: str = "processed"):
        super().__init__(dataframe, data_dir, level="l1d")

    def store_capture(self, satobj: Hypso2, target: str):
        capture_name = satobj.capture_name

        # Check if data directory exists, if not create it
        os.makedirs(os.path.join(DATA_DIR,target,self.data_dir), exist_ok=True)

        # Save the capture
        nc_file = f"{capture_name}-{self.level}.nc"
        l1d_path = os.path.join(DATA_DIR,target,self.data_dir,nc_file)
        write_l1d_nc_file(satobj=satobj, l1d_path=l1d_path, overwrite=True)

    def filter(self, func):
        mask = self.df.apply(func, axis=1)
        filtered_df = self.df.loc[mask]
        return ProcessedDataset(filtered_df, data_dir=self.data_dir)
    
    def sort(self, by: str):
        sorted_df = self.df.sort_values(by=by)
        return ProcessedDataset(sorted_df, data_dir=self.data_dir)