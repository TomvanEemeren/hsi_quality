import os
import numpy as np
import pandas as pd
from pathlib import Path

from hypso import Hypso2
from hypso.write import write_l1d_nc_file
from hsi_quality.utils import convert_timestamp

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

class Dataset:
    def __init__(self, dataframe: pd.DataFrame, data_dir: str = "reflectance", level: str = "l1d"):
        self.df = dataframe
        self.data_dir = data_dir
        self.level = level

    def get_capture(self, target: str, timestamp: str):
        # Create the path to the netcdf file
        file_name = f"{target}_{timestamp}-{self.level}.nc"
        path = f"datasets/{target}/{self.data_dir}/{file_name}"

        # Load the capture
        satobj = self._load_capture(path)

        return satobj

    def store_capture(self, satobj: Hypso2, target: str, capture_name: str, dir: str = "processed", level: str = "l1d"):
        # Check if data directory exists, if not create it
        os.makedirs(os.path.join(DATA_DIR,target,dir), exist_ok=True)

        # Save the capture
        nc_file = f"{capture_name}-{level}.nc"
        l1d_path = os.path.join(DATA_DIR,target,dir,nc_file)
        write_l1d_nc_file(satobj=satobj, l1d_path=l1d_path, overwrite=True)

    def filter(self, func):
        mask = self.df.apply(func, axis=1)
        filtered_df = self.df.loc[mask]
        return Dataset(filtered_df, data_dir=self.data_dir, level=self.level)

    def sort(self, by: str):
        sorted_df = self.df.sort_values(by=by)
        return Dataset(sorted_df, data_dir=self.data_dir, level=self.level)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.df[key]
        
        elif isinstance(key, int):
            row = self.df.iloc[key]

            # Get metadata for the capture
            target = row["location_description"]
            timestamp = row["timestamp_acquired_string"]
            timestamp = convert_timestamp(timestamp)

            # Create the path to the netcdf file
            file_name = f"{target}_{timestamp}-{self.level}.nc"
            path = f"datasets/{target}/{self.data_dir}/{file_name}"

            # Load the capture
            satobj = self._load_capture(path)

            return satobj
    
    def _load_capture(self, path: str):

        # Load the data and store it in a Hypso2 object
        satobj = Hypso2(path=path, verbose=False)

        return satobj
    


