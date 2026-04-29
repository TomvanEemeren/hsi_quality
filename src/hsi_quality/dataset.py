import pandas as pd

from hypso import Hypso2
from hsi_quality.utils import convert_timestamp

class Dataset:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def get_capture(self, target: str, timestamp: str):
        # Create the path to the netcdf file
        file_name = f"{target}_{timestamp}-l1d.nc"
        path = f"datasets/{target}/processed/{file_name}"

        # Load the capture
        satobj = self._load_nc_file(path)

        return satobj

    def filter(self, func):
        mask = self.df.apply(func, axis=1)
        filtered_df = self.df.loc[mask]
        return Dataset(filtered_df)

    def sort(self, by: str):
        sorted_df = self.df.sort_values(by=by)
        return Dataset(sorted_df)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get metadata for the capture
        target = row["location_description"]
        timestamp = row["timestamp_acquired_string"]
        timestamp = convert_timestamp(timestamp)

        # Create the path to the netcdf file
        file_name = f"{target}_{timestamp}-l1d.nc"
        path = f"datasets/{target}/processed/{file_name}"

        # Load the capture
        satobj = self._load_nc_file(path)

        return satobj

    def _load_nc_file(self, path: str):

        # Load the data and store it in a Hypso2 object
        satobj = Hypso2(path=path, verbose=False)

        return satobj
    


