import pandas as pd

from hsi_quality.utils import convert_timestamp
from hsi_quality.data import Pipeline, DataLoader

class Dataset:
    def __init__(self, loader: DataLoader, pipeline: Pipeline = None, metadata: pd.DataFrame = None):
        self.loader = loader
        self.pipeline = pipeline

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = self.loader.load_metadata()

    def apply_pipeline(self, processed_dir: str = "processed"):
        if self.pipeline is not None:
            areas = self.metadata["area"].to_numpy()

            # Iterate through Hypso-2 captures
            clean_rows = []
            for idx in range(len(self.metadata)):
                satobj = self._get_capture(idx)
                row = self.metadata.iloc[idx]
            
                satobj, has_error = self.pipeline.run(satobj, row, areas)

                if not has_error:
                    self.loader.store_capture(satobj, dir=processed_dir)
                    clean_rows.append(row)

                clean_metadata = pd.DataFrame(clean_rows)
                self.loader.store_metadata(clean_metadata, dir=processed_dir)

    def filter(self, func):
        mask = self.metadata.apply(func, axis=1)
        filtered_metadata = self.metadata.loc[mask]
        return Dataset(self.loader, self.pipeline, filtered_metadata)

    def sort(self, by: str):
        sorted_metadata = self.metadata.sort_values(by=by)
        return Dataset(self.loader, self.pipeline, sorted_metadata)

    def _get_capture(self, idx):
        row = self.metadata.iloc[idx]

        # Get metadata for the capture
        target = row["location_description"]
        timestamp = row["timestamp_acquired_string"]
        timestamp = convert_timestamp(timestamp)

        # Create the path to the netcdf file
        capture_name = f"{target}_{timestamp}"

        # Load the capture
        satobj = self.loader.load_capture(capture_name)

        return satobj

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        return self._get_capture(idx)