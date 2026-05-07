import pandas as pd

from hsi_quality.utils import convert_timestamp
from .storage import Storage
from .preprocessing import Pipeline


class Dataset:
    def __init__(self, storage: Storage, pipeline: Pipeline = None, metadata: pd.DataFrame = None):
        self.storage = storage
        self.pipeline = pipeline

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = self.storage.load_metadata()

    def apply_pipeline(self, processed_dir: str = "processed"):
        if self.pipeline is not None:
            areas = self.metadata["area"].to_numpy()

            # Iterate through Hypso-2 captures
            clean_rows = []
            for idx in range(len(self.metadata)):
                capture_name, row = self._get_capture_name(idx)
                satobj = self.get_capture(capture_name)

                satobj, has_error = self.pipeline.run(satobj, row, areas)

                if not has_error:
                    self.storage.store_capture(satobj, dir=processed_dir, level="l1d")
                    clean_rows.append(row)

                clean_metadata = pd.DataFrame(clean_rows)
                self.storage.store_metadata(clean_metadata, dir=processed_dir)

    def filter(self, func):
        mask = self.metadata.apply(func, axis=1)
        filtered_metadata = self.metadata.loc[mask]
        return Dataset(self.storage, self.pipeline, filtered_metadata)

    def sort(self, by: str):
        sorted_metadata = self.metadata.sort_values(by=by)
        return Dataset(self.storage, self.pipeline, sorted_metadata)

    def get_capture(self, capture_name: str):
        satobj = self.storage.load_capture(capture_name)
        return satobj

    def _get_capture_name(self, idx):
        row = self.metadata.iloc[idx]

        # Get metadata for the capture
        target = row["location_description"]
        timestamp = row["timestamp_acquired_string"]
        timestamp = convert_timestamp(timestamp)

        capture_name = f"{target}_{timestamp}"
        return capture_name, row

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, key):
        if isinstance(key, int):
            capture_name, row = self._get_capture_name(key)
            satobj = self.get_capture(capture_name)
            return satobj, row
        elif isinstance(key, str):
            return self.metadata[key].reset_index(drop=True)