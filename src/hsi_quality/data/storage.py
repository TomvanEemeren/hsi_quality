import os
import numpy as np
import pandas as pd
from pathlib import Path

from hypso import Hypso2
from hypso.write.l1d_nc_writer import write_l1d_nc_file

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")


class Storage:
    def __init__(self, target: str, level: str = "l1a", data_dir: str = "raw"):
        self.target = target
        self.data_dir = data_dir
        self.level = level

    def load_metadata(self) -> pd.DataFrame:
        metadata_path = os.path.join(DATA_DIR, self.target, self.data_dir, "metadata.csv")
        metadata = pd.read_csv(metadata_path)
        return metadata

    def load_capture(self, capture_name: str) -> Hypso2:
        # Load the data and store it in a Hypso2 object
        path = f"datasets/{self.target}/{self.data_dir}/{capture_name}-{self.level}.nc"
        satobj = Hypso2(path=path, verbose=False)

        if satobj.latitudes is None or satobj.longitudes is None:
            # Load the latitudes obtained from indirect georeferencing
            path = f"datasets/{self.target}/latitudes_indirect/{capture_name}.dat"
            latitudes = np.fromfile(path, dtype=np.float32)
            satobj.latitudes = latitudes.reshape(satobj.spatial_dimensions)

            # Load the longitudes obtained from indirect georeferencing
            path = f"datasets/{self.target}/longitudes_indirect/{capture_name}.dat"
            longitudes = np.fromfile(path, dtype=np.float32)
            satobj.longitudes = longitudes.reshape(satobj.spatial_dimensions)

        return satobj
    
    def store_capture(self, satobj: Hypso2, dir: str = "processed", level: str = "l1d"):
        store_path = os.path.join(DATA_DIR, self.target, dir)
        os.makedirs(store_path, exist_ok=True)

        # Save the capture
        capture_name = satobj.capture_name
        nc_file = f"{capture_name}-{level}.nc"
        l1d_path = os.path.join(store_path, nc_file)
        write_l1d_nc_file(satobj=satobj, l1d_path=l1d_path, overwrite=True)

    def store_metadata(self, metadata: pd.DataFrame, dir: str = "processed"):
        store_path = os.path.join(DATA_DIR, self.target, dir)
        os.makedirs(store_path, exist_ok=True)

        metadata_path = os.path.join(store_path, "metadata.csv")
        metadata.to_csv(metadata_path, index=False)