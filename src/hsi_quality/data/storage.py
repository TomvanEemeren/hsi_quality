import numpy as np
import pandas as pd
from pathlib import Path

from hypso import Hypso2
from hypso.write.l1d_nc_writer import write_l1d_nc_file

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "datasets"


class Storage:
    def __init__(self, target: str, level: str = "l1a", data_dir: str = "raw"):
        self.target = target
        self.data_dir = data_dir
        self.level = level

    def load_metadata(self) -> pd.DataFrame:
        metadata_path = DATA_DIR / self.target / self.data_dir / "metadata.csv"
        metadata = pd.read_csv(metadata_path)
        return metadata

    def load_capture(self, capture_name: str) -> Hypso2:
        # Load the data and store it in a Hypso2 object
        path = DATA_DIR / self.target / self.data_dir / f"{capture_name}-{self.level}.nc"
        satobj = Hypso2(path=path, verbose=False)

        if satobj.latitudes is None or satobj.longitudes is None:
            # Load the latitudes obtained from indirect georeferencing
            path = DATA_DIR / self.target / "latitudes_indirect" / f"{capture_name}.dat"
            latitudes = np.fromfile(path, dtype=np.float32)
            satobj.latitudes = latitudes.reshape(satobj.spatial_dimensions)

            # Load the longitudes obtained from indirect georeferencing
            path = DATA_DIR / self.target / "longitudes_indirect" / f"{capture_name}.dat"
            longitudes = np.fromfile(path, dtype=np.float32)
            satobj.longitudes = longitudes.reshape(satobj.spatial_dimensions)

        if satobj.cloud_mask is None:
            path = DATA_DIR / self.target / self.data_dir / "cloud_labels" / f"{capture_name}.labels"
            cloud_labels = np.fromfile(path, dtype=np.uint8)
            satobj.cloud_mask = cloud_labels.reshape(satobj.spatial_dimensions)

        return satobj

    def store_capture(self, satobj: Hypso2, dir: str = "processed", level: str = "l1d"):
        store_path = DATA_DIR / self.target / dir
        store_path.mkdir(parents=True, exist_ok=True)

        # Save the capture
        capture_name = satobj.capture_name
        nc_file = f"{capture_name}-{level}.nc"
        l1d_path = store_path / nc_file
        write_l1d_nc_file(satobj=satobj, l1d_path=l1d_path, overwrite=True)

        if satobj.cloud_mask is not None:
            cloud_labels_path = store_path / "cloud_labels"/ f"{capture_name}.labels"
            cloud_labels_path.mkdir(parents=True, exist_ok=True)
            np.asarray(satobj.cloud_mask.values, dtype=np.uint8).tofile(cloud_labels_path)

    def store_metadata(self, metadata: pd.DataFrame, dir: str = "processed"):
        store_path = DATA_DIR / self.target / dir
        store_path.mkdir(parents=True, exist_ok=True)

        metadata_path = store_path / "metadata.csv"
        metadata.to_csv(metadata_path, index=False)