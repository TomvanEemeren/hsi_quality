import os
import h5py
import pandas as pd
from pathlib import Path

from hypso import Hypso2

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

def store_capture(satobj: Hypso2, row: pd.Series, dir: str = "processed"):
    target = row["location_description"]
    base_path = os.path.join(DATA_DIR, target, dir)
    os.makedirs(base_path, exist_ok=True)

    file_path = os.path.join(base_path, f"{satobj.capture_name}-l1d.h5")

    with h5py.File(file_path, "w") as f:
        f.create_dataset("l1d_cube", data=satobj.l1d_cube, compression="gzip")
        f.create_dataset("latitudes", data=satobj.latitudes, compression="gzip")
        f.create_dataset("longitudes", data=satobj.longitudes, compression="gzip")
        f.attrs["name"] = satobj.capture_name
        f.attrs["satellite"] = row["off_nadir"]