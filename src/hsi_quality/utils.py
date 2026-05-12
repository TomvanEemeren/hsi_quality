import yaml
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"


def convert_timestamp(timestamp: str) -> str:
    format = "%Y-%m-%dT%H-%M-%SZ"

    # Convert the timestamp to a specific format
    timestamp = pd.to_datetime(timestamp, utc=True)
    timestamp = timestamp.strftime(format)

    return timestamp


def convert_zone(zone_str: str) -> tuple[int, bool]:
    zone = int(zone_str[:-1])
    south = zone_str.endswith("s")

    return zone, south


def normalize_cube(cube: xr.DataArray, method: str = "min_max") -> xr.DataArray:
    data = cube.values

    if method == "min_max":
        min_val = np.min(data, axis=(0, 1))
        max_val = np.max(data, axis=(0, 1))
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)

    elif method == "percentile":
        p2 = np.percentile(data, 2, axis=(0, 1))
        p98 = np.percentile(data, 98, axis=(0, 1))
        normalized_data = np.clip((data - p2) / (p98 - p2 + 1e-8), 0, 1)

    normalized_cube = xr.DataArray(normalized_data, dims=["y", "x", "band"])
    normalized_cube.attrs.update(cube.attrs)

    return normalized_cube


def clip_cube(cube: xr.DataArray) -> xr.DataArray:
    clipped_data = np.clip(cube.values, 0, 1)
    clipped_cube = xr.DataArray(clipped_data, dims=cube.dims)
    clipped_cube.attrs.update(cube.attrs)

    return clipped_cube


def load_parameters(config_name: str) -> dict:
    path = CONFIG_DIR / f"{config_name}.yaml"

    with open(path, "r") as file:
        cfg = yaml.safe_load(file)

    return cfg