import pandas as pd

from hypso import Hypso2

def convert_timestamp(timestamp: str) -> str:
    format = "%Y-%m-%dT%H-%M-%SZ"

    # Convert the timestamp string to a datetime object
    timestamp = pd.to_datetime(timestamp, utc=True)
    timestamp = timestamp.strftime(format)

    return timestamp

def get_longitude_latitude(satobj: Hypso2, x: int, y: int) -> tuple[float, float]:
    longitude = satobj.longitudes_direct[y][x]
    latitude = satobj.latitudes_direct[y][x]

    return longitude, latitude