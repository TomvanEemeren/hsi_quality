import pandas as pd


def convert_timestamp(timestamp: str) -> str:
    format = "%Y-%m-%dT%H-%M-%SZ"

    # Convert the timestamp to a specific format
    timestamp = pd.to_datetime(timestamp, utc=True)
    timestamp = timestamp.strftime(format)

    return timestamp


def convert_zone(zone: str) -> tuple[int, bool]:
    zone = int(zone[:-1])
    south = zone.endswith("s")

    return zone, south