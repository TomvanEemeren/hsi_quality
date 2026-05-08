import pandas as pd


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