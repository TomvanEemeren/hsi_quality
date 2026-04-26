import os
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timezone

from hypso import Hypso2

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

def load_data_from_url(location: str):
    """
    Loads the images and metadata from the local server at the NTNU 
    containing the image database for HYPSO 2.

    Args:
        location (str): The location of the images to load. Should match the 
                        name of the location in the database (e.g., "dubai").
    """

    # Check that the location is provided
    if location is None:
        raise ValueError("Location must be provided.")
    
    # Make a directory to save the data if it doesn't exist already
    os.makedirs(os.path.join(DATA_DIR, location, "radiance"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, location, "raw"), exist_ok=True)
    
    # Open the URL to main directory of a location
    url = f"http://129.241.2.147:8009/{location}/"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Store the HTML content of the page in a data structure
    soup = BeautifulSoup(response.text, "html.parser")

    # List to store meta data
    all_meta_data = []

    # Iterate through the URLs of the subdirectories and collect the data
    for link in soup.find_all("a"):
        href = link.get("href")

        if href:
            dir_url = urljoin(url, href)

            try:
                # Open the URL of the subdirectory
                dir_response = requests.get(dir_url, timeout=30)
                dir_response.raise_for_status()
                dir_soup = BeautifulSoup(dir_response.text, "html.parser")

                # Find the link to the meta data file
                meta_link = dir_soup.find("a", href=lambda h: h and h.endswith("-meta.json"))

                # Find the link to the hyperspectral image file
                image_link = dir_soup.find("a", href=lambda h: h and h.endswith("-scaled-radiance.png"))

                # Find the link to the raw data file
                raw_link = dir_soup.find("a", href=lambda h: h and h.endswith("-l1a.nc"))

                # Collect the data
                if meta_link and image_link and raw_link:
                    meta_url = urljoin(dir_url, meta_link.get("href"))
                    meta_response = requests.get(meta_url, timeout=30)
                    meta_response.raise_for_status()
                    meta_data = meta_response.json()
                    all_meta_data.append(meta_data)

                    timestamp = meta_data["timestamp_acquired"]
                    utc_dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
                    time_str = utc_dt.strftime("%Y-%m-%dT%H-%M-%SZ")

                    # Save image
                    image_url = urljoin(dir_url, image_link.get("href"))
                    image_response = requests.get(image_url, timeout=30)
                    image_response.raise_for_status()

                    image = image_response.content
                    with open(os.path.join(DATA_DIR, location, "radiance", time_str + ".png"), "wb") as f:
                        f.write(image)
                    print(f"Saved image to {os.path.join(DATA_DIR, location, 'radiance', time_str + '.png')}")

                    # Save raw data (streaming download for large files)
                    if raw_link:
                        raw_url = urljoin(dir_url, raw_link.get("href"))
                        raw_path = os.path.join(DATA_DIR, location, "raw", f"{location}_{time_str}-l1a.nc")
                        tmp_path = raw_path + ".part"

                        with requests.get(raw_url, timeout=60, stream=True) as raw_response:
                            raw_response.raise_for_status()
                            with open(tmp_path, "wb") as f:
                                for chunk in raw_response.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
                                    if chunk:
                                        f.write(chunk)

                        os.replace(tmp_path, raw_path)  # atomic rename when complete
                        print(f"Saved raw data to {raw_path}")

            except requests.exceptions.RequestException as e:
                print(f"Could not process {dir_url}: {e}")

    # Save metadata to a csv file
    if all_meta_data:
        df = pd.DataFrame(all_meta_data)
        df.to_csv(os.path.join(DATA_DIR, location, "metadata.csv"), index=False)
        print(f"Saved metadata to {os.path.join(DATA_DIR, location, 'metadata.csv')}.")

def load_nc_file(file_name: str):
    """
    Loads the HYPSO-2 hyperspectral captures at any level:
    - L1a (raw data)
    - L1b (top-of-atmosphere radiance)
    - L1c (top-of-atmosphere radiance with georeferencing)
    - L1d (top-of-atmosphere reflectance with georeferencing)

    Args:
        file_name (str): The name of the netCDF file to load. Should be in the format
                          "{location}_{timestamp}-{level}.nc"
    Returns:
        Hypso2 object containing the data from the netCDF file.
    """
    if not file_name:
        raise ValueError("file_name must be provided.")
    
    # Split filename into target + rest
    fields = file_name.removesuffix(".nc").split("_")

    # Get only the target (location)
    if len(fields) == 2:
        target, _ = fields
    else:
        raise ValueError("Unexpected filename format")

    # Create the file path
    file_path = os.path.join(DATA_DIR, target, "processed", file_name)

    # Load the data and store it in a Hypso2 object
    satobj_h2 = Hypso2(path=file_path, verbose=False)

    return satobj_h2