import os
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

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
    location = location
    all_metadata = []

    os.makedirs(os.path.join(DATA_DIR, location), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, location, "raw"), exist_ok=True)

    raw_dir = os.path.join(DATA_DIR, location, "raw")
    existing_files = set(os.listdir(raw_dir)) if os.path.exists(raw_dir) else set()

    # Open the URL to main directory of a location
    url = f"http://129.241.2.147:8009/{location}/"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Store all captures for a specific location
    soup = BeautifulSoup(response.text, "html.parser")
    captures = soup.find_all("a")

    # Iterate through the captures
    for link in captures:
        href = link.get("href")
        if href:
            capture_name = href.strip("/")

            # Skip if raw file already exists
            if capture_name in existing_files:
                print(f"Skipping {capture_name} - already downloaded")
                continue

            capture_url = urljoin(url, href)
            try:
                response = requests.get(capture_url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                metadata = load_metadata(capture_url, soup)

                load_raw_data(location, capture_url, capture_name, soup)

                load_radiance_image(location, capture_url, capture_name, soup)

                load_cloud_labels(location, capture_url, capture_name)

                all_metadata.append(metadata)

            except requests.exceptions.RequestException as e:
                print(f"Could not process {capture_url}: {e}")

    if all_metadata:
        df = pd.DataFrame(all_metadata)
        df.to_csv(os.path.join(DATA_DIR, location, "metadata.csv"), index=False)
        print(f"Saved metadata to {os.path.join(DATA_DIR, location, 'metadata.csv')}.")

def load_metadata(url: str, soup: BeautifulSoup):
    # Find the link to the meta data file
    meta_link = soup.find("a", href=lambda h: h and h.endswith("-meta.json"))

    if meta_link:
        meta_url = urljoin(url, meta_link.get("href"))
        response = requests.get(meta_url, timeout=30)
        response.raise_for_status()
    
    return response.json()

def load_raw_data(location: str, url: str, capture_name: str, soup: BeautifulSoup):
    # Find the link to the raw data file
    raw_link = soup.find("a", href=lambda h: h and h.endswith("-l1a.nc"))

    if raw_link:
        os.makedirs(os.path.join(DATA_DIR, location, "raw"), exist_ok=True)

        # Save raw data (streaming download for large files)
        raw_url = urljoin(url, raw_link.get("href"))
        raw_path = os.path.join(DATA_DIR, location, "raw", capture_name + "-l1a.nc")
        tmp_path = raw_path + ".part"

        with requests.get(raw_url, timeout=60, stream=True) as raw_response:
                            raw_response.raise_for_status()
                            with open(tmp_path, "wb") as f:
                                for chunk in raw_response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                                    if chunk:
                                        f.write(chunk)

        os.replace(tmp_path, raw_path)  # atomic rename when complete
        print(f"Saved raw data to {raw_path}")

def load_radiance_image(location: str, url: str, capture_name: str, soup: BeautifulSoup):
    # Find the link to the hyperspectral image file
    image_link = soup.find("a", href=lambda h: h and h.endswith("-scaled-radiance.png"))

    if image_link:
        os.makedirs(os.path.join(DATA_DIR, location, "radiance"), exist_ok=True)

        # Save image
        image_url = urljoin(url, image_link.get("href"))
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        image = response.content
        with open(os.path.join(DATA_DIR, location, "radiance", capture_name + ".png"), "wb") as f:
            f.write(image)
        print(f"Saved image to {os.path.join(DATA_DIR, location, 'radiance', capture_name + '.png')}")

def load_cloud_labels(location: str, url: str, capture_name: str):
    os.makedirs(os.path.join(DATA_DIR, location, "cloud_labels"), exist_ok=True)

    # Save cloud labels image
    cloud_url = urljoin(url, "processing-temp/sea-land-cloud.labels")
    response = requests.get(cloud_url, timeout=30)
    response.raise_for_status()

    cloud_labels = response.content
    with open(os.path.join(DATA_DIR, location, "cloud_labels", capture_name + ".labels"), "wb") as f:
        f.write(cloud_labels)
    print(f"Saved cloud labels to {os.path.join(DATA_DIR, location, 'cloud_labels', capture_name + '.labels')}")