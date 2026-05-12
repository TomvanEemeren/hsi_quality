import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "datasets"

logger = logging.getLogger("data_logger")

class DataLoader:
    BASE_URL = "http://129.241.2.147:8009"

    def __init__(self, location: str):
        self.location = location

        self.base_dir = Path(DATA_DIR) / location
        self.raw_dir = self.base_dir / "raw"
        self.radiance_dir = self.base_dir / "radiance"
        self.cloud_dir = self.raw_dir / "cloud_labels"
        self.lat_dir = self.base_dir / "latitudes_indirect"
        self.lon_dir = self.base_dir / "longitudes_indirect"

        self.session = requests.Session()

        self._create_directories()

    def load_data(self):
        """
        Loads the images and metadata for Hypso-2 from the local server at the NTNU.
        """
        metadata_list = []

        existing_captures = {
            file.stem.removesuffix("-l1a")
            for file in self.raw_dir.glob("*.nc")
        }

        location_url = f"{self.BASE_URL}/{self.location}/"

        # Open the URL to main directory of a location
        response = self.session.get(location_url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Iterate through the captures
        for link in tqdm(soup.find_all("a"), desc="Loading data from server"):
            href = link.get("href")

            if not href:
                 continue
            
            capture_name = href.strip("/")

            # Skip if raw file already exists
            if capture_name in existing_captures:
                logger.info(f"Skipping {capture_name} - already downloaded")
                continue
            
            capture_url = urljoin(location_url, href)

            try:
                metadata = self._load_metadata(capture_url, capture_name)

                self._load_cloud_labels(capture_url, capture_name)
                self._load_lonlat_indirect(capture_url, capture_name)
                self._load_radiance_image(capture_url, capture_name)
                self._load_raw_data(capture_url, capture_name)

                metadata_list.append(metadata)

            except requests.RequestException as e:
                logger.warning(f"Could not load {capture_url}: {e}")

        if metadata_list:
            metadata_path = self.raw_dir / "metadata.csv"
            pd.DataFrame(metadata_list).to_csv(metadata_path, index=False)
            logger.debug(f"Saved metadata to {metadata_path}")

    def _create_directories(self):
        for directory in [
            self.base_dir,
            self.raw_dir,
            self.radiance_dir,
            self.cloud_dir,
            self.lat_dir,
            self.lon_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _download_file(self, url: str, destination: Path, stream: bool = False):
        with self.session.get(url, timeout=60, stream=stream) as response:
            response.raise_for_status()

            tmp_path = destination.with_suffix(destination.suffix + ".part")

            with open(tmp_path, "wb") as f:
                if stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                else:
                    f.write(response.content)

            tmp_path.replace(destination)

        logger.debug(f"Saved {destination.name} to {destination}")


    def _load_metadata(self, url: str, capture_name: str):
        meta_url = urljoin(url, f"{capture_name}-meta.json")

        response = self.session.get(meta_url, timeout=30)
        response.raise_for_status()

        return response.json()

    def _load_raw_data(self, url: str, capture_name: str):
        raw_url = urljoin(url, f"{capture_name}-l1a.nc")
        destination = self.raw_dir / f"{capture_name}-l1a.nc"

        self._download_file(raw_url, destination, stream=True)

    def _load_radiance_image(self, url: str, capture_name: str):
        image_url = urljoin(url, f"{capture_name}-scaled-radiance.png")
        destination = self.radiance_dir / f"{capture_name}.png"

        self._download_file(image_url, destination)

    def _load_cloud_labels(self, url: str, capture_name: str):
        cloud_url = urljoin(url, "processing-temp/sea-land-cloud.labels")
        destination = self.cloud_dir / f"{capture_name}.labels"

        self._download_file(cloud_url, destination)

    def _load_lonlat_indirect(self, url: str, capture_name: str):
        latitude_url = urljoin(url, "processing-temp/latitudes_indirectgeoref.dat")
        longitude_url = urljoin(url, "processing-temp/longitudes_indirectgeoref.dat")

        lat_destination = self.lat_dir / f"{capture_name}.dat"
        lon_destination = self.lon_dir / f"{capture_name}.dat"

        self._download_file(latitude_url, lat_destination)
        self._download_file(longitude_url, lon_destination)
