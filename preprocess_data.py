import os
import sys
import argparse
import logging

path = os.path.abspath(os.path.join(os.path.dirname(__file__),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.path.dirname(__file__),"src","hypso"))
sys.path.append(path)

from hsi_quality.data import Dataset, DataLoader, Pipeline, Storage
from hsi_quality.utils import load_parameters
from hsi_quality.logger import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="dubai", help="Name of the target.")
    parser.add_argument("--full", action="store_true", help="Whether to apply full preprocessing pipeline.")
    parser.add_argument("--directory", type=str, default="processed", help="Directory to store the processed data.")
    
    args = parser.parse_args()

    logger = setup_logger("data_logger", log_file=f"preprocess_{args.location}.log", level=logging.INFO)

    # Load the raw data from the NTNU server
    data_loader = DataLoader(args.location)
    data_loader.load_data()

    # Preprocess the data
    cfg = load_parameters("preprocessing_params")
    pipeline = Pipeline(config=cfg, full=args.full)

    storage = Storage(target=args.location, data_dir="raw")

    raw_dataset = Dataset(storage=storage, pipeline=pipeline)
    raw_dataset.apply_pipeline(processed_dir=args.directory)

if __name__ == "__main__":
    main()

