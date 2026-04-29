import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.path.dirname(__file__),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.path.dirname(__file__),"src","hypso"))
sys.path.append(path)

from hsi_quality.data_loader import load_data_from_url
from hsi_quality.preprocessing import preprocess_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="dubai", help="Name of the target.")
    parser.add_argument("--full", action="store_true", help="Whether to apply full preprocessing pipeline.")
    
    args = parser.parse_args()

    # Load the raw data from the NTNU server
    load_data_from_url(args.location)

    # Preprocess the data
    preprocess_data(args.location, full=args.full)

if __name__ == "__main__":
    main()

