import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.getcwd(),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.getcwd(),"src","hypso"))
sys.path.append(path)

from hsi_quality.data import Dataset, Storage
from hsi_quality.analysis import Resampler, plot_metric, plot_band, plot_rgb
from hsi_quality.metrics import MeanSSIM, MvSSIM, QLambda

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="dubai", help="Name of the target.")
    parser.add_argument("--directory", type=str, default="processed", help="Directory where the processed data is stored.")
    
    args = parser.parse_args()

    storage = Storage(target=args.location, data_dir=args.directory, level="l1d")

    dataset = Dataset(storage=storage)

    plot_band(dataset, band=40, save=True)

    plot_rgb(dataset, save=True)

    # Define the region of interest for resampling in meters
    area_extent = (2.75e5, 2.6e6, 3.15e5, 2.7e6)

    # Initialize the resampler for the given roi
    resampler = Resampler(bbox=area_extent)

    # Plot and save the metric as a function of off-nadir angle
    plot_metric(dataset, MvSSIM(), resampler, save=True)
    plot_metric(dataset, MeanSSIM(), resampler, save=True)
    plot_metric(dataset, QLambda(), resampler, save=True)

if __name__ == "__main__":
    main()
