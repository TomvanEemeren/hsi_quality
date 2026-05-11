import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.getcwd(),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.getcwd(),"src","hypso"))
sys.path.append(path)

from hsi_quality.data import Dataset, Storage
from hsi_quality.analysis import Resampler, select_box, intersect_captures
from hsi_quality.analysis import plot_metric, plot_resampled_images, plot_full_images
from hsi_quality.metrics import MeanSSIM, MvSSIM, QLambda

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="dubai", help="Name of the target.")
    parser.add_argument("--zone", type=str, default="40N", help="UTM zone of the target (e.g., 33s, 40n).")
    parser.add_argument("--directory", type=str, default="processed", help="Directory where the processed data is stored.")
    
    args = parser.parse_args()

    storage = Storage(target=args.location, data_dir=args.directory, level="l1d")

    dataset = Dataset(storage=storage)

    plot_full_images(dataset, save=True)

    # Select bounding box
    overlap = intersect_captures(dataset, zone=args.zone, visualize=False)
    area_extent = select_box(overlap)

    # Initialize the resampler for the given roi
    resampler = Resampler(bbox=area_extent, zone=args.zone)

    # Save visualizations of the selected area
    plot_resampled_images(dataset, resampler, save=True)

    # Plot and save the metric as a function of off-nadir angle
    plot_metric(dataset, MvSSIM(), resampler, save=True)
    plot_metric(dataset, MeanSSIM(), resampler, save=True)
    plot_metric(dataset, QLambda(), resampler, save=True)

if __name__ == "__main__":
    main()
