import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.getcwd(),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.getcwd(),"src","hypso"))
sys.path.append(path)

from hsi_quality.data import Dataset, Storage, Resampler
from hsi_quality.plotting import select_box, intersect_captures
from hsi_quality.plotting import plot_metric, plot_resampled_images, plot_full_images, plot_cloud_images
from hsi_quality.metrics import MeanSSIM, MvSSIM, QLambda, GRD
from hsi_quality.utils import load_parameters

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
    plot_cloud_images(dataset, resampler, save=True)

    # Plot and save the metric as a function of off-nadir angle
    cfg = load_parameters("metric_params")
    plot_metric(dataset, MvSSIM(params=cfg["MvSSIM"]), resampler, save=True)
    plot_metric(dataset, MeanSSIM(params=cfg["MeanSSIM"]), resampler, save=True)
    plot_metric(dataset, QLambda(params=cfg["Qlambda"]), resampler, save=True)
    plot_metric(dataset, GRD(params=cfg["GRD"]), resampler, save=True)

if __name__ == "__main__":
    main()
