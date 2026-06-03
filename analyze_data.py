import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.getcwd(),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.getcwd(),"src","hypso"))
sys.path.append(path)

from hsi_quality.data import Dataset, Storage
from hsi_quality.analysis import Resampler, EdgeDetector, calculate_scores
from hsi_quality.plotting import select_box, intersect_captures
from hsi_quality.plotting import plot_resampled_images, plot_full_images, plot_metric, plot_resampled_cloud_images
from hsi_quality.plotting import make_grd_plots
from hsi_quality.metrics import MeanSSIM, MvSSIM, QLambda, GRD
from hsi_quality.utils import load_parameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="dubai", help="Name of the target.")
    parser.add_argument("--zone", type=str, default="40N", help="UTM zone of the target (e.g., 33s, 40n).")
    parser.add_argument("--directory", type=str, default="processed", help="Directory where the processed data is stored.")
    parser.add_argument("--metric", type=str, default="GRD")
    parser.add_argument("--images", action="store_true")

    args = parser.parse_args()

    storage = Storage(target=args.location, data_dir=args.directory, level="l1d")

    dataset = Dataset(storage=storage)

    cfg = load_parameters("metric_params")
    
    if args.images:
        plot_full_images(dataset, save=True)

    if args.metric == "GRD":
        grd = GRD(params=cfg["GRD"])

        # Initialize the edge detector
        detector_params = load_parameters("edge_params")
        ed = EdgeDetector(params=detector_params)

        scores = calculate_scores(dataset, grd, ed=ed, save=True)
        plot_metric(scores, save=True)

        make_grd_plots(dataset, ed, grd, save=True)

    elif args.metric == "SSIM":

        # Select bounding box
        overlap = intersect_captures(dataset, zone=args.zone, visualize=False)
        area_extent = select_box(overlap)

        # Initialize the resampler for the given roi
        resampler = Resampler(bbox=area_extent, zone=args.zone)

        # Save visualizations of the selected area
        if args.images:
            plot_resampled_images(dataset, resampler, save=True)
            plot_resampled_cloud_images(dataset, resampler, save=True)

        scores = calculate_scores(dataset, MvSSIM(params=cfg["MvSSIM"]), resampler=resampler, save=True)
        plot_metric(scores, save=True)

        scores = calculate_scores(dataset, MeanSSIM(params=cfg["MeanSSIM"]), resampler=resampler, save=True)
        plot_metric(scores, save=True)

        scores = calculate_scores(dataset, QLambda(params=cfg["Qlambda"]), resampler=resampler, save=True)
        plot_metric(scores, save=True)

if __name__ == "__main__":
    main()
