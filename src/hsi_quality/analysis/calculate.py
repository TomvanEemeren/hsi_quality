import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from hsi_quality.data import Dataset
from hsi_quality.metrics import Metric
from .resample import Resampler
from .edge import EdgeDetector

from hsi_quality import RESULTS_DIR


def calculate_scores(dataset: Dataset, metric: Metric, resampler: Resampler = None, ed: EdgeDetector = None, save: bool = False):
    dataset = dataset.sort(by="off_nadir")

    scores = pd.DataFrame(columns=["location", "off_nadir", "score", "fwhm", "gsd", "metric"])

    if metric.name in ["GRD"]:

        reference_edge = None
        for idx, (satobj, metadata) in enumerate(tqdm(dataset, desc=f"Calculating {metric}", leave=False)):

            angle = metadata["off_nadir"]
            location = metadata["location_description"]
            cube = satobj.l1d_cube.values
            cloud_mask = satobj.cloud_mask

            cloud_coverage = np.mean(cloud_mask == 2) * 100
            if cloud_coverage > 5:
                continue
            
            edges, _, _ = ed.detect_edges(satobj)
            if len(edges) == 0:
                continue

            if idx == 0:
                reference_edge = ed.select_edge(edges)
                edge = reference_edge
            else:
                edge = ed.find_closest_edge(edges, reference_edge.longitude, reference_edge.latitude)

            score, info = metric.calculate(cube, edge, metadata)

            fwhm = info["fwhm"]
            gsd = info["gsd"]
            scores = pd.concat(
                [scores, pd.DataFrame([{"location": location, "off_nadir": angle, "score": score, "fwhm": fwhm, "gsd": gsd, "metric": metric.name}])],
                ignore_index=True,
            )


    elif metric.name in ["MvSSIM", "MeanSSIM", "QLambda"] and resampler is not None:

        reference = None
        for idx, (satobj, metadata) in enumerate(tqdm(dataset, desc=f"Calculating {metric}", leave=False)):

            angle = metadata["off_nadir"]
            location = metadata["location_description"]
            resampled_cube, cloud_mask = resampler.resample_capture(satobj)

            cloud_coverage = np.mean(cloud_mask == 2) * 100
            if cloud_coverage > 5:
                continue

            if reference is None:
                reference = resampled_cube
            
            score, info = metric.calculate(reference, resampled_cube)

            scores = pd.concat(
                [scores, pd.DataFrame([{"location": location, "off_nadir": angle, "score": score, "metric": metric.name}])],
                ignore_index=True,
            )

    if save:
        target = dataset["location_description"].unique()[0]
        base_dir = Path(RESULTS_DIR) / target
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{metric.name}"
        scores.to_csv(path.with_suffix(".csv"), index=False)

    return scores