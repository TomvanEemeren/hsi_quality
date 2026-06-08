import numpy as np
import pandas as pd
from pathlib import Path    

from hsi_quality import RESULTS_DIR


def aggregate_scores(targets: list[str] | str, metric: str):
    if targets == "all":
        targets = [p.name for p in Path(RESULTS_DIR).iterdir() if p.is_dir() and p.name != "plots"]
    elif isinstance(targets, str):
        targets = [targets]
    
    aggregated_scores = pd.DataFrame(columns=["location", "off_nadir", "score"])

    for target in targets:
        csv_file = Path(RESULTS_DIR) / target / f"{metric}.csv"
        if csv_file.exists():
            scores = pd.read_csv(csv_file)
            aggregated_scores = pd.concat([aggregated_scores, scores], ignore_index=True)

    if metric == "GRD":
        # Normalize scores per location
        idx_min = aggregated_scores.groupby("location")["off_nadir"].idxmin()
        min_rows = aggregated_scores.loc[idx_min].set_index("location")
        baseline = min_rows["score"].astype(float) # * np.cos(np.deg2rad(min_rows["off_nadir"].astype(float)))**2
        aggregated_scores["norm_score"] = aggregated_scores["score"] / aggregated_scores["location"].map(baseline)

    return aggregated_scores