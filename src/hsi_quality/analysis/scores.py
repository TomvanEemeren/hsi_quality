import pandas as pd
from pathlib import Path    

from hsi_quality import RESULTS_DIR


def combine_scores(targets: list[str] | str, metric: str):
    if targets == "all":
        targets = [p.name for p in Path(RESULTS_DIR).iterdir() if p.is_dir() and p.name not in ["scores", "plots"]]
    elif isinstance(targets, str):
        targets = [targets]
    
    combined_scores = pd.DataFrame(columns=["location", "off_nadir", "score"])

    for target in targets:
        csv_file = Path(RESULTS_DIR) / target / f"{metric}.csv"
        if csv_file.exists():
            scores = pd.read_csv(csv_file)
            combined_scores = pd.concat([combined_scores, scores], ignore_index=True)
        
    return combined_scores


def normalize_scores(combined_scores: pd.DataFrame, scaling_factor: float = 100.0):
    metric = combined_scores["metric"].iloc[0]
    
    if metric in ["SSIMLambda", "MeanSSIM", "MvSSIM"]:
        combined_scores["norm_score"] = (1 - combined_scores["score"]) * scaling_factor
    elif metric == "GRD":
        # Normalize scores per location
        idx_min = combined_scores.groupby("location")["off_nadir"].idxmin()
        min_rows = combined_scores.loc[idx_min].set_index("location")
        baseline = min_rows["score"].astype(float) # * np.cos(np.deg2rad(min_rows["off_nadir"].astype(float)))**2
        combined_scores["norm_score"] = combined_scores["score"] / combined_scores["location"].map(baseline)

    return combined_scores


def remove_outliers(combined_scores: pd.DataFrame, threshold: float = 3.0):
    # Remove outliers based on z-score per location
    location_mean = combined_scores.groupby("location")["score"].transform("mean")
    location_std = combined_scores.groupby("location")["score"].transform("std")
    combined_scores["z_score"] = (combined_scores["score"] - location_mean) / location_std
    filtered_scores = combined_scores[combined_scores["z_score"].abs() < threshold].copy()
    filtered_scores.drop(columns=["z_score"], inplace=True)

    return filtered_scores


def remove_blacklist(combined_scores: pd.DataFrame, blacklist: list[str]):
    filtered_scores = combined_scores[~combined_scores["capture_name"].isin(blacklist)].copy()
    return filtered_scores


def save_combined_scores(combined_scores: pd.DataFrame):
    metric = combined_scores["metric"].iloc[0]
    base_dir = Path(RESULTS_DIR) / "scores"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{metric}_combined.csv"
    combined_scores.to_csv(path, index=False)


def load_combined_scores(metric: str) -> pd.DataFrame:
    path = Path(RESULTS_DIR) / "scores" / f"{metric}_combined.csv"
    if path.exists():
        combined_scores = pd.read_csv(path)
        return combined_scores
    else:
        raise FileNotFoundError(f"Combined scores file not found: {path}")