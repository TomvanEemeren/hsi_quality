import pandas as pd
from pathlib import Path    

from hsi_quality import RESULTS_DIR


def aggregate_scores(targets: list[str] | str, metric: str):
    if targets == "all":
        targets = [p.name for p in Path(RESULTS_DIR).iterdir() if p.is_dir()]
    elif isinstance(targets, str):
        targets = [targets]
    
    aggregated_scores = pd.DataFrame(columns=["location", "off_nadir", "score"])

    for target in targets:
        csv_file = Path(RESULTS_DIR) / target / f"{metric}.csv"
        if csv_file.exists():
            scores = pd.read_csv(csv_file)
            aggregated_scores = pd.concat([aggregated_scores, scores], ignore_index=True)

    return aggregated_scores