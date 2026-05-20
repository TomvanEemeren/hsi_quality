from matplotlib import pyplot as plt
from pathlib import Path

# Parameters for plotting
rcParams = {
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 7,
    'legend.title_fontsize': 7,
    'figure.titlesize': 7,
    # 'lines.linewidth': 0.8,
    'lines.markersize': 3,
}

plt.rcParams.update(rcParams)

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"