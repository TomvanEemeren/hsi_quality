# %%

# Run this first!
import os
import sys

path = os.path.abspath(os.path.join(os.getcwd(),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.getcwd(),"src","hypso"))
sys.path.append(path)

# %%
from hsi_quality.data import Dataset, Storage

loader = Storage(target="dubai", data_dir="cleaned", level="l1d")

dataset = Dataset(loader=loader)

# %%
from hsi_quality.analysis import plot_rgb

# Sort by date
dataset = dataset.sort(by="timestamp_acquired")

# Iterate through captures and save the RGB images
for idx in range(len(dataset)):
    satobj, _ = dataset[idx]

    image = plot_rgb(satobj, save=True, verbose=False)

# %%
from hsi_quality.analysis import Resampler

# Define the region of interest for resampling in meters
area_extent = (2.75e5, 2.6e6, 3.15e5, 2.7e6)

# Initialize the resampler for the given roi
resampler = Resampler(bbox=area_extent)

# %% 
from hsi_quality.analysis import plot_metric
from hsi_quality.metrics import MvSSIM

# Choose the metric to plot
metric = MvSSIM()

# Plot and save the metric as a function of off-nadir angle
plot_metric(dataset, metric, resampler, save=True)
