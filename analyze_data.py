# %%

# Run this first!
import os
import sys

path = os.path.abspath(os.path.join(os.getcwd(),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.getcwd(),"src","hypso"))
sys.path.append(path)

# %%
import pandas as pd
from hsi_quality.data import ProcessedDataset

# Load the metadata corresponding to a dataset
metadata = pd.read_csv("datasets/dubai/cleaned/clean_metadata.csv")

# Load the dataset object
dataset = ProcessedDataset(metadata, data_dir="cleaned")

# %%
from hsi_quality.visualize import plot_rgb

# Sort by date
dataset = dataset.sort(by="timestamp_acquired")

# Iterate through captures and save the RGB images
for idx in range(len(dataset)):
    satobj = dataset[idx]

    image = plot_rgb(satobj, save=True, verbose=False)

# %%
from hsi_quality.data import generate_area_def, resample_data

# Sort by off-nadir angle
dataset = dataset.sort(by="off_nadir")

# You have to select this manually for a given target
area_extent = (2.65e5, 2.6e6, 3.1e5, 2.7e6)

# Generate the desired area to resample to
area_def = generate_area_def(area_id = 'New area',
                            proj_id = 'id',
                            description = 'new area',
                            bbox = area_extent,
                            height = 512,
                            width = 512
                            )

# Resample all captures to the same area
resampled_data = resample_data(dataset, area_def)