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
from hsi_quality.data import Dataset

# Load the metadata corresponding to a dataset
metadata = pd.read_csv("datasets/dubai/cleaned/clean_metadata.csv")

# Load the dataset object
dataset = Dataset(metadata, data_dir="cleaned")

# %%
from hsi_quality.visualize import plot_rgb

# Sort by date
dataset = dataset.sort(by="timestamp_acquired")

# Iterate through captures and save the RGB images
for idx in range(len(dataset)):
    satobj = dataset[idx]

    image = plot_rgb(satobj, save=True, verbose=False)

# %%
