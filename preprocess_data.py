# %%

# Run this first!
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.path.dirname(__file__),"src","hypso"))
sys.path.append(path)

# %%
from hsi_quality.data_loader import load_data_from_url

# load_data_from_url("dubai")

# %%
from hsi_quality.preprocessing import preprocess_hyperspectral_image

# Hypso-2 capture
nc_file = "dubai_2026-01-14T07-09-26Z-l1a.nc"

# Preprocess the hyperspectral image and generate L1d datacube
l1d_cube, satobj_h2 = preprocess_hyperspectral_image(nc_file=nc_file, location="dubai")
