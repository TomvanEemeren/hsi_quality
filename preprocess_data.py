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

load_data_from_url("dubai")

# %%
from hsi_quality.preprocessing import preprocess_data

preprocess_data("dubai")

# %%
