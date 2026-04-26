# %%

# Run this first!
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__),"src"))
sys.path.append(path)

path = os.path.abspath(os.path.join(os.path.dirname(__file__),"src","hypso"))
sys.path.append(path)

# %%
from hsi_quality.data_loader import load_nc_file

netcdf_file = "dubai_2026-01-14T07-09-26Z-l1d.nc"

satobj_h2 = load_nc_file(netcdf_file)

# %%
from hsi_quality.visualize import plot_rgb

l1d_cube = satobj_h2.l1d_cube

plot_rgb(satobj_h2, l1d_cube, save=True)
