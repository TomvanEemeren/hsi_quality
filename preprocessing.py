import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__),"src","hypso"))
sys.path.append(path)

from hypso import Hypso2
from hypso.write import write_l1d_nc_file

def preprocess_hyperspectral_image(nc_file: str, location: str):
    """
    Preprocess a hyperspectral image by loading the data, generating L1b and L1c cubes.

    Parameters:
    l1a_nc_file (str): Path to the L1a NetCDF file of the hyperspectral image.

    Returns:
    tuple: A tuple containing the L1a cube, L1b cube, and L1c cube.
    """

    # Hypso-2 capture
    h2_l1a_nc_file = os.path.join("datasets",location,"raw",nc_file)

    # Load Hypso-2 capture
    satobj_h2 = Hypso2(path=h2_l1a_nc_file, verbose=False)

    # Apply preprocessing steps
    satobj_h2.generate_l1b_cube(coeff_type="moved")
    satobj_h2.generate_l1c_cube()
    satobj_h2.generate_l1d_cube(use_direct_georef=True)

    # Access datacube
    l1d_cube = satobj_h2.l1d_cube

    # Check if processed directory exists
    os.makedirs(os.path.join("datasets",location,"processed"), exist_ok=True)

    # Save the processed data
    nc_file_new = os.path.splitext(nc_file)[0].removesuffix("-l1a")+"-l1d.nc"
    l1d_path = os.path.join("datasets",location,"processed",nc_file_new)
    write_l1d_nc_file(satobj=satobj_h2, l1d_path=l1d_path, overwrite=True)

    return l1d_cube, satobj_h2