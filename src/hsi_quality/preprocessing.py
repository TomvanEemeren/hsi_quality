import os
from pathlib import Path

from hypso import Hypso2
from hypso.write import write_l1d_nc_file

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(ROOT_DIR, "datasets")

def preprocess_data(target: str):
    """
    Preprocess multiple hyperspectral images for a specific target location.
    The data is loaded from the raw directory and saved to the processed directory.

    Args:
        target (str): The target location for which to preprocess the data.
    """

    # Path to Hypso-2 captures
    raw_dir = os.path.join(DATA_DIR, target, "raw")
    nc_files = os.listdir(raw_dir)

    # Iterate through Hypso-2 captures
    for nc_file in nc_files:
        # Preprocess the hyperspectral image
        _, _ = preprocess_hyperspectral_image(
            nc_file=nc_file,
            target=target
        )

def preprocess_hyperspectral_image(nc_file: str, target: str):
    """
    Preprocess a hyperspectral image by loading the data, generating L1b and L1c cubes.

    Args:
        nc_file (str): The name of the NetCDF file containing the hyperspectral image data.
        target (str): The target location for which to preprocess the data.

    Returns:
        tuple: A tuple containing the L1a cube, L1b cube, and L1c cube.
    """

    # Hypso-2 capture
    h2_l1a_nc_file = os.path.join(DATA_DIR,target,"raw",nc_file)

    # Load Hypso-2 capture
    satobj_h2 = Hypso2(path=h2_l1a_nc_file, verbose=False)

    # Apply preprocessing steps
    satobj_h2.generate_l1b_cube(coeff_type="moved")
    satobj_h2.generate_l1c_cube()
    satobj_h2.generate_l1d_cube(use_direct_georef=True)

    # Access datacube
    l1d_cube = satobj_h2.l1d_cube

    # Check if processed directory exists
    os.makedirs(os.path.join(DATA_DIR,target,"processed"), exist_ok=True)

    # Save the processed data
    nc_file_new = os.path.splitext(nc_file)[0].removesuffix("-l1a")+"-l1d.nc"
    l1d_path = os.path.join(DATA_DIR,target,"processed",nc_file_new)
    write_l1d_nc_file(satobj=satobj_h2, l1d_path=l1d_path, overwrite=True)

    return l1d_cube, satobj_h2