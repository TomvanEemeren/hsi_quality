from hypso import Hypso2
from hypso.geometry.nearest import get_nearest_pixel

def get_spectrum(satobj_h2: Hypso2, latitude: float = None, longitude: float = None, x: int = None, y: int = None):
    """
    Compute the spectrum of the wavelengths for a specific pixel.

    Args:
        satobj_h2 (Hypso2): The Hypso2 satellite object.
        latitude (float, optional): The latitude of the pixel. Defaults to None.
        longitude (float, optional): The longitude of the pixel. Defaults to None.
        x (int, optional): The x-coordinate of the pixel. Defaults to None.
        y (int, optional): The y-coordinate of the pixel. Defaults to None.

    Returns:
        tuple: A tuple containing the spectrum and the corresponding wavelengths.
    """

    # Extract the datacube
    l1d_cube = satobj_h2.l1d_cube

    if latitude is not None and longitude is not None:
        # Get the corresponding pixel index for a given latitude and longitude
        idx = get_nearest_pixel(target_latitude=latitude, 
                                target_longitude=longitude,
                                latitudes=satobj_h2.latitudes_direct,
                                longitudes=satobj_h2.longitudes_direct)
    elif x is not None and y is not None:
        # Get the corresponding pixel index for a given pixel coordinate
        idx = (x, y)
    else:
        raise ValueError("Either (latitude, longitude) or (x, y) must be provided.")
    
    # Get the spectrum for a given pixel
    spectrum = l1d_cube[idx[0], idx[1], :]

    # Get the wavelengths of the capture
    bands = satobj_h2.wavelengths

    return spectrum, bands