
import numpy as np
import xarray as xr
from pyproj import Proj
import dask.array as da
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from pyresample import kd_tree, geometry
from pyresample.geometry import SwathDefinition

def intersect_captures(dataset, zone: int, visualize: bool = False) -> Polygon:

    # Define projection and datum for mapping 3D coordinates to 2D plane
    p = Proj(proj='utm', zone=zone, ellps='WGS84', datum='WGS84', preserve_units=False)

    for idx, satobj in enumerate(dataset):
        latitudes = satobj.latitudes_direct
        longitudes = satobj.longitudes_direct
        
        xs, ys = p(longitudes, latitudes)

        boundary = np.concatenate([
            np.column_stack((xs[0, :], ys[0, :])),
            np.column_stack((xs[:, -1], ys[:, -1])),
            np.column_stack((xs[-1, ::-1], ys[-1, ::-1])),
            np.column_stack((xs[::-1, 0], ys[::-1, 0]))
        ])

        polygon = Polygon(boundary)

        if idx == 0:
            overlap = polygon
        else:
            overlap = overlap.intersection(polygon)

    if visualize:
        plt.figure(figsize=(5, 5))
        plt.plot(*overlap.exterior.xy)
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        plt.show()

    return overlap

def generate_area_def(area_id: str, proj_id: str, description: str, bbox: tuple[float, float, float, float], height: int = None, width: int = None):

    # area_extent: (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
    area_extent = (bbox[0], bbox[1], bbox[2], bbox[3])

    projection = {"proj": "utm", "zone": 40, "ellps": "WGS84", "datum": "WGS84", "units": "m"}

    area_def = geometry.AreaDefinition(area_id, proj_id, description, projection,  width, height, area_extent)

    return area_def

def resample_capture(satobj, area_def):
    data = satobj.l1d_cube

    latitudes = satobj.latitudes_direct
    longitudes = satobj.longitudes_direct 

    swath_def = SwathDefinition(lons=longitudes, lats=latitudes)

    resampled_capture = kd_tree.resample_nearest(swath_def, data.values, area_def, fill_value=np.nan, radius_of_influence=500)
    resampled_capture = xr.DataArray(resampled_capture, dims=["y", "x", "band"])
    resampled_capture.attrs.update(data.attrs)

    return resampled_capture

def resample_data(dataset, area_def):
    resampled_data = []

    for satobj in dataset:
        resampled_capture = resample_capture(satobj, area_def)
        resampled_data.append(resampled_capture)

    return resampled_data