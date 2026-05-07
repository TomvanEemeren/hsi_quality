
import numpy as np
import xarray as xr
from pyproj import Proj
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from pyresample import kd_tree, geometry
from pyresample.geometry import SwathDefinition

from hypso import Hypso2
from hsi_quality.data import Dataset


class Resampler:
    def __init__(self, bbox: tuple[float, float, float, float], height: int = 512, width: int = 512):
        self.area_extent = bbox # (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
        self.height = height
        self.width = width
        self.projection = {"proj": "utm", "zone": 40, "ellps": "WGS84", "datum": "WGS84", "units": "m"}
        self.area_def = geometry.AreaDefinition(
                area_id='New area',
                proj_id='id',
                description='new area',
                projection=self.projection,
                width=self.width,
                height=self.height,
                area_extent=self.area_extent
            )

    def resample_capture(self, satobj: Hypso2):
        data = satobj.l1d_cube

        latitudes = satobj.latitudes
        longitudes = satobj.longitudes

        swath_def = SwathDefinition(lons=longitudes, lats=latitudes)

        resampled_capture = kd_tree.resample_nearest(swath_def, data.values, self.area_def, fill_value=np.nan, radius_of_influence=500)
        resampled_capture = xr.DataArray(resampled_capture, dims=["y", "x", "band"])
        resampled_capture.attrs.update(data.attrs)

        return resampled_capture


def intersect_captures(dataset: Dataset, zone: int, visualize: bool = False, bbox: tuple[float, float, float, float] = None) -> Polygon:

    # Define projection and datum for mapping 3D coordinates to 2D plane
    p = Proj(proj='utm', zone=zone, ellps='WGS84', datum='WGS84', preserve_units=False)

    for idx, (satobj, metadata) in enumerate(dataset):
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes

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
        if bbox is not None:
            box = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
            plt.plot(*box.exterior.xy, color='red')
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        plt.show()

    return overlap