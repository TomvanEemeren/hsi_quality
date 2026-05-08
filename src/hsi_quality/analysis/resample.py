
import numpy as np
import xarray as xr
from pyresample import kd_tree, geometry
from pyresample.geometry import SwathDefinition

from hypso import Hypso2


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