import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.interpolate as si
from scipy.ndimage import map_coordinates

from .metric import Metric
from hsi_quality.analysis import Edge


class GRD(Metric):
    def __init__(self, params: dict = None):
        super().__init__(name="GRD", params=params)
        self.length = self.params.get("length", 11)
        self.num_interp = self.params.get("num_interp", 1000)

        self.x = np.arange(self.length)
        self.x_interp = np.linspace(0, self.length-1, self.num_interp)

    def calculate(self, cube: np.ndarray, edge: Edge, metadata: pd.Series):
        gsd_along = metadata["gsd_along"]
        gsd_across = metadata["gsd_across"]
        line = edge.normal
        angle = edge.angle

        gsd = np.sqrt((gsd_across * np.sin(angle))**2 + (gsd_along  * np.cos(angle))**2)

        grd_list = []
        fwhm_list = []
        for band in range(cube.shape[2]):
            values = map_coordinates(cube[:, :, band], line, order=1, mode="nearest")

            # Skip if all intensities are zero
            if np.all(values == 0):
                continue

            esf, esf_norm, popt, _, _ = self.fit_edge_spread_function(values)

            lsf, lsf_norm = self.compute_line_spread_function(esf_norm, popt)

            fwhm, _, _ = self.compute_fwhm(lsf)

            grd = fwhm * gsd

            grd_list.append(grd)
            fwhm_list.append(fwhm)

        grd = np.mean(grd_list)
        fwhm = np.mean(fwhm_list)

        info = {"fwhm": fwhm, "gsd": gsd}

        return grd, info
    
    def fit_edge_spread_function(self, values):
        values_cubic = si.griddata(self.x, values, self.x_interp, method="cubic")
        values_linear = si.griddata(self.x, values, self.x_interp, method="linear")

        d = np.min(values)
        b = self.length // 2
        if values_linear[-1] > values_linear[0]:
            c = -0.5
        else:
            c = 0.5
        a = np.max(values_linear) - np.min(values_linear)

        popt, pcov = so.curve_fit(
            self.edge_function, 
            self.x_interp, 
            values_linear, 
            bounds = (
                [0, 0, -10, -np.inf],
                [np.inf, self.length, 10, np.inf]
            ),
            p0=[a, b, c, d]
        )

        esf = self.edge_function(self.x_interp, popt[0], popt[1], popt[2], popt[3])

        esf_range = esf.max() - esf.min()
        if np.isclose(esf_range, 0.0, atol=1e-12):
            esf_norm = np.zeros_like(esf)
        else:
            esf_norm = (esf - esf.min()) / esf_range

        return esf, esf_norm, popt, pcov, values_linear

    def compute_line_spread_function(self, esf_norm, popt):
        lsf = self.edge_function_grad(self.x_interp, popt[0], popt[1], popt[2], popt[3])
        lsf = np.abs(lsf)

        if np.isclose(lsf.max(), 0.0, atol=1e-12):
            lsf_norm = np.zeros_like(lsf)
        else:
            lsf_norm = lsf / lsf.max()

        return lsf, lsf_norm

    def compute_fwhm(self, lsf):
        half_max = lsf.max() / 2

        larger_than_indices = np.where(lsf > half_max)[0]
        if larger_than_indices.size == 0:
            return np.nan, None, None
        
        fwhm_0 = larger_than_indices[0]
        fwhm_1 = larger_than_indices[-1]
        fwhm = self.x_interp[fwhm_1] - self.x_interp[fwhm_0]

        return fwhm, fwhm_0, fwhm_1

    @staticmethod
    def edge_function(x, a, b, c, d):
        z = np.clip((x - b) / c, -500, 500)
        return d + a / (1 + np.exp(z))
    
    @staticmethod
    def edge_function_grad(x, a, b, c, d):
        z = np.clip((x - b) / c, -500, 500)
        exp_z = np.exp(z)
        return -(a * exp_z) / (c * (1 + exp_z)**2)
    
    def get_x_interp(self):
        return self.x_interp
