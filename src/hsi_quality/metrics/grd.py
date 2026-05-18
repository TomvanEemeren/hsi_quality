import numpy as np
import pandas as pd
from skimage import feature
import scipy.optimize as so
import scipy.interpolate as si
from sklearn.decomposition import PCA
from scipy.ndimage import binary_dilation, map_coordinates, sobel

from .metric import Metric

class GRD(Metric):
    def __init__(self, params: dict = None):
        super().__init__(name="GRD", params=params)
        self.cloud_margin = self.params.get("cloud_margin", 3)
        self.length = self.params.get("length", 11)
        self.num_interp = self.params.get("num_interp", 1000)
        self.x0 = np.arange(self.length)
        self.x0_interp = np.linspace(0, self.length-1, self.num_interp)

    def calculate(self, cube: np.ndarray, cloud_mask: np.ndarray, metadata: pd.Series):
        gsd_along = metadata["gsd_along"]
        gsd_across = metadata["gsd_across"]

        _, _, line, _, _, angle = self.get_sharpest_edge_line(cube, cloud_mask)

        gsd = np.hypot(gsd_along * np.cos(angle), gsd_across * np.sin(angle))

        grd_list = []
        for band in range(cube.shape[2]):
            intensities = map_coordinates(cube[:, :, band], line, order=1, mode="nearest")

            # Skip if all intensities are zero
            if np.all(intensities == 0):
                continue

            popt, pcov = self.fit_edge_response(intensities)
            
            fwhm = self.calculate_fwhm(popt)

            grd = fwhm * gsd
            grd_list.append(grd)

        grd = np.mean(grd_list)

        info = {"fwhm": fwhm, "gsd": gsd}

        return grd, info

    @staticmethod
    def edge_function(x, a, b, c, d):
        z = np.clip((x - b) / c, -500, 500)
        return d + a / (1 + np.exp(z))
    
    def calculate_fwhm(self, popt):
        edge_response = self.edge_function(self.x0_interp, popt[0], popt[1], popt[2], popt[3])

        normalized_esf = (edge_response - edge_response.min()) / (edge_response.max() - edge_response.min())

        line_spread_function = np.abs(np.diff(normalized_esf))
        line_spread_function_scaled = 0.5 * line_spread_function / line_spread_function.max()

        x1_interp = self.x0_interp[:-1]+(self.x0_interp[1]-self.x0_interp[0])/2
        
        half_max = line_spread_function.max() / 2
        larger_than_indices = np.where(line_spread_function > half_max)[0]
        fwhm_0 = larger_than_indices[0]
        fwhm_1 = larger_than_indices[-1]
        fwhm = x1_interp[fwhm_1] - x1_interp[fwhm_0]

        return fwhm
    
    def fit_edge_response(self, intensities):
        intensities_interp = si.griddata(self.x0, intensities, self.x0_interp, method="cubic")
        intensities_interp_linear = si.griddata(self.x0, intensities, self.x0_interp, method="linear")

        d = np.min(intensities)
        b = self.length // 2
        c = -0.5
        a = 2*(intensities_interp[self.num_interp//2] - d)

        popt, pcov = so.curve_fit(self.edge_function, self.x0_interp, intensities_interp_linear, p0=[a, b, c, d])

        return popt, pcov

    def get_sharpest_edge_line(self, cube: np.ndarray, cloud_mask: np.ndarray):
        # cube has shape (Height, Width, Bands) = (H, W, Q)
        H, W, Q = cube.shape

        # Reduce the image to 1 dimension
        X = cube.reshape(-1, Q)
        pc = PCA(n_components=1).fit_transform(X)
        pc_img = pc.reshape(H, W)

        # Extract the edges in the image
        edges = feature.canny(pc_img, sigma=1.0, low_threshold=2, high_threshold=3)

        # Remove any edges that are near clouds
        if cloud_mask is not None:
            cloud_region = cloud_mask == 2
            if self.cloud_margin > 0:
                cloud_region = binary_dilation(cloud_region, iterations=self.cloud_margin)
            edges = edges & ~cloud_region
        
        # Find the intensity gradients for each pixel
        gx = sobel(pc_img, axis=1)
        gy = sobel(pc_img, axis=0)

        magnitudes = np.hypot(gx, gy)
        directions = np.arctan2(gy, gx)

        # Get the sharpest edge 
        masked = np.full_like(magnitudes, -np.inf) 
        masked[edges] = magnitudes[edges] 

        y0, x0 = np.unravel_index(np.argmax(masked), masked.shape)

        # Extract the line along the sharpest edge
        half_length = self.length // 2
        t = np.linspace(-half_length, half_length, self.length)

        angle = directions[y0, x0]
        dx = np.cos(angle)
        dy = np.sin(angle)
        xs = x0 + t * dx
        ys = y0 + t * dy

        line = (ys, xs)

        return pc_img, edges, line, x0, y0, angle