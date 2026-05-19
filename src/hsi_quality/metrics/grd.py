import numpy as np
import pandas as pd
from skimage import feature
from scipy.ndimage import label
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

        self.min_len = self.params.get("min_len", 5)
        self.max_len = self.params.get("max_len", 50)

        self.alpha = self.params.get("alpha", 1.0)
        self.beta = self.params.get("beta", 1.0)
        self.gamma = self.params.get("gamma", 1.0)
        self.sample_dist = self.params.get("sample_dist", 5)

        self.num_interp = self.params.get("num_interp", 1000)
        self.x = np.arange(self.length)
        self.x_interp = np.linspace(0, self.length-1, self.num_interp)
        self.x_diff_interp = self.x_interp[:-1]+(self.x_interp[1]-self.x_interp[0])/2

    def calculate(self, cube: np.ndarray, cloud_mask: np.ndarray, metadata: pd.Series):
        gsd_along = metadata["gsd_along"]
        gsd_across = metadata["gsd_across"]

        edge_pixels, img = self.compute_edge_pixels(cube, cloud_mask)

        edges = self.compute_edges(edge_pixels, img)

        filtered_edges = self.filter_edges(edges, img)

        edges_normal = self.compute_edge_normal(filtered_edges)

        selected_edge = self.select_edge(edges_normal)
        line = selected_edge["normal"]
        angle = selected_edge["angle"]

        gsd = np.hypot(gsd_along * np.cos(angle), gsd_across * np.sin(angle))

        grd_list = []
        for band in range(cube.shape[2]):
            values = map_coordinates(cube[:, :, band], line, order=1, mode="nearest")

            # Skip if all intensities are zero
            if np.all(values == 0):
                continue
            
            esf, esf_norm, _, _, _ = self.fit_edge_spread_function(values)

            lsf, lsf_norm = self.compute_line_spread_function(esf_norm)

            fwhm, _, _ = self.compute_fwhm(lsf)

            grd = fwhm * gsd
            grd_list.append(grd)

        grd = np.mean(grd_list)

        info = {"fwhm": fwhm, "gsd": gsd}

        return grd, info

    def compute_edge_pixels(self, cube: np.ndarray, cloud_mask: np.ndarray):
        H, W, Q = cube.shape

        # Reduce the image to 1 dimension using PCA
        X = cube.reshape(-1, Q)
        pc = PCA(n_components=1).fit_transform(X)
        img = pc.reshape(H, W)

        # Compute edge pixels using Canny edge detection
        edge_pixels = feature.canny(img, sigma=1.0, low_threshold=2, high_threshold=3)

        # Remove edge pixels that are near clouds
        cloud_region = cloud_mask == 2
        cloud_region = binary_dilation(cloud_region, iterations=self.cloud_margin)
        edge_pixels = edge_pixels & ~cloud_region

        return edge_pixels, img

    def compute_edges(self, edge_pixels: np.ndarray, img: np.ndarray):
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)

        orientations = np.arctan2(gy, gx)
        magnitudes = np.hypot(gx, gy)

        labeled, ncomp = label(edge_pixels)
        edges = []

        for comp in range(1, ncomp + 1):
            edge = {}

            coords = np.column_stack(np.nonzero(labeled == comp))  # (y, x)
            length = coords.shape[0]

            # discard edges that are too short or too long
            if length < self.min_len or length > self.max_len:
                continue

            y_span = np.ptp(coords[:, 0])
            x_span = np.ptp(coords[:, 1])
            along_track = y_span > x_span
            
            edge["coords"] = coords
            edge["orientations"] = orientations[coords[:, 0], coords[:, 1]]
            edge["magnitudes"] = magnitudes[coords[:, 0], coords[:, 1]]
            edge["along_track"] = along_track
            edges.append(edge)

        return edges

    def filter_edges(self, edges: list, img: np.ndarray):
        H, W = img.shape

        filtered_edges = []
        for edge in edges:
            coords = edge["coords"]
            orientation = edge["orientations"]

            dx = np.cos(orientation)
            dy = np.sin(orientation)
            x1 = coords[:, 1] + self.sample_dist * dx
            y1 = coords[:, 0] + self.sample_dist * dy
            x2 = coords[:, 1] - self.sample_dist * dx
            y2 = coords[:, 0] - self.sample_dist * dy
            
            in_bounds = (
                (x1 >= 0) & (x1 < W) &
                (x2 >= 0) & (x2 < W) &
                (y1 >= 0) & (y1 < H) &
                (y2 >= 0) & (y2 < H)
            )
            if not np.all(in_bounds):
                continue

            side1_vals = map_coordinates(img, [y1, x1], order=1)
            side2_vals = map_coordinates(img, [y2, x2], order=1)

            if np.mean(side1_vals) >= np.mean(side2_vals):
                bright_vals = side1_vals
                dark_vals = side2_vals
            else:
                bright_vals = side2_vals
                dark_vals = side1_vals

            # Apply statistical checks
            bright_mean = bright_vals.mean()
            dark_mean = dark_vals.mean()
            sigma_grid = np.std(np.concatenate([bright_vals, dark_vals]))

            if (bright_mean > self.alpha * dark_mean and
                np.std(bright_vals) < self.beta * sigma_grid and
                np.std(dark_vals) < self.beta * sigma_grid and
                np.percentile(bright_vals, 10) > self.gamma * np.percentile(dark_vals, 90)):
                filtered_edges.append(edge)

        return filtered_edges

    def compute_edge_normal(self, edges: list):
        half_length = self.length // 2
        t = np.linspace(-half_length, half_length, self.length)

        for edge in edges:
            coords = edge["coords"]
            orientations = edge["orientations"]

            centroid = np.mean(coords, axis=0)
            angle = np.mean(orientations)

            nx = np.cos(angle)
            ny = np.sin(angle)

            ys = centroid[0] + t * ny
            xs = centroid[1] + t * nx
            
            edge["normal"] = (ys, xs)
            edge["angle"] = angle
        
        return edges

    def select_edge(self, edges: list):
        max_magnitude = -np.inf
        sharpest_edge = None

        for edge in edges:
            magnitude = edge["magnitudes"].mean()
            if magnitude > max_magnitude:
                max_magnitude = magnitude
                sharpest_edge = edge

        return sharpest_edge
    
    def fit_edge_spread_function(self, values):
        values_cubic = si.griddata(self.x, values, self.x_interp, method="cubic")
        values_linear = si.griddata(self.x, values, self.x_interp, method="linear")

        d = np.min(values)
        b = self.length // 2
        c = -0.5
        a = 2*(values_cubic[self.num_interp//2] - d)

        popt, pcov = so.curve_fit(self.edge_function, self.x_interp, values_linear, p0=[a, b, c, d])

        esf = self.edge_function(self.x_interp, popt[0], popt[1], popt[2], popt[3])

        esf_norm = (esf - esf.min()) / (esf.max() - esf.min())

        return esf, esf_norm, popt, pcov, values_linear

    def compute_line_spread_function(self, esf_norm):
        lsf = np.abs(np.diff(esf_norm))
        lsf_norm = lsf / lsf.max()

        return lsf, lsf_norm

    def compute_fwhm(self, lsf):
        half_max = lsf.max() / 2
        larger_than_indices = np.where(lsf > half_max)[0]
        fwhm_0 = larger_than_indices[0]
        fwhm_1 = larger_than_indices[-1]
        fwhm = self.x_diff_interp[fwhm_1] - self.x_diff_interp[fwhm_0]

        return fwhm, fwhm_0, fwhm_1

    @staticmethod
    def edge_function(x, a, b, c, d):
        z = np.clip((x - b) / c, -500, 500)
        return d + a / (1 + np.exp(z))
    
    def get_x_interp(self):
        return self.x_interp
    
    def get_x_diff_interp(self):
        return self.x_diff_interp
