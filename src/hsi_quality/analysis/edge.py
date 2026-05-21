import numpy as np
from skimage import feature
import scipy.interpolate as si
from dataclasses import dataclass
from scipy.ndimage import binary_dilation, map_coordinates, sobel, label

from hypso import Hypso2


@dataclass
class Edge:
    pixels: np.ndarray = None
    sub_pixels: np.ndarray = None
    orientations: np.ndarray = None
    magnitudes: np.ndarray = None
    along_track: bool = None
    tangent: tuple[np.ndarray, np.ndarray] = None
    normal: tuple[np.ndarray, np.ndarray] = None
    centroid: np.ndarray = None
    angle: float = None
    longitude: float = None
    latitude: float = None
    location: str = None
    name: str = None
    bright_perc: float = None
    dark_perc: float = None


class EdgeDetector:
    def __init__(self, params: dict = None):
        self.params = params if params is not None else {}
        self.low_threshold = self.params.get("low_threshold", 0.1)
        self.high_threshold = self.params.get("high_threshold", 0.4)
        self.cloud_margin = self.params.get("cloud_margin", 3)

        self.min_len = self.params.get("min_len", 5)
        self.max_len = self.params.get("max_len", 100)

        self.length = self.params.get("length", 11)

        self.sample_dist = self.params.get("sample_dist", 5)
        self.alpha = self.params.get("alpha", 1.0)
        self.beta = self.params.get("beta", 1.0)
        self.gamma = self.params.get("gamma", 1.0)

    def detect_edges(self, satobj: Hypso2):
        cube = satobj.l1d_cube.values
        cloud_mask = satobj.cloud_mask
        longitudes = satobj.longitudes
        latitudes = satobj.latitudes
        
        img = cube[:, :, 60]
        
        edge_pixels = self._compute_edge_pixels(img, cloud_mask)

        edges = self._compute_edges(img, edge_pixels, longitudes, latitudes)

        filtered_edges = self._filter_edges(img, edges)

        refined_edges = self._refine_sub_pixels(img, filtered_edges)

        for edge in refined_edges:
            edge.location = satobj.capture_target
            edge.name = satobj.capture_name

        return refined_edges, edge_pixels, img

    def select_edge(self, edges: list[Edge]):
        if len(edges) == 0:
            return None

        ranked_edges = sorted(edges, key=self._edge_score, reverse=True)
        return ranked_edges[0]
    
    def select_edges(self, edges: list[Edge], num_edges: int = 4):
        if len(edges) == 0:
            return []

        ranked_edges = sorted(edges, key=self._edge_score, reverse=True)
        return ranked_edges[:num_edges]

    def find_closest_edge(self, edges: list[Edge], longitude: float, latitude: float):
        if len(edges) == 0:
            return None

        closest_edge = min(edges, key=lambda edge: np.hypot(edge.longitude - longitude, edge.latitude - latitude))
        return closest_edge

    def _compute_edge_pixels(self, img: np.ndarray, cloud_mask: np.ndarray):

        # Compute edge pixels using Canny edge detection
        edge_pixels = feature.canny(img, sigma=1.0, low_threshold=self.low_threshold, high_threshold=self.high_threshold)

        # Remove edge pixels that are near clouds
        cloud_region = cloud_mask == 2
        cloud_region = binary_dilation(cloud_region, iterations=self.cloud_margin)
        edge_pixels = edge_pixels & ~cloud_region

        return edge_pixels

    def _compute_edges(self, img: np.ndarray, edge_pixels: np.ndarray, longitudes: np.ndarray, latitudes: np.ndarray):
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)

        orientations = np.arctan2(gy, gx)
        magnitudes = np.hypot(gx, gy)

        labeled, ncomp = label(edge_pixels)
        edges = []

        for comp in range(1, ncomp + 1):
            edge = Edge()

            coords = np.column_stack(np.nonzero(labeled == comp))  # (y, x)
            length = coords.shape[0]

            # discard edges that are too short or too long
            if length < self.min_len or length > self.max_len:
                continue

            y_span = np.ptp(coords[:, 0])
            x_span = np.ptp(coords[:, 1])
            along_track = y_span > x_span
        
            edge.pixels = coords
            edge.orientations = orientations[coords[:, 0], coords[:, 1]]
            edge.magnitudes = magnitudes[coords[:, 0], coords[:, 1]]
            edge.along_track = along_track
            edge.longitude = np.mean(longitudes[coords[:, 0], coords[:, 1]])
            edge.latitude = np.mean(latitudes[coords[:, 0], coords[:, 1]])

            edges.append(edge)

        return edges
    
    def _filter_edges(self, img: np.ndarray, edges: list[Edge]):
        H, W = img.shape

        filtered_edges = []
        for edge in edges:
            coords = edge.pixels
            orientation = edge.orientations

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
            bright_perc = np.percentile(bright_vals, 10)
            dark_perc = np.percentile(dark_vals, 90)

            edge.bright_perc = bright_perc
            edge.dark_perc = dark_perc

            if (bright_mean > self.alpha * dark_mean and
                np.std(bright_vals) < self.beta * sigma_grid and
                np.std(dark_vals) < self.beta * sigma_grid and
                bright_perc > self.gamma * dark_perc):
                filtered_edges.append(edge)

        return filtered_edges
    
    def _refine_sub_pixels(self, img: np.ndarray, filtered_edges: list[Edge]):
        offsets = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)

        refined_edges = []
        for edge in filtered_edges:
            coords = edge.pixels
            orientations = edge.orientations

            sub_coords = np.zeros((coords.shape[0], 2), dtype=float)
            for i, ((y, x), angle) in enumerate(zip(coords, orientations)):
                # Sample pixels along the gradient direction through the edge pixel
                dx = np.cos(angle)
                dy = np.sin(angle)
                ys = y + offsets * dy
                xs = x + offsets * dx
                vals = map_coordinates(img, [ys, xs], order=1, mode="nearest")

                # Fit a cubic spline and take second derivative to locate sub-pixel edge
                cs = si.CubicSpline(offsets, vals, extrapolate=True)
                second_derivative = cs.derivative(2)

                roots = second_derivative.roots()
                if roots.size == 0:
                    root = 0.0
                else:
                    root = roots[np.argmin(np.abs(roots))]

                sub_coords[i, 0] = y + root * dy
                sub_coords[i, 1] = x + root * dx

            # Fit a straight line through the sub-pixel points using orthogonal least squares
            centroid = sub_coords.mean(axis=0)
            centered = sub_coords - centroid

            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            direction = vt[0]

            # normalize tangent
            ty, tx = direction
            norm = np.hypot(tx, ty)
            tx /= norm
            ty /= norm

            # tangent line
            edge_length = len(coords)
            t = np.linspace(-edge_length//2, edge_length//2, edge_length)
            yt = centroid[0] + t * ty
            xt = centroid[1] + t * tx

            # normal line
            ny = -tx
            nx = ty

            t = np.linspace(-self.length//2, self.length//2, self.length)
            yn = centroid[0] + t * ny
            xn = centroid[1] + t * nx

            edge.sub_pixels = sub_coords
            edge.tangent = (yt, xt)
            edge.normal = (yn, xn)
            edge.centroid = centroid
            edge.angle = np.arctan2(ty, tx)
            refined_edges.append(edge)

        return refined_edges
    
    def _edge_score(self, edge: Edge):
        bright_perc = edge.bright_perc
        dark_perc = edge.dark_perc
        magnitudes = edge.magnitudes

        if bright_perc is None or dark_perc is None:
            return -np.inf

        contrast = bright_perc - dark_perc
        scale = np.abs(bright_perc) + np.abs(dark_perc) + 1e-12
        contrast_score = contrast / scale

        magnitude_score = np.mean(magnitudes)

        return contrast_score * magnitude_score