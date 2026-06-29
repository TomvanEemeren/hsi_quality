import numpy as np
import pandas as pd

from .canny import CannyDetector
from .hough import detect_lines
from .filtering import filter_edges, rank_edges
from .edge import Edge

from hypso import Hypso2


class EdgeDetector:
    def __init__(self, params: dict = None):
        self.params = params if params is not None else {}
        self.min_len = self.params.get("min_len", 50)
        self.threshold = self.params.get("threshold", 50)
        self.alpha = self.params.get("alpha", 1.0)
        self.beta = self.params.get("beta", 1.0)
        self.gamma = self.params.get("gamma", 1.0)

    def detect_edges(self, satobj: Hypso2, metadata: pd.DataFrame, canny_detector: CannyDetector) -> Edge:
        cube = satobj.l1d_cube.values
        cloud_mask = satobj.cloud_mask
        gsd_y = metadata["gsd_along"]
        gsd_x = metadata["gsd_across"]

        img = cube[:, :, 60]

        edge_pixels = self.get_edge_pixels(img, canny_detector, gsd_x, gsd_y)

        edges = self.get_edge_lines(edge_pixels, cloud_mask, gsd_x, gsd_y)

        filtered_edges = self.get_filtered_edges(img, edges)

        best_edge = self.get_best_edge(filtered_edges)

        return best_edge
    
    def get_edge_pixels(self, img: np.ndarray, canny_detector: CannyDetector, gsd_x: float, gsd_y: float) -> np.ndarray:
        edge_pixels = canny_detector.detect_edge_pixels(img, gsd_x, gsd_y)
        return edge_pixels
    
    def get_edge_lines(self, edge_pixels: np.ndarray, cloud_mask: np.ndarray, gsd_x: float, gsd_y: float) -> list:
        edges = detect_lines(edge_pixels, 
                             cloud_mask, 
                             gsd_x, gsd_y, 
                             threshold=self.threshold, 
                             line_length=self.min_len, 
                             line_gap=3, 
                             seed=42)
        return edges
    
    def get_filtered_edges(self, img: np.ndarray, edges: list) -> list:
        filtered_edges = filter_edges(img, 
                                      edges, 
                                      alpha=self.alpha, 
                                      beta=self.beta, 
                                      gamma=self.gamma)
        return filtered_edges
    
    def get_best_edge(self, edges: list) -> Edge:
        best_edge = rank_edges(edges)
        return best_edge