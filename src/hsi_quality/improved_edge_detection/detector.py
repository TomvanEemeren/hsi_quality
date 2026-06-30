import numpy as np
import pandas as pd

from .canny import CannyDetector
from .hough import detect_lines
from .filtering import filter_edges, rank_edges
from .edge import Edge
from .sub_pixel import sub_pixel_refinement

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

        longitudes = satobj.longitudes
        latitudes = satobj.latitudes

        for edge in filtered_edges:
            cx, cy = edge.centroid
            edge.longitude = longitudes[int(cy), int(cx)]
            edge.latitude = latitudes[int(cy), int(cx)]

        return filtered_edges, edge_pixels, img
    
    def select_closest_edge(self, satobj: Hypso2, img: np.ndarray, edges: list[Edge], ref_longitude: float, ref_latitude: float) -> Edge:
        closest_edge = None
        min_distance = float('inf')

        for edge in edges:
            distance = np.sqrt((edge.longitude - ref_longitude) ** 2 + (edge.latitude - ref_latitude) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_edge = edge

        if closest_edge is not None:
            refined_edge = self.get_refined_edge(img, closest_edge)

            refined_edge.location = satobj.capture_target
            refined_edge.name = satobj.capture_name

            return refined_edge
    
        return None

    def select_best_edge(self, satobj: Hypso2, img: np.ndarray, edges: list[Edge]) -> Edge:
        best_edge = self.get_best_edge(edges)

        refined_edge = self.get_refined_edge(img, best_edge)

        refined_edge.location = satobj.capture_target
        refined_edge.name = satobj.capture_name

        return refined_edge

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
    
    def get_refined_edge(self, img: np.ndarray, edge: Edge) -> Edge:
        refined_edge = sub_pixel_refinement(img, edge)
        return refined_edge