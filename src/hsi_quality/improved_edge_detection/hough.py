import numpy as np
from skimage.transform import probabilistic_hough_line
from scipy.ndimage import binary_dilation

from .edge import Edge


def detect_lines(edge_pixels: np.ndarray, cloud_mask: np.ndarray, gsd_x: float, gsd_y: float, cloud_margin: int = 3, 
                 threshold: int = 50, line_length: int = 50, line_gap: int = 3, seed: int = 42) -> list:

    # Remove edge pixels that are near clouds
    cloud_region = cloud_mask == 2
    cloud_region = binary_dilation(cloud_region, iterations=cloud_margin)
    edge_pixels = edge_pixels & ~cloud_region

    # Detect lines with Hough transform
    lines = probabilistic_hough_line(edge_pixels, 
                                     threshold=threshold, 
                                     line_length=line_length, 
                                     line_gap=line_gap, 
                                     rng=np.random.default_rng(seed))

    # Filter to keep across-track edges
    edges = []
    for line in lines:
        p0, p1 = line
        dx = (p1[0] - p0[0]) * gsd_x
        dy = (p1[1] - p0[1]) * gsd_y
        angle = np.arctan2(dy, dx)
        angle = np.abs(np.degrees(angle))
        #if 0 <= angle <= 30 or 150 <= angle <= 180:
        if 60 <= angle <= 120:
            norm = np.hypot(dx, dy) + 1e-8
            edge = Edge()
            edge.p0 = p0
            edge.p1 = p1
            edge.centroid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            edge.normal = (-dy / norm, dx / norm)
            edge.angle = angle
            edges.append(edge)

    return edges