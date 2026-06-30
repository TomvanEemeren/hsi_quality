import numpy as np
from dataclasses import dataclass


@dataclass
class Edge:
    p1: tuple = None
    p0: tuple = None
    centroid: tuple = None
    normal_dir: tuple = None
    angle: float = None
    score: float = None
    points: np.ndarray = None
    normal: tuple[np.ndarray, np.ndarray] = None
    longitude: float = None
    latitude: float = None
    location: str = None
    name: str = None
