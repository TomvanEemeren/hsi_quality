from dataclasses import dataclass


@dataclass
class Edge:
    p1: tuple = None
    p0: tuple = None
    centroid: tuple = None
    normal: tuple = None
    angle: float = None
    score: float = None
