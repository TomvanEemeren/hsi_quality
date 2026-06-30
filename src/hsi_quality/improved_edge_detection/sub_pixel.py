import numpy as np
from scipy import ndimage
import scipy.interpolate as si

from .edge import Edge


def sub_pixel_refinement(img: np.ndarray, edge: Edge):
    normal = edge.normal_dir

    # Sample points along the edge
    points = sample_points(edge, margin=0.1, num_points=10)

    # Refine each point
    for (i, point) in enumerate(points):
        refined_point = refine_point(img, point, normal)
        points[i] = refined_point
    
    refined_edge = refine_edge(edge, points)

    normal_points = calculate_normal_line(refined_edge)
    refined_edge.normal = normal_points

    return refined_edge


def sample_points(edge: Edge, margin: float = 0.1, num_points: int = 10):
    p0 = np.asarray(edge.p0, dtype=float)
    p1 = np.asarray(edge.p1, dtype=float)
    t = np.linspace(margin, 1 - margin, num_points)
    return p0[None, :] + t[:, None] * (p1 - p0)


def refine_point(img: np.ndarray, point: np.ndarray, normal: tuple, half_width: int = 10):
    nx, ny = normal
    x0, y0 = point

    t = np.linspace(-half_width, half_width, 2 * half_width + 1)

    xs = x0 + t * nx
    ys = y0 + t * ny

    profile = ndimage.map_coordinates(
        img,
        [ys, xs],
        order=1,
        mode="nearest"
    )

    # Fit a cubic spline and take second derivative to locate sub-pixel edge
    cs = si.CubicSpline(t, profile, extrapolate=True)
    second_derivative = cs.derivative(2)

    roots = second_derivative.roots()
    if roots.size == 0:
        root = 0.0
    else:
        root = roots[np.argmin(np.abs(roots))]

    refined_x = x0 + root * nx
    refined_y = y0 + root * ny

    return np.array([refined_x, refined_y])


def refine_edge(edge: Edge, points: np.ndarray):
    centroid = points.mean(axis=0)
    centered = points - centroid

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    direction /= np.linalg.norm(direction)

    dx, dy = direction

    normal = np.array([-dy, dx])
    normal /= np.linalg.norm(normal)

    edge.normal_dir = normal
    edge.centroid = centroid
    edge.points = points

    return edge
   

def calculate_normal_line(edge: Edge, length: float = 22):
    centroid = edge.centroid
    normal = edge.normal_dir

    nx, ny = normal
    cx, cy = centroid

    t = np.linspace(-length / 2, length / 2, int(length) + 1)
    normal_line_y = cy + t * nx
    normal_line_x = cx + t * ny

    return (normal_line_y, normal_line_x)