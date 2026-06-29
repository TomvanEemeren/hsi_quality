import numpy as np

from .edge import Edge

    
def filter_edges(img: np.ndarray, edges: list[Edge], 
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> list:

    filtered_edges = []
    for edge in edges:
        bright_vals, dark_vals = get_bright_dark_sides(img, edge)
        
        if bright_vals.size > 0 and dark_vals.size > 0:
            bright_mean = np.mean(bright_vals)
            dark_mean = np.mean(dark_vals)
            sigma_grid = np.std(np.concatenate([bright_vals, dark_vals]))
            bright_perc = np.percentile(bright_vals, 10)
            dark_perc = np.percentile(dark_vals, 90)

            cond1 = bright_mean > alpha * dark_mean
            cond2 = np.std(bright_vals) < beta * sigma_grid
            cond3 = np.std(dark_vals) < beta * sigma_grid
            cond4 = bright_perc > gamma * dark_perc

            if (cond1 and cond2 and cond3 and cond4):
                score = edge_score(
                    bright_mean, dark_mean,
                    np.std(bright_vals), np.std(dark_vals),
                    sigma_grid,
                    bright_perc, dark_perc
                )
                edge.score = score
                filtered_edges.append(edge)

    return filtered_edges


def rank_edges(edges: list[Edge]) -> Edge:
    sorted_edges = sorted(edges, key=lambda e: e.score, reverse=True)
    return sorted_edges[0]


def get_bright_dark_sides(img, edge: Edge, box_size: int = 15):
    (cx, cy) = int(edge.centroid[0]), int(edge.centroid[1])
    (nx, ny) = edge.normal

    half = box_size // 2

    bright_vals = []
    dark_vals = []

    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            x = cx + i
            y = cy + j

            if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
                continue

            # vector from center to pixel
            vx = i
            vy = j

            # project onto normal direction
            proj = vx * nx + vy * ny

            if proj > 0:
                bright_vals.append(img[y, x])
            else:
                dark_vals.append(img[y, x])

    return np.array(bright_vals), np.array(dark_vals)


def edge_score(bright_mean: float, dark_mean: float, bright_std: float, 
               dark_std: float, sigma_grid: float, bright_perc: float, dark_perc: float) -> float:

    score = 0.0

    # contrast strength
    score += bright_mean / (dark_mean + 1e-6)

    # low noise penalty
    score -= (bright_std + dark_std) / (sigma_grid + 1e-6)

    # percentile separation (robust contrast)
    score += bright_perc / (dark_perc + 1e-6)

    return score