from networkx import edges
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from hsi_quality.analysis import EdgeDetector, Edge
from hsi_quality import RESULTS_DIR 

from hypso import Hypso2


def plot_edge_pixels(satobj: Hypso2, ed: EdgeDetector, save: bool = False):
    edges, edge_pixels, img = ed.detect_edges(satobj)

    img_rot = np.rot90(img, k=1)
    edges_rot = np.rot90(edge_pixels, k=1)

    fig_pca, ax = plt.subplots(figsize=(4,4))
    ax.imshow(img_rot, cmap="gray", aspect=1/8)
    ax.axis("off")

    fig_edges, ax = plt.subplots(figsize=(4,4))
    ax.imshow(img_rot, cmap="gray", aspect=1/8)
    ax.contour(edges_rot, colors="red", linewidths=0.5)
    ax.axis("off")

    fig_mask, ax = plt.subplots(figsize=(4,4))
    ax.imshow(edges_rot, cmap="gray", aspect=1/8)
    ax.axis("off")

    if save:
        target = satobj.capture_target
        base_dir = Path(RESULTS_DIR) / target / "grd"
        base_dir.mkdir(parents=True, exist_ok=True)
        fig_pca.savefig(base_dir / f"pca.png", bbox_inches="tight")
        fig_edges.savefig(base_dir / f"edge_pixels.png", bbox_inches="tight")
        fig_mask.savefig(base_dir / f"mask.png", bbox_inches="tight")
        plt.close(fig_edges)
        plt.close(fig_pca)
        plt.close(fig_mask) 
    else:
        plt.show()

def plot_edge(edge: Edge, img: np.ndarray, save: bool = False):
    normal = edge.normal
    tangent = edge.tangent
    target = edge.location
    name = edge.name

    all_rows = np.concatenate([normal[0], tangent[0]])
    all_cols = np.concatenate([normal[1], tangent[1]])
    pad = 15
    r0 = max(0, int(np.floor(all_rows.min() - pad)))
    r1 = min(img.shape[0], int(np.ceil(all_rows.max() + pad)))
    c0 = max(0, int(np.floor(all_cols.min() - pad)))
    c1 = min(img.shape[1], int(np.ceil(all_cols.max() + pad)))

    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(img, cmap="gray", aspect="equal")
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.plot(normal[1], normal[0], color="deepskyblue", linewidth=2, label="Normal")
    ax.plot(tangent[1], tangent[0], color="red", linewidth=2, label="Edge")
    ax.axis("off")

    if save:
        base_dir = Path(RESULTS_DIR) / target / "grd" / name
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"edge"
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()