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

    mm = 1/25.4
    fig_edges, ax = plt.subplots(figsize=(74*mm, 74*mm))
    ax.imshow(img_rot, cmap="gray", aspect=1/8)
    ax.contour(edges_rot, colors="red", linewidths=0.5)
    ax.axis("off")

    fig_mask, ax = plt.subplots(figsize=(74*mm, 74*mm))
    ax.imshow(edges_rot, cmap="gray", aspect=1/8)
    ax.axis("off")

    fig_img, ax = plt.subplots(figsize=(74*mm, 74*mm))
    ax.imshow(img_rot, cmap="gray", aspect=1/8)
    ax.axis("off")

    if save:
        target = satobj.capture_target
        base_dir = Path(RESULTS_DIR) / target / "grd"
        base_dir.mkdir(parents=True, exist_ok=True)

        out_path = base_dir / f"edge_pixels.png"
        fig_edges.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, dpi=300)
        fig_edges.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig_edges)

        out_path = base_dir / f"edge_mask.png"
        fig_mask.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, dpi=300)
        fig_mask.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig_mask) 

        out_path = base_dir / f"original_image.png"
        fig_img.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, dpi=300)
        fig_img.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig_img) 
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
    r0 = int(np.floor(all_rows.min() - pad))
    r1 = int(np.ceil(all_rows.max() + pad))
    c0 = int(np.floor(all_cols.min() - pad))
    c1 = int(np.ceil(all_cols.max() + pad))

    height = r1 - r0
    width = c1 - c0
    side = max(height, width)

    r_center = (r0 + r1) / 2
    c_center = (c0 + c1) / 2
    r0 = max(0, int(np.floor(r_center - side / 2)))
    r1 = min(img.shape[0], r0 + side)
    r0 = max(0, r1 - side)
    c0 = max(0, int(np.floor(c_center - side / 2)))
    c1 = min(img.shape[1], c0 + side)
    c0 = max(0, c1 - side)

    mm = 1/25.4
    fig, ax = plt.subplots(figsize=(30*mm, 30*mm))
    ax.imshow(img, cmap="gray", aspect="equal")
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.plot(normal[1], normal[0], color="deepskyblue", label="Normal")
    ax.plot(tangent[1], tangent[0], color="red", label="Edge")
    ax.axis("off")

    if save:
        base_dir = Path(RESULTS_DIR) / target / "grd" / name
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / f"edge"
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, dpi=300)
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
    else:
        plt.show()