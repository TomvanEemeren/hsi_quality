import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from hsi_quality import RESULTS_DIR

from .edge import Edge


def plot_edge_pixels(img: np.ndarray, edge_pixels: np.ndarray, save: bool = False):
    rot_img = np.rot90(img, k=1)
    rot_edge_pixels = np.rot90(edge_pixels, k=1)

    mm = 1/25.4
    fig, ax = plt.subplots(2, 1, figsize=(60*mm, 30*mm), constrained_layout=True)
    ax[0].imshow(rot_img, cmap='gray', aspect=1/8)
    ax[0].axis('off')
    ax[1].imshow(rot_edge_pixels, cmap='gray', aspect=1/8)
    ax[1].axis('off')
    
    if save:
        base_dir = Path(RESULTS_DIR) / "plots"
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / f"canny_new.png"
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, dpi=300)
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
    else:
        plt.show()

def plot_lines(img: np.ndarray, edge_pixels: np.ndarray, edges: list, save: bool = False):

    mm = 1/25.4
    fig, ax = plt.subplots(1, 2, figsize=(60*mm, 120*mm), constrained_layout=True)

    ax[0].imshow(img, cmap='gray', aspect=8)
    ax[0].axis('off')
    ax[1].imshow(edge_pixels, cmap='gray', aspect=8)
    for edge in edges:
        p0 = edge.p0
        p1 = edge.p1
        ax[1].plot([p0[0], p1[0]], [p0[1], p1[1]], color='red', linewidth=1)
    ax[1].axis('off')

    if save:
        base_dir = Path(RESULTS_DIR) / "plots"
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / f"canny_edge_new.png"
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, dpi=300)
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
    else:
        plt.show()

def plot_refined_edge(edge: Edge, img: np.ndarray, edge_pixels: np.ndarray, save: bool = False):
    mm = 1/25.4
    fig, ax = plt.subplots(1, 2, figsize=(30*mm, 60*mm), constrained_layout=True)

    ax[0].imshow(img, cmap='gray', aspect=8)
    ax[0].axis('off')
    ax[1].imshow(edge_pixels, cmap='gray', aspect=8)
    ax[1].plot(edge.points[:, 0], edge.points[:, 1], 'r-', linewidth=1)
    ax[1].plot(edge.centroid[0], edge.centroid[1], 'yo', markersize=2)
    ax[1].plot(edge.normal[1], edge.normal[0], 'g-', linewidth=1)
    ax[1].axis('off')

    if save:
        base_dir = Path(RESULTS_DIR) / "plots"
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / f"edge_normal_new.png"
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, dpi=300)
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)
    else:
        plt.show()