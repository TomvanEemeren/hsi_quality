import numpy as np
from matplotlib import pyplot as plt


def plot_edge_pixels(img: np.ndarray, edge_pixels: np.ndarray):
    rot_img = np.rot90(img, k=1)
    rot_edge_pixels = np.rot90(edge_pixels, k=1)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    ax[0].imshow(rot_img, cmap='gray', aspect=1/8)
    ax[0].axis('off')
    ax[1].imshow(rot_edge_pixels, cmap='gray', aspect=1/8)
    ax[1].axis('off')
    plt.show()

def plot_lines(img: np.ndarray, edge_pixels: np.ndarray, edges: list):
    fig, ax = plt.subplots(1, 2, figsize=(6, 12), constrained_layout=True)
    ax[0].imshow(img, cmap='gray', aspect=8)
    ax[0].axis('off')
    ax[1].imshow(edge_pixels, cmap='gray', aspect=8)
    for edge in edges:
        p0 = edge.p0
        p1 = edge.p1
        ax[1].plot([p0[0], p1[0]], [p0[1], p1[1]], color='red', linewidth=1)
    ax[1].axis('off')
    plt.show()