import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates

from hsi_quality.metrics import GRD

from hypso import Hypso2

ROOT_DIR = Path(__file__).resolve().parents[3]
PLOTS_DIR = ROOT_DIR / "plots"


def plot_edge_pixels(satobj: Hypso2, grd: GRD, save: bool = False):
    cube = satobj.l1d_cube.values
    cloud_mask = satobj.cloud_mask
    
    edge_pixels, img = grd.compute_edge_pixels(cube, cloud_mask)

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
        base_dir = Path(PLOTS_DIR) / target / "fwhm"
        base_dir.mkdir(parents=True, exist_ok=True)
        fig_pca.savefig(base_dir / f"pca.png", bbox_inches="tight")
        fig_edges.savefig(base_dir / f"edge_pixels.png", bbox_inches="tight")
        fig_mask.savefig(base_dir / f"mask.png", bbox_inches="tight")
        plt.close(fig_edges)
        plt.close(fig_pca)
        plt.close(fig_mask) 
    else:
        plt.show()

def plot_esf(satobj, grd: GRD, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube.values
    cloud_mask = satobj.cloud_mask

    edge_pixels, img = grd.compute_edge_pixels(cube, cloud_mask)

    edges = grd.compute_edges(edge_pixels, img)

    filtered_edges = grd.filter_edges(edges, img)

    edges_normal = grd.compute_edge_normal(filtered_edges)

    selected_edge = grd.select_edge(edges_normal)
    line = selected_edge["normal"]

    values = map_coordinates(cube[:, :, band], line, order=1, mode="nearest")

    esf, esf_norm, _, _, values_linear = grd.fit_edge_spread_function(values)

    x_interp = grd.get_x_interp()

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(x_interp, values_linear, "--", label="Edge response")
    ax.plot(x_interp, esf, label="ESF fit")
    ax.set_xlabel("Pixel along edge")
    ax.set_ylabel("Reflectance")
    ax.grid()
    ax.legend()

    if save:
        target = satobj.capture_target
        base_dir = Path(PLOTS_DIR) / target / "fwhm"
        base_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(base_dir / f"esf.png", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_lsf(satobj, grd: GRD, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube.values
    cloud_mask = satobj.cloud_mask

    edge_pixels, img = grd.compute_edge_pixels(cube, cloud_mask)

    edges = grd.compute_edges(edge_pixels, img)

    filtered_edges = grd.filter_edges(edges, img)

    edges_normal = grd.compute_edge_normal(filtered_edges)

    selected_edge = grd.select_edge(edges_normal)
    line = selected_edge["normal"]

    values = map_coordinates(cube[:, :, band], line, order=1, mode="nearest")

    esf, esf_norm, _, _, values_linear = grd.fit_edge_spread_function(values)

    lsf, lsf_norm = grd.compute_line_spread_function(esf_norm)

    x_interp = grd.get_x_interp()
    x_diff_interp = grd.get_x_diff_interp()

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(x_interp, esf_norm,label="Normalized ESF")
    ax.plot(x_diff_interp, lsf_norm*0.5, label="Scaled LSF")
    ax.set_xlabel("Pixel along edge")
    ax.set_ylabel("Normalized ESF")
    ax.legend()
    ax.grid()

    if save:
        target = satobj.capture_target
        base_dir = Path(PLOTS_DIR) / target / "fwhm"
        base_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(base_dir / f"lsf.png", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_fwhm(satobj, grd: GRD, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube.values
    cloud_mask = satobj.cloud_mask

    edge_pixels, img = grd.compute_edge_pixels(cube, cloud_mask)

    edges = grd.compute_edges(edge_pixels, img)

    filtered_edges = grd.filter_edges(edges, img)

    edges_normal = grd.compute_edge_normal(filtered_edges)

    selected_edge = grd.select_edge(edges_normal)
    line = selected_edge["normal"]

    values = map_coordinates(cube[:, :, band], line, order=1, mode="nearest")

    esf, esf_norm, _, _, values_linear = grd.fit_edge_spread_function(values)

    lsf, lsf_norm = grd.compute_line_spread_function(esf_norm)

    fwhm, fwhm_0, fwhm_1 = grd.compute_fwhm(lsf)

    x_interp = grd.get_x_interp()
    x_diff_interp = grd.get_x_diff_interp()

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(x_diff_interp, lsf_norm, label="Normalized LSF")
    ax.set_xlabel("Pixel along edge")
    ax.set_ylabel("Normalized LSF")
    ax.plot([x_diff_interp[fwhm_0], x_diff_interp[fwhm_1]], [0.5, 0.5], '--', label='FWHM')
    ax.plot([x_diff_interp[fwhm_0], x_diff_interp[fwhm_1]], [0.5, 0.5], 'kx')
    ax.text(
        (x_diff_interp[fwhm_1] + x_diff_interp[fwhm_0]) / 2,
        0.7 * max(lsf_norm[fwhm_0], lsf_norm[fwhm_1]),
        f'{fwhm:6.3f}',
        fontsize=8,
        ha='center',
        va='bottom',
    )
    ax.set_xlim([x_diff_interp[fwhm_0] - 2, x_diff_interp[fwhm_1] + 2])
    ax.grid()

    if save:
        target = satobj.capture_target
        base_dir = Path(PLOTS_DIR) / target / "fwhm"
        base_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(base_dir / f"fwhm.png", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()