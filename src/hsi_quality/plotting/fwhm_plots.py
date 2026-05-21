import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates

from .edge_plots import plot_edge
from hsi_quality.data import Dataset
from hsi_quality.metrics import GRD
from hsi_quality.analysis import EdgeDetector, Edge
from hsi_quality import RESULTS_DIR

from hypso import Hypso2


def plot_esf(satobj: Hypso2, edge: Edge, grd: GRD, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube.values
    line = edge.normal
    target = edge.location
    name = edge.name

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
        base_dir = Path(RESULTS_DIR) / target / "grd" / name
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"esf"
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_lsf(satobj: Hypso2, edge: Edge, grd: GRD, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube.values
    line = edge.normal
    target = edge.location
    name = edge.name

    values = map_coordinates(cube[:, :, band], line, order=1, mode="nearest")

    esf, esf_norm, popt, _, _ = grd.fit_edge_spread_function(values)

    lsf, lsf_norm = grd.compute_line_spread_function(esf_norm, popt)

    x_interp = grd.get_x_interp()

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(x_interp, esf_norm,label="Normalized ESF")
    ax.plot(x_interp, lsf_norm*0.5, label="Scaled LSF")
    ax.set_xlabel("Pixel along edge")
    ax.set_ylabel("Normalized ESF")
    ax.legend()
    ax.grid()

    if save:
        base_dir = Path(RESULTS_DIR) / target / "grd" / name
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"lsf"
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_fwhm(satobj: Hypso2, edge: Edge, grd: GRD, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube.values
    line = edge.normal
    target = edge.location
    name = edge.name

    values = map_coordinates(cube[:, :, band], line, order=1, mode="nearest")

    esf, esf_norm, popt, _, _ = grd.fit_edge_spread_function(values)

    lsf, lsf_norm = grd.compute_line_spread_function(esf_norm, popt)

    fwhm, fwhm_0, fwhm_1 = grd.compute_fwhm(lsf)

    x_interp = grd.get_x_interp()

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(x_interp, lsf_norm, label="Normalized LSF")
    ax.set_xlabel("Pixel along edge")
    ax.set_ylabel("Normalized LSF")
    ax.plot([x_interp[fwhm_0], x_interp[fwhm_1]], [0.5, 0.5], '--', label='FWHM')
    ax.plot([x_interp[fwhm_0], x_interp[fwhm_1]], [0.5, 0.5], 'kx')
    ax.text(
        (x_interp[fwhm_1] + x_interp[fwhm_0]) / 2,
        0.7 * max(lsf_norm[fwhm_0], lsf_norm[fwhm_1]),
        f'{fwhm:6.3f}',
        fontsize=8,
        ha='center',
        va='bottom',
    )
    ax.set_xlim([x_interp[fwhm_0] - 2, x_interp[fwhm_1] + 2])
    ax.grid()

    if save:
        base_dir = Path(RESULTS_DIR) / target / "grd" / name
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"fwhm"
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def make_grd_plots(dataset: Dataset, ed: EdgeDetector, grd: GRD, save: bool = False):
    reference_edge = None
    for idx, (satobj, metadata) in enumerate(tqdm(dataset, desc=f"Plotting edges", leave=False)):
        cloud_mask = satobj.cloud_mask

        cloud_coverage = np.mean(cloud_mask == 2) * 100
        if cloud_coverage > 5:
            continue
        
        edges, _, img = ed.detect_edges(satobj)
        if len(edges) == 0:
            continue

        if reference_edge is None:
            reference_edge = ed.select_edge(edges)
            edge = reference_edge
        else:
            edge = ed.find_closest_edge(edges, reference_edge.longitude, reference_edge.latitude)

        if edge is None:
            continue

        plot_edge(edge, img, save=save)

        plot_esf(satobj, edge, grd, band=40, save=save)

        plot_lsf(satobj, edge, grd, band=40, save=save)

        plot_fwhm(satobj, edge, grd, band=40, save=save)