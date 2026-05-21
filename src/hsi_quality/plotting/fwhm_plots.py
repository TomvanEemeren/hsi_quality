import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates

from hsi_quality.metrics import GRD
from hsi_quality.analysis import Edge
from hsi_quality import RESULTS_DIR

from hypso import Hypso2


def plot_esf(satobj: Hypso2, edge: Edge, grd: GRD, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube.values
    line = edge.normal

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
        base_dir = Path(RESULTS_DIR) / target / "fwhm"
        base_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(base_dir / f"esf.png", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_lsf(satobj: Hypso2, edge: Edge, grd: GRD, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube.values
    line = edge.normal

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
        target = satobj.capture_target
        base_dir = Path(RESULTS_DIR) / target / "fwhm"
        base_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(base_dir / f"lsf.png", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_fwhm(satobj: Hypso2, edge: Edge, grd: GRD, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube.values
    line = edge.normal

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
        target = satobj.capture_target
        base_dir = Path(RESULTS_DIR) / target / "fwhm"
        base_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(base_dir / f"fwhm.png", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()