import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates

from hsi_quality.metrics import GRD

from hypso import Hypso2

ROOT_DIR = Path(__file__).resolve().parents[3]
PLOTS_DIR = ROOT_DIR / "plots"


def plot_edge_line(satobj: Hypso2, grd: GRD, save: bool = False):
    cube = satobj.l1d_cube.values
    cloud_mask = satobj.cloud_mask
    pc_img, edges, line, x0, y0, angle = grd.get_sharpest_edge_line(cube, cloud_mask)

    img_rot = np.rot90(pc_img, k=1)
    edges_rot = np.rot90(edges, k=1)

    line_y, line_x = line

    H, W = pc_img.shape
    x0_rot = y0
    y0_rot = W - 1 - x0

    line_x_rot = line_y
    line_y_rot = W - 1 - line_x

    # Rotated PC image with marker + line
    fig_pca, ax = plt.subplots()
    ax.imshow(img_rot, cmap="gray", aspect=1/8)
    ax.axis("off")

    # Rotated edges with marker + line
    fig_edges, ax = plt.subplots()
    ax.imshow(edges_rot, cmap="gray", aspect=1/8)
    ax.plot(x0_rot, y0_rot, "o", color="orange", markersize=4, label="Selected edge point")
    ax.axis("off")
    ax.legend()

    zoom = 100

    x_center = int(x0_rot)
    y_center = int(y0_rot)

    x_start = max(0, x_center - zoom // 8)
    x_end = min(img_rot.shape[1], x_center + zoom // 8 + 1)
    y_start = max(0, y_center - zoom)
    y_end = min(img_rot.shape[0], y_center + zoom + 1)

    line_mask = (
        (line_x_rot >= x_start) & (line_x_rot < x_end) &
        (line_y_rot >= y_start) & (line_y_rot < y_end)
    )

    fig_line, ax = plt.subplots(1, 2, figsize=(4, 4))

    ax[0].imshow(img_rot[y_start:y_end, x_start:x_end], cmap="gray", aspect=1/8)
    ax[0].plot(line_x_rot[line_mask] - x_start, line_y_rot[line_mask] - y_start, color="deepskyblue", linewidth=2, label="Transect")
    ax[0].plot(x0_rot - x_start, y0_rot - y_start, "o", color="orange", markersize=6, label="Selected edge point")
    ax[0].set_title("Zoomed PC image")
    ax[0].axis("off")
    ax[0].legend()

    ax[1].imshow(edges_rot[y_start:y_end, x_start:x_end], cmap="gray", aspect=1/8)
    ax[1].plot(line_x_rot[line_mask] - x_start, line_y_rot[line_mask] - y_start, color="deepskyblue", linewidth=2, label="Transect")
    ax[1].plot(x0_rot - x_start, y0_rot - y_start, "o", color="orange", markersize=6, label="Selected edge point")
    ax[1].set_title("Zoomed edges")
    ax[1].axis("off")
    ax[1].legend()

    if save:
        target = satobj.capture_target
        base_dir = Path(PLOTS_DIR) / target / "fwhm"
        base_dir.mkdir(parents=True, exist_ok=True)
        fig_pca.savefig(base_dir / f"pca.png", bbox_inches="tight")
        fig_edges.savefig(base_dir / f"edges.png", bbox_inches="tight")
        fig_line.savefig(base_dir / f"line.png", bbox_inches="tight")
        plt.close(fig_pca)
        plt.close(fig_edges)
        plt.close(fig_line)
    else:
        plt.show()

def plot_edge_response(satobj, grd: GRD, band: int = 40, save: bool = False):
    cube = satobj.l1d_cube.values
    cloud_mask = satobj.cloud_mask
    pc_img, edges, line, x_edge, y_edge, angle = grd.get_sharpest_edge_line(cube, cloud_mask)

    intensities = map_coordinates(cube[:, :, band], line, order=1, mode="nearest")
    
    popt, pcov, intensities_interp_linear = grd.fit_edge_response(intensities)

    fwhm, esf, esf_norm, lsf_norm, fwhm_0, fwhm_1 = grd.calculate_fwhm(popt)

    x_interp = grd.get_x_interp()

    fig_fit, ax = plt.subplots(figsize=(5, 2))
    ax.plot(x_interp, intensities_interp_linear, "--", label="Edge response")
    ax.plot(x_interp, esf, label="ESF fit")
    ax.set_xlabel("Pixel along edge")
    ax.set_ylabel("Reflectance")
    ax.grid()
    ax.legend()

    fig_esf, ax = plt.subplots(figsize=(5, 2))
    ax.plot(x_interp, esf_norm,label="Normalized ESF")
    ax.plot(x_interp[:-1], lsf_norm*0.5, label="Scaled LSF")
    ax.set_xlabel("Pixel along edge")
    ax.set_ylabel("Normalized ESF")
    ax.legend()
    ax.grid()

    x_interp = x_interp[:-1]

    fig_lsf, ax = plt.subplots(figsize=(5, 2))
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
        base_dir = Path(PLOTS_DIR) / target / "fwhm"
        base_dir.mkdir(parents=True, exist_ok=True)
        fig_fit.savefig(base_dir / f"fit.png", bbox_inches="tight")
        fig_esf.savefig(base_dir / f"esf.png", bbox_inches="tight")
        fig_lsf.savefig(base_dir / f"lsf.png", bbox_inches="tight")
        plt.close(fig_fit)
        plt.close(fig_esf)
        plt.close(fig_lsf)
    else:
        plt.show()