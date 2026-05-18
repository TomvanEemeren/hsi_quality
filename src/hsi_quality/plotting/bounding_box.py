import numpy as np
from tqdm import tqdm
from pyproj import Proj
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from hsi_quality.data import Dataset
from hsi_quality.utils import convert_zone

class RectangleSelector:
    def __init__(self, ax):
        self.ax = ax
        self.start_point = None
        self.rect = None
        self.bbox = None
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self._on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self._on_motion)

    def _on_press(self, event):
        if event.inaxes == self.ax:
            self.start_point = (event.xdata, event.ydata)
            self.rect = Rectangle(self.start_point, 0, 0, edgecolor='black', facecolor='orange', alpha=0.5)
            self.ax.add_patch(self.rect)

    def _on_motion(self, event):
        if self.start_point is not None and event.inaxes == self.ax:
            x0, y0 = self.start_point
            x1, y1 = event.xdata, event.ydata

            self.rect.set_x(min(x0, x1))
            self.rect.set_y(min(y0, y1))
            self.rect.set_width(abs(x1 - x0))
            self.rect.set_height(abs(y1 - y0))

            self.ax.figure.canvas.draw()

    def _on_release(self, event):
        if self.start_point is not None:
            self.bbox = self._calculate_bbox()
            self.start_point = None
            self.rect.remove()
            self.ax.figure.canvas.draw()
            plt.close(self.ax.figure)

    def _calculate_bbox(self):
        x0 = self.rect.get_x()
        y0 = self.rect.get_y()

        w = self.rect.get_width()
        h = self.rect.get_height()

        return (x0, y0, x0 + w, y0 + h)

    def get_bbox(self):
        return self.bbox


def select_box(overlap: Polygon) -> tuple[float, float, float, float]:
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(*overlap.exterior.xy)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    print("Select bounding box")
    selector = RectangleSelector(ax)

    plt.show(block=False)

    # Wait until figure is closed
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)

    bbox = selector.get_bbox()
    print(
        f"(x_min, y_min, x_max, y_max) = "
        f"({bbox[0]:.2f}, {bbox[1]:.2f}, "
        f"{bbox[2]:.2f}, {bbox[3]:.2f})"
    )

    return bbox


def intersect_captures(dataset: Dataset, zone: str, visualize: bool = False, bbox: tuple[float, float, float, float] = None) -> Polygon:
    zone, south = convert_zone(zone)

    # Define projection and datum for mapping 3D coordinates to 2D plane
    p = Proj(proj='utm', zone=zone, south=south, ellps='WGS84', datum='WGS84', preserve_units=False)

    for idx, (satobj, metadata) in enumerate(tqdm(dataset, desc="Calculating overlap")):
        latitudes = satobj.latitudes
        longitudes = satobj.longitudes

        xs, ys = p(longitudes, latitudes)

        boundary = np.concatenate([
            np.column_stack((xs[0, :], ys[0, :])),
            np.column_stack((xs[:, -1], ys[:, -1])),
            np.column_stack((xs[-1, ::-1], ys[-1, ::-1])),
            np.column_stack((xs[::-1, 0], ys[::-1, 0]))
        ])

        polygon = Polygon(boundary).buffer(0)

        if idx == 0:
            overlap = polygon
        else:
            overlap = overlap.intersection(polygon).buffer(0)

    if visualize:
        plt.figure(figsize=(5, 5))
        plt.plot(*overlap.exterior.xy)
        if bbox is not None:
            box = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
            plt.plot(*box.exterior.xy, color='red')
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        plt.show()

    return overlap