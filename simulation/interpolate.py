from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree as KDTree
from pyproj import Transformer

def build_tree_from_dataset(ds, src_epsg: str = "EPSG:4326", dst_epsg: str = "EPSG:32631") -> Tuple[KDTree, np.ndarray]:
    """
    Build a KDTree over NetCDF grid centers using LAT/LON → UTM transformation.
    Assumes ds has variables 'LAT' and 'LON' with matching shapes.
    Returns (tree, grid_points_utm[N,2]).
    """
    if "LAT" not in ds or "LON" not in ds:
        raise KeyError("Dataset must contain 'LAT' and 'LON' variables for grid coordinates.")
    lat = ds["LAT"].values
    lon = ds["LON"].values
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    tf = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    xg, yg = tf.transform(lon_flat, lat_flat)
    grid_xy = np.column_stack([xg, yg])
    tree = KDTree(grid_xy)
    return tree, grid_xy

def interpolate_with_tree(
    positions_utm: np.ndarray,
    u2z_flat: np.ndarray,
    v2z_flat: np.ndarray,
    tree: KDTree,
    k: int = 4,
) -> np.ndarray:
    """
    Inverse-distance weighted kNN interpolation of wind speed from grid to turbine positions.
    - positions_utm: (N,2) in same UTM as grid_xy used to build tree.
    - u2z_flat, v2z_flat: flattened component arrays aligned with tree points.
    Returns (N,) wind speeds in m/s.
    """
    positions_utm = np.asarray(positions_utm, float)
    dists, idxs = tree.query(positions_utm, k=k)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    w = 1.0 / (dists + 1e-6)
    w /= w.sum(axis=1, keepdims=True)

    u = (w * u2z_flat[idxs]).sum(axis=1)
    v = (w * v2z_flat[idxs]).sum(axis=1)
    return np.sqrt(u**2 + v**2)
