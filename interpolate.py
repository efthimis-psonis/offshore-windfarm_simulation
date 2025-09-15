from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree as KDTree

def wind_speed_from_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.asarray(u, float)**2 + np.asarray(v, float)**2)

def build_kdtree(grid_xy: np.ndarray) -> KDTree:
    grid_xy = np.asarray(grid_xy, float)
    if grid_xy.ndim != 2 or grid_xy.shape[1] != 2:
        raise ValueError("grid_xy must be (M,2)")
    return KDTree(grid_xy)

def knn_interpolate_scalar(
    tree: KDTree,
    grid_xy: np.ndarray,
    grid_values: np.ndarray,
    query_xy: np.ndarray,
    k: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse-distance weighted k-NN interpolation for scalar grid values."""
    query_xy = np.asarray(query_xy, float)
    dists, idxs = tree.query(query_xy, k=k)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    w = 1.0 / np.maximum(dists, 1e-12)
    w = w / w.sum(axis=1, keepdims=True)
    vals = (w * grid_values[idxs]).sum(axis=1)
    return vals, idxs

def interpolate_turbine_conditions(
    positions: np.ndarray,
    U2Z: np.ndarray,
    V2Z: np.ndarray,
    grid_xy: Optional[np.ndarray] = None,
    tree: Optional[KDTree] = None,
) -> np.ndarray:
    """
    Placeholder/utility interpolation. Returns wind speed at each turbine position.

    - If `grid_xy` (M,2) and KDTree are provided: performs k-NN IDW from grid to turbines.
    - Otherwise: returns the spatial mean wind speed (smoke test only).
    """
    speed = wind_speed_from_uv(U2Z, V2Z)
    N = len(positions)
    if grid_xy is not None and tree is not None and speed.size == grid_xy.shape[0]:
        ws, _ = knn_interpolate_scalar(tree, grid_xy, speed, positions, k=4)
        return ws
    # Fallback: mean speed for all turbines
    return np.full(N, float(np.nanmean(speed)))
