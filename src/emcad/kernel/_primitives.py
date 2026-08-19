"""
Low-level edge-intersection / point-in-polygon primitives.

Pure array-in, array-out functions with no dependency on `Polygon` or
anything else in the package that depends on `Polygon`. This is
deliberate: `poly.py` imports these (via the thin wrappers in
`api.py`), so nothing in this file may import `Polygon` -- doing so
would reintroduce the circular import `api.py`'s `join_polygons`
wrapper is specifically structured to avoid (see `api.py`'s module
docstring for the full picture).
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

from ._intersect_kernels import compute_intersections
from ._inside import _is_inside
from ._profiling import format_runtime


def edge_self_intersections(pts_start: np.ndarray, pts_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the self intersection points of a set of edges defined by start and end coordinates

    Args:
        pts_start: (2,N) array of coordinates
        pts_end: (2,N) array of coordinates

    Returns:
        np.ndarray: The edge intersection ids (2,N) of the first and second edge index
        np.ndarray: The corresponding crossing coordinate.
    """
    pts_start = np.array(pts_start)
    pts_end = np.array(pts_end)

    n = pts_start.shape[1]
    t0 = time.time()
    result = compute_intersections(pts_start, pts_end, pts_start, pts_end)
    t1 = time.time()
    logger.trace(f"{format_runtime(t0, t1)} intersections time metric: {n / max(t1 - t0, 1e-300):.4f} pts/s")
    return result


def edge_cross_intersections(
    pts1_start: np.ndarray, pts1_end: np.ndarray, pts2_start: np.ndarray, pts2_end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute all edge intersections between two distinct edge sets.

    Args:
        pts1_start: (2,N) array of coordinates
        pts1_end: (2,N) array of coordinates
        pts2_start: (2,M) array of coordinates
        pts2_end: (2,M) array of coordinates

    Returns:
        np.ndarray: The edge intersection ids (2,N) of the first and second edge index
        np.ndarray: The corresponding crossing coordinate.
    """
    pts1_start = np.array(pts1_start)
    pts1_end = np.array(pts1_end)
    pts2_start = np.array(pts2_start)
    pts2_end = np.array(pts2_end)

    n = pts1_start.shape[1] + pts2_start.shape[1]
    t0 = time.time()
    result = compute_intersections(pts1_start, pts1_end, pts2_start, pts2_end)
    t1 = time.time()
    logger.trace(f"{format_runtime(t0, t1)} intersections time metric: {n / max(t1 - t0, 1e-300):.4f} pts/s")
    return result


def is_inside(xs: np.ndarray, ys: np.ndarray, x: float, y: float, include_boundary: bool = True) -> bool:
    """Checks if a coordinate (x, y) is inside the polygon specified by
    the arrays xs, ys.
    """
    return _is_inside(xs, ys, x, y, include_boundary)
