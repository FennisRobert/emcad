from numba import njit, f8, i8, types, prange
import numpy as np
import time
from typing import Iterable
from loguru import logger

from ._profiling import format_runtime

@njit(cache=True)
def _cross(xs, ys, i1, i2, i3):
    return ((xs[i2] - xs[i1]) * (ys[i3] - ys[i2]) -
             (ys[i2] - ys[i1]) * (xs[i3] - xs[i2]))

@njit(cache=True)
def _clean_loop(ids: np.ndarray) -> np.ndarray:
    """Cleans duplicate points

    Args:
        ids (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    ptr = 1
    ids_out = np.empty_like(ids)
    ids_out[0] = ids[0]
    for i in range(1, ids.shape[0]):
        if ids[i] == ids_out[ptr]:
            continue
        ids_out[ptr] = ids[i]
        ptr += 1
    return ids_out[:ptr]
     
@njit(i8[:](f8[:], f8[:]), cache=True)
def _convex_top_hull(xs: np.ndarray, ys: np.ndarray):
    Nin = xs.shape[0]
    
    ids_ltr = np.argsort(xs + ys*1j)
    
    stack = np.empty((Nin,), dtype=np.int64)
    size = 2
    
    stack[0] = ids_ltr[0]
    stack[1] = ids_ltr[1]
    
    for i in range(2,Nin):
        idx = ids_ltr[i]
        
        while size >= 2 and _cross(xs, ys, stack[size-2], stack[size-1], idx) <= 0:
            size -= 1
        stack[size] = idx
        size += 1
    
    return stack[:size]
            
@njit(i8[:](f8[:], f8[:]), cache=True)
def _convex_bot_hull(xs: np.ndarray, ys: np.ndarray):
    Nin = xs.shape[0]
    
    ids_ltr = np.argsort(xs + ys*1j)
    
    ys = -ys
    stack = np.empty((Nin,), dtype=np.int64)
    size = 2
    
    stack[0] = ids_ltr[0]
    stack[1] = ids_ltr[1]
    
    for i in range(2,Nin):
        idx = ids_ltr[i]
        
        while size >= 2 and _cross(xs, ys, stack[size-2], stack[size-1], idx) <=0:
            size -= 1
            
        stack[size] = idx
        size += 1
    
    return stack[:size]

@njit(i8[:](f8[:], f8[:]), cache=True)
def _convex_hull(xs: np.ndarray, ys: np.ndarray):
    """ Computes the convex hull has a list of point indices that form the convex hull of a set of points"""
    ids_top = _convex_top_hull(xs, ys)
    ids_bot = _convex_top_hull(-xs, -ys)
    N1 = ids_top.shape[0]
    N2 = ids_bot.shape[0]
    ids_out = np.empty((N1+N2,), dtype=np.int64)
    ids_out[:N1] = ids_top
    ids_out[N1:] = ids_bot#[::-1]
    
    ids_out = _clean_loop(ids_out)
    if ids_out[0] == ids_out[-1]:
        ids_out = ids_out[:-1]
    return ids_out


# --------------------------------------------------------------------------
# Public wrapper (api.py just delegates to this)
# --------------------------------------------------------------------------

def convex_hull(xs: Iterable[float], ys: Iterable[float]) -> np.ndarray:
    """Compute the convex hull for a set of points.

    Args:
        xs, ys: point coordinates.

    Returns:
        np.ndarray: indices into xs/ys forming the convex hull, in order.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = xs.shape[0]
    t0 = time.time()
    ids = _convex_hull(xs, ys)
    t1 = time.time()
    logger.trace(f"{format_runtime(t0, t1)} convex hull time metric: {n / max(t1 - t0, 1e-300):.4f} pts/s")
    return ids
