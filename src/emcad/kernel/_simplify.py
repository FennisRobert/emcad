"""Numba-accelerated polyline simplification (Ramer-Douglas-Peucker) and
polygon sanitization.

Public wrappers `simplify_polyline` / `sanitize_polygon` live here
alongside their numba kernels, following the same "one file per
topic" pattern as `_ch.py`/`_intersect_kernels.py`+`_primitives.py`;
`api.py` just delegates to them.

Usage:
    from emcad.kernel.api import simplify_polyline
    xs2, ys2 = simplify_polyline(xs, ys, delta=0.001)   # delta in the same
                                                          # units as xs/ys
                                                          # (e.g. meters)
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Perpendicular distance from point (px, py) to segment (a -> b)."""
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-30:   # a and b coincide -> distance to the point itself
        ddx = px - ax
        ddy = py - ay
        return (ddx * ddx + ddy * ddy) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t * dx
    cy = ay + t * dy
    ddx = px - cx
    ddy = py - cy
    return (ddx * ddx + ddy * ddy) ** 0.5


@njit(cache=True)
def _rdp_core(xs: np.ndarray, ys: np.ndarray, delta: float) -> np.ndarray:
    """Iterative (non-recursive, numba-friendly) Douglas-Peucker. Returns a
    boolean keep-mask over the input points."""
    n = xs.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    keep[0] = True
    keep[n - 1] = True

    # Explicit stack of (start_idx, end_idx) segment ranges still to check.
    # Each split pushes 2 and pops 1 (net +1), and there are at most n-2
    # possible splits, so capacity n is always sufficient.
    stack = np.empty((n, 2), dtype=np.int64)
    top = 0
    stack[top, 0] = 0
    stack[top, 1] = n - 1
    top += 1

    while top > 0:
        top -= 1
        i0 = stack[top, 0]
        i1 = stack[top, 1]
        if i1 - i0 < 2:
            continue

        ax, ay = xs[i0], ys[i0]
        bx, by = xs[i1], ys[i1]

        max_dist = -1.0
        max_idx = -1
        for i in range(i0 + 1, i1):
            d = _point_segment_distance(xs[i], ys[i], ax, ay, bx, by)
            if d > max_dist:
                max_dist = d
                max_idx = i

        if max_dist > delta:
            keep[max_idx] = True
            stack[top, 0] = i0
            stack[top, 1] = max_idx
            top += 1
            stack[top, 0] = max_idx
            stack[top, 1] = i1
            top += 1

    return keep


def _filter_min_step(xs, ys, min_step: float):
    """Greedily drop points that are closer than `min_step` to the last kept point.
 
    Parameters
    ----------
    xs, ys : sequences of floats, same length, same units as your geometry.
    min_step : float
        Minimum allowed distance between consecutive kept points.
 
    Returns
    -------
    (xs_out, ys_out) : numpy arrays. Always keeps the first and last input
        points (even if the last point ends up closer than min_step to the
        previous kept point, since dropping the actual endpoint would change
        what the polyline represents).
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.shape != ys.shape:
        raise ValueError(f"xs and ys must be the same shape, got {xs.shape} vs {ys.shape}")
    if xs.ndim != 1:
        raise ValueError("xs/ys must be 1D")
    if min_step < 0:
        raise ValueError(f"min_step must be >= 0, got {min_step}")
    n = xs.shape[0]
    if n < 3:
        return xs.copy(), ys.copy()
 
    keep_idx = [0]
    last_x, last_y = xs[0], ys[0]
    for i in range(1, n - 1):
        if ((xs[i] - last_x) ** 2 + (ys[i] - last_y) ** 2) ** 0.5 >= min_step:
            keep_idx.append(i)
            last_x, last_y = xs[i], ys[i]
    keep_idx.append(n - 1)
 
    keep_idx = np.array(keep_idx, dtype=np.int64)
    return xs[keep_idx], ys[keep_idx]

def _filter_min_dist(xs, ys, min_step: float):
    n = len(xs)
    for i in range(n):
        n1 = len(xs)
        xs, ys = _filter_min_step(xs, ys, min_step)
        if len(xs)==n1:
            return xs, ys
    return xs, ys


# --------------------------------------------------------------------------
# Public wrappers (api.py just delegates to these)
# --------------------------------------------------------------------------

def sanitize_polygon(xs, ys, tol=1e-9, closed=True):
    """
    Robustly sanitize a polygon/polyline point sequence so every consecutive
    segment (including the closing segment, if `closed`) has non-zero length.

    Removes:
      - consecutive duplicate/near-duplicate points (distance < tol)
      - a degenerate explicit closing point (last == first, within tol) is
        normalized to a single, exact closure
      - repeats the sweep until stable, since removing a duplicate can
        expose a *new* adjacent duplicate (e.g. after a simplification pass
        collapses several points onto the same spot) -- this is exactly the
        "fix one, a new one appears" issue.

    Parameters
    ----------
    xs, ys : sequences of floats, same length.
    tol : float
        Points closer than this (Euclidean distance) are treated as the
        same point. Same units as your geometry. This is a "these are
        numerically the same point" tolerance (e.g. 1e-9 m), not a shape
        tolerance -- use simplify_polyline/filter_min_step for that.
    closed : bool
        True (default) for a polygon ring (board outline, hole, island
        boundary) where the last point should coincide with the first and
        the wrap-around edge also needs to be non-degenerate. False for an
        open polyline.

    Returns
    -------
    (xs_out, ys_out) : numpy arrays. For closed rings, the output always
        ends with a point exactly equal to (xs_out[0], ys_out[0]) --
        matching ODB++'s own explicit-closure convention.

    Raises
    ------
    ValueError if the input has non-finite values, or if fewer than 3
    (closed) / 2 (open) distinct points remain after sanitizing -- i.e. the
    polygon collapsed to a point or a line and isn't usable geometry anymore.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.shape != ys.shape or xs.ndim != 1:
        raise ValueError("xs and ys must be 1D arrays of the same length")
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        raise ValueError("xs/ys contain NaN or infinite values")
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    if xs.shape[0] == 0:
        raise ValueError("empty polygon")

    pts = list(zip(xs.tolist(), ys.tolist()))

    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    if closed and len(pts) > 1 and dist(pts[0], pts[-1]) <= tol:
        pts = pts[:-1]   # work on the bare ring; re-close explicitly at the end

    # Fixed-point sweep: keep collapsing consecutive/wrap-around duplicates
    # until a full pass makes no change. A single pass isn't enough because
    # dropping points can bring previously-distant points next to each other.
    changed = True
    while changed and len(pts) > 1:
        changed = False
        cleaned = [pts[0]]
        for p in pts[1:]:
            if dist(p, cleaned[-1]) > tol:
                cleaned.append(p)
            else:
                changed = True
        if closed and len(cleaned) > 1 and dist(cleaned[0], cleaned[-1]) <= tol:
            cleaned.pop()
            changed = True
        pts = cleaned

    min_pts = 3 if closed else 2
    if len(pts) < min_pts:
        raise ValueError(
            f"Polygon degenerated to {len(pts)} distinct point(s) after sanitizing "
            f"(tol={tol}); not enough left for a valid {'ring' if closed else 'polyline'}."
        )

    if closed:
        pts.append(pts[0])   # restore exact explicit closure

    xs_out = np.array([p[0] for p in pts], dtype=np.float64)
    ys_out = np.array([p[1] for p in pts], dtype=np.float64)
    return xs_out, ys_out


def simplify_polyline(xs, ys, delta: float):
    """Simplify a polyline with Ramer-Douglas-Peucker.

    Parameters
    ----------
    xs, ys : sequences of floats (lists, tuples, or numpy arrays), same
        length, >= 2 points. Same units as your geometry (e.g. meters).
    delta : float
        Maximum perpendicular deviation (same units as xs/ys) a dropped
        point is allowed to have from the simplified line. Larger delta ->
        more aggressive simplification, fewer points.

    Returns
    -------
    (xs_out, ys_out) : numpy arrays, the simplified point lists. Always
        includes the first and last input points.

    Notes
    -----
    - This treats the input as an open polyline. For a closed polygon loop
      (first point == last point, e.g. one island boundary from an ODB++
      profile), pass it as-is — the shared first/last point is always kept,
      so the loop stays closed. If your polygon *doesn't* explicitly repeat
      the first point at the end, append it before calling and drop the
      duplicate afterward if you need it back in ODB++'s non-repeating form.
    - delta=0 (or very small) returns (close to) all points back.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.shape != ys.shape:
        raise ValueError(f"xs and ys must be the same shape, got {xs.shape} vs {ys.shape}")
    if xs.ndim != 1:
        raise ValueError("xs/ys must be 1D")
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    if xs.shape[0] < 3:
        return xs.copy(), ys.copy()

    keep = _rdp_core(xs, ys, float(delta))
    return sanitize_polygon(xs[keep], ys[keep], 1e-9, closed=False)


# --------------------------------------------------------------------------
# De-zigzag: collapse short "kink" segments sandwiched between two much
# longer, near-parallel segments -- e.g. a round trace end-cap's polygon
# approximation (an n-gon) slightly under/overshooting the tangent point
# where it should cleanly meet a straight segment, leaving a tiny
# sideways step. Picture "---z____" (long run, short near-perpendicular
# kink, another long run): each such kink is removed by extending the
# two long segments' own lines until they meet, replacing the kink's two
# endpoints with that single intersection point, turning it into
# "-----------".
#
# Pure Python, not numba: the core loop mutates a variable-length point
# list (collapsing two points into one, restarting from the freshly
# emitted point) and repeats until a full pass makes no more changes --
# the same "dynamic, Python-object-shaped algorithm" category as
# `viaconnect.py`'s `merge_collinear_chains`, which stays pure Python
# for the identical reason. Per-point cost is a handful of scalar ops,
# so this is not a bottleneck at the vertex counts a single polygon ring
# has (tens to low hundreds) -- unlike the kernel's genuinely
# quadratic-over-thousands-of-elements hot paths.
# --------------------------------------------------------------------------

def _angle_between_deg(d1: tuple[float, float], d2: tuple[float, float]) -> float:
    len1 = math.hypot(d1[0], d1[1])
    len2 = math.hypot(d2[0], d2[1])
    if len1 < 1e-300 or len2 < 1e-300:
        return 180.0
    cos_a = (d1[0] * d2[0] + d1[1] * d2[1]) / (len1 * len2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def _line_intersect(a1, a2, b1, b2) -> tuple[float, float] | None:
    """Intersection of the infinite line through (a1,a2) with the
    infinite line through (b1,b2), or None if they're too close to
    parallel to intersect stably.
    """
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def dezigzag_polyline(
    xs, ys, max_kink_length: float, max_angle_deg: float = 20.0, min_neighbor_factor: float = 3.0
):
    """Remove short zigzag/step artifacts from a polyline.

    A segment counts as a KINK if it's shorter than `max_kink_length`
    AND both of its own neighboring segments (the ones right before and
    right after it) are at least `max_kink_length * min_neighbor_factor`
    long AND within `max_angle_deg` of each other in direction -- i.e.
    the path is basically going straight except for this one short
    sideways step. A genuine small corner (the two neighbors pointing in
    clearly different directions) is left alone.

    Parameters
    ----------
    xs, ys : sequences of floats, an open OR closed polyline (if closed,
        i.e. first point == last point -- same convention as
        `Polygon.cxs`/`cys` -- the shared first/last point is always
        kept, exactly like `simplify_polyline`; a kink straddling that
        exact seam index is not detected, the same pinned-endpoint
        limitation `simplify_polyline` already has).
    max_kink_length : float
        Same units as xs/ys. Should be well under your smallest genuine
        feature size -- e.g. a fraction of your trace width.
    max_angle_deg : float
        Maximum direction difference (degrees) allowed between a kink's
        two neighboring segments for it to still count as "basically
        straight". Default 20.
    min_neighbor_factor : float
        Each neighboring segment must be at least this many times
        `max_kink_length` long. Default 3 -- guards against collapsing
        a run of several genuinely small features that merely happen to
        be near-parallel.

    Returns
    -------
    (xs_out, ys_out) : numpy arrays, same open/closed convention as the input.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.shape != ys.shape:
        raise ValueError(f"xs and ys must be the same shape, got {xs.shape} vs {ys.shape}")
    if xs.ndim != 1:
        raise ValueError("xs/ys must be 1D")
    if max_kink_length < 0:
        raise ValueError(f"max_kink_length must be >= 0, got {max_kink_length}")
    if xs.shape[0] < 5:
        return xs.copy(), ys.copy()

    pts = list(zip(xs.tolist(), ys.tolist()))
    min_neighbor = max_kink_length * min_neighbor_factor

    changed = True
    while changed and len(pts) >= 5:
        changed = False
        n = len(pts)
        out = [pts[0]]
        i = 1
        while i < n - 2:
            p_prev = out[-1]
            p_i = pts[i]
            p_next = pts[i + 1]
            p_next2 = pts[i + 2]

            kink_len = math.hypot(p_next[0] - p_i[0], p_next[1] - p_i[1])
            if 0.0 < kink_len <= max_kink_length:
                d1 = (p_i[0] - p_prev[0], p_i[1] - p_prev[1])
                d2 = (p_next2[0] - p_next[0], p_next2[1] - p_next[1])
                len1 = math.hypot(*d1)
                len2 = math.hypot(*d2)
                if (len1 >= min_neighbor and len2 >= min_neighbor
                        and _angle_between_deg(d1, d2) <= max_angle_deg):
                    # The common real case is two near-PARALLEL runs
                    # offset sideways by the kink -- their lines have
                    # no true intersection (or, for a merely-close-to-
                    # parallel pair, one so far away it'd be numerically
                    # unstable to use). Midpoint-of-the-kink is always
                    # local and stable, so it's the fallback whenever
                    # the "true" intersection isn't a safe, nearby
                    # point -- and for a genuine shallow-angle corner
                    # (the two runs actually converging nearby), the
                    # two choices differ by at most about the kink's
                    # own tiny length anyway.
                    midpoint = ((p_i[0] + p_next[0]) / 2.0, (p_i[1] + p_next[1]) / 2.0)
                    ipt = _line_intersect(p_prev, p_i, p_next, p_next2)
                    if ipt is None or math.hypot(ipt[0] - p_i[0], ipt[1] - p_i[1]) > 2.0 * max(len1, len2):
                        ipt = midpoint
                    out.append(ipt)
                    i += 2
                    changed = True
                    continue
            out.append(p_i)
            i += 1
        out.extend(pts[i:])
        pts = out

    xs_out = np.array([p[0] for p in pts], dtype=np.float64)
    ys_out = np.array([p[1] for p in pts], dtype=np.float64)
    return xs_out, ys_out