"""
Exact, grid-snapped integer geometry predicates -- the numerical
foundation the rest of the arrangement engine (`_arrangement.py`)
builds on.

Why this exists
----------------
Every hard bug this kernel has hit (a T-junction the crossing detector
doesn't see; two rings that overlap "the same side" of a shared edge
undetected; a `point_inside()` failure on a degenerate sliver) traces
back to the same root cause: *combinatorial* decisions -- is point C
left of, right of, or exactly on line AB? do two segments cross, touch,
or overlap? -- were being made with float64 arithmetic and ad hoc
epsilon thresholds, independently, in more than one place, with no
guarantee any two of those places agree with each other.

The fix used throughout this module is the same one Clipper/Clipper2
(the most widely deployed 2D polygon-boolean library) and CGAL's exact
kernels both rely on: snap every input coordinate to a fixed integer
grid once, up front, then make every subsequent combinatorial decision
with *exact* integer arithmetic on that grid. Two segments either
share a grid point or they don't -- there is no "almost", so no two
predicates can ever disagree. `merge_tol` (the same tolerance already
used everywhere else in this package) sets the grid spacing, so this
keeps the same practical precision as before -- the difference is that
every decision at that precision is now exact, not fuzzy.

Overflow safety
----------------
With `scale = 1 / merge_tol` (default merge_tol=1e-7 -> scale=1e7),
grid coordinates stay within +-1e9 for any input geometry up to +-100
meters in extent -- far beyond anything this package's PCB/EM-geometry
callers produce. `orient2d_i64`'s cross product is then bounded by
(2e9)*(2e9) = 4e18, safely under int64's ~9.22e18 max. Callers with
larger geometry should pass a coarser `merge_tol` (i.e. a smaller
scale) accordingly.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def make_scale(merge_tol: float) -> float:
    """Grid points-per-unit-length for `to_grid`, from the same
    `merge_tol` used elsewhere in this package."""
    return 1.0 / merge_tol


def to_grid(xy: np.ndarray, scale: float) -> np.ndarray:
    """Snap (2, N) float64 coordinates to the int64 grid at `scale`
    points per unit length. Two inputs within `1/scale` of each other
    snap to the same grid point -- same practical meaning as
    `merge_tol`, but now a decidable integer equality instead of a
    distance threshold.
    """
    return np.round(xy * scale).astype(np.int64)


@njit("int64(int64, int64, int64, int64, int64, int64)", cache=True)
def orient2d_i64(ax, ay, bx, by, cx, cy):
    """Exact sign of the cross product (B-A) x (C-A): >0 if A,B,C turn
    left (C strictly left of directed line A->B), <0 if right, ==0 iff
    A, B, C are exactly collinear on the grid. No epsilon anywhere --
    grid-integer arithmetic, so this is exact by construction.
    """
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit("boolean(int64, int64, int64, int64, int64, int64)", cache=True)
def on_segment_i64(ax, ay, bx, by, px, py):
    """True iff P is collinear with A->B (caller must have already
    checked `orient2d_i64(ax,ay,bx,by,px,py) == 0`) AND lies within
    the closed bounding box of A and B -- i.e. P is on the closed
    segment [A, B], exactly.
    """
    return (min(ax, bx) <= px <= max(ax, bx)) and (min(ay, by) <= py <= max(ay, by))


# --------------------------------------------------------------------------
# Segment relation classification. Every two-edge relationship this
# kernel needs to reason about -- proper crossing, endpoint touch, a
# vertex landing on the other's open interior (a T-junction), or a
# collinear overlap of positive length -- comes out of ONE consistent
# exact test, replacing what used to be two independently-epsilon'd
# detectors (`_primitives.edge_self_intersections`'s crossing-only
# test, and `_join.py`'s separate T-junction/collinear-overlap finder).
#
# Encoding of the returned `kind`:
#   0 = no relation (segments don't meet at all)
#   1 = proper transversal crossing, interior to both -- genuine
#       interior overlap between whatever these two segments bound
#   2 = collinear overlap of positive length -- `out` holds the two
#       endpoints of the shared sub-segment (in an arbitrary but
#       consistent order)
#   (endpoint touches and T-junctions need no separate code: they fall
#   out as split points at the relevant vertex, handled by the caller
#   via `on_segment_i64` checks on the four endpoints against the
#   OTHER segment -- see `_arrangement.find_all_splits`.)
# --------------------------------------------------------------------------

@njit(
    "Tuple((int64, int64[:]))(int64, int64, int64, int64, int64, int64, int64, int64)",
    cache=True,
)
def classify_segment_pair_i64(ax, ay, bx, by, cx, cy, dx, dy):
    """Exact relation between closed segments A->B and C->D.

    Returns (kind, out) where `out` is a flat [x0, y0, x1, y1] array
    (unused entries zero): for kind==1, (x0,y0) is the unique crossing
    point (integer-exact -- see note below); for kind==2, (x0,y0) and
    (x1,y1) are the two endpoints of the shared collinear sub-segment.

    The crossing point for kind==1 is computed by intersecting the two
    *lines* using exact-integer numerator/denominator and rounding to
    the nearest grid point -- this is the one place a sub-grid-unit
    rounding choice is made (two segments crossing near-exactly at a
    grid point round to that point consistently, since both callers
    see the same A,B,C,D). This is the "tiny gap" of the exact scheme:
    the *combinatorial* fact "these segments cross" is exact; the
    *reported coordinate* of a transversal crossing is grid-rounded,
    exactly like every input vertex already is.
    """
    out = np.zeros(4, dtype=np.int64)

    d1 = orient2d_i64(cx, cy, dx, dy, ax, ay)
    d2 = orient2d_i64(cx, cy, dx, dy, bx, by)
    d3 = orient2d_i64(ax, ay, bx, by, cx, cy)
    d4 = orient2d_i64(ax, ay, bx, by, dx, dy)

    if d1 == 0 and d2 == 0 and d3 == 0 and d4 == 0:
        # Collinear -- overlap is a 1D interval problem projected onto
        # whichever axis has the larger extent (avoids a degenerate
        # zero-length projection for axis-aligned segments).
        if abs(bx - ax) >= abs(by - ay):
            lo = min(ax, bx)
            hi = max(ax, bx)
            plo = min(cx, dx)
            phi = max(cx, dx)
            olo = max(lo, plo)
            ohi = min(hi, phi)
            if olo > ohi:
                return 0, out
            if olo == ohi:
                # single shared point, not a positive-length overlap --
                # treat as a touch, not an overlap (kind 2 requires
                # positive length so callers can split cleanly).
                return 0, out
            out[0] = olo
            out[2] = ohi
            if bx == ax:
                out[1] = ay
                out[3] = ay
            else:
                out[1] = ay + (by - ay) * (olo - ax) // (bx - ax)
                out[3] = ay + (by - ay) * (ohi - ax) // (bx - ax)
            return 2, out
        else:
            lo = min(ay, by)
            hi = max(ay, by)
            plo = min(cy, dy)
            phi = max(cy, dy)
            olo = max(lo, plo)
            ohi = min(hi, phi)
            if olo > ohi:
                return 0, out
            if olo == ohi:
                return 0, out
            out[1] = olo
            out[3] = ohi
            if by == ay:
                out[0] = ax
                out[2] = ax
            else:
                out[0] = ax + (bx - ax) * (olo - ay) // (by - ay)
                out[2] = ax + (bx - ax) * (ohi - ay) // (by - ay)
            return 2, out

    if d3 == 0 and on_segment_i64(ax, ay, bx, by, cx, cy):
        # C lies on segment AB (endpoint touch or T-junction) -- not a
        # transversal crossing; the caller discovers this by testing
        # each endpoint against the other segment directly, so nothing
        # to report here.
        return 0, out
    if d4 == 0 and on_segment_i64(ax, ay, bx, by, dx, dy):
        return 0, out
    if d1 == 0 and on_segment_i64(cx, cy, dx, dy, ax, ay):
        return 0, out
    if d2 == 0 and on_segment_i64(cx, cy, dx, dy, bx, by):
        return 0, out

    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        # Genuine transversal crossing: A,B strictly straddle line CD
        # and C,D strictly straddle line AB. Parametrize along A->B
        # (t_num/denom, both built ONLY from coordinate *differences*,
        # never raw grid coordinates) and round the resulting point to
        # the grid -- this is the one place a sub-grid-unit rounding
        # choice is made; see this function's docstring.
        #
        # Deliberately NOT the classical "ax*by - ay*bx" cross-multiply
        # form: that formula multiplies two already-large raw
        # coordinates together before combining, which overflows
        # int64 for realistic geometry (e.g. a 10-unit-wide polygon at
        # the default 1e-7 grid already scales coordinates to ~1e8,
        # and that formula's intermediate products reach ~1e24). Using
        # only differences keeps every intermediate product bounded by
        # (grid extent)^2, matching `orient2d_i64`'s own overflow
        # analysis in the module docstring.
        denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
        # denom != 0 here: a zero denominator means AB parallel to CD,
        # which is only consistent with the all-collinear branch above
        # (already handled) since two DISTINCT parallel, non-collinear
        # lines can never have d1..d4 straddle both ways.
        t_num = (cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)
        t = t_num / denom
        out[0] = round(ax + t * (bx - ax))
        out[1] = round(ay + t * (by - ay))
        return 1, out

    return 0, out
