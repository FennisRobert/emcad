"""
Public kernel API -- thin, fully-typed wrappers over the compiled Rust
kernel (`emcad._rs`, built from `rust/`).

No implementation logic lives in this file on purpose: it's the
interface boundary the rest of the package and outside callers use, and
the place each operation's contract is documented. Rust source map:

    rust/src/primitives.rs   convex_hull, edge_*_intersections, sanitize/simplify/dezigzag
    rust/src/geom.rs         is_inside, Polygon.point_inside / is_inside
    rust/src/exact.rs        exact grid-integer predicates + split finder
    rust/src/arrangement.rs  arrangement engine, is_simple_ring
    rust/src/boolean.rs      add / intersect / subtract / join (+ bbox clustering)
    rust/src/fragment.rs     poly_fragment
    rust/src/keyhole.rs      dekeyhole_polygon(s)
    rust/src/regularize.rs   regularize_polyline
    rust/src/via.rs          via proximity graph kernels (used by viaconnect.py)

Every polygon-level operation is ONE call into Rust over the whole
input: Rust reads `Polygon.xs`/`.ys`/`.holes` directly and builds the
result `Polygon` objects itself (see `benchmarks/RESULTS.md` for why
that boundary was drawn there).
"""

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import numpy as np

from .. import _rs
from ._constants import DEFAULT_MERGE_TOL, DEFAULT_T_TOL, DEFAULT_AREA_TOL

if TYPE_CHECKING:
    from ..poly import Polygon


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def convex_hull(xs: Iterable[float], ys: Iterable[float]) -> np.ndarray:
    """Convex hull of a point set (monotone chain).

    Returns:
        int64 array of indices into xs/ys forming the hull, in order,
        without a repeated closing index.
    """
    return _rs.convex_hull(_f64(xs), _f64(ys))


def edge_self_intersections(pts_start: np.ndarray, pts_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All crossings among one set of edges (every ordered pair (i, j)
    whose closed segments intersect; parallel pairs are skipped).

    Args:
        pts_start, pts_end: (2, N) edge start / end coordinates.

    Returns:
        (ids, coords): (2, K) int64 edge-index pairs and (2, K) float64
        crossing coordinates.
    """
    s, e = _f64(pts_start), _f64(pts_end)
    return _rs.edge_intersections(s, e, s, e)


def edge_cross_intersections(
    pts1_start: np.ndarray, pts1_end: np.ndarray, pts2_start: np.ndarray, pts2_end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """All crossings between two distinct edge sets ((2, N) and (2, M)
    start/end arrays). Same return shape as `edge_self_intersections`,
    ids[0] indexing the first set and ids[1] the second.
    """
    return _rs.edge_intersections(_f64(pts1_start), _f64(pts1_end), _f64(pts2_start), _f64(pts2_end))


def is_inside(xs: np.ndarray, ys: np.ndarray, x: float, y: float, include_boundary: bool = True) -> bool:
    """Point-in-polygon test (even-odd ray casting) for a single open
    ring. A point exactly on an edge returns `include_boundary`.
    """
    return _rs.is_inside(xs, ys, x, y, include_boundary)


def sanitize_polygon(xs, ys, tol: float = 1e-9, closed: bool = True):
    """Remove consecutive duplicate / near-duplicate points (distance <=
    tol) until stable, so every segment has non-zero length.

    For `closed=True` the input may or may not repeat its first point;
    the output always does (ODB++'s explicit-closure convention).

    Raises:
        ValueError: non-finite input, or fewer than 3 (closed) / 2
            (open) distinct points left.
    """
    return _rs.sanitize_polygon(xs, ys, tol, closed)


def simplify_polyline(xs, ys, delta: float):
    """Ramer-Douglas-Peucker simplification of an open polyline (a closed
    loop passed with its repeated first point stays closed -- the end
    points are always kept). `delta` is the maximum perpendicular
    deviation a dropped point may have, in the same units as xs/ys.

    Note RDP keeps *extreme* points: on an outline with small bumps (via
    pads on a straight edge) it keeps the bump tips, which skews long
    straight edges -- use `regularize_polyline` for that kind of input.
    """
    return _rs.simplify_polyline(xs, ys, delta)


def dezigzag_polyline(
    xs, ys, max_kink_length: float, max_angle_deg: float = 20.0, min_neighbor_factor: float = 3.0
):
    """Remove short zigzag/step artifacts: a segment shorter than
    `max_kink_length` whose two neighbors are both at least
    `max_kink_length * min_neighbor_factor` long and within
    `max_angle_deg` of each other in direction is collapsed (its two end
    points replaced by the neighbors' line intersection, or its midpoint
    when that intersection isn't nearby). Repeats until stable. Open or
    closed input; the first/last point is always kept.
    """
    return _rs.dezigzag_polyline(xs, ys, max_kink_length, max_angle_deg, min_neighbor_factor)


def regularize_polyline(
    xs,
    ys,
    tol: float,
    dangle_deg: float = 5.0,
    angle_tol_deg: float = 0.05,
    min_anchor_len: float | None = None,
    offset_tol: float = 5e-6,
    vw_area: float | None = None,
):
    """Map-making style outline regularization of a closed ring: snap
    the outline back onto its dominant straight lines and drop small
    protrusions (e.g. via pads poking out of a copper edge), WITHOUT
    skewing the long edges the way RDP does.

    1. Anchors: edges whose direction is a multiple of `dangle_deg`
       (within `angle_tol_deg`). An edge counts once the total length of
       all collinear edges on its snapped line (offsets within
       `offset_tol`) reaches `min_anchor_len` -- pieces of one real edge
       chopped up by pads vote together; a lone arc segment doesn't.
    2. Merge: consecutive anchors on the same line fuse when everything
       between them stays within `tol` of that line (and progresses
       forward along it) -- the bump is dropped, the edge is exact again.
    3. Corners: two consecutive non-parallel lines are joined at their
       intersection when the detour between them stays within `tol` of
       the resulting corner -- rebuilds corners hidden under a pad.
    4. Everything else (real arcs: trace end caps, round pads) is kept,
       Visvalingam-Whyatt simplified with area threshold `vw_area`, its
       end points projected exactly onto the neighboring lines.

    Args:
        xs, ys: one ring, open or closed (first point repeated); the
            output keeps the same convention.
        tol: largest detour (same units as xs/ys) that may be flattened
            -- bigger than the bumps to remove, smaller than the
            smallest real feature to keep (e.g. a slot width).
        dangle_deg: snap-angle step; edges are only anchored on
            multiples of it.
        min_anchor_len: default `2 * tol`.
        vw_area: default `tol**2`.

    Returns:
        (xs, ys) float64 arrays.
    """
    return _rs.regularize_polyline(xs, ys, tol, dangle_deg, angle_tol_deg, min_anchor_len, offset_tol, vw_area)


def is_simple_ring(xs, ys, merge_tol: float = DEFAULT_MERGE_TOL) -> bool:
    """True if a closed ring (open form: first point != last) has no
    transversal crossing, T-junction, or collinear overlap between any
    two of its own edges, other than the vertex consecutive edges share.
    Uses the same exact grid-integer split finder as the arrangement
    engine (grid spacing `merge_tol`).
    """
    return _rs.is_simple_ring(xs, ys, merge_tol)


# --------------------------------------------------------------------------
# polygon operations
# --------------------------------------------------------------------------

def poly_fragment(
    polys: list["Polygon"],
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
    keep: str = "positive",
    filter_to_originals: bool = True,
    debug: bool = False,
) -> list["Polygon"]:
    """Fragment N polygons into mutually disjoint sub-polygons of their
    combined planar arrangement.

    Args:
        polys: input polygons. May overlap each other and/or be
            individually self-intersecting; holes are part of the
            arrangement.
        merge_tol: absolute distance below which two points are
            considered the same vertex.
        t_tol: parametric (0..1, per-edge) tolerance for treating a
            crossing as landing exactly on an existing endpoint.
        area_tol: faces with |signed area| below this are dropped as
            numerical noise.
        keep: "positive" (default -- the interior fragments, nested
            into polygons-with-holes via a containment forest),
            "negative" (outer/component boundaries, mostly for
            debugging), or "all".
        filter_to_originals: drop any fragment that doesn't fall inside
            at least one input polygon.
        debug: if True, plot the returned fragments.
    """
    result = _rs.poly_fragment(polys, merge_tol, t_tol, area_tol, keep, filter_to_originals)
    if debug:
        from ..plot import GeometryPlotter

        pl = GeometryPlotter()
        for poly in result:
            pl.add_polygon(poly, alpha=0.2)
        pl.show()
    return result


def add_polygons(
    *polys: "Polygon",
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Boolean union (fuse) of N polygons: keep anything inside at least
    one of them. Holes (at any nesting depth) of every input are
    respected; the result is nested polygons-with-holes.

    Built on an exact-predicate arrangement: coordinates snap to a
    `merge_tol` integer grid and every touch/crossing/overlap decision
    is exact. Operands whose bounding boxes are farther apart than
    `merge_tol` are processed as independent clusters. `t_tol` is
    accepted for API compatibility and unused.
    """
    return _rs.boolean_op(polys, "add", 0, merge_tol, area_tol)


def intersect_polygons(
    *polys: "Polygon",
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Boolean intersection of N polygons: keep only what's inside every
    one of them. See `add_polygons` for the engine and parameters.
    """
    return _rs.boolean_op(polys, "intersect", 0, merge_tol, area_tol)


def subtract_polygons(
    add: tuple["Polygon", ...],
    subtract: tuple["Polygon", ...],
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Boolean subtraction: union(add) minus union(subtract), in one pass
    (not pairwise) -- a region survives iff it's inside at least one
    `add` polygon and inside none of the `subtract` polygons. See
    `add_polygons` for the engine and parameters.
    """
    add = list(add)
    return _rs.boolean_op(add + list(subtract), "subtract", len(add), merge_tol, area_tol)


def join_polygons(
    polys: list["Polygon"],
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Fuse polygons that meet edge-to-edge into fewer, larger polygons,
    dissolving every boundary shared between two of them -- and
    assembling whatever holes/islands that touching implies (e.g. four
    strips forming a picture frame come back as one polygon with a hole).

    Unlike `add_polygons`, join refuses overlapping input: it raises
    `GeometryException` if any two inputs' interiors genuinely overlap
    (a transversal crossing, a same-side collinear overlap, or one input
    fully inside another), since it won't guess how to resolve that.
    A single input is returned unchanged (same object).
    """
    if len(polys) <= 1:
        return list(polys)
    return _rs.boolean_op(polys, "join", 0, merge_tol, area_tol)


def dekeyhole_polygon(poly: "Polygon", tol: float = DEFAULT_MERGE_TOL) -> "Polygon":
    """Replace keyhole bridges with real `Polygon.holes`.

    Keyhole encoding (common in ODB++/Gerber) represents a hole by
    letting the outline dip in, trace the hole, and come back out --
    via a single shared vertex or a two-point slit. Both show up as a
    repeated vertex (within `tol`); each one splits off a sub-ring, and
    the sub-rings are nested by geometric containment. Recurses into
    `poly.holes` too.

    Raises:
        GeometryException: an extracted sub-ring isn't nested inside the
            outer boundary (malformed input).
    """
    return _rs.dekeyhole_polygon(poly, tol)


def dekeyhole_polygons(polys: list["Polygon"], tol: float = DEFAULT_MERGE_TOL) -> list["Polygon"]:
    """Batch version of `dekeyhole_polygon` (processed in parallel)."""
    return _rs.dekeyhole_polygons(polys, tol)
