"""
Public kernel API -- thin, fully-typed wrappers only.

No implementation logic lives in this file on purpose: it's the
interface boundary the rest of the package and outside callers use.
Every actual implementation lives in an underscore-prefixed module:

    _ch.py               convex_hull
    _primitives.py        edge_self_intersections / edge_cross_intersections / is_inside
    _simplify.py          sanitize_polygon / simplify_polyline / dezigzag_polyline
    _fragment.py           poly_fragment
    _boolean_ops.py        add_polygons / intersect_polygons / subtract_polygons
    _arrangement.py         is_simple_ring (plus the underlying engine build_arrangement/label_and_assemble use)
    _join.py                join_polygons
    _keyhole.py             dekeyhole_polygon / dekeyhole_polygons

Import structure, and why: `_fragment.py`, `_boolean_ops.py`, and
`_primitives.py`/`_ch.py`/`_simplify.py` don't need `Polygon` at
module-load time (any type hints resolve lazily thanks to
`from __future__ import annotations`, and their one genuine runtime
use of `Polygon` -- reconstructing a result -- is deferred inside the
function body that needs it), so they're all safe to import normally,
right here at the top.

`_join.py` is the one exception: it constructs `Polygon` and raises
`GeometryException` throughout its logic, so it needs `Polygon`
imported at ITS OWN module top level. `poly.py` imports the primitives
above from THIS module, so if this module imported `_join` at its own
top level too, loading `poly.py` would transitively try to load
`_join.py` -- which would try to import from `poly.py` -- while
`poly.py` is still mid-load. That's a circular import. `_boolean_ops.py`
already defers its own `_join` import for the same reason; `join_polygons`
below does the same.
"""

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import numpy as np

from ._ch import convex_hull as _convex_hull_impl
from ._primitives import (
    edge_self_intersections as _edge_self_intersections_impl,
    edge_cross_intersections as _edge_cross_intersections_impl,
    is_inside as _is_inside_impl,
)
from ._simplify import (
    sanitize_polygon as _sanitize_polygon_impl,
    simplify_polyline as _simplify_polyline_impl,
    dezigzag_polyline as _dezigzag_polyline_impl,
)
from ._fragment import poly_fragment as _poly_fragment_impl
from ._boolean_ops import (
    add_polygons as _add_polygons_impl,
    intersect_polygons as _intersect_polygons_impl,
    subtract_polygons as _subtract_polygons_impl,
)
from ._arrangement import is_simple_ring as _is_simple_ring_impl
from ._keyhole import (
    dekeyhole_polygon as _dekeyhole_polygon_impl,
    dekeyhole_polygons as _dekeyhole_polygons_impl,
)
from ._constants import DEFAULT_MERGE_TOL, DEFAULT_T_TOL, DEFAULT_AREA_TOL

if TYPE_CHECKING:
    from ..poly import Polygon


def convex_hull(xs: Iterable[float], ys: Iterable[float]) -> np.ndarray:
    """Compute the convex hull for a set of points. See `_ch.convex_hull`."""
    return _convex_hull_impl(xs, ys)


def edge_self_intersections(pts_start: np.ndarray, pts_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the self intersection points of a set of edges. See
    `_primitives.edge_self_intersections`.
    """
    return _edge_self_intersections_impl(pts_start, pts_end)


def edge_cross_intersections(
    pts1_start: np.ndarray, pts1_end: np.ndarray, pts2_start: np.ndarray, pts2_end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute all edge intersections between two distinct edge sets. See
    `_primitives.edge_cross_intersections`.
    """
    return _edge_cross_intersections_impl(pts1_start, pts1_end, pts2_start, pts2_end)


def is_inside(xs: np.ndarray, ys: np.ndarray, x: float, y: float, include_boundary: bool = True) -> bool:
    """Point-in-polygon test for a single ring. See `_primitives.is_inside`."""
    return _is_inside_impl(xs, ys, x, y, include_boundary)


def sanitize_polygon(xs, ys, tol: float = 1e-9, closed: bool = True):
    """Remove degenerate/duplicate points from a polygon or polyline. See
    `_simplify.sanitize_polygon`.
    """
    return _sanitize_polygon_impl(xs, ys, tol, closed)


def simplify_polyline(xs, ys, delta: float):
    """Simplify a polyline with Ramer-Douglas-Peucker. See
    `_simplify.simplify_polyline`.
    """
    return _simplify_polyline_impl(xs, ys, delta)


def dezigzag_polyline(
    xs, ys, max_kink_length: float, max_angle_deg: float = 20.0, min_neighbor_factor: float = 3.0
):
    """Remove short zigzag/step artifacts from a polyline. See
    `_simplify.dezigzag_polyline`.
    """
    return _dezigzag_polyline_impl(xs, ys, max_kink_length, max_angle_deg, min_neighbor_factor)


def is_simple_ring(xs, ys, merge_tol: float = DEFAULT_MERGE_TOL) -> bool:
    """True if a closed ring has no self-intersections. See
    `_arrangement.is_simple_ring`.
    """
    return _is_simple_ring_impl(xs, ys, merge_tol)


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
    combined planar arrangement. See `_fragment.poly_fragment` for the
    full docstring and parameter details.
    """
    return _poly_fragment_impl(
        polys,
        merge_tol=merge_tol,
        t_tol=t_tol,
        area_tol=area_tol,
        keep=keep,
        filter_to_originals=filter_to_originals,
        debug=debug,
    )


def add_polygons(
    *polys: "Polygon",
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Boolean union (fuse) of N polygons. See `_boolean_ops.add_polygons`."""
    return _add_polygons_impl(*polys, merge_tol=merge_tol, t_tol=t_tol, area_tol=area_tol)


def intersect_polygons(
    *polys: "Polygon",
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Boolean intersection of N polygons. See `_boolean_ops.intersect_polygons`."""
    return _intersect_polygons_impl(*polys, merge_tol=merge_tol, t_tol=t_tol, area_tol=area_tol)


def subtract_polygons(
    add: tuple["Polygon", ...],
    subtract: tuple["Polygon", ...],
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Boolean subtraction: union(add) minus union(subtract). See
    `_boolean_ops.subtract_polygons`.
    """
    return _subtract_polygons_impl(add, subtract, merge_tol=merge_tol, t_tol=t_tol, area_tol=area_tol)


def join_polygons(
    polys: list["Polygon"],
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Fuse polygons that meet edge-to-edge into fewer, larger polygons.
    See `_join.join_polygons` for the full docstring, including exactly
    what it refuses to do and why. Imported lazily -- see this module's
    docstring.
    """
    from ._join import join_polygons as _impl

    return _impl(polys, merge_tol=merge_tol, t_tol=t_tol, area_tol=area_tol)


def dekeyhole_polygon(poly: "Polygon", tol: float = DEFAULT_MERGE_TOL) -> "Polygon":
    """Replace a polygon's keyhole bridges with real `Polygon.holes`.
    See `_keyhole.dekeyhole_polygon` for the full docstring.
    """
    return _dekeyhole_polygon_impl(poly, tol=tol)


def dekeyhole_polygons(polys: list["Polygon"], tol: float = DEFAULT_MERGE_TOL) -> list["Polygon"]:
    """Batch version of `dekeyhole_polygon`. See `_keyhole.dekeyhole_polygons`."""
    return _dekeyhole_polygons_impl(polys, tol=tol)