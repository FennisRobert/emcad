"""
Convert keyhole-encoded polygons (a single self-touching ring that
represents holes via a bridge/slit to the outer boundary, rather than
Polygon's native `holes=[...]`) into proper multi-loop Polygon objects.

Keyhole encoding is common in formats without first-class hole
support (ODB++, Gerber, and plenty of older CAD interchange formats):
instead of a separate hole ring, the outline "dips in" to trace the
hole boundary and comes back out, either at a single shared vertex (a
zero-length bridge / pinch point) or via an explicit two-point bridge
edge traversed forward then immediately back (a slit). Both styles
show up as a repeated vertex coordinate somewhere in the ring -- that
repetition is the only signal this module needs; it doesn't care which
style produced it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._constants import DEFAULT_MERGE_TOL

if TYPE_CHECKING:
    from ..poly import Polygon


def _signed_area(xs: np.ndarray, ys: np.ndarray) -> float:
    x1 = np.roll(xs, -1)
    y1 = np.roll(ys, -1)
    return 0.5 * float(np.sum(xs * y1 - x1 * ys))


def _dekeyhole_ring(xs: list, ys: list, tol: float):
    """Split one ring into its outer path plus a list of extracted
    sub-rings, using a repeated-vertex coordinate as the splitting
    signal.

    Each extracted sub-ring *includes* the shared vertex (it's a
    genuine corner of that sub-shape too, whichever keyhole style
    produced it), and a bridge that closes without enclosing any real
    area (fewer than 3 points once extracted -- the leftover of an
    explicit P-Q-...-Q-P slit after its hole loop is pulled out) is
    silently discarded rather than kept as a fake hole.
    """

    def key(x: float, y: float) -> tuple[int, int]:
        return (round(x / tol), round(y / tol))

    stack: list[tuple[float, float]] = []
    pos_of: dict[tuple[int, int], int] = {}
    extracted: list[list[tuple[float, float]]] = []

    for x, y in zip(xs, ys):
        k = key(x, y)
        if k in pos_of:
            p = pos_of[k]
            sub = stack[p:]  # includes the shared vertex -- real corner of the sub-shape too
            for sv in stack[p + 1:]:
                pos_of.pop(key(*sv), None)
            del stack[p + 1:]
            if len(sub) >= 3:
                extracted.append(sub)
            # else: a <3-point remnant is just bridge cleanup, not a real hole -- discard
        else:
            pos_of[k] = len(stack)
            stack.append((x, y))

    return stack, extracted


def _dekeyhole_recursive(xs: list, ys: list, tol: float):
    """Recursively de-keyhole, in case an extracted sub-ring has its own
    embedded keyhole (e.g. a hole with an island bridged inside it).
    Returns (outer_points, flat_list_of_all_extracted_subrings) --
    flattened regardless of nesting depth; nesting is reconstructed
    afterward from geometry (see `_assemble_holes`), not from how deep
    the keyhole recursion went.
    """
    outer, extracted = _dekeyhole_ring(xs, ys, tol)
    flat: list[list[tuple[float, float]]] = []
    for sub in extracted:
        sub_xs = [p[0] for p in sub]
        sub_ys = [p[1] for p in sub]
        sub_outer, sub_flat = _dekeyhole_recursive(sub_xs, sub_ys, tol)
        flat.append(sub_outer)
        flat.extend(sub_flat)
    return outer, flat


def _assemble_holes(outer: "Polygon", sub_rings: list["Polygon"]) -> "Polygon":
    """Nest a flat list of extracted rings under the outer ring by
    geometric containment (smallest enclosing ring wins -- the same
    containment-forest approach `_join.py` uses to assemble holes).
    Same reasoning applies here as there: the raw keyhole path's
    winding isn't a reliable solid/hole signal, containment is.
    """
    from ..poly import Polygon, GeometryException

    if not sub_rings:
        return outer

    all_polys = [outer] + sub_rings
    n = len(all_polys)
    areas = [abs(_signed_area(p.axs, p.ays)) for p in all_polys]
    points = [p.point_inside() for p in all_polys]

    parent: list[int | None] = [None] * n
    for i in range(1, n):  # index 0 (the outer ring) is always the root
        xi, yi = points[i]
        ai = areas[i]
        best_area = None
        best_j = None
        for j in range(n):
            if j == i or areas[j] <= ai:
                continue
            if all_polys[j].is_inside(xi, yi):
                if best_area is None or areas[j] < best_area:
                    best_area, best_j = areas[j], j
        parent[i] = best_j
        if parent[i] is None:
            raise GeometryException(
                "dekeyhole_polygon: an extracted sub-ring isn't nested inside the "
                "outer boundary (or any other extracted ring) -- this doesn't look "
                "like a valid keyhole polygon."
            )

    children_of: dict[int, list[int]] = {}
    for i, p in enumerate(parent):
        if p is not None:
            children_of.setdefault(p, []).append(i)

    order = sorted(range(n), key=lambda i: areas[i])  # children (smaller) before parents
    built: dict[int, "Polygon"] = {}
    for i in order:
        my_holes = [built[c] for c in children_of.get(i, [])]
        # _construct_verified, not Polygon(...): we've already verified
        # nesting above via containment, and this ring legitimately
        # touches its parent at the keyhole junction it came from --
        # exactly the case Polygon's own crossing-based hole validation
        # exists to reject for hand-built `holes=`, but not appropriate
        # here.
        built[i] = Polygon._construct_verified(all_polys[i].xs, all_polys[i].ys, holes=my_holes or None)

    return built[0]


def dekeyhole_polygon(poly: "Polygon", tol: float = DEFAULT_MERGE_TOL) -> "Polygon":
    """Replace a polygon's keyhole bridges with real `Polygon.holes`.

    Detects keyhole junctions as repeated vertex coordinates in
    `poly.xs`/`poly.ys` (see the module docstring for the two common
    keyhole styles this covers) and reconstructs the proper nested
    hole/island structure from geometry. Recurses into `poly.holes`
    too, in case a polygon that already has real holes also has one of
    them keyhole-encoded internally.

    A polygon with no repeated vertices at all has nothing to do and is
    returned as an equivalent Polygon.

    Args:
        poly: input polygon. Its own `xs`/`ys` are scanned for keyhole
            junctions; any existing `poly.holes` are recursively
            processed the same way, not just passed through unchanged.
        tol: absolute distance below which two vertices are considered
            the same point.

    Returns:
        An equivalent Polygon with keyhole bridges replaced by real
        `holes=[...]` entries.

    Raises:
        GeometryException: if a detected keyhole junction doesn't
            resolve into a properly nested sub-ring (malformed input).
    """
    from ..poly import Polygon

    outer_pts, sub_rings_pts = _dekeyhole_recursive(poly.xs, poly.ys, tol)

    outer = Polygon([p[0] for p in outer_pts], [p[1] for p in outer_pts])
    sub_polys = [Polygon([p[0] for p in ring], [p[1] for p in ring]) for ring in sub_rings_pts]

    result = _assemble_holes(outer, sub_polys)

    if poly.has_holes:
        extra_holes = [dekeyhole_polygon(h, tol) for h in poly.holes]
        result = Polygon._construct_verified(result.xs, result.ys, holes=list(result.holes) + extra_holes)

    return result


def dekeyhole_polygons(polys: list["Polygon"], tol: float = DEFAULT_MERGE_TOL) -> list["Polygon"]:
    """Batch version of `dekeyhole_polygon` -- maps it over a list."""
    return [dekeyhole_polygon(p, tol) for p in polys]