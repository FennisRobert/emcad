"""Shared shape builders and area helpers for the kernel boolean-op tests.

Every test in this directory verifies boolean-op results primarily by
*area* (shoelace, recursively net of holes) plus targeted structural
checks (hole depth, point containment) where area alone can't tell two
different topologies apart -- e.g. "one polygon with two separate
holes" vs "two polygons" can have the same total area.
"""
from __future__ import annotations

import math

from emcad.poly import Polygon


def square(cx: float, cy: float, size: float) -> Polygon:
    """Axis-aligned square centered at (cx, cy) with the given side length."""
    h = size / 2.0
    return Polygon([cx - h, cx + h, cx + h, cx - h], [cy - h, cy - h, cy + h, cy + h])


def rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    """Axis-aligned rectangle from corner (x0, y0) to corner (x1, y1)."""
    return Polygon([x0, x1, x1, x0], [y0, y0, y1, y1])


def ring_area(xs, ys) -> float:
    """Shoelace area of a single open ring (no holes)."""
    xs = list(xs)
    ys = list(ys)
    n = len(xs)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(a) / 2.0


def polygon_area(poly: Polygon) -> float:
    """Net area of a polygon: its own ring, minus every (recursively
    nested) hole's own net area -- so a hole-of-a-hole (an island) adds
    back, exactly matching Polygon's own even-odd containment rule.
    """
    a = ring_area(poly.xs, poly.ys)
    for hole in poly.holes:
        a -= polygon_area(hole)
    return a


def total_area(polys) -> float:
    if isinstance(polys, Polygon):
        polys = [polys]
    return sum(polygon_area(p) for p in polys)


def assert_area(polys, expected: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> None:
    actual = total_area(polys)
    assert math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol), (
        f"area {actual!r} != expected {expected!r} (polys={polys!r})"
    )


def hole_depth(poly: Polygon) -> int:
    """0 for a plain polygon, 1 for one with holes, 2 for a hole that
    itself has holes (islands), etc. -- the deepest nesting present.
    """
    if not poly.holes:
        return 0
    return 1 + max(hole_depth(h) for h in poly.holes)


def find_containing(polys, x: float, y: float) -> Polygon | None:
    """The (first) top-level polygon in `polys` whose material contains
    (x, y), or None. Useful for asserting *which* output polygon a
    particular feature ended up as part of, when several are returned.
    """
    for p in polys:
        if p.is_inside(x, y):
            return p
    return None
