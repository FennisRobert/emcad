"""Layer 10: `kernel.api.regularize_polyline` / `Polygon.regularize` --
map-making style outline regularization of copper outlines whose
straight edges are littered with small protrusions (via pads poking out
of an edge after the union), which RDP (`simplify`) can only remove by
skewing the long edges.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from emcad.kernel.api import add_polygons, regularize_polyline
from emcad.poly import Polygon

from helpers import polygon_area, rect
from workloads import circle, trace

MM = 1e-3
PAD_R = 0.4 * MM  # via pad radius: protrudes 0.4 mm - (inset) past the edge


def _via_bar(x0, y0, x1, y1, pitch=2.8 * MM, inset=0.25 * MM, corner_pads=True):
    """A copper bar lined with via pads along both long (vertical) edges,
    pads centred `inset` inside the edge so each pokes out as a bump."""
    pads = []
    # corner pads are inset in BOTH directions (as on a real board), so
    # they poke out past the end edges by the same small amount
    y = y0 + (inset if corner_pads else pitch)
    while y <= y1 - (inset if corner_pads else 0) + 1e-12:
        pads.append(circle(x0 + inset, y, PAD_R, 24))
        pads.append(circle(x1 - inset, y, PAD_R, 24))
        y += pitch
    (merged,) = add_polygons(rect(x0, y0, x1, y1), *pads)
    return merged


def _edges_on_axes(poly: Polygon) -> bool:
    n = len(poly.xs)
    for i in range(n):
        dx = poly.xs[(i + 1) % n] - poly.xs[i]
        dy = poly.ys[(i + 1) % n] - poly.ys[i]
        if abs(dx) > 1e-12 and abs(dy) > 1e-12:
            return False
    return True


def _corners(poly: Polygon) -> set:
    return {(round(x / 1e-9), round(y / 1e-9)) for x, y in zip(poly.xs, poly.ys)}


def test_via_lined_bar_becomes_its_exact_rectangle():
    bar = _via_bar(0, 0, 6 * MM, 40 * MM, corner_pads=False)
    assert len(bar.xs) > 200  # every pad bump tessellated into the outline
    bar.regularize(0.25 * MM)
    assert len(bar.xs) == 4
    assert _edges_on_axes(bar)
    assert _corners(bar) == _corners(rect(0, 0, 6 * MM, 40 * MM))


def test_corner_hidden_under_a_pad_is_rebuilt():
    bar = _via_bar(0, 0, 6 * MM, 39.2 * MM + 0.5 * MM, corner_pads=True)  # pads sit on all 4 corners
    bar.regularize(0.25 * MM)
    assert len(bar.xs) == 4
    assert _corners(bar) == _corners(rect(0, 0, 6 * MM, 39.2 * MM + 0.5 * MM))


def test_rdp_skews_the_same_bar():
    # the motivating failure: RDP keeps the extreme (bump-tip) points
    bar = _via_bar(0, 0, 6 * MM, 40 * MM, corner_pads=False)
    bar.simplify(0.2 * MM)
    assert not _edges_on_axes(bar)


def test_short_edge_chopped_by_close_pads_still_snaps():
    # three pads across a 3 mm end: every straight piece left between
    # them is shorter than min_anchor_len on its own -- they must still
    # vote together as one anchored edge
    body = rect(0, 0, 3 * MM, 12 * MM)
    pads = [circle(x, 0.25 * MM, PAD_R, 24) for x in (0.3 * MM, 1.5 * MM, 2.7 * MM)]
    (merged,) = add_polygons(body, *pads)
    merged.regularize(0.25 * MM)
    assert len(merged.xs) == 4
    assert _corners(merged) == _corners(body)


def test_real_arcs_survive():
    # an isolated round pad has no straight anchors: it stays a circle
    # (lightly simplified), not a polygon collapsed onto a few chords
    c = circle(0, 0, 2 * MM, 64)
    a0 = polygon_area(c)
    c.regularize(0.25 * MM)
    assert len(c.xs) >= 8
    assert math.isclose(polygon_area(c), a0, rel_tol=0.05)


def test_45_degree_trace_keeps_its_direction_and_width():
    t = trace(0, 0, 10 * MM, 10 * MM, 1.57 * MM)
    t.regularize(0.25 * MM)
    # the two long sides are exactly 45 degrees, 1.57 mm apart
    n = len(t.xs)
    long_sides = []
    for i in range(n):
        dx, dy = t.xs[(i + 1) % n] - t.xs[i], t.ys[(i + 1) % n] - t.ys[i]
        if math.hypot(dx, dy) > 5 * MM:
            long_sides.append((t.xs[i], t.ys[i], math.degrees(math.atan2(dy, dx))))
    assert len(long_sides) == 2
    for _, _, ang in long_sides:
        assert min(abs(ang % 45), 45 - abs(ang % 45)) < 1e-9
    (x0, y0, _), (x1, y1, _) = long_sides
    gap = abs((x1 - x0) * -math.sqrt(0.5) + (y1 - y0) * math.sqrt(0.5))
    assert math.isclose(gap, 1.57 * MM, rel_tol=1e-9)


def test_features_larger_than_tol_are_kept():
    # a 1 mm slot cut into a block: much bigger than tol, must survive
    u = Polygon([0, 6 * MM, 6 * MM, 3.5 * MM, 3.5 * MM, 2.5 * MM, 2.5 * MM, 0],
                [0, 0, 8 * MM, 8 * MM, 3 * MM, 3 * MM, 8 * MM, 8 * MM])
    a0 = polygon_area(u)
    u.regularize(0.25 * MM)
    assert len(u.xs) == 8
    assert math.isclose(polygon_area(u), a0, rel_tol=1e-12)


def test_holes_are_regularized_too():
    outer = rect(0, 0, 20 * MM, 20 * MM)
    hole = _via_bar(5 * MM, 5 * MM, 15 * MM, 15 * MM, corner_pads=False)
    p = Polygon(outer.xs, outer.ys, holes=[hole])
    p.regularize(0.25 * MM)
    assert len(p.xs) == 4
    assert len(p.holes) == 1 and len(p.holes[0].xs) == 4
    assert _corners(p.holes[0]) == _corners(rect(5 * MM, 5 * MM, 15 * MM, 15 * MM))


def test_open_closed_convention_is_preserved():
    bar = _via_bar(0, 0, 6 * MM, 40 * MM, corner_pads=False)
    xs, ys = regularize_polyline(bar.cxs, bar.cys, 0.25 * MM)
    assert xs[0] == xs[-1] and ys[0] == ys[-1] and len(xs) == 5
    xs, ys = regularize_polyline(bar.xs, bar.ys, 0.25 * MM)
    assert len(xs) == 4


def test_off_grid_edges_are_not_snapped():
    # a 7-degree edge isn't on the 5-degree grid: nothing to anchor on,
    # so the shape comes back untouched (no area change)
    ang = math.radians(7)
    pts = [(0, 0), (10 * MM * math.cos(ang), 10 * MM * math.sin(ang))]
    pts.append((pts[1][0] - 3 * MM * math.sin(ang), pts[1][1] + 3 * MM * math.cos(ang)))
    pts.append((-3 * MM * math.sin(ang), 3 * MM * math.cos(ang)))
    p = Polygon([q[0] for q in pts], [q[1] for q in pts])
    a0 = polygon_area(p)
    p.regularize(0.25 * MM)
    assert math.isclose(polygon_area(p), a0, rel_tol=1e-12)


def test_bad_parameters_raise():
    with pytest.raises(ValueError):
        regularize_polyline([0, 1, 1], [0, 0, 1], tol=0.0)
    with pytest.raises(ValueError):
        regularize_polyline([0, 1, 1], [0, 0, 1], tol=0.1, dangle_deg=0.0)
    with pytest.raises(ValueError):
        regularize_polyline(np.zeros(3), np.zeros(4), tol=0.1)
