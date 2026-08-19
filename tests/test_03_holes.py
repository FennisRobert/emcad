"""Layer 3: holes -- explicit `Polygon(holes=...)`, holes carved by
`subtract_polygons`, and holes that only emerge from gluing several
*touching* pieces together into a ring with a gap (`join_polygons` /
`add_polygons`). No islands-inside-holes yet -- that's layer 4.
"""
from __future__ import annotations

import pytest

from emcad.kernel.api import add_polygons, join_polygons, subtract_polygons
from emcad.poly import GeometryException, Polygon

from helpers import assert_area, hole_depth, rect, square


def test_explicit_hole_area():
    outer = square(0, 0, 6)          # area 36
    hole = square(0, 0, 2)           # area 4, nested, untouched
    poly = Polygon(outer.xs, outer.ys, holes=[hole])
    assert poly.has_holes
    assert hole_depth(poly) == 1
    assert_area(poly, 36.0 - 4.0)


def test_polygon_rejects_hole_crossing_its_own_boundary():
    outer = square(0, 0, 4)
    bad_hole = square(2, 2, 4)  # straddles outer's own edge, not nested
    with pytest.raises(GeometryException):
        Polygon(outer.xs, outer.ys, holes=[bad_hole])


def test_polygon_rejects_hole_outside_its_boundary():
    outer = square(0, 0, 2)
    far_hole = square(10, 10, 1)
    with pytest.raises(GeometryException):
        Polygon(outer.xs, outer.ys, holes=[far_hole])


def test_polygon_rejects_two_holes_crossing_each_other():
    outer = square(0, 0, 10)
    h1 = square(-1, 0, 3)
    h2 = square(1, 0, 3)  # overlaps h1
    with pytest.raises(GeometryException):
        Polygon(outer.xs, outer.ys, holes=[h1, h2])


def test_subtract_two_disjoint_untouched_holes():
    outer = square(0, 0, 10)          # area 100
    hole_a = rect(-4, -4, -2, -2)     # area 4, untouched
    hole_b = rect(2, 2, 4, 4)         # area 4, untouched, disjoint from hole_a
    result = subtract_polygons((outer,), (hole_a, hole_b))
    assert len(result) == 1
    assert result[0].has_holes
    assert len(result[0].holes) == 2
    assert_area(result, 100.0 - 4.0 - 4.0)


def test_join_ring_of_touching_pieces_forms_a_hole():
    # four rectangles tiling a picture-frame around a [1,2] x [1,2] gap,
    # touching each other only along partial, sub-segment edges.
    bottom = rect(0, 0, 3, 1)
    top = rect(0, 2, 3, 3)
    left = rect(0, 1, 1, 2)
    right = rect(2, 1, 3, 2)
    result = join_polygons([bottom, top, left, right])
    assert len(result) == 1
    assert result[0].has_holes
    assert_area(result, 9.0 - 1.0)
    # the gap is really there, not just an area coincidence
    assert not result[0].is_inside(1.5, 1.5)
    assert result[0].is_inside(0.5, 0.5)


def test_add_polygons_ring_of_touching_pieces_forms_a_hole():
    bottom = rect(0, 0, 3, 1)
    top = rect(0, 2, 3, 3)
    left = rect(0, 1, 1, 2)
    right = rect(2, 1, 3, 2)
    result = add_polygons(bottom, top, left, right)
    assert len(result) == 1
    assert result[0].has_holes
    assert_area(result, 9.0 - 1.0)


def test_add_many_touching_pieces_forming_a_ring_with_untouched_boundary_loops():
    # Regression: when a shape assembled from MANY small touching
    # pieces forms an annulus (a ring with a hole), the ring's OUTER
    # boundary and INNER (hole) boundary loops are typically two
    # SEPARATE, mutually non-touching connected components in the
    # reduced (kept-vs-not-kept) graph -- each is its own isolated
    # loop with a CCW/CW twin pair, exactly like a lone untouched
    # polygon. Left un-deduped, both twins of a pair carry the same
    # |area|, so an unrelated nested face's "smallest strictly bigger
    # container" search has a tie to break arbitrarily by face-array
    # order -- silently producing a WRONG, ORDER-DEPENDENT result (the
    # hole stranded as its own extra top-level piece instead of
    # nesting under the outer boundary) depending only on which twin
    # happened to be traced first. This shape -- a hexagonal ring of
    # circles joined by connecting bars, closing into a loop -- is the
    # minimal case that exposes it.
    def circle(cx, cy, r, n=12):
        import math
        pts = [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
        return Polygon([p[0] for p in pts], [p[1] for p in pts])

    def bar(p0, p1, half_width):
        import math
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        xs = [p0[0] + nx * half_width, p1[0] + nx * half_width, p1[0] - nx * half_width, p0[0] - nx * half_width]
        ys = [p0[1] + ny * half_width, p1[1] + ny * half_width, p1[1] - ny * half_width, p0[1] - ny * half_width]
        return Polygon(xs, ys)

    import math
    n_hubs = 6
    radius = 2.0
    hub_r = 0.15
    bar_hw = 0.135
    centers = [
        (radius * math.cos(2 * math.pi * k / n_hubs), radius * math.sin(2 * math.pi * k / n_hubs))
        for k in range(n_hubs)
    ]
    circles = [circle(cx, cy, hub_r, 16) for cx, cy in centers]
    bars = [bar(centers[i], centers[(i + 1) % n_hubs], bar_hw) for i in range(n_hubs)]

    result = add_polygons(*circles, *bars)
    assert len(result) == 1
    assert result[0].has_holes
    assert len(result[0].holes) == 1
    # material well inside the ring, and the hole's own middle, resolve
    # to solid / empty regardless of which twin an internal tie would
    # have arbitrarily picked.
    assert result[0].is_inside(*centers[0])
    assert not result[0].is_inside(0, 0)


def test_add_polygon_with_hole_plus_disjoint_square_stays_two_pieces():
    plane = Polygon(square(0, 0, 6).xs, square(0, 0, 6).ys, holes=[square(0, 0, 2)])
    elsewhere = square(20, 20, 2)
    result = add_polygons(plane, elsewhere)
    assert len(result) == 2
    with_hole = next(p for p in result if p.has_holes)
    assert_area(with_hole, 36.0 - 4.0)
    without_hole = next(p for p in result if not p.has_holes)
    assert_area(without_hole, 4.0)
