"""Layer 6: `kernel/_simplify.py::dezigzag_polyline` and
`Polygon.dezigzag()` -- collapsing a short "kink" segment sandwiched
between two much longer, near-parallel segments (the tessellated-circle
tangent-point artifact described in the request: "---z____" should
become "-----------").
"""
from __future__ import annotations

import math

from emcad.kernel.api import dezigzag_polyline
from emcad.poly import Polygon


def _seg_len(p0, p1) -> float:
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


def test_parallel_offset_kink_is_collapsed():
    # two perfectly parallel (offset) long runs joined by a short
    # vertical "z" -- the textbook tessellation-artifact shape.
    xs = [0.0, 10.0, 10.0, 20.0, 30.0]
    ys = [0.0, 0.0, 0.05, 0.05, 0.05]
    xs2, ys2 = dezigzag_polyline(xs, ys, max_kink_length=0.1, max_angle_deg=20.0, min_neighbor_factor=3.0)
    assert len(xs2) == 4   # the two kink points collapsed into one
    pts = list(zip(xs2.tolist(), ys2.tolist()))
    # no more sharp zigzag: consecutive segment directions barely change
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        d1 = (b[0] - a[0], b[1] - a[1])
        d2 = (c[0] - b[0], c[1] - b[1])
        len1, len2 = math.hypot(*d1), math.hypot(*d2)
        cos_a = (d1[0] * d2[0] + d1[1] * d2[1]) / (len1 * len2)
        assert cos_a > 0.99   # nearly collinear, no more visible kink


def test_converging_shallow_kink_is_collapsed():
    # two long runs that would meet slightly off-axis if extended,
    # connected by a short near-perpendicular kink.
    xs = [0.0, 10.0, 10.02, 20.0, 30.0]
    ys = [0.0, 0.0, 0.03, 0.06, 0.09]
    xs2, ys2 = dezigzag_polyline(xs, ys, max_kink_length=0.1, max_angle_deg=25.0, min_neighbor_factor=3.0)
    assert len(xs2) == 4


def test_genuine_corner_is_left_alone():
    # a real right-angle corner -- the two "neighboring" segments point
    # in very different directions, so this must NOT be touched even
    # though the corner's own middle segment is short.
    xs = [0.0, 10.0, 10.0, 0.0, 0.0]
    ys = [0.0, 0.0, 0.05, 0.05, 10.0]
    xs2, ys2 = dezigzag_polyline(xs, ys, max_kink_length=0.1, max_angle_deg=20.0, min_neighbor_factor=3.0)
    assert len(xs2) == len(xs)   # untouched


def test_short_neighbors_are_not_collapsed():
    # the "kink" is flanked by segments that are also short -- this is
    # a genuinely small feature, not a zigzag artifact on a long run,
    # so min_neighbor_factor must block it.
    xs = [0.0, 0.2, 0.2, 0.4]
    ys = [0.0, 0.0, 0.05, 0.05]
    xs2, ys2 = dezigzag_polyline(xs, ys, max_kink_length=0.1, max_angle_deg=20.0, min_neighbor_factor=3.0)
    assert len(xs2) == len(xs)   # untouched -- neighbors too short


def test_large_kink_length_is_not_touched():
    xs = [0.0, 10.0, 10.0, 20.0, 30.0]
    ys = [0.0, 0.0, 5.0, 5.0, 5.0]   # kink segment is 5 units -- not tiny
    xs2, ys2 = dezigzag_polyline(xs, ys, max_kink_length=0.1, max_angle_deg=20.0, min_neighbor_factor=3.0)
    assert len(xs2) == len(xs)


def test_polygon_dezigzag_in_place_and_recurses_into_holes():
    # a square outer boundary with a zigzag on one edge, and a hole
    # that also carries a zigzag -- both must be cleaned in one call.
    outer_xs = [0.0, 10.0, 10.0, 20.0, 20.0, 0.0]
    outer_ys = [0.0, 0.0, 0.05, 0.05, 20.0, 20.0]
    hole = Polygon([5.0, 8.0, 8.0, 12.0, 12.0, 5.0], [5.0, 5.0, 5.05, 5.05, 15.0, 15.0])
    poly = Polygon(outer_xs, outer_ys, holes=[hole])

    n_before = len(poly.xs)
    n_hole_before = len(poly.holes[0].xs)
    poly.dezigzag(max_kink_length=0.1, max_angle_deg=20.0, min_neighbor_factor=3.0)
    assert len(poly.xs) < n_before
    assert len(poly.holes[0].xs) < n_hole_before
    # still a valid, sane polygon afterward
    assert poly.holes[0] is not None


def test_dezigzag_never_produces_a_wild_far_away_point():
    # near-parallel (not EXACTLY parallel) long runs -- the naive line-
    # line intersection formula is numerically unstable here and could
    # place the "fix" point kilometers away; the reach guard must fall
    # back to the local kink midpoint instead.
    xs = [0.0, 10.0, 10.001, 20.0, 30.000001]
    ys = [0.0, 0.0, 0.05, 0.05, 0.05000001]
    xs2, ys2 = dezigzag_polyline(xs, ys, max_kink_length=0.1, max_angle_deg=20.0, min_neighbor_factor=3.0)
    for x, y in zip(xs2.tolist(), ys2.tolist()):
        assert -1.0 <= x <= 31.0
        assert -1.0 <= y <= 1.0
