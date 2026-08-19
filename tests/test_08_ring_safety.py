"""Layer 8: `kernel.api.is_simple_ring` and the shared fail-safe
machinery `Polygon._apply_ring_transform` gives `.simplify()`/
`.dezigzag()` -- neither may ever turn a valid polygon into a
self-intersecting one, even though both transforms can, in principle,
move vertices close enough to another part of the same ring (or a
hole/sibling) to fold it over itself (the "extremely thin sliver
self-intersection" failure mode reported on text-trace geometry).

Rather than hand-crafting real coordinates that happen to trigger a
bad candidate (fragile, and a poor test of the MECHANISM itself), most
of these tests inject a deliberately-bad `ring_fn` directly into
`_apply_ring_transform` -- this exercises exactly the code path a real
geometric edge case would hit, without depending on getting delicate
float arithmetic exactly right.
"""
from __future__ import annotations

from emcad.kernel.api import is_simple_ring, dezigzag_polyline
from emcad.poly import Polygon

from helpers import square


# --------------------------------------------------------------------------
# is_simple_ring
# --------------------------------------------------------------------------

def test_simple_convex_ring_is_simple():
    assert is_simple_ring([0, 10, 10, 0], [0, 0, 10, 10])


def test_simple_concave_many_vertex_ring_is_simple():
    # a staircase boundary -- many consecutive edges sharing vertices
    # at right angles. This is exactly the shape that a naive
    # self-intersection check (flagging every shared vertex between
    # adjacent edges) would incorrectly reject.
    xs = [0, 1, 1, 2, 2, 3, 3, 0]
    ys = [0, 0, 1, 1, 2, 2, 3, 3]
    assert is_simple_ring(xs, ys)


def test_bowtie_ring_is_not_simple():
    xs = [0, 10, 0, 10]
    ys = [0, 10, 10, 0]
    assert not is_simple_ring(xs, ys)


def test_spike_folding_back_on_itself_is_not_simple():
    # a ring with a degenerate spike: goes out and immediately doubles
    # back along (past) the same line.
    xs = [0, 10, 10, 5, 10, 0]
    ys = [0, 0, 10, 0, -5, 10]
    assert not is_simple_ring(xs, ys)


# --------------------------------------------------------------------------
# _apply_ring_transform: the shared fail-safe used by simplify/dezigzag
# --------------------------------------------------------------------------

def test_apply_ring_transform_rejects_self_intersecting_candidate():
    poly = square(5, 5, 10)
    original_xs, original_ys = list(poly.xs), list(poly.ys)

    def bowtie_transform(xs, ys):
        return [0, 10, 0, 10], [0, 10, 10, 0]

    poly._apply_ring_transform(bowtie_transform)
    assert poly.xs == original_xs
    assert poly.ys == original_ys


def test_apply_ring_transform_commits_a_valid_candidate():
    poly = square(5, 5, 10)

    def shrink_transform(xs, ys):
        return [x * 0.5 for x in xs], [y * 0.5 for y in ys]

    poly._apply_ring_transform(shrink_transform)
    # committed -- no longer the original square
    assert max(poly.xs) < 5.5


def test_apply_ring_transform_rejects_a_self_intersecting_hole_only():
    # the outer ring's own transform is fine; only the hole's
    # candidate self-intersects -- the hole must be left at its
    # original shape while the outer boundary still gets updated.
    outer = square(0, 0, 20)
    hole = square(0, 0, 4)
    poly = Polygon(outer.xs, outer.ys, holes=[hole])
    original_hole_xs = list(poly.holes[0].xs)

    def transform(xs, ys):
        # bowtie only for the small (hole-sized) ring; pass through
        # the large outer ring unchanged.
        if max(xs) - min(xs) < 10:
            return [-2, 2, -2, 2], [-2, 2, 2, -2]
        return xs, ys

    poly._apply_ring_transform(transform)
    assert poly.holes[0].xs == original_hole_xs
    assert max(poly.xs) - min(poly.xs) == 20   # outer untouched by the guard


def test_dezigzag_never_produces_a_self_intersecting_result():
    # exercises the real dezigzag_polyline function (not an injected
    # fake) against Polygon.dezigzag()'s own guard, across a batch of
    # thin, tightly-cornered "glyph stroke"-like shapes.
    import math
    shapes = []
    # a thin zigzag stroke with a kink near a sharp turn
    shapes.append((
        [0, 0, 0.3, 0.3, 0.35, 0.35, 6, 6],
        [0, 6, 6, 0.35, 0.35, 0.3, 0.3, 0],
    ))
    # a thin hairpin with a kink on one arm
    shapes.append((
        [0, 0, 0.3, 0.3, 0.32, 0.32, 0.3, 0.3, 0],
        [0, 10, 10, 6.02, 6.0, 5.98, 6, 0, 0],
    ))
    for xs, ys in shapes:
        poly = Polygon(xs, ys)
        assert is_simple_ring(poly.xs, poly.ys)   # input must be valid to start
        poly.dezigzag(max_kink_length=0.06, max_angle_deg=25.0, min_neighbor_factor=2.0)
        assert is_simple_ring(poly.xs, poly.ys)
