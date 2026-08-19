"""Layer 4: islands sitting inside holes without touching anything --
the regression class that broke `emcad.emerge_interface.ODBImport.generate_traces()`
on real ODB++ data (a copper island/pad landing inside another
polygon's antipad/clearance cutout with no shared edge).

Two distinct code paths can produce this shape, and both need their
own coverage:

- `poly_fragment`'s own containment-forest assembly nests an untouched
  ring purely by geometric depth (see `fragment_tools.
  assemble_nested_polygons`), which conflates "pre-existing hole" with
  "another genuinely separate, independently-selected solid input" --
  `_boolean_ops._reclassify` is what has to tell those apart.
- `join_polygons`'s OWN, separate containment forest (over whatever
  needs gluing by touching) can *dynamically* produce a hole out of
  several originally-unrelated touching pieces (see
  test_03_holes.py's picture-frame case) -- and an untouched island
  can land inside THAT hole too, which only `_join.py`'s own
  `_assign_holes` can resolve, since it has no membership/arrangement
  information to fall back on the way `_boolean_ops.py` does.
"""
from __future__ import annotations

from emcad.kernel.api import add_polygons, join_polygons, subtract_polygons
from emcad.poly import Polygon

from helpers import assert_area, hole_depth, rect, square


# --------------------------------------------------------------------------
# via _boolean_ops._reclassify (poly_fragment's own nesting)
# --------------------------------------------------------------------------

def test_add_untouched_island_into_a_subtract_carved_hole():
    outer = square(0, 0, 10)   # area 100
    hole = square(0, 0, 4)     # area 16, untouched hole
    island = square(0, 0, 1)   # area 1, untouched, fully inside hole

    plane = subtract_polygons((outer,), (hole,))[0]
    assert plane.has_holes

    result = add_polygons(plane, island)
    assert len(result) == 1
    assert hole_depth(result[0]) == 2
    assert_area(result, 100.0 - 16.0 + 1.0)
    assert result[0].is_inside(0, 0)          # island material
    assert result[0].is_inside(3, 0)          # plane's own solid annulus
    assert not result[0].is_inside(1.5, 0)    # remaining hole gap around the island


def test_add_island_touching_the_hole_boundary_still_works():
    outer = square(0, 0, 10)
    hole = rect(-1, -1, 1, 1)          # area 4
    island = rect(-1, -1, 1, -0.5)     # area 1, bottom edge is a sub-segment
    #                                    of hole's bottom edge -- touching.

    plane = subtract_polygons((outer,), (hole,))[0]
    result = add_polygons(plane, island)
    assert len(result) == 1
    assert_area(result, 100.0 - 4.0 + 1.0)


def test_two_disjoint_islands_in_the_same_hole():
    outer = square(0, 0, 10)
    hole = square(0, 0, 6)             # area 36
    island_a = rect(-2, -2, -1, -1)    # area 1
    island_b = rect(1, 1, 2, 2)        # area 1, disjoint from island_a

    plane = subtract_polygons((outer,), (hole,))[0]
    result = add_polygons(plane, island_a, island_b)
    assert len(result) == 1
    the_hole = result[0].holes[0]
    assert len(the_hole.holes) == 2
    assert_area(result, 100.0 - 36.0 + 1.0 + 1.0)


def test_islands_in_two_separate_holes():
    outer = square(0, 0, 20)                # area 400
    hole_a = rect(-8, -8, -2, -2)           # area 36
    hole_b = rect(2, 2, 8, 8)               # area 36
    island_a = rect(-6, -6, -4, -4)          # area 4, inside hole_a
    island_b = rect(4, 4, 6, 6)              # area 4, inside hole_b

    plane = subtract_polygons((outer,), (hole_a, hole_b))[0]
    result = add_polygons(plane, island_a, island_b)
    assert len(result) == 1
    assert len(result[0].holes) == 2
    assert_area(result, 400.0 - 36.0 - 36.0 + 4.0 + 4.0)


def test_deeply_nested_alternating_islands_and_holes():
    # outer(solid) > h1(hole) > i1(island) > h2(hole-of-island) > i2(island),
    # built the way a tree-reduce union pipeline actually would: each
    # boolean op feeds its already-holed result back in as an operand of
    # the next one.
    big = square(0, 0, 20)     # area 400
    h1 = square(0, 0, 10)      # area 100
    i1 = square(0, 0, 6)       # area 36
    h2 = square(0, 0, 3)       # area 9
    i2 = square(0, 0, 1)       # area 1

    plane = subtract_polygons((big,), (h1,))[0]                 # 400 - 100
    step2 = add_polygons(plane, i1)[0]                          # + 36
    step3 = subtract_polygons((step2,), (h2,))[0]                # - 9
    result = add_polygons(step3, i2)

    assert len(result) == 1
    assert hole_depth(result[0]) == 4
    assert_area(result, 400.0 - 100.0 + 36.0 - 9.0 + 1.0)
    assert result[0].is_inside(0, 0)        # i2, the deepest island
    assert result[0].is_inside(2, 0)        # i1's solid material (outside h2 entirely)
    assert not result[0].is_inside(1.0, 0)  # h2's remaining gap around i2


# --------------------------------------------------------------------------
# via join_polygons's OWN forest (a hole that only emerges dynamically
# from gluing several originally-unrelated touching pieces together --
# the exact shape of the real ODB++ regression, reproduced with plain
# rectangles instead of parsed copper features).
# --------------------------------------------------------------------------

def _picture_frame():
    # tiles [0,3] x [0,3] minus the [1,2] x [1,2] gap in the middle,
    # touching only along partial sub-segment edges -- same shape as
    # test_03_holes.py's ring-of-touching-pieces case.
    return [
        rect(0, 0, 3, 1),   # bottom
        rect(0, 2, 3, 3),   # top
        rect(0, 1, 1, 2),   # left
        rect(2, 1, 3, 2),   # right
    ]


def test_join_polygons_untouched_island_in_a_dynamically_formed_gap():
    frame = _picture_frame()
    island = square(1.5, 1.5, 0.5)   # well inside the [1,2] x [1,2] gap, untouched
    result = join_polygons(frame + [island])
    assert len(result) == 1
    assert_area(result, (9.0 - 1.0) + 0.25)
    assert result[0].is_inside(1.5, 1.5)         # the island itself
    assert not result[0].is_inside(1.1, 1.1)     # gap material around it


def test_add_polygons_untouched_island_in_a_dynamically_formed_gap():
    frame = _picture_frame()
    island = square(1.5, 1.5, 0.5)
    result = add_polygons(*frame, island)
    assert len(result) == 1
    assert_area(result, (9.0 - 1.0) + 0.25)


def test_join_polygons_two_islands_in_a_dynamically_formed_gap():
    frame = _picture_frame()
    island_a = rect(1.1, 1.1, 1.4, 1.4)
    island_b = rect(1.6, 1.6, 1.9, 1.9)
    result = join_polygons(frame + [island_a, island_b])
    assert len(result) == 1
    the_hole = result[0].holes[0]
    assert len(the_hole.holes) == 2
    assert_area(result, (9.0 - 1.0) + 0.09 + 0.09)
