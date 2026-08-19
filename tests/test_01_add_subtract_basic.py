"""Layer 1: add_polygons / subtract_polygons / intersect_polygons on
plain, hole-free polygons -- disjoint, overlapping, identical, and
fully-nested-without-touching pairs. No shared edges anywhere in this
file; that's layer 2 (test_02_touching.py).
"""
from __future__ import annotations

from emcad.kernel.api import add_polygons, intersect_polygons, subtract_polygons

from helpers import assert_area, find_containing, square, rect, total_area


# --------------------------------------------------------------------------
# add_polygons (union)
# --------------------------------------------------------------------------

def test_add_disjoint_squares_stays_two_pieces():
    a = square(0, 0, 2)
    b = square(10, 0, 2)
    result = add_polygons(a, b)
    assert len(result) == 2
    assert_area(result, 4.0 + 4.0)
    assert find_containing(result, 0, 0) is not None
    assert find_containing(result, 10, 0) is not None


def test_add_overlapping_squares_merges_to_one():
    a = square(0, 0, 2)   # [-1, 1] x [-1, 1]
    b = square(1, 0, 2)   # [0, 2] x [-1, 1]
    result = add_polygons(a, b)
    assert len(result) == 1
    # overlap region is [0,1] x [-1,1] = area 2
    assert_area(result, 4.0 + 4.0 - 2.0)


def test_add_identical_squares_collapses_to_one():
    a = square(0, 0, 2)
    b = square(0, 0, 2)
    result = add_polygons(a, b)
    assert len(result) == 1
    assert_area(result, 4.0)


def test_add_fully_nested_no_touch_drops_inner():
    outer = square(0, 0, 10)   # area 100
    inner = square(0, 0, 2)    # area 4, fully inside, no shared boundary
    result = add_polygons(outer, inner)
    assert len(result) == 1
    assert_area(result, 100.0)


def test_add_three_way_chain_overlap():
    # a-b overlap, b-c overlap, a-c do not touch -- exercises fragment
    # merging across more than 2 inputs at once.
    a = square(0, 0, 2)    # [-1, 1]
    b = square(1, 0, 2)    # [0, 2]
    c = square(2, 0, 2)    # [1, 3]
    result = add_polygons(a, b, c)
    assert len(result) == 1
    # a, b, c share the same [-1,1] y-range, so the union is the single
    # contiguous rectangle [-1,3] x [-1,1]: width 4, height 2.
    assert_area(result, 4.0 * 2.0)


# --------------------------------------------------------------------------
# subtract_polygons (add - subtract)
# --------------------------------------------------------------------------

def test_subtract_partial_overlap_notches_corner():
    outer = square(0, 0, 10)          # [-5, 5] x [-5, 5], area 100
    cut = rect(3, -1, 7, 1)           # only [3,5] x [-1,1] actually overlaps outer
    result = subtract_polygons((outer,), (cut,))
    assert len(result) == 1
    assert_area(result, 100.0 - 4.0)


def test_subtract_full_containment_creates_untouched_hole():
    outer = square(0, 0, 10)          # area 100
    hole = square(0, 0, 2)            # area 4, fully inside, untouched
    result = subtract_polygons((outer,), (hole,))
    assert len(result) == 1
    assert result[0].has_holes
    assert_area(result, 100.0 - 4.0)


def test_subtract_disjoint_is_a_no_op():
    outer = square(0, 0, 10)
    elsewhere = square(20, 20, 2)
    result = subtract_polygons((outer,), (elsewhere,))
    assert len(result) == 1
    assert not result[0].has_holes
    assert_area(result, 100.0)


def test_subtract_everything_leaves_nothing():
    outer = square(0, 0, 2)
    result = subtract_polygons((outer,), (square(0, 0, 10),))
    assert result == []


# --------------------------------------------------------------------------
# intersect_polygons
# --------------------------------------------------------------------------

def test_intersect_overlapping_squares():
    a = square(0, 0, 2)
    b = square(1, 0, 2)
    result = intersect_polygons(a, b)
    assert len(result) == 1
    assert_area(result, 2.0)  # [0,1] x [-1,1]


def test_intersect_disjoint_squares_is_empty():
    a = square(0, 0, 2)
    b = square(10, 0, 2)
    assert intersect_polygons(a, b) == []


def test_intersect_nested_no_touch_equals_inner():
    outer = square(0, 0, 10)
    inner = square(2, 2, 2)   # fully inside, offset, untouched
    result = intersect_polygons(outer, inner)
    assert len(result) == 1
    assert_area(result, 4.0)
    assert result[0].is_inside(2, 2)
    assert not result[0].is_inside(4.9, 4.9)


def test_total_area_helper_sanity():
    # helpers.py isn't part of the kernel, but a broken area helper
    # would make every other assertion in this file meaningless -- a
    # direct check against a hand-computed value catches that.
    assert total_area([square(0, 0, 2)]) == 4.0
