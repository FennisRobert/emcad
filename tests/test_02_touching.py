"""Layer 2: polygons that touch (share boundary) rather than overlap or
sit fully apart -- exact shared edges and partial collinear overlaps,
exercised through both `join_polygons` directly and `add_polygons`
(which calls it internally). Also pins down `join_polygons`'s own
refusal contract: genuine interior crossings, and solid-in-solid
untouched nesting from two different top-level inputs, both still have
to raise -- only a *hole's* untouched nesting is allowed to pass
(see test_04_islands_in_holes.py).
"""
from __future__ import annotations

import pytest

from emcad.kernel.api import add_polygons, join_polygons
from emcad.poly import GeometryException, Polygon

from helpers import assert_area, rect, square


def test_join_shared_edge_merges_into_one_rectangle():
    a = rect(0, 0, 1, 1)
    b = rect(1, 0, 2, 1)
    result = join_polygons([a, b])
    assert len(result) == 1
    assert not result[0].has_holes
    assert_area(result, 2.0)


def test_join_partial_collinear_overlap():
    # b's left edge (x=1, y in [0,1]) is a sub-segment of a's right
    # edge (x=1, y in [0,2]) -- a's edge needs to be split at (1,1)
    # before the shared portion can be recognized as touching.
    a = rect(0, 0, 1, 2)
    b = rect(1, 0, 2, 1)
    result = join_polygons([a, b])
    assert len(result) == 1
    assert_area(result, 2.0 + 1.0)


def test_add_polygons_touching_squares_matches_join():
    a = rect(0, 0, 1, 1)
    b = rect(1, 0, 2, 1)
    result = add_polygons(a, b)
    assert len(result) == 1
    assert_area(result, 2.0)


def test_join_single_polygon_passthrough():
    a = rect(0, 0, 1, 1)
    assert join_polygons([a]) == [a]


def test_join_empty_list():
    assert join_polygons([]) == []


def test_join_raises_on_genuine_crossing():
    # a triangle poking transversally through a square's edges (not just
    # touching a corner to an edge) -- a real interior overlap with
    # actual edge-edge crossings, which _check_no_interior_crossings is
    # built to detect.
    a = rect(0, 0, 2, 2)
    b = Polygon([1.3, 2.9, 1.1], [-0.7, 0.6, 2.8])
    with pytest.raises(GeometryException):
        join_polygons([a, b])


def test_join_detects_corner_only_overlap_as_a_genuine_crossing():
    # two axis-aligned squares overlapping so that each one's corner
    # pokes into the OTHER's edge interior (four T-junctions, never a
    # transversal edge-edge crossing) -- `_check_no_interior_crossings`
    # alone can't see this, since it only looks for transversal
    # crossings; `_check_no_same_side_overlap` catches it via the
    # shared collinear sub-edge that this kind of overlap always
    # produces (see that function's docstring in kernel/_join.py).
    a = square(0, 0, 2)
    b = square(1, 0, 2)  # overlap region [0,1] x [-1,1]; every touch point
    # is a corner-into-edge T-junction, not a transversal edge crossing.
    with pytest.raises(GeometryException):
        join_polygons([a, b])


def test_join_raises_on_untouched_solid_in_solid_different_source():
    # direct misuse: join_polygons is not a general union. Two solid,
    # untouched, unrelated inputs where one sits fully inside the
    # other must raise -- that's the boolean-op precondition
    # ("interior-disjoint inputs") being violated, not a hole.
    outer = square(0, 0, 10)
    inner = square(0, 0, 2)
    with pytest.raises(GeometryException):
        join_polygons([outer, inner])


def test_join_raises_on_segment_shared_by_three():
    a = rect(0, 0, 1, 1)
    b = rect(1, 0, 2, 1)
    b_dup = rect(1, 0, 2, 1)  # duplicate of b -- its left edge now
    # claims the (1,0)-(1,1) segment a third time, alongside a and b.
    with pytest.raises(GeometryException):
        join_polygons([a, b, b_dup])
