"""Layer 5: bounding-box clustering (`kernel/_cluster.py`), wired into
`_boolean_ops.py::_run()` and `_join.py::join_polygons()` so a call
batching thousands of mostly-disjoint operands (e.g. every pad on one
copper layer) runs each spatially-independent group through its own,
much smaller arrangement instead of one global O(E^2) pass.

Every test here doubles as a correctness check: clustering must be
*invisible* to the result -- same output whether an op processes one
big batch or the same inputs pre-split by hand.
"""
from __future__ import annotations

import time

import pytest

from emcad.kernel.api import add_polygons, intersect_polygons, join_polygons, subtract_polygons
from emcad.poly import GeometryException

from helpers import assert_area, find_containing, rect, square


# --------------------------------------------------------------------------
# add_polygons: disjoint clusters stay independent, touching ones still merge
# --------------------------------------------------------------------------

def test_add_many_disjoint_squares_stays_separate():
    squares = [square(i * 100.0, 0.0, 2.0) for i in range(50)]
    result = add_polygons(*squares)
    assert len(result) == 50
    assert_area(result, 50 * 4.0)


def test_add_straddles_a_cluster_boundary():
    # a, b overlap (one cluster); c is far away (its own cluster) --
    # exercises a batch that's genuinely split across >1 cluster AND
    # has real merging work to do inside one of them.
    a = square(0, 0, 2)      # [-1, 1]
    b = square(1, 0, 2)      # [0, 2], overlaps a
    c = square(100, 100, 2)  # disjoint from both
    result = add_polygons(a, b, c)
    assert len(result) == 2
    merged = find_containing(result, 0.5, 0)
    assert merged is not None
    assert_area([merged], 4.0 + 4.0 - 2.0)   # overlap region [0,1]x[-1,1] = 2
    far = find_containing(result, 100, 100)
    assert far is not None
    assert_area([far], 4.0)


def test_add_island_in_a_dynamically_formed_gap_far_from_a_second_cluster():
    # the picture-frame-plus-untouched-island shape (regression for the
    # hull-containment pass in _cluster.py: no single frame strip's
    # bbox reaches the island, only the frame's overall extent does),
    # PLUS a second, entirely unrelated frame+island pair far away --
    # makes sure the hull pass converges independently per component
    # instead of accidentally merging the two.
    def frame_and_island(ox, oy):
        frame = [
            rect(ox + 0, oy + 0, ox + 3, oy + 1),
            rect(ox + 0, oy + 2, ox + 3, oy + 3),
            rect(ox + 0, oy + 1, ox + 1, oy + 2),
            rect(ox + 2, oy + 1, ox + 3, oy + 2),
        ]
        island = square(ox + 1.5, oy + 1.5, 0.5)
        return frame, island

    frame_a, island_a = frame_and_island(0, 0)
    frame_b, island_b = frame_and_island(1000, 1000)

    result = add_polygons(*frame_a, island_a, *frame_b, island_b)
    assert len(result) == 2
    for ox, oy in ((0, 0), (1000, 1000)):
        piece = find_containing(result, ox + 1.5, oy + 1.5)
        assert piece is not None
        assert_area([piece], (9.0 - 1.0) + 0.25)
        assert not piece.is_inside(ox + 1.1, oy + 1.1)   # gap material, not the island


# --------------------------------------------------------------------------
# intersect_polygons: operands split across >1 cluster can never all
# simultaneously overlap, so the result must be empty -- even when a
# SUBSET of the operands do genuinely overlap each other.
# --------------------------------------------------------------------------

def test_intersect_across_disjoint_clusters_is_empty():
    a = square(0, 0, 2)
    b = square(1, 0, 2)          # overlaps a -- same cluster as a
    c = square(100, 100, 2)      # its own, disjoint cluster
    assert intersect_polygons(a, b, c) == []


def test_intersect_within_one_cluster_still_works():
    a = square(0, 0, 2)
    b = square(1, 0, 2)
    assert_area(intersect_polygons(a, b), 2.0)


# --------------------------------------------------------------------------
# subtract_polygons: a "remove" operand with no nearby "add" operand
# contributes nothing (and must not be treated as an error), while one
# that DOES overlap an add operand still carves it correctly.
# --------------------------------------------------------------------------

def test_subtract_irrelevant_remove_only_cluster_is_a_noop():
    add_a = square(0, 0, 10)          # area 100
    add_b = square(100, 100, 10)      # area 100, its own cluster
    cut = rect(3, -1, 7, 1)           # overlaps only add_a: notches 4.0 off it
    stray_remove = square(-500, -500, 2)   # nowhere near any add operand

    result = subtract_polygons((add_a, add_b), (cut, stray_remove))
    assert len(result) == 2
    notched = find_containing(result, 0, 0)
    assert_area([notched], 100.0 - 4.0)
    untouched = find_containing(result, 100, 100)
    assert_area([untouched], 100.0)


# --------------------------------------------------------------------------
# join_polygons: independent touching-groups scattered far apart still
# each fuse correctly, AND a genuine overlap within one cluster is
# still detected (clustering must not accidentally swallow it).
# --------------------------------------------------------------------------

def _picture_frame(ox: float, oy: float) -> list:
    return [
        rect(ox + 0, oy + 0, ox + 3, oy + 1),
        rect(ox + 0, oy + 2, ox + 3, oy + 3),
        rect(ox + 0, oy + 1, ox + 1, oy + 2),
        rect(ox + 2, oy + 1, ox + 3, oy + 2),
    ]


def test_join_two_independent_frames_far_apart():
    frame_a = _picture_frame(0, 0)
    frame_b = _picture_frame(1000, 1000)
    result = join_polygons(frame_a + frame_b)
    assert len(result) == 2
    for ox, oy in ((0, 0), (1000, 1000)):
        piece = find_containing(result, ox + 0.5, oy + 0.5)
        assert piece is not None
        assert piece.has_holes
        assert_area([piece], 9.0 - 1.0)


def test_join_still_raises_on_genuine_overlap_within_a_cluster():
    a = square(0, 0, 2)
    b = square(0.5, 0, 2)   # overlaps a's interior -- not a clean touch
    with pytest.raises(GeometryException):
        join_polygons([a, b])


def test_join_raises_even_with_an_unrelated_far_away_cluster_present():
    a = square(0, 0, 2)
    b = square(0.5, 0, 2)          # genuinely overlaps a
    elsewhere = square(1000, 1000, 2)   # its own, unrelated cluster
    with pytest.raises(GeometryException):
        join_polygons([a, b, elsewhere])


# --------------------------------------------------------------------------
# Performance sanity check: without clustering this is an O(E^2) pass
# over ~1000 tessellated primitives at once, which is the exact shape
# of the real-world slowdown (a dense copper layer's worth of pads
# batched into one subtract_polygons call) that motivated this file.
# Bound is generous -- this isn't a benchmark, just a guard against
# silently regressing back to the global O(E^2) path.
# --------------------------------------------------------------------------

def test_add_many_disjoint_squares_is_fast():
    squares = [square(i * 10.0, (i % 7) * 10.0, 2.0) for i in range(1500)]
    t0 = time.time()
    result = add_polygons(*squares)
    elapsed = time.time() - t0
    assert len(result) == 1500
    assert elapsed < 10.0, f"1500 disjoint squares took {elapsed:.2f}s -- clustering may not be kicking in"
