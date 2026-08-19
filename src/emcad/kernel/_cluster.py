"""Bounding-box connected-components clustering for the boolean-op
operand list.

Two operands whose axis-aligned bounding boxes are farther apart than
`merge_tol` cannot share so much as a vertex, let alone cross or
overlap -- `merge_tol` is exactly the distance below which two points
are considered the same vertex (see `_constants.py`), and any genuine
edge crossing/touch requires actual point-level proximity. That makes
"AABBs overlap within merge_tol" a safe, sound partition: operands in
different clusters are PROVABLY inert to each other under every
boolean op (add/intersect/subtract/join), so each cluster can be run
through the O(E^2) arrangement engine independently and its results
just concatenated -- no cross-cluster interaction is possible, by
construction.

This is what actually matters once a caller batches thousands of
mostly-disjoint primitives into one call (e.g. every pad on a dense
copper layer, via `PCBView.resolve_layer_polygons`): cost becomes
sum(e_i^2) over clusters instead of E^2 over the whole batch at once,
and for real-world layouts (where most primitives don't touch most
other primitives) that's the difference between "slow" and "instant."

Kept in pure Python -- dict/list-of-lists union-find, not a numba
target; see `_arrangement.py`'s own note on which pieces are easy
numba wins and which aren't.
"""
from __future__ import annotations

import numpy as np


def compute_operand_bboxes(
    rings: list[tuple[np.ndarray, np.ndarray]],
    ring_operand_id: list[int],
    n_operands: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-operand (xmin, xmax, ymin, ymax), merged across every ring
    (outer boundary + holes, at any nesting depth) tagged with that
    operand -- a hole's bbox is always a subset of its own outer
    boundary's, so folding every ring in unconditionally is exact, not
    an approximation.
    """
    xmin = np.full(n_operands, np.inf)
    xmax = np.full(n_operands, -np.inf)
    ymin = np.full(n_operands, np.inf)
    ymax = np.full(n_operands, -np.inf)
    for (xs, ys), op in zip(rings, ring_operand_id):
        if len(xs) == 0:
            continue
        xs_arr = np.asarray(xs, dtype=np.float64)
        ys_arr = np.asarray(ys, dtype=np.float64)
        x0, x1 = float(xs_arr.min()), float(xs_arr.max())
        y0, y1 = float(ys_arr.min()), float(ys_arr.max())
        if x0 < xmin[op]:
            xmin[op] = x0
        if x1 > xmax[op]:
            xmax[op] = x1
        if y0 < ymin[op]:
            ymin[op] = y0
        if y1 > ymax[op]:
            ymax[op] = y1
    return xmin, xmax, ymin, ymax


def cluster_by_bbox_overlap(
    xmin: np.ndarray, xmax: np.ndarray, ymin: np.ndarray, ymax: np.ndarray, pad: float,
) -> list[list[int]]:
    """Connected components of the "AABBs come within `pad` of each
    other" graph, via a left-to-right sweep over xmin with an active-
    interval list -- O(n log n) typical for the sort, plus O(active
    size) per step for the y-overlap check. Degrades toward O(n^2)
    only if many boxes share a very wide common x-range (e.g. one long
    trace touching hundreds of pads along its length) -- even then
    this is never worse than skipping clustering entirely, since that
    's exactly the case where they'd all end up in one cluster anyway.

    `pad` should be `merge_tol` (or larger): operands farther apart
    than that cannot interact under any boolean op, so using anything
    smaller risks splitting operands that the arrangement engine
    itself would still snap together.

    Returns:
        List of operand-index groups (each a list of indices into the
        original xmin/xmax/ymin/ymax arrays). Every operand appears in
        exactly one group.
    """
    n = xmin.shape[0]
    if n == 0:
        return []

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    order = sorted(range(n), key=lambda i: xmin[i])
    active: list[int] = []
    for i in order:
        xi = xmin[i]
        active = [j for j in active if xmax[j] + pad >= xi]
        yi0, yi1 = ymin[i] - pad, ymax[i] + pad
        for j in active:
            if yi0 <= ymax[j] and ymin[j] <= yi1:
                union(i, j)
        active.append(i)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


#: safety valve for the fixed-point loop in `cluster_operands` -- real
#: layouts converge in 2-3 rounds (see its docstring); this only ever
#: bites on an adversarial chain-of-ever-growing-hulls input, where it
#: falls back to one merged cluster (equivalent to no clustering at
#: all) rather than let iteration count blow up.
_MAX_HULL_ROUNDS = 64


def cluster_operands(
    rings: list[tuple[np.ndarray, np.ndarray]],
    ring_operand_id: list[int],
    n_operands: int,
    merge_tol: float,
) -> list[list[int]]:
    """Bbox + cluster in one call, INCLUDING containment.

    A single pass of `cluster_by_bbox_overlap` over the raw per-operand
    boxes only catches EDGE-level interaction (two operands whose own
    boxes actually come close) -- it misses containment: an untouched
    island sitting inside a hole that only exists dynamically, from
    the UNION of several other operands, none of which individually
    reaches the island's location (e.g. a picture-frame made of 4
    separate touching strips, with an island floating in the middle
    gap -- no single strip's bbox comes anywhere near the island, but
    the frame's own overall extent obviously does). Missing that
    would silently return the island as its own disconnected result
    instead of nesting it, so a second kind of pass is needed.

    The fix: re-cluster, but on each GROUP's overall (hull) bbox
    instead of each raw operand's -- any operand truly nested inside
    a group's hole must have a bbox that's a subset of (hence
    overlaps) that group's hull bbox, so hull-overlap clustering is
    guaranteed to catch it. Repeat until a round produces no new
    merges (a fixed point) -- typically 2-3 rounds for a real layout
    (one pass to glue directly-touching pieces, one more to catch
    islands nested in what that first pass produced), since each round
    is still the same O(n log n) sweep, just over however many groups
    the previous round left -- never an O(n^2) pairwise scan over the
    original operand count, even though the *effect* is the same as
    checking every operand against every other cluster's true extent.
    """
    if n_operands <= 1:
        return [[0]] if n_operands == 1 else []

    xmin, xmax, ymin, ymax = compute_operand_bboxes(rings, ring_operand_id, n_operands)
    groups: list[list[int]] = [[i] for i in range(n_operands)]

    for _round in range(_MAX_HULL_ROUNDS):
        hxmin = np.array([min(xmin[i] for i in g) for g in groups])
        hxmax = np.array([max(xmax[i] for i in g) for g in groups])
        hymin = np.array([min(ymin[i] for i in g) for g in groups])
        hymax = np.array([max(ymax[i] for i in g) for g in groups])
        merge_groups = cluster_by_bbox_overlap(hxmin, hxmax, hymin, hymax, pad=merge_tol)
        if len(merge_groups) == len(groups):
            return groups
        groups = [
            [op for gi in merge_group for op in groups[gi]]
            for merge_group in merge_groups
        ]

    return [[i for g in groups for i in g]]
