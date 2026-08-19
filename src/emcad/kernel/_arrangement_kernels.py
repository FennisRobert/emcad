"""
Numba-jitted O(E^2) hot loop for the arrangement engine: find every
split point every edge needs, using only exact grid-integer predicates
from `_exact.py`. Two-pass (count, then fill) so the output arrays are
allocated exactly once, same pattern as `viaconnect.py`'s
`_proximity_edges` and `fragment_kernels.py`'s other kernels.

A pair of edges can only need a split from one of two situations,
checked uniformly for EVERY pair regardless of orientation:
  - one edge's endpoint lands exactly on the OTHER edge's open
    interior (collinear, strictly between its two endpoints) -- this
    single check covers T-junctions, collinear partial overlaps, and
    collinear full containment all at once, since an overlap
    interval's bounds are always one segment's own endpoint (see
    `_exact.classify_segment_pair_i64`'s docstring for why);
  - the two edges cross transversally at a point that is neither
    edge's own endpoint -- needs a genuinely new point inserted into
    BOTH edges.

No other case exists: two exactly-equal edges, or edges sharing an
endpoint already, contribute nothing here (they need no NEW split
point) -- see `dedupe_and_chain` in `_arrangement.py` for how exact
duplicate segments collapse afterward.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._exact import classify_segment_pair_i64, on_segment_i64, orient2d_i64


@njit("boolean(int64, int64, int64, int64, int64, int64)", cache=True)
def _strictly_interior_i64(ax, ay, bx, by, px, py):
    if orient2d_i64(ax, ay, bx, by, px, py) != 0:
        return False
    if px == ax and py == ay:
        return False
    if px == bx and py == by:
        return False
    return on_segment_i64(ax, ay, bx, by, px, py)


@njit(cache=True)
def _pair_split_count(ax, ay, bx, by, cx, cy, dx, dy):
    n = 0
    if _strictly_interior_i64(ax, ay, bx, by, cx, cy):
        n += 1
    if _strictly_interior_i64(ax, ay, bx, by, dx, dy):
        n += 1
    if _strictly_interior_i64(cx, cy, dx, dy, ax, ay):
        n += 1
    if _strictly_interior_i64(cx, cy, dx, dy, bx, by):
        n += 1
    kind, _out = classify_segment_pair_i64(ax, ay, bx, by, cx, cy, dx, dy)
    if kind == 1:
        n += 2
    return n


@njit(cache=True)
def find_all_splits_i64(ex0, ey0, ex1, ey1):
    """
    ex0, ey0, ex1, ey1: int64[E] edge endpoints on the exact grid.

    Returns (split_edge, split_x, split_y): every point (possibly with
    duplicates across different discovering pairs -- the caller dedupes
    when chaining) that edge `split_edge[k]` needs inserted at its own
    interior. Never includes an edge's own two endpoints.
    """
    E = ex0.shape[0]

    total = 0
    for i in range(E):
        ax, ay, bx, by = ex0[i], ey0[i], ex1[i], ey1[i]
        for j in range(i + 1, E):
            cx, cy, dx, dy = ex0[j], ey0[j], ex1[j], ey1[j]
            if max(ax, bx) < min(cx, dx) or min(ax, bx) > max(cx, dx):
                continue
            if max(ay, by) < min(cy, dy) or min(ay, by) > max(cy, dy):
                continue
            total += _pair_split_count(ax, ay, bx, by, cx, cy, dx, dy)

    out_edge = np.empty(total, dtype=np.int64)
    out_x = np.empty(total, dtype=np.int64)
    out_y = np.empty(total, dtype=np.int64)
    ctr = 0

    for i in range(E):
        ax, ay, bx, by = ex0[i], ey0[i], ex1[i], ey1[i]
        for j in range(i + 1, E):
            cx, cy, dx, dy = ex0[j], ey0[j], ex1[j], ey1[j]
            if max(ax, bx) < min(cx, dx) or min(ax, bx) > max(cx, dx):
                continue
            if max(ay, by) < min(cy, dy) or min(ay, by) > max(cy, dy):
                continue

            if _strictly_interior_i64(ax, ay, bx, by, cx, cy):
                out_edge[ctr] = i; out_x[ctr] = cx; out_y[ctr] = cy; ctr += 1
            if _strictly_interior_i64(ax, ay, bx, by, dx, dy):
                out_edge[ctr] = i; out_x[ctr] = dx; out_y[ctr] = dy; ctr += 1
            if _strictly_interior_i64(cx, cy, dx, dy, ax, ay):
                out_edge[ctr] = j; out_x[ctr] = ax; out_y[ctr] = ay; ctr += 1
            if _strictly_interior_i64(cx, cy, dx, dy, bx, by):
                out_edge[ctr] = j; out_x[ctr] = bx; out_y[ctr] = by; ctr += 1

            kind, cross = classify_segment_pair_i64(ax, ay, bx, by, cx, cy, dx, dy)
            if kind == 1:
                out_edge[ctr] = i; out_x[ctr] = cross[0]; out_y[ctr] = cross[1]; ctr += 1
                out_edge[ctr] = j; out_x[ctr] = cross[0]; out_y[ctr] = cross[1]; ctr += 1

    return out_edge[:ctr], out_x[:ctr], out_y[:ctr]


@njit("boolean(int64[:], int64[:], int64, int64)", cache=True)
def point_in_ring_i64(ring_x, ring_y, px, py):
    """Exact even-odd point-in-polygon test for a single ring, via
    integer-only ray casting (horizontal ray from P to +x infinity,
    exact orientation-based crossing count -- no float ray-casting
    epsilon anywhere). Caller must ensure P does not lie exactly on
    the ring's own boundary (true by construction for every caller in
    this package: components never share a vertex with a ring they're
    being tested against).
    """
    n = ring_x.shape[0]
    inside = False
    for i in range(n):
        j = (i - 1) % n
        yi = ring_y[i]
        yj = ring_y[j]
        if (yi > py) != (yj > py):
            xi = ring_x[i]
            xj = ring_x[j]
            # exact sign of (x-intersection of edge (j,i) with y=py) - px,
            # via cross product instead of a float division
            # x_int = xj + (xi - xj) * (py - yj) / (yi - yj)
            # x_int > px  <=>  (xi-xj)*(py-yj)  ><  (px-xj)*(yi-yj), sign per (yi-yj)
            lhs = (xi - xj) * (py - yj)
            rhs = (px - xj) * (yi - yj)
            if yi > yj:
                crosses = lhs > rhs
            else:
                crosses = lhs < rhs
            if crosses:
                inside = not inside
    return inside


# --------------------------------------------------------------------------
# Half-edge face walks. Both mirror `fragment_kernels.trace_faces`'s own
# "next half-edge = clockwise predecessor of the twin in the rotation
# system" rule exactly -- they just record different things while doing
# the identical walk, so it made sense to give them explicit signatures
# and compile them the same way, rather than leaving them as a second,
# uncompiled copy of a pattern `trace_faces` already proves compiles
# cleanly. Pure array-in/array-out, no Polygon/list-of-list involved,
# which is what makes these "easy" numba wins -- unlike the winding-
# number membership propagation (needs Python `frozenset` per face) or
# the containment-forest searches (operate on Python lists of
# variable-length ring arrays), which stay in `_arrangement.py` as
# plain Python.
# --------------------------------------------------------------------------

# signature: same he_start/he_end/he_twin/sorted_he/v_offset/pos_in_block
# convention as `fragment_kernels.trace_faces`, plus `face_starts` (its
# own output) as an extra input -- returns he_face_id: int64[H], which
# traced face each half-edge belongs to.
@njit(
    "int64[:](int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64[:])",
    cache=True,
)
def face_id_per_halfedge(face_starts, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block):
    H = he_start.shape[0]
    he_face_id = np.full(H, -1, dtype=np.int64)
    for fi in range(face_starts.shape[0]):
        h0 = face_starts[fi]
        h = h0
        while True:
            he_face_id[h] = fi
            vv = he_end[h]
            t = he_twin[h]
            deg = v_offset[vv + 1] - v_offset[vv]
            p = pos_in_block[t]
            next_pos = p - 1
            if next_pos < 0:
                next_pos += deg
            h_next = sorted_he[v_offset[vv] + next_pos]
            if h_next == h0:
                break
            h = h_next
    return he_face_id


# signature: same inputs as above -- returns (verts, offsets), a
# CSR-style flattening of every face's vertex loop at once: face fi's
# loop is verts[offsets[fi]:offsets[fi+1]]. Two-pass (count, then
# fill), same allocate-once pattern as `find_all_splits_i64`, so this
# replaces N separate Python-level calls (one per face) with a single
# compiled pass over all of them.
@njit(
    "Tuple((int64[:], int64[:]))(int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64[:])",
    cache=True,
)
def face_vertex_loops(face_starts, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block):
    n_faces = face_starts.shape[0]
    offsets = np.empty(n_faces + 1, dtype=np.int64)
    offsets[0] = 0
    for fi in range(n_faces):
        h0 = face_starts[fi]
        h = h0
        count = 0
        while True:
            count += 1
            vv = he_end[h]
            t = he_twin[h]
            deg = v_offset[vv + 1] - v_offset[vv]
            p = pos_in_block[t]
            next_pos = p - 1
            if next_pos < 0:
                next_pos += deg
            h_next = sorted_he[v_offset[vv] + next_pos]
            if h_next == h0:
                break
            h = h_next
        offsets[fi + 1] = offsets[fi] + count

    verts = np.empty(offsets[n_faces], dtype=np.int64)
    for fi in range(n_faces):
        h0 = face_starts[fi]
        h = h0
        idx = offsets[fi]
        while True:
            verts[idx] = he_start[h]
            idx += 1
            vv = he_end[h]
            t = he_twin[h]
            deg = v_offset[vv + 1] - v_offset[vv]
            p = pos_in_block[t]
            next_pos = p - 1
            if next_pos < 0:
                next_pos += deg
            h_next = sorted_he[v_offset[vv] + next_pos]
            if h_next == h0:
                break
            h = h_next
    return verts, offsets


# --------------------------------------------------------------------------
# Containment-forest search: "nearest (smallest-area) container" queries,
# batched and fully numba-compiled -- replaces the two O(n^2) PURE PYTHON
# double loops that used to live in `_arrangement.py` (`build_arrangement`'s
# cross-component forest, `label_and_assemble`'s reduced-graph forest).
#
# Neither loop was ever expensive because of the exact point-in-ring walk
# itself -- it's expensive because of what surrounds it: millions of
# Python-level loop iterations, each paying interpreter overhead PLUS a
# Python->numba call boundary crossing for `point_in_ring_i64`, for a
# real board where the vast majority of those calls are for a candidate
# whose bounding box doesn't even contain the query point. Moving the
# whole nested loop into one compiled function removes essentially all
# of that per-iteration overhead, and adding a bbox pre-filter (cheap,
# safe -- a ring's bbox always encloses it, so this can only reject,
# never wrongly accept) skips the actual ring walk for the near-totality
# of candidates on a real layout, where most bounding boxes don't
# contain any given query point at all.
#
# This does NOT change the algorithm's O(n^2) worst case -- a proper
# spatial index (grid/interval tree) would be needed for that -- but for
# real board data (one sparse, board-spanning polygon like a copper pour
# among thousands of small, scattered pads) the constant-factor win from
# compiling + bbox-pruning is what actually matters: the O(n^2) pass
# becomes millions of near-free bbox comparisons instead of millions of
# Python-dispatched exact ring walks.
# --------------------------------------------------------------------------

@njit(
    "boolean(int64[:], int64[:], int64[:], int64, int64, int64, int64)",
    cache=True,
    inline="always",
)
def _point_in_ring_idx_i64(vx, vy, verts, start, end, px, py):
    """Same exact even-odd test as `point_in_ring_i64`, but walking a
    CSR slice `verts[start:end]` of GLOBAL vertex indices (resolved
    through `vx`/`vy`) instead of a pre-materialized (ring_x, ring_y)
    pair -- avoids allocating a fresh little array per candidate face
    per query, which the old Python-side `vx[loop]` did on every single
    call.
    """
    n = end - start
    inside = False
    for ii in range(n):
        i = start + ii
        jj = ii - 1
        if jj < 0:
            jj = n - 1
        j = start + jj
        yi = vy[verts[i]]
        yj = vy[verts[j]]
        if (yi > py) != (yj > py):
            xi = vx[verts[i]]
            xj = vx[verts[j]]
            lhs = (xi - xj) * (py - yj)
            rhs = (px - xj) * (yi - yj)
            if yi > yj:
                crosses = lhs > rhs
            else:
                crosses = lhs < rhs
            if crosses:
                inside = not inside
    return inside


@njit(cache=True, parallel=True)
def nearest_container_batch(
    query_x, query_y, query_exclude_group, query_min_area,
    cand_group, cand_area, cand_xmin, cand_xmax, cand_ymin, cand_ymax,
    cand_offsets, cand_verts, vx, vy,
):
    """For each query point, find the SMALLEST-area candidate ring that
    contains it, among candidates whose `cand_group` differs from that
    query's `query_exclude_group` and whose `cand_area` is strictly
    greater than that query's `query_min_area`.

    `query_min_area` lets one kernel serve both containment-forest
    sites: pass -1.0 (always satisfied, areas are non-negative) for
    `build_arrangement`'s cross-component forest, which has no
    self-area constraint; pass each query's own abs(area) for
    `label_and_assemble`'s reduced-graph forest, which only ever wants
    a STRICTLY bigger container.

    Every array is a flat batch: `query_*` has one entry per query,
    `cand_*` one entry per candidate face, `cand_offsets`/`cand_verts`
    the CSR ring-loop encoding (`face_vertex_loops`'s own output
    shape) shared by every candidate, resolved through the shared
    `vx`/`vy` global vertex coordinate arrays. Independent per query,
    so this runs across threads (`prange`) for free.

    Returns:
        int64[Q]: index into the `cand_*` arrays of the nearest
        container for each query, or -1 if none contains it.
    """
    Q = query_x.shape[0]
    C = cand_group.shape[0]
    result = np.full(Q, -1, dtype=np.int64)
    for qi in prange(Q):
        qx = query_x[qi]
        qy = query_y[qi]
        excl = query_exclude_group[qi]
        min_area = query_min_area[qi]
        best_area = 1.0e300
        best_c = -1
        for c in range(C):
            if cand_group[c] == excl:
                continue
            a = cand_area[c]
            if a <= min_area or a >= best_area:
                continue
            if qx < cand_xmin[c] or qx > cand_xmax[c] or qy < cand_ymin[c] or qy > cand_ymax[c]:
                continue
            if _point_in_ring_idx_i64(vx, vy, cand_verts, cand_offsets[c], cand_offsets[c + 1], qx, qy):
                best_area = a
                best_c = c
        result[qi] = best_c
    return result


@njit(cache=True)
def face_bboxes_i64(offsets, verts, vx, vy):
    """Per-face (xmin, xmax, ymin, ymax) in the same CSR shape as
    `face_vertex_loops`'s own output -- the bbox pre-filter input for
    `nearest_container_batch`.
    """
    n = offsets.shape[0] - 1
    xmin = np.empty(n, dtype=np.int64)
    xmax = np.empty(n, dtype=np.int64)
    ymin = np.empty(n, dtype=np.int64)
    ymax = np.empty(n, dtype=np.int64)
    for i in range(n):
        s, e = offsets[i], offsets[i + 1]
        x0 = vx[verts[s]]
        x1 = x0
        y0 = vy[verts[s]]
        y1 = y0
        for k in range(s + 1, e):
            xv = vx[verts[k]]
            yv = vy[verts[k]]
            if xv < x0:
                x0 = xv
            if xv > x1:
                x1 = xv
            if yv < y0:
                y0 = yv
            if yv > y1:
                y1 = yv
        xmin[i] = x0
        xmax[i] = x1
        ymin[i] = y0
        ymax[i] = y1
    return xmin, xmax, ymin, ymax
