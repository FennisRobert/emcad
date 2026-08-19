"""
Unified arrangement engine: exact-predicate segment splitting +
half-edge tracing + winding-number membership propagation.

This replaces THREE previously-independent, previously-buggy pieces of
logic with one construction:

  - `poly_fragment`'s crossing-only split detection (missed T-junctions
    and collinear overlaps -- see `_exact.py`'s module docstring), plus
    its "always alternate solid/hole by depth" containment-forest
    assembly (conflated a genuine pre-existing hole with a second,
    independently-selected overlapping solid input);
  - `_join.py`'s separate T-junction/collinear-overlap embedding, plus
    its "same top-level source" heuristic for deciding whether an
    untouched nested face is a legitimate hole (wrong whenever the
    nesting is more than one level deep -- an island inside a
    dynamically-formed hole);
  - every `Polygon.point_inside()` + `is_inside()` re-test used to
    classify a traced face's membership (fragile on degenerate
    slivers, and a second, redundant way to get containment wrong).

Architecture
------------
Every operand (a flat list of rings -- a polygon's outer boundary plus
every hole, at every depth) contributes edges tagged with its own
operand id to ONE combined arrangement, built on the exact
grid-integer predicates in `_exact.py` / `_arrangement_kernels.py` --
so there is exactly one place that decides "do these two edges meet",
not two independently-tuned detectors that can disagree.

Each traced face's membership (which operands' material it falls
inside) is derived by WINDING-NUMBER PARITY PROPAGATION over the
face-adjacency graph: crossing a segment flips membership for every
operand that contributed an edge there (an even number of
contributions from the same operand cancels, exactly matching the
even-odd rule `Polygon.is_inside` already uses everywhere else in this
package). This is a structural fact about the arrangement, computed
once via BFS -- no representative point is ever sampled or tested
against a polygon's raw boundary, so there is no "degenerate sliver"
failure mode.

Boolean ops become label functions over these membership sets:
    add       -> len(membership) > 0
    intersect -> len(membership) == n_operands
    subtract  -> any(add operands) and not any(subtract operands)
    join      -> same as add, PLUS validate no face has membership
                  from more than one operand (the fully general,
                  exact replacement for both of `_join.py`'s old,
                  epsilon-based "genuine crossing" and "same-side
                  overlap" checks -- any kind of real interior overlap,
                  however it's produced, shows up as membership size
                  >= 2 somewhere).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from loguru import logger

from . import fragment_kernels as fkernels
from .fragment_tools import build_rotation_system
from ._exact import make_scale, to_grid
from ._arrangement_kernels import (
    find_all_splits_i64,
    face_id_per_halfedge,
    face_vertex_loops,
    face_bboxes_i64,
    nearest_container_batch,
)
from ._profiling import format_runtime
from ._constants import DEFAULT_MERGE_TOL


class ArrangementError(Exception):
    pass


def is_simple_ring(xs, ys, merge_tol: float = DEFAULT_MERGE_TOL) -> bool:
    """True if the closed ring (open form: first point != last, same
    convention as `Polygon.xs`/`ys`) has no self-intersections -- no
    transversal crossing, T-junction, or collinear overlap between any
    two of its own edges, other than the trivial vertex each pair of
    CONSECUTIVE edges naturally shares.

    Uses the same exact grid-integer split-finder the arrangement
    engine itself is built on (`find_all_splits_i64`/`_exact.py`), not
    the legacy `_primitives.edge_self_intersections` -- that one has no
    way to distinguish a genuine crossing from the vertex two adjacent
    ring edges always share, so calling it on a single ring's own edge
    list against itself flags every vertex as a "self-intersection".
    The exact split-finder doesn't have that problem: a point exactly
    at one of an edge's own two endpoints is never "strictly interior"
    to it (see `_strictly_interior_i64`), so a shared vertex between
    neighbors is correctly never reported as a split.

    Used as a validity check for anything that can MOVE a ring's own
    vertices (unlike a boolean op or `poly_fragment`, which only ever
    combine already-simple inputs) -- see `Polygon._ring_transform_candidate`,
    which `simplify()`/`dezigzag()` share.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = xs.shape[0]
    if n < 3:
        return False
    scale = make_scale(merge_tol)
    gxy = to_grid(np.vstack([xs, ys]), scale)
    ex0 = gxy[0]
    ey0 = gxy[1]
    ex1 = np.roll(gxy[0], -1)
    ey1 = np.roll(gxy[1], -1)
    split_edge, _split_x, _split_y = find_all_splits_i64(ex0, ey0, ex1, ey1)
    return split_edge.shape[0] == 0


# --------------------------------------------------------------------------
# Step 1: flatten operand rings into one tagged, grid-snapped edge list.
# --------------------------------------------------------------------------

def _flatten_edges(rings: list[tuple[np.ndarray, np.ndarray]], ring_operand_id: list[int], scale: float):
    all_x: list[int] = []
    all_y: list[int] = []
    edge_u_raw: list[int] = []
    edge_v_raw: list[int] = []
    edge_operand: list[int] = []

    for ring_idx, (xs, ys) in enumerate(rings):
        gxy = to_grid(np.vstack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)]), scale)
        n = gxy.shape[1]
        base = len(all_x)
        all_x.extend(int(v) for v in gxy[0])
        all_y.extend(int(v) for v in gxy[1])
        for i in range(n):
            edge_u_raw.append(base + i)
            edge_v_raw.append(base + (i + 1) % n)
            edge_operand.append(ring_operand_id[ring_idx])

    return (
        np.asarray(all_x, dtype=np.int64), np.asarray(all_y, dtype=np.int64),
        np.asarray(edge_u_raw, dtype=np.int64), np.asarray(edge_v_raw, dtype=np.int64),
        np.asarray(edge_operand, dtype=np.int64),
    )


def _exact_merge_vertices(raw_x: np.ndarray, raw_y: np.ndarray):
    """Exact dict-keyed vertex merge -- two raw points merge iff their
    grid coordinates are bit-for-bit equal. No clustering, no
    tolerance, no ambiguity: integer equality is decidable.
    """
    key_to_id: dict[tuple[int, int], int] = {}
    raw_to_merged = np.empty(raw_x.shape[0], dtype=np.int64)
    xs_out: list[int] = []
    ys_out: list[int] = []
    for i in range(raw_x.shape[0]):
        key = (int(raw_x[i]), int(raw_y[i]))
        vid = key_to_id.get(key)
        if vid is None:
            vid = len(xs_out)
            key_to_id[key] = vid
            xs_out.append(key[0])
            ys_out.append(key[1])
        raw_to_merged[i] = vid
    return raw_to_merged, np.asarray(xs_out, dtype=np.int64), np.asarray(ys_out, dtype=np.int64), key_to_id


# --------------------------------------------------------------------------
# Step 2: split every edge at every exact coincidence, then chain each
# edge's own splits (sorted along the edge with a purely integer key --
# no parametric t needed) into atomic sub-segments.
# --------------------------------------------------------------------------

def _chain_edge_splits(
    edge_u: np.ndarray, edge_v: np.ndarray, edge_operand: np.ndarray,
    vx: np.ndarray, vy: np.ndarray,
    split_edge: np.ndarray, split_x: np.ndarray, split_y: np.ndarray,
    key_to_id: dict[tuple[int, int], int], xs_out: list[int], ys_out: list[int],
):
    """Returns seg_u, seg_v, seg_operand (atomic, directed, multiplicity
    intact -- two operands sharing a sub-segment both contribute their
    own instance of it)."""
    by_edge: dict[int, list[int]] = {}
    for k in range(split_edge.shape[0]):
        by_edge.setdefault(int(split_edge[k]), []).append(k)

    seg_u: list[int] = []
    seg_v: list[int] = []
    seg_operand: list[int] = []

    for e in range(edge_u.shape[0]):
        u, v = int(edge_u[e]), int(edge_v[e])
        ax, ay = vx[u], vy[u]
        bx, by = vx[v], vy[v]
        chain_vids = [u]

        ks = by_edge.get(e)
        if ks:
            # sort along the edge's dominant axis, in the edge's own
            # direction -- exact integer keys, no t parameter.
            dx, dy = bx - ax, by - ay
            if abs(dx) >= abs(dy):
                keyfn = lambda k: (split_x[k] - ax) * (1 if dx >= 0 else -1)
            else:
                keyfn = lambda k: (split_y[k] - ay) * (1 if dy >= 0 else -1)
            ks.sort(key=keyfn)
            for k in ks:
                key = (int(split_x[k]), int(split_y[k]))
                vid = key_to_id.get(key)
                if vid is None:
                    vid = len(xs_out)
                    key_to_id[key] = vid
                    xs_out.append(key[0])
                    ys_out.append(key[1])
                if vid != chain_vids[-1]:
                    chain_vids.append(vid)

        if chain_vids[-1] != v:
            chain_vids.append(v)

        for a, b in zip(chain_vids[:-1], chain_vids[1:]):
            if a == b:
                continue
            seg_u.append(a)
            seg_v.append(b)
            seg_operand.append(int(edge_operand[e]))

    return (np.asarray(seg_u, dtype=np.int64), np.asarray(seg_v, dtype=np.int64),
            np.asarray(seg_operand, dtype=np.int64))


# --------------------------------------------------------------------------
# Step 3: dedupe atomic segments by undirected key, computing each
# unique segment's flip-set (which operands' membership toggles when
# crossing it -- parity of how many of that operand's own edges landed
# here, so a redundant doubled-back edge from the same operand cancels
# out, exactly like the even-odd rule already used everywhere else).
# --------------------------------------------------------------------------

def _dedupe_segments(seg_u: np.ndarray, seg_v: np.ndarray, seg_operand: np.ndarray):
    counts: dict[tuple[int, int], dict[int, int]] = {}
    order: list[tuple[int, int]] = []
    for k in range(seg_u.shape[0]):
        u, v = int(seg_u[k]), int(seg_v[k])
        key = (u, v) if u < v else (v, u)
        d = counts.get(key)
        if d is None:
            d = {}
            counts[key] = d
            order.append(key)
        op = int(seg_operand[k])
        d[op] = d.get(op, 0) + 1

    uniq_u = np.empty(len(order), dtype=np.int64)
    uniq_v = np.empty(len(order), dtype=np.int64)
    flip_sets: list[frozenset[int]] = []
    for i, key in enumerate(order):
        uniq_u[i], uniq_v[i] = key
        d = counts[key]
        flip_sets.append(frozenset(op for op, c in d.items() if c % 2 == 1))
    return uniq_u, uniq_v, flip_sets


# --------------------------------------------------------------------------
# Step 4: half-edges, rotation system, face tracing (reusing the
# existing, already-correct, purely-combinatorial machinery), plus
# per-half-edge face id and flip-set.
# --------------------------------------------------------------------------

def _build_halfedges(uniq_u: np.ndarray, uniq_v: np.ndarray, flip_sets: list[frozenset[int]]):
    M = uniq_u.shape[0]
    he_start = np.empty(2 * M, dtype=np.int64)
    he_end = np.empty(2 * M, dtype=np.int64)
    he_twin = np.empty(2 * M, dtype=np.int64)
    he_start[0::2] = uniq_u
    he_end[0::2] = uniq_v
    he_start[1::2] = uniq_v
    he_end[1::2] = uniq_u
    he_twin[0::2] = np.arange(1, 2 * M, 2, dtype=np.int64)
    he_twin[1::2] = np.arange(0, 2 * M, 2, dtype=np.int64)
    he_flip: list[frozenset[int]] = [None] * (2 * M)  # type: ignore[list-item]
    for m in range(M):
        he_flip[2 * m] = flip_sets[m]
        he_flip[2 * m + 1] = flip_sets[m]
    return he_start, he_end, he_twin, he_flip


# --------------------------------------------------------------------------
# Step 5: connected components, via union-find over vertices linked by
# segments -- used both to seed per-component membership propagation
# and to restrict the cross-component containment forest to genuinely
# different components (two faces in the SAME component are connected
# by shared edges, not nesting).
# --------------------------------------------------------------------------

def _union_find(n: int):
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return find, union


# --------------------------------------------------------------------------
# Main entry point.
# --------------------------------------------------------------------------

class Arrangement:
    """The full half-edge structure, kept around (not flattened to a
    per-face list) because assembling a boolean op's result needs a
    SECOND pass over it (see `label_and_assemble`) once we know which
    faces that particular op keeps -- merging adjacent same-keep-status
    faces into fewer, bigger output polygons isn't knowable from a
    per-face list alone; it needs the adjacency structure itself.
    """

    def __init__(self, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block,
                 he_face_id, face_starts, areas, membership, vx, vy, scale):
        self.he_start = he_start
        self.he_end = he_end
        self.he_twin = he_twin
        self.sorted_he = sorted_he
        self.v_offset = v_offset
        self.pos_in_block = pos_in_block
        self.he_face_id = he_face_id
        self.face_starts = face_starts
        self.areas = areas
        self.membership: list[frozenset] = membership  # one entry per face
        self.vx = vx
        self.vy = vy
        self.scale = scale

    @classmethod
    def empty(cls, scale: float) -> "Arrangement":
        z = np.empty(0, dtype=np.int64)
        return cls(z, z, z, z, np.zeros(1, dtype=np.int64), z, z, z, np.empty(0), [], z, z, scale)

    @property
    def n_faces(self) -> int:
        return len(self.face_starts)


def build_arrangement(
    rings: list[tuple[np.ndarray, np.ndarray]],
    ring_operand_id: list[int],
    n_operands: int,
    merge_tol: float,
) -> Arrangement:
    scale = make_scale(merge_tol)
    if len(rings) == 0:
        return Arrangement.empty(scale)
    n_edges = sum(len(xs) for xs, _ in rings)
    t0 = time.time()
    # the split-finder below is O(n_edges^2) -- this is the one number
    # to watch for a call that's about to become the bottleneck (e.g.
    # a union fed hundreds of already-tessellated primitives at once).
    logger.debug(f"build_arrangement: {n_operands} operand(s), {len(rings)} ring(s), {n_edges} edge(s)")
    raw_x, raw_y, edge_u_raw, edge_v_raw, edge_operand = _flatten_edges(rings, ring_operand_id, scale)
    raw_to_merged, xs_out, ys_out, key_to_id = _exact_merge_vertices(raw_x, raw_y)
    xs_out = list(xs_out)
    ys_out = list(ys_out)

    edge_u = raw_to_merged[edge_u_raw]
    edge_v = raw_to_merged[edge_v_raw]

    vx = np.asarray(xs_out, dtype=np.int64)
    vy = np.asarray(ys_out, dtype=np.int64)
    ex0, ey0 = vx[edge_u], vy[edge_u]
    ex1, ey1 = vx[edge_v], vy[edge_v]

    _t_split0 = time.time()
    split_edge, split_x, split_y = find_all_splits_i64(ex0, ey0, ex1, ey1)
    _t_split1 = time.time()
    logger.trace(f"{format_runtime(_t_split0, _t_split1)} find_all_splits_i64: {ex0.shape[0]} edge(s), {split_edge.shape[0]} split(s) found")

    seg_u, seg_v, seg_operand = _chain_edge_splits(
        edge_u, edge_v, edge_operand, vx, vy, split_edge, split_x, split_y, key_to_id, xs_out, ys_out
    )

    n_verts = len(xs_out)
    vx = np.asarray(xs_out, dtype=np.int64)
    vy = np.asarray(ys_out, dtype=np.int64)

    if seg_u.shape[0] == 0:
        return Arrangement.empty(scale)

    uniq_u, uniq_v, flip_sets = _dedupe_segments(seg_u, seg_v, seg_operand)

    he_start, he_end, he_twin, he_flip = _build_halfedges(uniq_u, uniq_v, flip_sets)

    xy_float = np.vstack([vx.astype(np.float64), vy.astype(np.float64)])
    sorted_he, v_offset, pos_in_block = build_rotation_system(he_start, he_end, xy_float, n_verts)

    face_starts = fkernels.trace_faces(he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block)
    areas = fkernels.face_areas(face_starts, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block, xy_float)
    he_face_id = face_id_per_halfedge(face_starts, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block)

    find, union = _union_find(n_verts)
    for u, v in zip(uniq_u, uniq_v):
        union(int(u), int(v))

    # -- relative (per-component) membership via BFS parity propagation --
    # Every traced face gets a membership value here, including both
    # faces of a trivial isolated-ring "twin pair" -- there is no
    # separate dedup step: `label_and_assemble`'s reduced-graph pass
    # (see below) merges same-keep-status adjacency generically,
    # which subsumes what twin-deduping used to handle as a special
    # case.
    n_faces_all = len(face_starts)
    component_of = [find(int(he_start[face_starts[k]])) for k in range(n_faces_all)]
    relative_membership: list[frozenset] = [None] * n_faces_all  # type: ignore[list-item]

    faces_by_component: dict[int, list[int]] = {}
    for k in range(n_faces_all):
        faces_by_component.setdefault(component_of[k], []).append(k)

    outer_face_of_component: dict[int, int] = {}
    for comp, face_idxs in faces_by_component.items():
        outer = min(face_idxs, key=lambda k: areas[k])  # the CW (most negative) face
        outer_face_of_component[comp] = outer

        relative_membership[outer] = frozenset()
        frontier = [outer]
        while frontier:
            nxt = []
            for fk in frontier:
                h0 = int(face_starts[fk])
                h = h0
                while True:
                    nb_h = int(he_twin[h])
                    nb_face = int(he_face_id[nb_h])
                    if relative_membership[nb_face] is None:
                        relative_membership[nb_face] = relative_membership[fk] ^ he_flip[h]
                        nxt.append(nb_face)
                    vv = he_end[h]
                    t = he_twin[h]
                    deg = v_offset[vv + 1] - v_offset[vv]
                    p = pos_in_block[t]
                    next_pos = p - 1
                    if next_pos < 0:
                        next_pos += deg
                    h_next = int(sorted_he[v_offset[vv] + next_pos])
                    if h_next == h0:
                        break
                    h = h_next
            frontier = nxt

    # -- cross-component containment forest over each component's outer face --
    # Batched, fully numba-compiled (see _arrangement_kernels.py's
    # nearest_container_batch docstring for why this replaced a pure
    # Python O(n_components * n_faces) double loop): one query point
    # per component (its outer face's first vertex), candidates are
    # every face NOT in that component, looking for the smallest-area
    # one whose ring contains the point.
    verts, offsets = face_vertex_loops(face_starts, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block)
    cand_xmin, cand_xmax, cand_ymin, cand_ymax = face_bboxes_i64(offsets, verts, vx, vy)

    components = list(faces_by_component.keys())
    outer_faces = np.asarray([outer_face_of_component[c] for c in components], dtype=np.int64)
    query_x = vx[verts[offsets[outer_faces]]]
    query_y = vy[verts[offsets[outer_faces]]]
    query_exclude_group = np.asarray(components, dtype=np.int64)
    query_min_area = np.full(len(components), -1.0, dtype=np.float64)

    cand_group = np.asarray(component_of, dtype=np.int64)
    cand_area = np.abs(areas).astype(np.float64)

    best = nearest_container_batch(
        query_x, query_y, query_exclude_group, query_min_area,
        cand_group, cand_area, cand_xmin, cand_xmax, cand_ymin, cand_ymax,
        offsets, verts, vx, vy,
    )
    parent_component: dict[int, int | None] = {
        comp: (int(best[ci]) if best[ci] >= 0 else None) for ci, comp in enumerate(components)
    }

    resolved_membership: dict[int, frozenset] = {}
    remaining = set(components)
    while remaining:
        progressed = False
        for comp in list(remaining):
            pf = parent_component[comp]
            if pf is None:
                resolved_membership[comp] = frozenset()
                remaining.discard(comp)
                progressed = True
            else:
                pf_comp = component_of[pf]
                if pf_comp in resolved_membership:
                    outer_true = resolved_membership[pf_comp] ^ relative_membership[pf]
                    resolved_membership[comp] = outer_true
                    remaining.discard(comp)
                    progressed = True
        if not progressed:
            raise ArrangementError("containment forest over components did not resolve -- unexpected cycle")

    # -- absolute membership per face --
    membership: list[frozenset] = [None] * n_faces_all  # type: ignore[list-item]
    for k in range(n_faces_all):
        comp = component_of[k]
        membership[k] = resolved_membership[comp] ^ relative_membership[k]

    t1 = time.time()
    logger.debug(f"{format_runtime(t0, t1)} build_arrangement: {len(face_starts)} face(s) traced")

    return Arrangement(he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block,
                        he_face_id, face_starts, areas, membership, vx, vy, scale)


# --------------------------------------------------------------------------
# Step 6: label every ORIGINAL face by a boolean-op keep function, then
# build a REDUCED graph containing only the segments where keep-status
# actually differs on the two sides (a segment between two same-status
# faces isn't a boundary of anything -- the two faces are the same
# output region and must merge, which the first version of this
# function got wrong: a containment forest alone only ever merges
# NESTED redundancy, never SIDE-BY-SIDE adjacency). Re-tracing that
# reduced graph with the exact same half-edge machinery automatically
# produces the correctly-merged output faces; the remaining
# containment forest step (still needed) now only has to resolve
# CROSS-COMPONENT nesting -- e.g. an island that's its own separate
# untouched component, sitting inside a hole that belongs to an
# entirely different component -- using the same "render iff differs
# from nearest rendered ancestor" rule as before, now driven by exact,
# already-merged keep-status instead of point-sampled membership.
# --------------------------------------------------------------------------

def label_and_assemble(arrangement: Arrangement, keep_fn: Callable[[frozenset], bool], Polygon: type,
                        area_tol: float) -> list["Polygon"]:
    n_faces_all = arrangement.n_faces
    if n_faces_all == 0:
        return []

    he_start, he_end, he_twin = arrangement.he_start, arrangement.he_end, arrangement.he_twin
    he_face_id = arrangement.he_face_id
    vx, vy, scale = arrangement.vx, arrangement.vy, arrangement.scale
    n_verts = vx.shape[0]

    keep = [keep_fn(m) for m in arrangement.membership]

    # -- reduced boundary-only edge set, with a lookup from each
    # directed (u, v) pair back to the ORIGINAL face on that side --
    M = he_start.shape[0] // 2
    red_u: list[int] = []
    red_v: list[int] = []
    orig_face_of_dir: dict[tuple[int, int], int] = {}
    for m in range(M):
        h, ht = 2 * m, 2 * m + 1
        fa, fb = int(he_face_id[h]), int(he_face_id[ht])
        if keep[fa] == keep[fb]:
            continue
        u, v = int(he_start[h]), int(he_end[h])
        red_u.append(u)
        red_v.append(v)
        orig_face_of_dir[(u, v)] = fa
        orig_face_of_dir[(v, u)] = fb

    if len(red_u) == 0:
        return []

    red_u_arr = np.asarray(red_u, dtype=np.int64)
    red_v_arr = np.asarray(red_v, dtype=np.int64)

    rhe_start = np.empty(2 * len(red_u), dtype=np.int64)
    rhe_end = np.empty(2 * len(red_u), dtype=np.int64)
    rhe_twin = np.empty(2 * len(red_u), dtype=np.int64)
    rhe_start[0::2] = red_u_arr
    rhe_end[0::2] = red_v_arr
    rhe_start[1::2] = red_v_arr
    rhe_end[1::2] = red_u_arr
    rhe_twin[0::2] = np.arange(1, 2 * len(red_u), 2, dtype=np.int64)
    rhe_twin[1::2] = np.arange(0, 2 * len(red_u), 2, dtype=np.int64)

    xy_float = np.vstack([vx.astype(np.float64), vy.astype(np.float64)])
    r_sorted_he, r_v_offset, r_pos_in_block = build_rotation_system(rhe_start, rhe_end, xy_float, n_verts)

    r_face_starts = fkernels.trace_faces(rhe_start, rhe_end, rhe_twin, r_sorted_he, r_v_offset, r_pos_in_block)
    r_areas = fkernels.face_areas(r_face_starts, rhe_start, rhe_end, rhe_twin, r_sorted_he, r_v_offset, r_pos_in_block, xy_float)

    # A boundary loop that doesn't touch any OTHER boundary loop (e.g.
    # a simple hole, or a simple outer boundary, neither one touching
    # anything else) is its own isolated connected component in this
    # reduced graph -- and, same as any isolated loop, traces to a
    # CCW/CW twin pair describing the identical ring. Left un-deduped,
    # both twins independently enter the containment forest below with
    # the SAME |area|, so neither can be the other's parent -- and
    # which one an unrelated nested face picks as ITS parent becomes
    # an arbitrary tie-break on face-array order, silently producing
    # different (and sometimes wrong -- a nested hole's parent
    # resolving to the twin that never renders, stranding the hole as
    # a spurious extra top-level face instead of nesting it) results
    # depending only on primitive insertion order. Keep just the CCW
    # (positive-area) twin -- `trace_faces`' own convention for which
    # one is "the bounded enclosed region" -- so there is exactly one
    # face, with one unambiguous keep-status, per physical ring.
    find_r, union_r = _union_find(n_verts)
    for u, v in zip(red_u_arr, red_v_arr):
        union_r(int(u), int(v))

    groups: dict[int, list[int]] = {}
    for k in range(len(r_face_starts)):
        comp = find_r(int(rhe_start[r_face_starts[k]]))
        groups.setdefault(comp, []).append(k)

    keep_idx = []
    for idxs in groups.values():
        if len(idxs) == 2 and abs(abs(r_areas[idxs[0]]) - abs(r_areas[idxs[1]])) <= 1e-6 * max(abs(r_areas[idxs[0]]), abs(r_areas[idxs[1]]), 1.0):
            keep_idx.append(idxs[0] if r_areas[idxs[0]] > r_areas[idxs[1]] else idxs[1])
        else:
            keep_idx.extend(idxs)

    r_face_starts = r_face_starts[keep_idx]
    r_areas = r_areas[keep_idx]

    n = len(r_face_starts)
    r_verts, r_offsets = face_vertex_loops(r_face_starts, rhe_start, rhe_end, rhe_twin, r_sorted_he, r_v_offset, r_pos_in_block)
    r_loops = [r_verts[r_offsets[i]:r_offsets[i + 1]] for i in range(n)]
    # every reduced face's keep-status is uniform across its whole
    # loop by construction -- look it up via any one constituent edge.
    r_keep = []
    for loop in r_loops:
        u = int(loop[0])
        v = int(loop[1]) if len(loop) > 1 else u
        r_keep.append(keep[orig_face_of_dir[(u, v)]])

    rings_x = [vx[loop] for loop in r_loops]
    rings_y = [vy[loop] for loop in r_loops]

    # Same batched, numba-compiled containment query as the
    # cross-component forest above (see nearest_container_batch's
    # docstring) -- one query per reduced face, candidates are every
    # OTHER reduced face with strictly greater area (query_min_area =
    # this face's own abs area), looking for the smallest one whose
    # ring contains it.
    cand_xmin, cand_xmax, cand_ymin, cand_ymax = face_bboxes_i64(r_offsets, r_verts, vx, vy)
    query_x = vx[r_verts[r_offsets[:n]]]
    query_y = vy[r_verts[r_offsets[:n]]]
    face_id = np.arange(n, dtype=np.int64)
    cand_area = np.abs(r_areas).astype(np.float64)

    best = nearest_container_batch(
        query_x, query_y, face_id, cand_area,
        face_id, cand_area, cand_xmin, cand_xmax, cand_ymin, cand_ymax,
        r_offsets, r_verts, vx, vy,
    )
    parent: list[int | None] = [int(best[i]) if best[i] >= 0 else None for i in range(n)]

    order_desc = sorted(range(n), key=lambda i: abs(r_areas[i]), reverse=True)
    rendered_ancestor_keep: dict[int, bool] = {}
    is_rendered = [False] * n
    render_target: list[int | None] = [None] * n

    for i in order_desc:
        p = parent[i]
        ancestor_keep = False if p is None else rendered_ancestor_keep[p]
        is_rendered[i] = r_keep[i] != ancestor_keep
        rendered_ancestor_keep[i] = r_keep[i] if is_rendered[i] else ancestor_keep
        render_target[i] = None if p is None else (p if is_rendered[p] else render_target[p])

    children_of: dict[int | None, list[int]] = {}
    for i in range(n):
        if is_rendered[i]:
            children_of.setdefault(render_target[i], []).append(i)

    order_asc = sorted(range(n), key=lambda i: abs(r_areas[i]))
    built: dict[int, "Polygon"] = {}
    for i in order_asc:
        if not is_rendered[i] or abs(r_areas[i]) <= area_tol:
            continue
        my_holes = [built[c] for c in children_of.get(i, []) if c in built]
        xs = [x / scale for x in rings_x[i]]
        ys = [y / scale for y in rings_y[i]]
        built[i] = Polygon._construct_verified(xs, ys, holes=my_holes or None)

    return [built[i] for i in children_of.get(None, []) if i in built]
