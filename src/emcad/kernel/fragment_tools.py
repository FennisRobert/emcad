"""
Plain-Python/NumPy helpers for polygon fragmentation.

Everything here operates on `Polygon` objects, Python lists, or does
one-shot NumPy work (sorting, `np.unique`, grouping) that isn't worth
handing to numba. The O(E^2)/O(V) hot loops that *are* worth jitting
live in `kernel/fragment_kernels.py` instead; this file is what glues
those kernels to the rest of the geometry package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# NOTE: deliberately NOT importing Polygon at module level. poly.py
# imports low-level primitives from kernel/api.py, which imports this
# module -- so importing Polygon up here would be circular. The only
# runtime (not just type-hint) use is inside `reconstruct_polygon`,
# which imports it locally instead; `from __future__ import
# annotations` above already makes every type-hint use of `Polygon` in
# this file safe without a real import, so this guarded one is purely
# for static type checkers. This module lives in kernel/, so it's two
# levels up to poly.py at the package root.
if TYPE_CHECKING:
    from ..poly import Polygon


# --------------------------------------------------------------------------
# Step 1: pool every edge of every polygon into one flat edge list
# --------------------------------------------------------------------------

def _flatten_rings(polys: list["Polygon"]) -> list[tuple]:
    """Recursively expand every polygon -- and its holes, and their
    holes, and so on -- into a flat list of (xs, ys) rings.

    Needed so `poly_fragment`'s arrangement actually sees hole
    boundaries as places to split, not just each polygon's outer
    boundary. Without this, a polygon that already carries `holes=`
    (e.g. the output of one boolean op fed back in as input to
    another, which is exactly what a batched/tree-reduce union does)
    would silently have its holes ignored by the arrangement -- the
    hole gets swallowed and the result comes back solid.
    `Polygon.is_inside`/`point_inside` already handle holes correctly
    for membership testing elsewhere in the pipeline; this is the
    other half, needed so the arrangement itself splits around them.
    """
    rings = []
    for poly in polys:
        rings.append((poly.xs, poly.ys))
        if poly.has_holes:
            rings.extend(_flatten_rings(poly.holes))
    return rings


def collect_edges(polys: list[Polygon]):
    """Flatten N polygons -- and, recursively, every hole they carry --
    into one (2, E) start/end edge array.

    By construction, edge `e`'s *start* vertex is global vertex id `e`
    (edges are emitted in the same order as vertices, ring by ring),
    which lets everything downstream avoid a separate edge -> vertex-id
    indirection for edge starts.

    Returns:
        pts_start, pts_end   : (2, E) float64 edge endpoint coordinates
        edge_poly_id         : (E,) int64, which ring (a polygon's own
                                outer boundary, or one of its holes at
                                any depth) each edge came from
        edge_local_id        : (E,) int64, edge index within that ring
        poly_vert_offset     : (n_rings + 1,) int64 prefix-sum of vertex
                                counts per ring
    """
    rings = _flatten_rings(polys)

    n_edges_total = sum(len(rxs) for rxs, _ in rings)
    pts_start = np.empty((2, n_edges_total), dtype=np.float64)
    pts_end = np.empty((2, n_edges_total), dtype=np.float64)
    edge_poly_id = np.empty(n_edges_total, dtype=np.int64)
    edge_local_id = np.empty(n_edges_total, dtype=np.int64)

    poly_vert_offset = np.zeros(len(rings) + 1, dtype=np.int64)

    ectr = 0
    for pid, (rxs, rys) in enumerate(rings):
        n = len(rxs)
        poly_vert_offset[pid + 1] = poly_vert_offset[pid] + n

        xs = np.asarray(rxs, dtype=np.float64)
        ys = np.asarray(rys, dtype=np.float64)
        xs_next = np.roll(xs, -1)
        ys_next = np.roll(ys, -1)

        pts_start[0, ectr:ectr + n] = xs
        pts_start[1, ectr:ectr + n] = ys
        pts_end[0, ectr:ectr + n] = xs_next

        pts_end[1, ectr:ectr + n] = ys_next
        edge_poly_id[ectr:ectr + n] = pid
        edge_local_id[ectr:ectr + n] = np.arange(n)
        ectr += n

    return pts_start, pts_end, edge_poly_id, edge_local_id, poly_vert_offset


# --------------------------------------------------------------------------
# Step 3: snap-merge near-duplicate points (this is what fixes most of the
# original instability -- independently computed "same" crossing points
# never landed on exactly the same float64 coordinate)
# --------------------------------------------------------------------------

def merge_points(xy: np.ndarray, tol: float):
    """Grid-snap merge of near-duplicate points.

    Args:
        xy: (2, N) coordinates.
        tol: absolute distance below which two points are the same.

    Returns:
        inverse: (N,) int64, per input point -> merged point id.
        merged_xy: (2, n_unique) float64, centroid of each merged group.
    """
    keys = np.round(xy.T / tol).astype(np.int64)  # (N, 2)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n_unique = counts.shape[0]

    merged_xy = np.zeros((2, n_unique), dtype=np.float64)
    np.add.at(merged_xy[0], inverse, xy[0])
    np.add.at(merged_xy[1], inverse, xy[1])
    merged_xy /= counts

    return inverse.astype(np.int64), merged_xy


# --------------------------------------------------------------------------
# Step 4: chain each original edge's interior split points (sorted by t)
# into atomic segments, and dedupe exact duplicate segments (e.g. two
# input polygons sharing an edge outright).
# --------------------------------------------------------------------------

def chain_edges(n_edges: int, edge_start_vid: np.ndarray, edge_end_vid: np.ndarray,
                 split_edge_id: np.ndarray, split_t: np.ndarray, split_vid: np.ndarray):
    """Chain each edge's interior split points into atomic (u, v)
    sub-segments. Does NOT deduplicate -- if two different input edges
    happen to chain into the exact same sub-segment, both copies come
    back. `build_segments` (below) throws that multiplicity away
    because `poly_fragment` doesn't need it, but `_join.py` does (a
    segment coming back twice is exactly how it tells a shared/internal
    edge from a boundary one), so it's kept here as the shared
    lower-level building block.

    Args:
        n_edges: number of original edges.
        edge_start_vid, edge_end_vid: (n_edges,) int64 merged vertex ids
            of each edge's endpoints.
        split_edge_id: (n_splits,) int64, which edge each split point is on.
        split_t: (n_splits,) float64, parametric position along that edge.
        split_vid: (n_splits,) int64, merged vertex id of each split point.

    Returns:
        seg_u, seg_v: (n_segments,) int64 arrays of undirected,
            zero-length-filtered, NOT deduplicated atomic segments.
    """
    if split_edge_id.shape[0] > 0:
        order = np.lexsort((split_t, split_edge_id))
        split_edge_id = split_edge_id[order]
        split_t = split_t[order]
        split_vid = split_vid[order]

    seg_u = []
    seg_v = []

    ptr = 0
    n_splits = split_edge_id.shape[0]
    for e in range(n_edges):
        chain = [edge_start_vid[e]]
        while ptr < n_splits and split_edge_id[ptr] == e:
            chain.append(split_vid[ptr])
            ptr += 1
        chain.append(edge_end_vid[e])

        for a, b in zip(chain[:-1], chain[1:]):
            if a == b:
                continue  # zero-length after merging -- drop
            seg_u.append(a)
            seg_v.append(b)

    return np.asarray(seg_u, dtype=np.int64), np.asarray(seg_v, dtype=np.int64)


def build_segments(n_edges: int, edge_start_vid: np.ndarray, edge_end_vid: np.ndarray,
                    split_edge_id: np.ndarray, split_t: np.ndarray, split_vid: np.ndarray):
    """Same as `chain_edges`, plus deduplication of exact-duplicate
    segments (e.g. two input polygons sharing an edge outright) --
    appropriate for `poly_fragment`'s own arrangement-building, where a
    shared edge should just become one graph edge. See `chain_edges` if
    you need the raw, multiplicity-preserving version instead.

    Returns:
        seg_u, seg_v: (n_segments,) int64 arrays of undirected,
            deduplicated, zero-length-filtered atomic segments.
    """
    seg_u, seg_v = chain_edges(n_edges, edge_start_vid, edge_end_vid, split_edge_id, split_t, split_vid)

    if seg_u.shape[0] == 0:
        return seg_u, seg_v

    key = np.stack([np.minimum(seg_u, seg_v), np.maximum(seg_u, seg_v)], axis=1)
    _, first_idx = np.unique(key, axis=0, return_index=True)
    first_idx.sort()
    return seg_u[first_idx], seg_v[first_idx]


# --------------------------------------------------------------------------
# Step 5: build the half-edge rotation system (per-vertex, angle-sorted
# outgoing half-edges) -- this replaces the O(n_points) linear scan the
# original code did at every single step of every walk.
# --------------------------------------------------------------------------

def build_rotation_system(he_start: np.ndarray, he_end: np.ndarray, xy: np.ndarray, n_verts: int):
    """Build the CSR-style angle-sorted rotation system consumed by the
    `trace_faces` / `face_areas` numba kernels.

    Args:
        he_start, he_end: (H,) int64 half-edge start/end vertex ids.
        xy: (2, n_verts) float64 vertex coordinates.
        n_verts: total number of merged vertices.

    Returns:
        sorted_he: (H,) int64, half-edge ids grouped by start vertex and
            angle-sorted within each group.
        v_offset: (n_verts + 1,) int64, CSR block boundaries into sorted_he.
        pos_in_block: (H,) int64, each half-edge's index within its own
            start vertex's sorted block.
    """
    H = he_start.shape[0]
    dx = xy[0, he_end] - xy[0, he_start]
    dy = xy[1, he_end] - xy[1, he_start]
    angle = np.arctan2(dy, dx)

    # primary key = he_start, secondary key = angle (lexsort's *last*
    # key is primary)
    sorted_he = np.lexsort((angle, he_start)).astype(np.int64)

    counts = np.bincount(he_start, minlength=n_verts)
    v_offset = np.zeros(n_verts + 1, dtype=np.int64)
    v_offset[1:] = np.cumsum(counts)

    pos_in_block = np.empty(H, dtype=np.int64)
    pos_in_block[sorted_he] = np.arange(H, dtype=np.int64) - v_offset[he_start[sorted_he]]

    return sorted_he, v_offset, pos_in_block


# --------------------------------------------------------------------------
# Step 6b: rebuild an actual Polygon from a traced face (cheap, and only
# needed for the faces we're keeping, so it stays in plain Python).
# --------------------------------------------------------------------------

def reconstruct_polygon(h0: int, he_start: np.ndarray, he_end: np.ndarray, he_twin: np.ndarray,
                         sorted_he: np.ndarray, v_offset: np.ndarray, pos_in_block: np.ndarray,
                         merged_xy: np.ndarray) -> Polygon:
    """Walk a traced face starting at half-edge `h0` and build a Polygon
    from the vertex sequence. Uses the same next-half-edge rule as the
    `trace_faces`/`face_areas` kernels -- see their comments for why it's
    correct.
    """
    verts = []
    h = h0
    while True:
        verts.append(he_start[h])
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
    from ..poly import Polygon  # deferred -- see the top-of-file circular-import note

    xs = merged_xy[0, verts]
    ys = merged_xy[1, verts]
    return Polygon(xs, ys)


# --------------------------------------------------------------------------
# Step 7: keep only fragments that actually fall inside (at least one of)
# the original input polygons -- discards anything the arrangement traced
# that wasn't part of the original polygon set (e.g. faces formed purely
# by how unrelated polygons happen to cross each other).
# --------------------------------------------------------------------------

def filter_fragments_to_originals(fragments: list[Polygon], original_polys: list[Polygon]) -> list[Polygon]:
    """Keep a fragment iff its interior point falls inside at least one
    of the original polygons (union semantics). Swap the `any(...)` for
    `all(p.is_inside(...) for p in original_polys)` if you specifically
    want boolean intersection instead.
    """
    kept = []
    for frag in fragments:
        try:
            px, py = frag.point_inside()
        except ValueError:
            # numerically degenerate sliver -- unclassifiable, skip
            # rather than crash (same stance as elsewhere in this file)
            continue
        for original in original_polys:
            if original.is_inside(px, py):
                kept.append(frag)
                break
    return kept


# --------------------------------------------------------------------------
# Step 8: assemble ALL traced faces (both signs) into properly nested
# Polygon(holes=...) objects via a containment forest, instead of
# poly_fragment's old plain "keep positive-area faces" filter.
#
# Why this matters: an isolated loop (one that shares no vertex with
# anything else in this specific arrangement -- which is exactly what
# a Polygon's own `.holes` ring looks like once it's fed back into a
# *new* arrangement as one operand of a further boolean op) always
# traces as a self-paired twin: one CCW pass and one CW pass of the
# identical shape, since nothing else in this arrangement disambiguates
# which one is "solid" and which is "the hole". Trusting the raw sign
# for such a loop is meaningless. Containment (does face A's point fall
# inside face B's raw boundary, and is B the smallest such face) is a
# reliable signal regardless of connectivity, so a face's role
# (solid/hole/island/...) is derived from its DEPTH in that forest
# instead. This is the same technique `_join.py`/`_keyhole.py` already
# use for their own hole assembly -- moving it in here means every
# consumer of poly_fragment (add_polygons, intersect_polygons,
# subtract_polygons, and anything built on top of them) gets a
# polygon-with-a-hole that's safe to feed into another boolean op,
# rather than only join_polygons/dekeyhole_polygon getting it.
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


def _signed_area(xs: np.ndarray, ys: np.ndarray) -> float:
    x1 = np.roll(xs, -1)
    y1 = np.roll(ys, -1)
    return 0.5 * float(np.sum(xs * y1 - x1 * ys))


def _dedupe_isolated_loop_twins(face_starts: np.ndarray, areas: np.ndarray,
                                 he_start: np.ndarray, find) -> list[int]:
    """Group traced faces by connected component (via the vertex they
    start at); a component that produced exactly two faces of matching
    |area| is an isolated simple loop's self-paired twin -- keep one
    (arbitrary, since both describe the identical shape and
    `is_inside`/`point_inside` don't care about winding) and drop the
    other. Any other grouping (more faces, or mismatched areas) is left
    untouched.
    """
    groups: dict[int, list[int]] = {}
    for k in range(len(face_starts)):
        comp = find(int(he_start[face_starts[k]]))
        groups.setdefault(comp, []).append(k)

    keep = []
    for idxs in groups.values():
        if len(idxs) == 2:
            a0, a1 = abs(areas[idxs[0]]), abs(areas[idxs[1]])
            if abs(a0 - a1) / max(a0, a1, 1e-300) < 1e-6:
                keep.append(idxs[0])
                continue
        keep.extend(idxs)
    return keep


def assemble_nested_polygons(
    face_starts: np.ndarray, he_start: np.ndarray, he_end: np.ndarray, he_twin: np.ndarray,
    sorted_he: np.ndarray, v_offset: np.ndarray, pos_in_block: np.ndarray, merged_xy: np.ndarray,
    n_verts: int, seg_u: np.ndarray, seg_v: np.ndarray, areas: np.ndarray, area_tol: float,
) -> list[Polygon]:
    """Trace every face (regardless of sign), deduplicate isolated-loop
    twin pairs, and build a containment forest to correctly nest the
    survivors into `Polygon(holes=[...])` objects. Returns only the
    top-level (no parent), positive-signed faces -- a top-level
    negative-area face is just the harmless outer/unbounded artifact of
    an isolated component (same role `keep="positive"` used to filter
    out directly), not a real hole of anything, so it's silently
    dropped rather than returned.
    """
    from ..poly import Polygon  # deferred -- see the top-of-file circular-import note

    find, union = _union_find(n_verts)
    for u, v in zip(seg_u, seg_v):
        union(int(u), int(v))

    keep_idx = _dedupe_isolated_loop_twins(face_starts, areas, he_start, find)

    raw_faces: list[Polygon] = []
    signed_areas: list[float] = []
    components: list[int] = []
    for k in keep_idx:
        if abs(areas[k]) <= area_tol:
            continue
        poly = reconstruct_polygon(face_starts[k], he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block, merged_xy)
        raw_faces.append(poly)
        signed_areas.append(float(areas[k]))
        components.append(find(int(he_start[face_starts[k]])))

    n = len(raw_faces)
    if n == 0:
        return []

    abs_areas = [abs(a) for a in signed_areas]
    # a face that cleared area_tol but is still numerically degenerate
    # for point_inside's own algorithm (e.g. a needle-thin sliver) is
    # unclassifiable either way -- drop it rather than letting one such
    # face take down the whole fragmentation, same defensive stance as
    # _classify_and_select's own point_inside guard downstream.
    points: list[tuple[float, float]] = [None] * n
    keep_mask = [True] * n
    for i, f in enumerate(raw_faces):
        try:
            points[i] = f.point_inside()
        except ValueError:
            keep_mask[i] = False
    if not all(keep_mask):
        raw_faces = [f for f, k in zip(raw_faces, keep_mask) if k]
        signed_areas = [a for a, k in zip(signed_areas, keep_mask) if k]
        components = [c for c, k in zip(components, keep_mask) if k]
        points = [p for p, k in zip(points, keep_mask) if k]
        abs_areas = [abs(a) for a in signed_areas]
        n = len(raw_faces)
        if n == 0:
            return []

    # A face's parent candidates are restricted to faces in a
    # *different* connected component. Two faces in the same component
    # share edges by definition (that's what connects them) -- e.g.
    # sq1-only, the overlap, and sq2-only for two overlapping squares
    # are three siblings that together partition one region, not a
    # nesting relationship, even though the region's own outer/negative
    # artifact face technically "contains" each of their points. A
    # genuine hole, by contrast, is topologically isolated from its
    # parent (no shared edges at all) -- that's a different component,
    # which is exactly the signal this restriction keys off.
    parent: list[int | None] = [None] * n
    for i in range(n):
        xi, yi = points[i]
        ai = abs_areas[i]
        ci = components[i]
        best_area = None
        best_j = None
        for j in range(n):
            if j == i or abs_areas[j] <= ai or components[j] == ci:
                continue
            if raw_faces[j].is_inside(xi, yi):
                if best_area is None or abs_areas[j] < best_area:
                    best_area, best_j = abs_areas[j], j
        parent[i] = best_j

    children_of: dict[int, list[int]] = {}
    for i, p in enumerate(parent):
        if p is not None:
            children_of.setdefault(p, []).append(i)

    order = sorted(range(n), key=lambda i: abs_areas[i])  # children (smaller) before parents
    built: dict[int, Polygon] = {}
    for i in order:
        my_holes = [built[c] for c in children_of.get(i, [])]
        built[i] = Polygon._construct_verified(raw_faces[i].xs, raw_faces[i].ys, holes=my_holes or None)

    return [built[i] for i in range(n) if parent[i] is None and signed_areas[i] > 0]