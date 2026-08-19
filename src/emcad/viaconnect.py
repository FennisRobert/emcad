"""
via_wall.py

Generate a thickened "wall" polygon (with correctly-formed holes for any
loop in the via network) from a cloud of via XY coordinates -- a
practical alternative to meshing every individual via in an FEM
simulation when there are hundreds to thousands of them forming a
via-fence around a trace.

Pipeline
--------
1. Connect every pair of vias within `max_dist` of each other. The
   O(n^2) proximity check is numba-jitted -- this is the part that
   actually matters once you're north of ~1000 vias.
2. Contract the resulting graph: any vertex of degree 2 whose two
   incident edges are collinear gets dissolved, merging its two edges
   into one longer straight edge. Iterated until stable, so a long
   straight run of vias collapses to a single segment instead of one
   unit per via-to-via hop. This is the "merger routine."
3. Two kinds of primitive:
     - a full circle at `thickness / 2` at every via that's part of
       the network,
     - a plain rectangle (no caps) at every surviving edge, by default
       at the same half-width as the circles (`edge_width_factor=1.0`
       -- see that parameter's own docstring on `via_wall_polygons`
       for the option to draw it narrower instead). No delicate
       angle-alignment needed at a joint the way there is with capsule
       end-caps, which is what made the previous (capsule-based)
       approach fragile at junctions where 3+ segments meet close
       together -- the exact-predicate arrangement engine resolves
       whatever overlap/tangency results at a joint robustly either way.
4. Fuse incrementally, respecting the graph's actual topology, via
   union-find over the vias: each connected component tracks a single
   live Polygon. For each edge, union its rectangle with the (at most
   two) components it touches -- never rectangle-with-rectangle, and a
   via's circle is never duplicated across separate pending pieces.
   Closing a loop is just the case where an edge's two endpoints are
   already in the same component: unioning the closing rectangle into
   that single already-fused piece is exactly what `add_polygons`/
   `join_polygons` correctly turns into a hole.

Standalone, single-file script: imports `emcad`'s existing `Polygon` /
`add_polygons` / `GeometryPlotter` rather than reimplementing polygon
boolean ops from scratch. Run directly (`python via_wall.py`) to see
the test case plotted.
"""

from __future__ import annotations

import math
import time

import numpy as np
from numba import njit
from loguru import logger

from emcad.poly import Polygon
from emcad.kernel.api import add_polygons
from emcad.plot import GeometryPlotter
from emcad.kernel import fragment_kernels as fkernels
from emcad.kernel.fragment_tools import build_rotation_system
from emcad.kernel._arrangement_kernels import face_id_per_halfedge, face_vertex_loops
from emcad.kernel._profiling import format_runtime


# --------------------------------------------------------------------------
# Step 1: proximity graph -- the one genuinely O(n^2) part, numba-jitted.
# Two passes (count, then fill) so we allocate the output array exactly
# once rather than growing a list inside the jitted loop.
# --------------------------------------------------------------------------

@njit(cache=True)
def _proximity_edges(xs: np.ndarray, ys: np.ndarray, max_dist: float) -> np.ndarray:
    """Every pair (i, j), i < j, with distance <= max_dist. Returns a
    (2, K) int64 array of vertex-index pairs.

    O(n^2), same "TODO: spatial index for very large N" caveat as the
    rest of this package's intersection kernels -- fine up to several
    thousand vias, since it's a tight, compiled loop; would want
    bucketing for tens of thousands.
    """
    n = xs.shape[0]
    max_d2 = max_dist * max_dist

    count = 0
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, n):
            dx = xs[j] - xi
            dy = ys[j] - yi
            if dx * dx + dy * dy <= max_d2:
                count += 1

    out = np.empty((2, count), dtype=np.int64)
    ctr = 0
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, n):
            dx = xs[j] - xi
            dy = ys[j] - yi
            if dx * dx + dy * dy <= max_d2:
                out[0, ctr] = i
                out[1, ctr] = j
                ctr += 1

    return out


def build_proximity_graph(via_xy: np.ndarray, max_dist: float) -> list[tuple[int, int]]:
    """via_xy: (N, 2) array. Returns a list of (i, j) index-pair edges,
    after pruning redundant "skip" edges (see `_prune_redundant_edges`).
    """
    xs = np.ascontiguousarray(via_xy[:, 0], dtype=np.float64)
    ys = np.ascontiguousarray(via_xy[:, 1], dtype=np.float64)
    edges = _proximity_edges(xs, ys, max_dist)
    keep = _prune_redundant_edges(xs, ys, edges[0], edges[1], max_dist)
    return [(int(edges[0, k]), int(edges[1, k])) for k in range(edges.shape[1]) if keep[k]]


@njit(cache=True)
def _prune_redundant_edges(
    xs: np.ndarray, ys: np.ndarray, edges_i: np.ndarray, edges_j: np.ndarray, max_dist: float
) -> np.ndarray:
    """Relative-Neighborhood-Graph-style pruning: drop edge (i, j) if
    some other via k is within max_dist of both endpoints and strictly
    closer to each of them than i and j are to each other.

    Without this, connecting "every pair within max_dist" naively
    creates redundant edges parallel to (or nearly along) a more
    direct chain whenever via spacing is smaller than max_dist -- e.g.
    a regularly-spaced straight run of vias would otherwise also get
    "skip" edges to next-nearest and next-next-nearest neighbors. That
    breaks the collinear-merge step (those vertices end up degree
    4-6, not 2) and produces redundant overlapping primitives.
    """
    n_edges = edges_i.shape[0]
    n = xs.shape[0]
    keep = np.ones(n_edges, dtype=np.bool_)
    max_d2 = max_dist * max_dist

    for e in range(n_edges):
        i = edges_i[e]
        j = edges_j[e]
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]
        dij2 = (xj - xi) ** 2 + (yj - yi) ** 2

        for k in range(n):
            if k == i or k == j:
                continue
            xk, yk = xs[k], ys[k]
            dik2 = (xk - xi) ** 2 + (yk - yi) ** 2
            if dik2 > max_d2 or dik2 >= dij2:
                continue
            djk2 = (xk - xj) ** 2 + (yk - yj) ** 2
            if djk2 > max_d2 or djk2 >= dij2:
                continue
            keep[e] = False
            break

    return keep


# --------------------------------------------------------------------------
# Step 2: the merger routine -- dissolve collinear degree-2 vertices.
# Graph-shaped (dicts of sets), so this stays plain Python rather than
# numba; edge counts here are O(n), not O(n^2), so it's not the
# bottleneck at any via count this script is meant for.
# --------------------------------------------------------------------------

def merge_collinear_chains(
    via_xy: np.ndarray,
    edges: list[tuple[int, int]],
    angle_tol: float = 1e-3,
) -> list[tuple[int, int]]:
    """Repeatedly dissolve any vertex of degree 2 whose two incident
    edges are collinear, replacing its two edges with one longer edge.

    A long straight run of vias collapses to a single segment;
    junctions (degree != 2) and genuine corners (non-collinear) are
    left alone, since those need to stay explicit vertices in the
    output wall.

    Args:
        via_xy: (N, 2) via coordinates.
        edges: (i, j) index pairs, as from `build_proximity_graph`.
        angle_tol: maximum allowed angular deviation (radians) between
            the two edge directions to still count as collinear.

    Returns:
        The reduced list of (i, j) edges.
    """
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def is_collinear(u: int, v: int, w: int) -> bool:
        pu, pv, pw = via_xy[u], via_xy[v], via_xy[w]
        d1 = pv - pu
        d2 = pw - pv
        n1 = math.hypot(d1[0], d1[1])
        n2 = math.hypot(d2[0], d2[1])
        if n1 < 1e-300 or n2 < 1e-300:
            return False
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        dot = d1[0] * d2[0] + d1[1] * d2[1]
        angle = math.atan2(abs(cross), dot)
        # dot > 0 rules out a "doubled-back" spike (u, v, w folding back
        # on itself) from being treated as collinear-straight
        return angle <= angle_tol and dot > 0

    changed = True
    while changed:
        changed = False
        for v in list(adj.keys()):
            nbrs = adj.get(v)
            if nbrs is None or len(nbrs) != 2:
                continue
            u, w = tuple(nbrs)
            if u == w:
                continue
            if not is_collinear(u, v, w):
                continue
            adj[u].discard(v)
            adj[w].discard(v)
            adj[u].add(w)
            adj[w].add(u)
            del adj[v]
            changed = True

    seen: set[tuple[int, int]] = set()
    final_edges: list[tuple[int, int]] = []
    for a, nbrs in adj.items():
        for b in nbrs:
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                final_edges.append(key)
    return final_edges


# --------------------------------------------------------------------------
# Step 3: primitives. A full circle per via, a narrower plain rectangle
# per edge -- the rectangle's ends are always strictly inside its own
# two via-circles, by construction, so there's nothing delicate to get
# right at a joint.
# --------------------------------------------------------------------------

def make_circle_polygon(center: np.ndarray, radius: float, n_segments: int = 16,
                         angle_offset: float = 0.0) -> Polygon:
    """A regular n-gon approximating a circle.

    `angle_offset` rotates the polygon's vertex placement -- a real
    circle is rotationally symmetric, so this changes nothing about
    the shape, only where its discretized vertices land. Not applied
    by default here: an earlier attempt at giving every via a
    deterministic per-via offset (to avoid axis-aligned rectangle edges
    exactly coinciding with circle vertices at 0/90/180/270 degrees,
    which any `n_segments` that's a multiple of 4 puts there) was
    reverted after it introduced a *different*, confirmed failure --
    with all-fresh rotations, some junctions ended up with new
    near-coincident (not exact) edges instead, which is just as
    unsafe. Left available as a parameter in case you want to
    experiment with it for a specific geometry, but not applied
    automatically.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False) + angle_offset
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return Polygon(list(xs), list(ys))


def make_edge_rectangle(p0: np.ndarray, p1: np.ndarray, half_width: float) -> Polygon | None:
    """A plain 4-point rectangle running p0->p1, no end caps. Returns
    None for a degenerate (coincident-point) edge -- caller should
    just rely on the via circles alone in that case.
    """
    d = p1 - p0
    length = math.hypot(d[0], d[1])
    if length < 1e-12:
        return None

    ux, uy = d[0] / length, d[1] / length
    nx, ny = -uy, ux

    xs = [
        p0[0] + nx * half_width, p1[0] + nx * half_width,
        p1[0] - nx * half_width, p0[0] - nx * half_width,
    ]
    ys = [
        p0[1] + ny * half_width, p1[1] + ny * half_width,
        p1[1] - ny * half_width, p0[1] - ny * half_width,
    ]
    return Polygon(xs, ys)


# --------------------------------------------------------------------------
# Step 4: fuse, incrementally, respecting the graph's actual topology --
# this is the part that matters. Union-find over the vias: each root
# tracks the single Polygon representing everything fused into its
# connected component so far. For each edge, we union its box with the
# (at most two) *components* it touches -- never box-with-box, and a
# via's circle is never duplicated across pending pieces, since there's
# only ever one live Polygon per component. Closing a loop is just the
# case where an edge's two endpoints are already in the same component:
# unioning the closing box into that single already-fused piece is
# exactly the situation add_polygons/join_polygons correctly turns into
# a hole (same mechanism already exercised by every hole-producing test
# elsewhere in this package) -- it never has to reconcile two different,
# independently-fused copies of the same shared geometry, which is what
# a generic, topology-blind batching strategy could stumble into.
# --------------------------------------------------------------------------

def _fuse_network(via_xy: np.ndarray, merged_edges: list[tuple[int, int]], connected: set[int],
                   radius: float, edge_half_width: float, circle_segments: int,
                   merge_tol: float, area_tol: float,
                   snapshot_at: set[int] | None = None,
                   snapshots: list[tuple[int, list[Polygon]]] | None = None) -> list[Polygon]:
    parent: dict[int, int] = {v: v for v in connected}

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    piece_of_root: dict[int, Polygon] = {
        v: make_circle_polygon(via_xy[v], radius, circle_segments)
        for v in connected
    }
    leftover_pieces: list[Polygon] = []

    for i, (a, b) in enumerate(merged_edges):
        ra, rb = find(a), find(b)
        d = via_xy[b] - via_xy[a]
        length = math.hypot(d[0], d[1])
        # if the two vias are closer together than their own radius,
        # their circles already substantially overlap on their own --
        # a connecting rectangle at that scale is both redundant and,
        # in practice, prone to producing degenerate sliver fragments
        # (this matters for densely-packed via clouds where spacing
        # can be well under the wall thickness).
        box = make_edge_rectangle(via_xy[a], via_xy[b], edge_half_width) if length >= radius else None

        if ra == rb:
            pieces = [piece_of_root[ra]] if box is None else [box, piece_of_root[ra]]
        else:
            pieces = (
                [piece_of_root[ra], piece_of_root[rb]] if box is None
                else [box, piece_of_root[ra], piece_of_root[rb]]
            )

        merged = (
            add_polygons(*pieces, merge_tol=merge_tol, area_tol=area_tol)
            if len(pieces) > 1 else pieces
        )

        if ra != rb:
            del piece_of_root[rb]
            parent[rb] = ra

        # normally exactly 1 piece comes back (the box/circles all
        # touch, by construction); if something genuinely stayed
        # disconnected, keep it rather than silently dropping it
        piece_of_root[ra] = merged[0]
        leftover_pieces.extend(merged[1:])

        if snapshot_at is not None and (i + 1) in snapshot_at:
            snapshots.append((i + 1, list(piece_of_root.values()) + list(leftover_pieces)))

    return list(piece_of_root.values()) + leftover_pieces


# --------------------------------------------------------------------------
# Step 4b: topology-first assembly -- the default, and the fast path for
# large fenced/gridded networks.
#
# The previous default (`incremental=False`) ran every circle/rectangle
# for the WHOLE network through one `add_polygons` call. That's a single
# exact-arrangement computation over every tessellated primitive at
# once -- correct, but its cost is driven by the network's total edge
# count, and a grid-stitching via fence with hundreds of enclosed cells
# hands that arrangement O(10x-20x) more edges than the underlying via
# graph actually has (each via becomes a `circle_segments`-gon, each
# connection a 4-point rectangle), on top of which every enclosed cell
# becomes a hole that a downstream CAD kernel then has to process.
#
# But the via network's own connectivity graph -- vias as nodes, merged
# edges as graph edges -- already tells us, combinatorially and for
# free, exactly which loops exist and which primitives bound each one:
# tracing ITS faces (same half-edge/rotation-system machinery used
# throughout this package, just on ~N_via nodes instead of the ~20x
# larger tessellated arrangement) finds every enclosed loop directly,
# with no geometric guessing needed at all. Each face's own boundary
# walk then tells us exactly which vias/edges bound it -- so instead of
# one large union over everything, we run many SMALL, LOCAL
# `add_polygons` calls, one per face, each over only the handful of
# primitives on that one loop's own boundary. Total work scales with
# the network's size (roughly one small union per hole), not with
# size-squared over the whole tessellated network at once.
#
# Assumption, and its limit: every bounded face within one connected
# component is treated as a direct, flat hole of that component's
# outer boundary -- correct for the overwhelming common cases (a
# simple loop, a grid/mesh of adjacent cells sharing edges, several
# disconnected sub-networks), and the same assumption
# `fragment_tools.assemble_nested_polygons` already makes elsewhere in
# this package. It under-resolves one exotic topology: a "bridge" edge
# connecting two otherwise-separate loops within the SAME connected
# component (a lollipop shape) traces a single complex face threading
# both loops rather than two independent ones -- rare for a real via
# fence, not specially handled here.
# --------------------------------------------------------------------------

def _via_graph_faces(via_xy: np.ndarray, merged_edges: list[tuple[int, int]]):
    """Trace the via connectivity graph's own faces -- cheap, since this
    graph has ~N_via nodes and ~N_via edges, not the ~20x larger
    tessellated arrangement the primitives would produce.

    Returns:
        n_faces: number of traced faces.
        component_of: (n_faces,) which connected component each face belongs to.
        outer_of_component: dict comp -> face index of that component's
            unbounded/exterior face (never itself a hole).
        face_vias: dict face_idx -> set of via indices on its boundary walk.
        face_edges: dict face_idx -> set of indices into `merged_edges` on its boundary walk.
    """
    n = via_xy.shape[0]
    M = len(merged_edges)

    edge_u = np.asarray([a for a, b in merged_edges], dtype=np.int64)
    edge_v = np.asarray([b for a, b in merged_edges], dtype=np.int64)

    he_start = np.empty(2 * M, dtype=np.int64)
    he_end = np.empty(2 * M, dtype=np.int64)
    he_twin = np.empty(2 * M, dtype=np.int64)
    he_edge_idx = np.empty(2 * M, dtype=np.int64)
    he_start[0::2] = edge_u
    he_end[0::2] = edge_v
    he_start[1::2] = edge_v
    he_end[1::2] = edge_u
    he_twin[0::2] = np.arange(1, 2 * M, 2, dtype=np.int64)
    he_twin[1::2] = np.arange(0, 2 * M, 2, dtype=np.int64)
    he_edge_idx[0::2] = np.arange(M, dtype=np.int64)
    he_edge_idx[1::2] = np.arange(M, dtype=np.int64)

    xy_float = np.vstack([via_xy[:, 0], via_xy[:, 1]]).astype(np.float64)
    sorted_he, v_offset, pos_in_block = build_rotation_system(he_start, he_end, xy_float, n)

    face_starts = fkernels.trace_faces(he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block)
    areas = fkernels.face_areas(face_starts, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block, xy_float)
    he_face_id = face_id_per_halfedge(face_starts, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block)
    verts, offsets = face_vertex_loops(face_starts, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block)

    find, union = _union_find_simple(n)
    for a, b in merged_edges:
        union(a, b)

    n_faces = len(face_starts)
    component_of = [find(int(he_start[face_starts[k]])) for k in range(n_faces)]

    faces_by_component: dict[int, list[int]] = {}
    for k in range(n_faces):
        faces_by_component.setdefault(component_of[k], []).append(k)
    outer_of_component = {comp: min(idxs, key=lambda k: areas[k]) for comp, idxs in faces_by_component.items()}

    # per-face participating vias/edges, read directly off each face's
    # own half-edge walk (`he_edge_idx` for edges, `he_start` for vias).
    face_vias: dict[int, set[int]] = {k: set() for k in range(n_faces)}
    face_edges: dict[int, set[int]] = {k: set() for k in range(n_faces)}
    for h in range(2 * M):
        fk = int(he_face_id[h])
        face_vias[fk].add(int(he_start[h]))
        face_edges[fk].add(int(he_edge_idx[h]))

    return n_faces, component_of, outer_of_component, face_vias, face_edges, faces_by_component


def _union_find_simple(n: int):
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


def _assemble_by_topology(
    via_xy: np.ndarray,
    merged_edges: list[tuple[int, int]],
    radius: float,
    edge_half_width: float,
    circle_segments: int,
    merge_tol: float,
    area_tol: float,
    min_hole_area: float | None = None,
) -> list[Polygon]:
    if len(merged_edges) == 0:
        return []

    n_faces, component_of, outer_of_component, face_vias, face_edges, faces_by_component = _via_graph_faces(
        via_xy, merged_edges
    )
    # this is the size driver for the loop below: one small add_polygons
    # call per bounded face, so a large n_faces here (not n_vias) is
    # what to watch for a slow via-wall call.
    logger.debug(
        f"_assemble_by_topology: {len(faces_by_component)} component(s), {n_faces} face(s) "
        f"({n_faces - len(faces_by_component)} bounded/hole-candidate)"
    )

    all_vias = {v for a, b in merged_edges for v in (a, b)}
    circle_of = {v: make_circle_polygon(via_xy[v], radius, circle_segments) for v in all_vias}
    rect_of: list[Polygon | None] = [
        make_edge_rectangle(via_xy[a], via_xy[b], edge_half_width)
        if math.hypot(*(via_xy[b] - via_xy[a])) >= radius else None
        for a, b in merged_edges
    ]

    def build_face_union(k: int) -> list[Polygon]:
        prims = [circle_of[v] for v in face_vias[k]]
        prims.extend(rect_of[e] for e in face_edges[k] if rect_of[e] is not None)
        if len(prims) == 0:
            return []
        if len(prims) == 1:
            return prims
        return add_polygons(*prims, merge_tol=merge_tol, area_tol=area_tol)

    component_polys: list[list[Polygon]] = []
    for comp, idxs in faces_by_component.items():
        outer_k = outer_of_component[comp]
        outer_pieces = build_face_union(outer_k)
        if not outer_pieces:
            continue

        # A bounded face's own local union is a solid FRAME around that
        # face's own empty middle -- e.g. one grid cell's 4 walls form a
        # small closed square with a hole, not the hole itself (the
        # same "ring of touching pieces" pattern this package's own
        # test suite covers for join/add_polygons). What we actually
        # want attached is that empty middle, i.e. `piece.holes`, not
        # `piece`. The exterior face's own union has exactly the same
        # shape: unioning ITS OWN primitives alone forms a frame around
        # a placeholder "hole" that's really the network's whole
        # interior (everything the exterior doesn't directly touch,
        # e.g. for a grid, all of it) -- an artifact of only having the
        # perimeter's own primitives on hand, not the interior
        # structure. Discard it unconditionally and rebuild the real
        # hole list from every bounded face's own correctly-local
        # result instead.
        hole_pieces: list[Polygon] = []
        for k in idxs:
            if k == outer_k:
                continue
            for piece in build_face_union(k):
                for hole in piece.holes:
                    if min_hole_area is not None and _ring_area(hole.axs, hole.ays) < min_hole_area:
                        continue
                    hole_pieces.append(hole)

        if len(outer_pieces) == 1:
            main = outer_pieces[0]
            component_polys.append([Polygon._construct_verified(main.xs, main.ys, holes=hole_pieces or None)])
        else:
            # Rare: this component's own outer-loop primitives didn't
            # collapse to one touching piece. Discard each piece's own
            # placeholder self-hole(s) the same way, then attach each
            # real hole to whichever piece actually contains it,
            # dropping a hole that (numerically) matches none rather
            # than silently discarding a whole piece.
            outer_pieces = [Polygon._construct_verified(p.xs, p.ys, holes=None) for p in outer_pieces]
            for hole in hole_pieces:
                hx, hy = hole.point_inside()
                for i, piece in enumerate(outer_pieces):
                    if piece.is_inside(hx, hy):
                        outer_pieces[i] = Polygon._construct_verified(piece.xs, piece.ys, holes=list(piece.holes) + [hole])
                        break
            component_polys.append(outer_pieces)

    # Cross-component nesting: a geometrically-separate component (e.g.
    # an isolated via cluster) sitting inside another component's hole
    # becomes an island there instead of its own top-level result.
    flat = [(comp_i, poly) for comp_i, polys in enumerate(component_polys) for poly in polys]
    consumed = [False] * len(flat)
    for i, (_, poly) in enumerate(flat):
        px, py = poly.point_inside()
        best_area = None
        best_target = None
        for j, (_, other) in enumerate(flat):
            if i == j:
                continue
            for hole in other.holes:
                if hole.is_inside(px, py):
                    area = _ring_area(hole.axs, hole.ays)
                    if best_area is None or area < best_area:
                        best_area, best_target = area, (j, hole)
        if best_target is not None:
            j, hole = best_target
            _, target_poly = flat[j]
            new_hole = Polygon._construct_verified(hole.xs, hole.ys, holes=list(hole.holes) + [poly])
            new_holes = [new_hole if h is hole else h for h in target_poly.holes]
            flat[j] = (flat[j][0], Polygon._construct_verified(target_poly.xs, target_poly.ys, holes=new_holes))
            consumed[i] = True

    return [poly for i, (_, poly) in enumerate(flat) if not consumed[i]]


def _ring_area(xs: np.ndarray, ys: np.ndarray) -> float:
    x1 = np.roll(xs, -1)
    y1 = np.roll(ys, -1)
    return 0.5 * abs(float(np.sum(xs * y1 - x1 * ys)))


# --------------------------------------------------------------------------
# Debug visualization (_view_process=True). Purely illustrative -- shows
# the proximity graph (with pruned "skip" edges dimmed), the graph after
# collinear merging, the raw circle/rectangle primitives, a couple of
# snapshots partway through the incremental fuse, and the final result.
# --------------------------------------------------------------------------

def _plot_process(
    via_xy: np.ndarray,
    raw_edges: list[tuple[int, int]],
    pruned_edges: list[tuple[int, int]],
    merged_edges: list[tuple[int, int]],
    primitives: list[Polygon],
    snapshots: list[tuple[int, list[Polygon]]],
    final_pieces: list[Polygon],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("via_wall_polygons -- process overview", fontsize=14)

    # 1. proximity graph: pruned "skip" edges dimmed, kept edges bold
    ax = axes[0, 0]
    ax.set_title(f"1. Proximity graph\n({len(raw_edges)} raw -> {len(pruned_edges)} kept after RNG pruning)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    pruned_set = {(min(a, b), max(a, b)) for a, b in pruned_edges}
    for a, b in raw_edges:
        if (min(a, b), max(a, b)) not in pruned_set:
            ax.plot([via_xy[a, 0], via_xy[b, 0]], [via_xy[a, 1], via_xy[b, 1]],
                     color="lightgray", linewidth=0.8, zorder=1)
    for a, b in pruned_edges:
        ax.plot([via_xy[a, 0], via_xy[b, 0]], [via_xy[a, 1], via_xy[b, 1]],
                 color="tab:blue", linewidth=1.3, zorder=2)
    ax.scatter(via_xy[:, 0], via_xy[:, 1], color="black", s=10, zorder=3)

    # 2. after collinear merge: dissolved vias dimmed, surviving joints bold
    ax = axes[0, 1]
    ax.set_title(f"2. After collinear merge\n({len(pruned_edges)} -> {len(merged_edges)} edges)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    merged_via_ids = {v for e in merged_edges for v in e}
    for a, b in merged_edges:
        ax.plot([via_xy[a, 0], via_xy[b, 0]], [via_xy[a, 1], via_xy[b, 1]],
                 color="tab:blue", linewidth=1.5, zorder=2)
    dissolved = [i for i in range(via_xy.shape[0]) if i not in merged_via_ids]
    if dissolved:
        ax.scatter(via_xy[dissolved, 0], via_xy[dissolved, 1], color="lightgray", s=6,
                   zorder=2, label="dissolved (collinear)")
    kept_ids = list(merged_via_ids)
    if kept_ids:
        ax.scatter(via_xy[kept_ids, 0], via_xy[kept_ids, 1], color="black", s=16,
                   zorder=3, label="surviving joints")
    ax.legend(fontsize=8)

    # 3. raw primitives, before any fusion
    ax = axes[0, 2]
    ax.set_title(f"3. Primitives before fusion\n({len(primitives)} circles + rectangles)")
    plotter = GeometryPlotter(ax=ax, equal_aspect=True)
    for p in primitives:
        plotter.add_polygon(p, alpha=0.35, linewidth=0.5)

    # 4/5. fusion snapshots partway through (only populated when
    # incremental=True; a flat union has no intermediate steps to show)
    for panel in range(2):
        ax = axes[1, panel]
        if panel < len(snapshots):
            n_done, pieces = snapshots[panel]
            ax.set_title(f"{4 + panel}. Fusing... {n_done}/{len(merged_edges)} edges\n({len(pieces)} piece(s) so far)")
            plotter = GeometryPlotter(ax=ax, equal_aspect=True)
            for p in pieces:
                plotter.add_polygon(p, alpha=0.5)
        else:
            ax.set_title(f"{4 + panel}. (single flat union --\nno intermediate steps)")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # 6. final result
    ax = axes[1, 2]
    ax.set_title(f"6. Final result\n({len(final_pieces)} polygon(s))")
    plotter = GeometryPlotter(ax=ax, equal_aspect=True)
    for p in final_pieces:
        plotter.add_polygon(p, alpha=0.6)

    plt.tight_layout()
    plt.show()


def via_wall_polygons(
    via_xy,
    max_dist: float,
    thickness: float,
    circle_segments: int = 16,
    edge_width_factor: float = 1.0,
    angle_tol: float = 1e-3,
    include_isolated_vias: bool = False,
    merge_tol: float | None = None,
    area_tol: float | None = None,
    min_hole_area: float | None = None,
    incremental: bool = False,
    _view_process: bool = False,
) -> list[Polygon]:
    """Build thickened wall polygon(s) -- with holes where the via
    network loops back on itself -- from a cloud of via coordinates.

    Args:
        via_xy: (N, 2) array-like of via center coordinates.
        max_dist: vias closer than this are connected by a wall segment.
        thickness: wall thickness. Via circles are drawn at the full
            `thickness / 2` radius; edge rectangles are drawn at the
            same width by default (see `edge_width_factor`).
        circle_segments: segments approximating each via's circle --
            higher is smoother but adds edges (and therefore union
            cost); 8-12 is enough for a coarse mesh-prep wall.
        edge_width_factor: edge rectangle half-width as a fraction of
            the via circle's radius (default 1.0 -- full radius, same
            thickness as the circles). The exact-predicate arrangement
            engine (`kernel/_arrangement.py`) handles the resulting
            tangent/overlapping joints robustly regardless of this
            value, unlike the old kernel this default used to work
            around (hence the old default of 0.9 and the old `< 1.0`
            restriction -- both gone). One real, purely cosmetic
            side effect at 1.0: `make_circle_polygon`'s n-gon is
            *inscribed* in the true circle (its straight edges sit
            strictly inside the true radius except exactly at its own
            vertices), so a rectangle corner placed at exactly the
            true radius generally pokes a tiny "ear" past the circle's
            polygon approximation between two of its vertices --
            magnitude bounded by the polygon's own sagitta,
            `radius * (1 - cos(pi / circle_segments))` (about 2% of
            radius at the default 16 segments) -- negligible for most
            purposes; raise `circle_segments` if it isn't for yours.
            Lower this below 1.0 for the old behavior (rectangle
            strictly inside the circles, no ears, slightly thinner
            connecting walls).
        angle_tol: collinearity tolerance (radians) for the chain-merge
            step.
        include_isolated_vias: if True, a via with no neighbor within
            max_dist gets its own circle instead of being dropped
            entirely.
        merge_tol, area_tol: passed straight through to every
            `add_polygons` call this makes. Left as None (the default),
            these scale with `thickness` (radius * 1e-3 and
            (pi * radius^2) * 1e-4 respectively) instead of using the
            package's fixed absolute defaults (merge_tol=1e-7,
            area_tol=1e-10) -- those assume "ordinary-scale" geometry
            and are dangerously close to the actual feature size once
            `thickness` gets down into the micron range (e.g. PCB via
            walls), which shows up as "Could not find an interior
            point (degenerate or near-degenerate polygon)" from a
            sliver fragment that cleared area_tol but is still
            numerically degenerate. Pass explicit values if your units
            or geometry don't fit that heuristic well.
        min_hole_area: if set, drop any hole (a loop in the via
            network) whose own area is below this threshold instead of
            carving it out -- that region is left solid. Useful for a
            densely-packed or irregular network where some enclosed
            gaps are too small to matter for meshing/simulation
            purposes but still cost real geometry (a vertex loop, a
            distinct hole for the downstream CAD kernel to process)
            to keep. None (the default) keeps every hole regardless of
            size. Compared against each hole's own gross ring area, in
            the same units as `via_xy`/`thickness` (i.e. area units,
            not a length -- e.g. pass `max_dist**2` if you want
            "smaller than the glue distance, squared" as a rule of
            thumb, or any other absolute area cutoff that fits your
            geometry's scale).
        incremental: if True, fuse edge-by-edge via the union-find
            approach (`_fuse_network`) instead of the topology-first
            default. Kept for comparison/debugging; the default path
            (`incremental=False`) is both faster and at least as
            correct for any number of holes, since it derives every
            loop directly from the via graph's own topology (see
            `_assemble_by_topology`) rather than depending on
            geometric boolean ops to discover holes edge-by-edge.
            `min_hole_area` has no effect on this path.
        _view_process: if True, pops up a matplotlib figure
            illustrating the pipeline: the proximity graph (pruned
            "skip" edges dimmed), the graph after collinear merging,
            the raw circle/rectangle primitives, and the final result.
            With `incremental=True`, also shows two snapshots partway
            through the fuse. Purely for understanding/debugging -- has
            no effect on the returned polygons.

    Returns:
        List of top-level Polygon(s), each possibly carrying
        `holes=[...]` for any loop in the via network, ready to extrude.
    """
    via_xy = np.asarray(via_xy, dtype=np.float64)
    if via_xy.ndim != 2 or via_xy.shape[1] != 2:
        raise ValueError(f"via_xy must be shaped (N, 2), got {via_xy.shape}")
    n = via_xy.shape[0]
    if n == 0:
        return []
    if not (0.0 < edge_width_factor <= 1.0):
        raise ValueError(f"edge_width_factor must be in (0, 1], got {edge_width_factor}")

    t0 = time.time()
    pruned_edges = build_proximity_graph(via_xy, max_dist)
    merged_edges = merge_collinear_chains(via_xy, pruned_edges, angle_tol)
    logger.debug(
        f"via_wall_polygons: {n} via(s) -> {len(pruned_edges)} proximity edge(s) -> "
        f"{len(merged_edges)} edge(s) after collinear merge"
    )

    radius = thickness / 2.0
    edge_half_width = edge_width_factor * radius

    if merge_tol is None:
        merge_tol = radius * 1e-3
    if area_tol is None:
        area_tol = (math.pi * radius * radius) * 1e-4

    merged_connected: set[int] = set()
    for a, b in merged_edges:
        merged_connected.add(a)
        merged_connected.add(b)

    # every distinct via's circle, once, plus one rectangle per edge --
    # used by both the flat-union path below and the visualization.
    primitives: list[Polygon] = [
        p for a, b in merged_edges
        if (p := make_edge_rectangle(via_xy[a], via_xy[b], edge_half_width)) is not None
    ]
    for idx in merged_connected:
        primitives.append(
            make_circle_polygon(via_xy[idx], radius, circle_segments)
        )

    snapshots: list[tuple[int, list[Polygon]]] = []

    if incremental:
        snapshot_at: set[int] | None = None
        if _view_process:
            n_edges = len(merged_edges)
            snapshot_at = {max(1, round(n_edges / 3)), max(1, round(2 * n_edges / 3))}
        final_pieces = _fuse_network(
            via_xy, merged_edges, merged_connected, radius, edge_half_width, circle_segments,
            merge_tol, area_tol, snapshot_at=snapshot_at, snapshots=snapshots if _view_process else None,
        )
    elif len(primitives) == 0:
        final_pieces = []
    elif len(primitives) == 1:
        final_pieces = primitives
    else:
        # topology-first assembly: trace the via graph's own faces
        # (cheap -- ~N_via nodes, not the tessellated arrangement's ~20x
        # larger edge count) to find every loop directly, then run one
        # small, local `add_polygons` call per loop instead of one huge
        # call over the whole network. See `_assemble_by_topology`'s
        # module-level comment for why this is correct and fast.
        final_pieces = _assemble_by_topology(
            via_xy, merged_edges, radius, edge_half_width, circle_segments, merge_tol, area_tol, min_hole_area
        )

    # Deliberately NOT running isolated-via circles through one more
    # add_polygons call: they're geometrically disjoint from everything
    # else by definition (no neighbor within max_dist), and an
    # already-holed piece unioned with anything else -- even something
    # unrelated and non-touching -- risks re-triggering the same
    # hole-loss gap described above. Appending them directly avoids
    # ever re-entering that code path.
    if include_isolated_vias:
        # "isolated" means no neighbor in the *raw* proximity graph --
        # not merged_connected, which also excludes a straight run's
        # interior vias (correctly dissolved by the collinear merge,
        # but they still have neighbors; they're not isolated, just no
        # longer their own vertex in the output).
        raw_connected = {v for edge in pruned_edges for v in edge}
        for i in range(n):
            if i not in raw_connected:
                final_pieces.append(
                    make_circle_polygon(via_xy[i], radius, circle_segments)
                )

    if _view_process:
        xs_ = np.ascontiguousarray(via_xy[:, 0], dtype=np.float64)
        ys_ = np.ascontiguousarray(via_xy[:, 1], dtype=np.float64)
        raw_edge_arr = _proximity_edges(xs_, ys_, max_dist)
        raw_edges_unpruned = [(int(raw_edge_arr[0, k]), int(raw_edge_arr[1, k]))
                               for k in range(raw_edge_arr.shape[1])]

        _plot_process(via_xy, raw_edges_unpruned, pruned_edges, merged_edges,
                       primitives, snapshots, final_pieces)

    t1 = time.time()
    n_holes = sum(len(p.holes) for p in final_pieces)
    logger.debug(
        f"{format_runtime(t0, t1)} via_wall_polygons: {len(final_pieces)} output polygon(s), {n_holes} hole(s)"
    )
    return final_pieces


# --------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------

def _make_test_vias() -> np.ndarray:
    """A synthetic via layout exercising all three interesting cases at
    once: a closed ring (-> a hole in the union), a long straight run
    (-> the collinear-merge step collapsing many hops into one segment),
    and one isolated via (-> include_isolated_vias).
    """
    loop_pts = []
    radius = 5.0
    n_loop = 24
    for k in range(n_loop):
        a = 2.0 * math.pi * k / n_loop
        loop_pts.append((radius * math.cos(a), radius * math.sin(a)))

    straight_run = [(6.0 + i * 0.4, 0.0) for i in range(10)]
    isolated = [(-10.0, -10.0)]

    return np.array(loop_pts + straight_run + isolated)


def _test():
    via_xy = _make_test_vias()

    raw_edges = build_proximity_graph(via_xy, max_dist=1.5)
    merged_edges = merge_collinear_chains(via_xy, raw_edges)
    print(f"{len(via_xy)} vias -> {len(raw_edges)} proximity edges -> "
          f"{len(merged_edges)} edges after collinear merging")

    polys = via_wall_polygons(
        via_xy,
        max_dist=1.5,
        thickness=0.3,
        circle_segments=16,
        include_isolated_vias=True,
    )

    print(f"-> {len(polys)} top-level wall polygon(s)")
    for i, p in enumerate(polys):
        print(f"   polygon {i}: {len(p.xs)} verts, {len(p.holes)} hole(s)")

    plotter = GeometryPlotter(title="Via wall thickening test")
    plotter.add_scatter(via_xy[:, 0], via_xy[:, 1], zorder=5)
    for p in polys:
        plotter.add_polygon(p, alpha=0.5)
    plotter.show()


if __name__ == "__main__":
    _test()