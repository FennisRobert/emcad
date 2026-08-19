"""
Numba-jitted hot loops for polygon fragmentation.

Everything in this file is pure array-index arithmetic on primitive
numpy arrays (int64 / float64) so every function is eagerly compiled
with an explicit signature -- no Polygon/Vertex/Edge objects, Python
lists, or anything else numba can't natively type. That's the dividing
line between this file and `fragment_tools.py`: if it needs numba, it
lives here; if it needs the Polygon class, it doesn't.

Using explicit signatures (rather than bare `@njit(cache=True)`) means
compilation happens at import time, not on first call, and a mistyped
argument fails loudly with a TypingError instead of silently
recompiling a second specialization.
"""

from __future__ import annotations

import numpy as np
from numba import njit


# --------------------------------------------------------------------------
# Crossing classification
# --------------------------------------------------------------------------

# signature: (int64[:,:], float64[:,:], float64[:,:], float64[:,:], float64)
#            -> (int64[:], float64[:], float64[:,:])
#
#   idspair    : int64[2, K]     edge-id pairs (i, j) for each raw crossing event
#   coords     : float64[2, K]   (x, y) of each raw crossing event
#   pts_start  : float64[2, E]   edge start coordinates, indexed by edge id
#   pts_end    : float64[2, E]   edge end coordinates, indexed by edge id
#   t_tol      : float64         parametric (0..1) tolerance for "at the endpoint"
#
#   returns (out_edge, out_t, out_xy):
#     out_edge : int64[:]        edge id each surviving split point belongs to
#     out_t    : float64[:]      parametric position (0..1) along that edge
#     out_xy   : float64[2, :]   coordinate of the split point
@njit(
    "Tuple((int64[:], float64[:], float64[:,:]))"
    "(int64[:,:], float64[:,:], float64[:,:], float64[:,:], float64)",
    cache=True,
)
def classify_crossings(idspair, coords, pts_start, pts_end, t_tol):
    """Turn raw (edge_i, edge_j, xy) crossing events into per-edge split
    points, dropping crossings that just coincide with an already
    existing shared vertex on *both* sides (those aren't new topology,
    they're the vertex that's already there).
    """
    K = idspair.shape[1]
    out_edge = np.empty(2 * K, dtype=np.int64)
    out_t = np.empty(2 * K, dtype=np.float64)
    out_xy = np.empty((2, 2 * K), dtype=np.float64)
    ctr = 0

    for k in range(K):
        i = idspair[0, k]
        j = idspair[1, k]
        cx = coords[0, k]
        cy = coords[1, k]

        dxi = pts_end[0, i] - pts_start[0, i]
        dyi = pts_end[1, i] - pts_start[1, i]
        len2_i = dxi * dxi + dyi * dyi
        if len2_i <= 0.0:
            continue
        ti = ((cx - pts_start[0, i]) * dxi + (cy - pts_start[1, i]) * dyi) / len2_i

        dxj = pts_end[0, j] - pts_start[0, j]
        dyj = pts_end[1, j] - pts_start[1, j]
        len2_j = dxj * dxj + dyj * dyj
        if len2_j <= 0.0:
            continue
        tj = ((cx - pts_start[0, j]) * dxj + (cy - pts_start[1, j]) * dyj) / len2_j

        i_at_end = (ti <= t_tol) or (ti >= 1.0 - t_tol)
        j_at_end = (tj <= t_tol) or (tj >= 1.0 - t_tol)

        if i_at_end and j_at_end:
            # This "crossing" is just two edges touching at a vertex
            # they already share. No new split point needed.
            continue

        if not i_at_end:
            out_edge[ctr] = i
            out_t[ctr] = ti
            out_xy[0, ctr] = cx
            out_xy[1, ctr] = cy
            ctr += 1

        if not j_at_end:
            out_edge[ctr] = j
            out_t[ctr] = tj
            out_xy[0, ctr] = cx
            out_xy[1, ctr] = cy
            ctr += 1

    return out_edge[:ctr], out_t[:ctr], out_xy[:, :ctr]


# --------------------------------------------------------------------------
# Face tracing
# --------------------------------------------------------------------------
#
# At half-edge (u -> v), the next half-edge of the same face is the
# clockwise predecessor of the twin (v -> u) in v's angle-sorted
# outgoing list. This is the standard planar-subdivision face-tracing
# rule: it visits every half-edge exactly once, produces every bounded
# face CCW (positive shoelace area), and produces exactly one CW
# (negative area) face per connected component -- its outer boundary.
# Vertices of degree 1 (dangling edges) self-resolve: the walk goes out
# and immediately back, contributing zero net area.

# signature: (int64[:], int64[:], int64[:], int64[:], int64[:], int64[:]) -> int64[:]
#
#   he_start, he_end : int64[H]  half-edge start / end vertex ids
#   he_twin          : int64[H]  he_twin[h] = the reverse half-edge of h
#   sorted_he        : int64[H]  half-edge ids, grouped by start vertex and
#                                 sorted by outgoing angle within each group
#                                 (a CSR-style rotation system; see
#                                 `fragment_tools.build_rotation_system`)
#   v_offset         : int64[V+1] sorted_he[v_offset[v]:v_offset[v+1]] is
#                                  vertex v's angle-sorted outgoing block
#   pos_in_block     : int64[H]  half-edge h's index within its own
#                                 start-vertex's sorted block
#
#   returns face_starts : int64[:]  one representative half-edge id per
#                                    traced face
@njit(
    "int64[:](int64[:], int64[:], int64[:], int64[:], int64[:], int64[:])",
    cache=True,
)
def trace_faces(he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block):
    H = he_start.shape[0]
    visited = np.zeros(H, dtype=np.bool_)
    face_starts = np.empty(H, dtype=np.int64)
    n_faces = 0

    for h0 in range(H):
        if visited[h0]:
            continue
        face_starts[n_faces] = h0
        h = h0
        while True:
            visited[h] = True
            v = he_end[h]
            t = he_twin[h]
            deg = v_offset[v + 1] - v_offset[v]
            p = pos_in_block[t]
            next_pos = p - 1
            if next_pos < 0:
                next_pos += deg
            h_next = sorted_he[v_offset[v] + next_pos]
            if h_next == h0:
                break
            h = h_next
        n_faces += 1

    return face_starts[:n_faces]


# --------------------------------------------------------------------------
# Signed face areas
# --------------------------------------------------------------------------

# signature: (int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], float64[:,:]) -> float64[:]
#
#   face_starts : int64[F]       one representative half-edge id per face
#                                 (as returned by `trace_faces`)
#   he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block : see `trace_faces`
#   xy          : float64[2, V]  vertex coordinates
#
#   returns areas : float64[F]   signed shoelace area of each face --
#                                 positive for bounded (CCW) faces,
#                                 negative for outer-boundary (CW) faces
@njit(
    "float64[:](int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], float64[:,:])",
    cache=True,
)
def face_areas(face_starts, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block, xy):
    n_faces = face_starts.shape[0]
    areas = np.zeros(n_faces, dtype=np.float64)

    for fi in range(n_faces):
        h0 = face_starts[fi]
        h = h0
        area2 = 0.0
        while True:
            u = he_start[h]
            v = he_end[h]
            area2 += xy[0, u] * xy[1, v] - xy[0, v] * xy[1, u]

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
        areas[fi] = 0.5 * area2

    return areas