"""
`poly_fragment` implementation. `api.py` just wraps `poly_fragment`
below with a thin, typed pass-through -- this is where the actual
planar-arrangement fragmentation logic lives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from . import fragment_tools as ftools
from . import fragment_kernels as fkernels
from . import _primitives
from ._constants import DEFAULT_MERGE_TOL, DEFAULT_T_TOL, DEFAULT_AREA_TOL

if TYPE_CHECKING:
    from ..poly import Polygon


def poly_fragment(
    polys: list["Polygon"],
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
    keep: str = "positive",
    filter_to_originals: bool = True,
    debug: bool = False,
) -> list["Polygon"]:
    """Fragment N polygons into mutually disjoint sub-polygons of their
    combined planar arrangement.

    Args:
        polys: input polygons. May overlap each other and/or be
            individually self-intersecting.
        merge_tol: absolute distance below which two points are
            considered the same vertex.
        t_tol: parametric (0..1, per-edge) tolerance for treating a
            crossing as landing exactly on an existing endpoint.
        area_tol: faces with |signed area| below this are dropped as
            numerical noise.
        keep: which faces to return -- "positive" (default, the actual
            interior sub-polygons), "negative" (outer/component
            boundaries, mostly useful for debugging), or "all".
        filter_to_originals: if True (default), drop any traced fragment
            that doesn't fall inside at least one of the input polygons.
        debug: if True, plot the arrangement and returned faces.

    Returns:
        List of disjoint Polygon fragments.
    """
    if len(polys) == 0:
        return []

    pts_start, pts_end, edge_poly_id, edge_local_id, poly_vert_offset = ftools.collect_edges(polys)
    n_edges = pts_start.shape[1]
    n_orig_verts = n_edges  # one vertex per edge start, by construction

    # edge e's start vertex is global vertex id e (see collect_edges docstring)
    edge_start_vid = np.arange(n_edges, dtype=np.int64)
    n_verts_per_poly = np.diff(poly_vert_offset)
    edge_end_local = (edge_local_id + 1) % n_verts_per_poly[edge_poly_id]
    edge_end_vid = poly_vert_offset[edge_poly_id] + edge_end_local

    idspair, coords = _primitives.edge_self_intersections(pts_start, pts_end)
    logger.debug(f"poly_fragment: {n_edges} input edges, {idspair.shape[1]} raw crossing events")

    if idspair.shape[1] == 0:
        split_edge_id = np.empty(0, dtype=np.int64)
        split_t = np.empty(0, dtype=np.float64)
        split_xy = np.empty((2, 0), dtype=np.float64)
    else:
        split_edge_id, split_t, split_xy = fkernels.classify_crossings(
            idspair, coords, pts_start, pts_end, t_tol
        )

    logger.debug(f"poly_fragment: {split_xy.shape[1]} genuine new split points")

    # merge original vertices + new split points into one deduplicated set
    n_new = split_xy.shape[1]
    all_xy = np.empty((2, n_orig_verts + n_new), dtype=np.float64)
    all_xy[:, :n_orig_verts] = pts_start
    all_xy[:, n_orig_verts:] = split_xy

    raw_to_merged, merged_xy = ftools.merge_points(all_xy, merge_tol)
    n_verts = merged_xy.shape[1]

    edge_start_vid_m = raw_to_merged[edge_start_vid]
    edge_end_vid_m = raw_to_merged[edge_end_vid]
    split_vid_m = raw_to_merged[n_orig_verts:] if n_new > 0 else np.empty(0, dtype=np.int64)

    seg_u, seg_v = ftools.build_segments(
        n_edges, edge_start_vid_m, edge_end_vid_m, split_edge_id, split_t, split_vid_m
    )
    logger.debug(f"poly_fragment: {seg_u.shape[0]} atomic segments after splitting/merging")

    if seg_u.shape[0] == 0:
        return []

    M = seg_u.shape[0]
    he_start = np.empty(2 * M, dtype=np.int64)
    he_end = np.empty(2 * M, dtype=np.int64)
    he_twin = np.empty(2 * M, dtype=np.int64)
    he_start[0::2] = seg_u
    he_end[0::2] = seg_v
    he_start[1::2] = seg_v
    he_end[1::2] = seg_u
    he_twin[0::2] = np.arange(1, 2 * M, 2, dtype=np.int64)
    he_twin[1::2] = np.arange(0, 2 * M, 2, dtype=np.int64)

    sorted_he, v_offset, pos_in_block = ftools.build_rotation_system(he_start, he_end, merged_xy, n_verts)

    face_starts = fkernels.trace_faces(he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block)
    areas = fkernels.face_areas(
        face_starts, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block, merged_xy
    )
    logger.debug(f"poly_fragment: {face_starts.shape[0]} faces traced")

    if keep == "positive":
        # containment-forest assembly, not a plain sign filter: an
        # isolated loop's sign is meaningless (see
        # ftools.assemble_nested_polygons's docstring), which matters
        # whenever any input polygon already carries holes= -- e.g.
        # the result of a previous boolean op fed back in as an
        # operand of this one.
        poly_out = ftools.assemble_nested_polygons(
            face_starts, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block, merged_xy,
            n_verts, seg_u, seg_v, areas, area_tol,
        )
    elif keep == "negative":
        mask = areas < -area_tol
        poly_out = [
            ftools.reconstruct_polygon(h0, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block, merged_xy)
            for h0, keep_flag in zip(face_starts, mask) if keep_flag
        ]
    elif keep == "all":
        mask = np.abs(areas) > area_tol
        poly_out = [
            ftools.reconstruct_polygon(h0, he_start, he_end, he_twin, sorted_he, v_offset, pos_in_block, merged_xy)
            for h0, keep_flag in zip(face_starts, mask) if keep_flag
        ]
    else:
        raise ValueError(f"Unknown keep={keep!r}, expected 'positive', 'negative', or 'all'")

    logger.debug(f"poly_fragment: {len(poly_out)} faces kept (keep={keep!r})")

    if filter_to_originals:
        poly_out = ftools.filter_fragments_to_originals(poly_out, polys)
        logger.debug(f"poly_fragment: {len(poly_out)} fragments kept after original-polygon filter")

    if debug:
        from ..plot import GeometryPlotter
        from ..poly import Vertex, Edge

        pl = GeometryPlotter()
        for poly in poly_out:
            pl.add_polygon(poly, alpha=0.2)
        pl.add_scatter(merged_xy[0, :], merged_xy[1, :])
        for h in range(0, 2 * M, 2):
            v1 = Vertex(merged_xy[0, he_start[h]], merged_xy[1, he_start[h]])
            v2 = Vertex(merged_xy[0, he_end[h]], merged_xy[1, he_end[h]])
            pl.add_edge(Edge(v1, v2))
        pl.show()

    return poly_out