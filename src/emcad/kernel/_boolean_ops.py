"""
add_polygons / intersect_polygons / subtract_polygons implementation.
`api.py` just wraps these with a thin, typed pass-through.

Built directly on `_arrangement.py`'s exact-predicate, winding-number
engine -- no more `poly_fragment` + point-sampled re-classification +
`join_polygons` three-stage pipeline. Each op is just:
    1. flatten every input polygon (and its holes, at every depth)
       into operand-tagged rings,
    2. build the arrangement once,
    3. label faces by a boolean function of membership,
    4. assemble.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from ._arrangement import build_arrangement, label_and_assemble
from ._cluster import cluster_operands
from ._constants import DEFAULT_MERGE_TOL, DEFAULT_T_TOL, DEFAULT_AREA_TOL

if TYPE_CHECKING:
    from ..poly import Polygon


def _flatten_operands(polys: list["Polygon"]) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
    """Flatten every polygon -- and, recursively, every hole it
    carries -- into a flat list of rings, each tagged with which
    top-level operand (its position in `polys`) it belongs to. A
    polygon's own pre-existing holes are part of the SAME operand as
    its outer boundary, exactly like every other ring of that input.
    """
    rings: list[tuple[np.ndarray, np.ndarray]] = []
    ring_operand_id: list[int] = []

    def walk(poly: "Polygon", operand: int) -> None:
        rings.append((poly.axs, poly.ays))
        ring_operand_id.append(operand)
        for hole in poly.holes:
            walk(hole, operand)

    for i, poly in enumerate(polys):
        walk(poly, i)

    return rings, ring_operand_id


def _run(polys: list["Polygon"], keep_fn, merge_tol: float, area_tol: float) -> list["Polygon"]:
    from ..poly import Polygon  # deferred -- see api.py's module docstring

    if len(polys) == 0:
        return []

    rings, ring_operand_id = _flatten_operands(polys)
    n_operands = len(polys)
    clusters = cluster_operands(rings, ring_operand_id, n_operands, merge_tol)

    if len(clusters) <= 1:
        arrangement = build_arrangement(rings, ring_operand_id, n_operands, merge_tol)
        result = label_and_assemble(arrangement, keep_fn, Polygon, area_tol)
        logger.debug(f"boolean op: {n_operands} input polygon(s) -> {len(result)} output polygon(s)")
        return result

    # Operands in different bbox clusters are provably inert to each
    # other (see _cluster.py) -- run each cluster through its own,
    # much smaller arrangement and just concatenate. keep_fn is
    # written in terms of GLOBAL operand ids, so wrap it per cluster
    # to translate local (0..cluster_size-1) membership back to the
    # original global ids before calling it.
    logger.debug(f"boolean op: {n_operands} input polygon(s) -> {len(clusters)} bbox cluster(s)")
    rings_by_operand: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for ring, op in zip(rings, ring_operand_id):
        rings_by_operand.setdefault(op, []).append(ring)

    result: list["Polygon"] = []
    for global_ids in clusters:
        local_rings: list[tuple[np.ndarray, np.ndarray]] = []
        local_ring_operand_id: list[int] = []
        for local_idx, g in enumerate(global_ids):
            for ring in rings_by_operand.get(g, []):
                local_rings.append(ring)
                local_ring_operand_id.append(local_idx)
        local_keep_fn = (lambda m, _gids=global_ids: keep_fn(frozenset(_gids[i] for i in m)))
        arrangement = build_arrangement(local_rings, local_ring_operand_id, len(global_ids), merge_tol)
        result.extend(label_and_assemble(arrangement, local_keep_fn, Polygon, area_tol))

    logger.debug(f"boolean op: {n_operands} input polygon(s) -> {len(result)} output polygon(s) ({len(clusters)} cluster(s))")
    return result


def add_polygons(
    *polys: "Polygon",
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Boolean union (fuse) of N polygons: keep anything inside at least
    one of them.

    `t_tol` is accepted for API compatibility but unused: the
    arrangement engine makes every touch/crossing/overlap decision with
    exact grid-integer predicates (see `_exact.py`), so there is no
    parametric "close enough to an endpoint" tolerance left to tune.
    """
    return _run(list(polys), lambda m: len(m) > 0, merge_tol, area_tol)


def intersect_polygons(
    *polys: "Polygon",
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Boolean intersection of N polygons: keep only what's inside
    every one of them.
    """
    n = len(polys)
    return _run(list(polys), lambda m: len(m) == n, merge_tol, area_tol)


def subtract_polygons(
    add: tuple["Polygon", ...],
    subtract: tuple["Polygon", ...],
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Boolean subtraction: union(add) minus union(subtract).

    Not pairwise -- every polygon in `subtract` is removed from the
    *combined* area of every polygon in `add` at once. A face survives
    iff it's inside at least one `add` polygon and inside none of the
    `subtract` polygons.
    """
    add = list(add)
    subtract = list(subtract)
    n_add = len(add)

    def keep_fn(membership):
        in_add = any(k < n_add for k in membership)
        in_sub = any(k >= n_add for k in membership)
        return in_add and not in_sub

    return _run(add + subtract, keep_fn, merge_tol, area_tol)
