"""
join_polygons: fuse polygons that meet edge-to-edge into fewer, larger
polygons -- and, more generally now, correctly assemble whatever
holes/islands the touching implies.

Built directly on `_arrangement.py`'s exact-predicate, winding-number
engine: each input polygon is its own operand (its own pre-existing
holes, at any depth, are part of that SAME operand -- exactly like
every other ring belonging to it). The result is the union, with ONE
validation on top of the ordinary boolean-op machinery: every face's
membership must never include more than one operand. That single,
exact, general check replaces what used to be two separate,
independently-epsilon'd detectors (a transversal-crossing test and a
same-side-collinear-overlap test) -- ANY kind of genuine interior
overlap between two different inputs, however it's geometrically
produced, shows up as membership size >= 2 somewhere, because two
operands both claiming the same region *is* what "interior overlap"
means. It also means join's classic "solid fully nested inside
another, untouched, different inputs" error case falls out for free:
the nested face's membership includes both the inner and outer
operand, size 2, same check.

An island floating untouched inside a hole -- at any nesting depth,
regardless of which operand the hole came from -- is never flagged:
that region's membership is exactly {the island's own operand}, size
1, same as any other ordinary solid material.
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


def join_polygons(
    polys: list["Polygon"],
    merge_tol: float = DEFAULT_MERGE_TOL,
    t_tol: float = DEFAULT_T_TOL,
    area_tol: float = DEFAULT_AREA_TOL,
) -> list["Polygon"]:
    """Fuse polygons that meet edge-to-edge into fewer, larger polygons
    by dissolving every boundary shared between exactly two of them.

    Raises `GeometryException` if any two inputs' interiors genuinely
    overlap -- whether via a transversal crossing, a same-side
    collinear overlap, or one input sitting fully inside another with
    no shared boundary of its own anywhere; all of these mean two
    different inputs both claim the same material, which `join_polygons`
    refuses to guess how to resolve (use `add_polygons` instead if the
    inputs' interiors genuinely overlap).

    Args:
        polys: polygons to fuse. Order doesn't matter.
        merge_tol: grid spacing for the exact arrangement (see
            `_exact.py`) -- absolute distance below which two points
            are the same vertex.
        t_tol: accepted for API compatibility, unused -- every
            touch/crossing/overlap decision is made with exact
            grid-integer predicates, so there's no parametric
            "close enough" tolerance left to tune.
        area_tol: faces with |signed area| below this are dropped as
            numerical noise.

    Returns:
        List of top-level fused Polygon(s), with holes attached where
        the join produced them.
    """
    from ..poly import Polygon, GeometryException  # deferred -- see api.py's module docstring

    if len(polys) == 0:
        return []
    if len(polys) == 1:
        return list(polys)

    rings: list[tuple[np.ndarray, np.ndarray]] = []
    ring_operand_id: list[int] = []

    def walk(poly: "Polygon", operand: int) -> None:
        rings.append((poly.axs, poly.ays))
        ring_operand_id.append(operand)
        for hole in poly.holes:
            walk(hole, operand)

    for i, poly in enumerate(polys):
        walk(poly, i)

    n_operands = len(polys)
    clusters = cluster_operands(rings, ring_operand_id, n_operands, merge_tol)

    def _check_and_assemble(rings_, ring_operand_id_, n_) -> list["Polygon"]:
        arrangement = build_arrangement(rings_, ring_operand_id_, n_, merge_tol)
        if any(len(m) >= 2 for m in arrangement.membership):
            raise GeometryException(
                "join_polygons: found inputs whose interiors genuinely overlap -- whether via a "
                "transversal crossing, a same-side collinear overlap, or one input sitting fully "
                "inside another with no shared boundary of its own anywhere. join only fuses "
                "polygons that meet as complementary (touching, non-overlapping) neighbors -- use "
                "a boolean operation (add_polygons / intersect_polygons / subtract_polygons) "
                "instead if the inputs' interiors actually overlap."
            )
        return label_and_assemble(arrangement, lambda m: len(m) > 0, Polygon, area_tol)

    if len(clusters) <= 1:
        result = _check_and_assemble(rings, ring_operand_id, n_operands)
        logger.debug(f"join_polygons: {n_operands} input polygon(s) -> {len(result)} output polygon(s)")
        return result

    # Operands in different bbox clusters are provably inert to each
    # other (see _cluster.py) -- neither a genuine overlap NOR a
    # legitimate touching-edge fuse can happen across clusters, so
    # each one gets its own overlap check + assembly, independently.
    logger.debug(f"join_polygons: {n_operands} input polygon(s) -> {len(clusters)} bbox cluster(s)")
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
        result.extend(_check_and_assemble(local_rings, local_ring_operand_id, len(global_ids)))

    logger.debug(f"join_polygons: {n_operands} input polygon(s) -> {len(result)} output polygon(s) ({len(clusters)} cluster(s))")
    return result
