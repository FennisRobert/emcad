"""The product model matrix: layer stack order, layer types, and drill spans.

Reference: ODB++Design Format Specification, "matrix/matrix (Matrix)" (p.62).
This file has no coordinates, so nothing here needs unit conversion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger

from .features import _open_text


@dataclass
class MatrixStep:
    """One ``STEP {...}`` block: a column of the matrix."""
    col: int
    name: str
    id: Optional[int] = None


@dataclass
class MatrixLayer:
    """One ``LAYER {...}`` block: a row of the matrix.

    ``row`` gives top-to-bottom physical stacking order (not necessarily
    sequential integers — sort by it to get the buildup order).
    """
    row: int
    context: str            # BOARD | MISC
    type: str                # SIGNAL | POWER_GROUND | DIELECTRIC | MIXED | SOLDER_MASK | ...
    name: str
    polarity: str = "POSITIVE"
    id: Optional[int] = None
    old_name: str = ""
    dielectric_type: Optional[str] = None
    dielectric_name: Optional[str] = None
    form: str = "RIGID"
    cu_top: Optional[int] = None
    cu_bottom: Optional[int] = None
    ref: Optional[int] = None
    start_name: str = ""
    end_name: str = ""
    add_type: Optional[str] = None
    color: Optional[str] = None


@dataclass
class Matrix:
    """A fully-parsed ``matrix/matrix`` file."""
    steps: List[MatrixStep] = field(default_factory=list)
    layers: List[MatrixLayer] = field(default_factory=list)

    def layers_top_to_bottom(self) -> List[MatrixLayer]:
        return sorted(self.layers, key=lambda l: l.row)

    def layer_by_name(self, name: str) -> Optional[MatrixLayer]:
        for l in self.layers:
            if l.name == name:
                return l
        return None

    def structural_layers(self) -> List[MatrixLayer]:
        """Layers that form the physical copper/dielectric stack (what you'd
        extrude in Z for a mechanical/FEM model)."""
        structural_types = {"SIGNAL", "POWER_GROUND", "DIELECTRIC", "MIXED"}
        return [l for l in self.layers_top_to_bottom() if l.type in structural_types]


_INT_FIELDS = {"COL", "ROW"}
_ID_FIELDS = {"ID", "CU_TOP", "CU_BOTTOM", "REF"}


def _parse_blocks(text: str, keyword: str):
    for block in re.finditer(rf"{keyword}\s*\{{(.*?)\}}", text, re.DOTALL):
        fields: Dict[str, str] = {}
        for line in block.group(1).splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
        yield fields


def parse_matrix_file(path: Union[str, Path]) -> Matrix:
    text = _open_text(path).read()

    steps: List[MatrixStep] = []
    for fields in _parse_blocks(text, "STEP"):
        steps.append(MatrixStep(
            col=int(fields.get("COL", "0")),
            name=fields.get("NAME", ""),
            id=int(fields["ID"]) if fields.get("ID") else None,
        ))

    layers: List[MatrixLayer] = []
    for fields in _parse_blocks(text, "LAYER"):
        def opt_int(key: str) -> Optional[int]:
            v = fields.get(key)
            return int(v) if v else None

        layers.append(MatrixLayer(
            row=int(fields.get("ROW", "0")),
            context=fields.get("CONTEXT", "BOARD"),
            type=fields.get("TYPE", ""),
            name=fields.get("NAME", ""),
            polarity=fields.get("POLARITY", "POSITIVE"),
            id=opt_int("ID"),
            old_name=fields.get("OLD_NAME", ""),
            dielectric_type=fields.get("DIELECTRIC_TYPE") or None,
            dielectric_name=fields.get("DIELECTRIC_NAME") or None,
            form=fields.get("FORM", "RIGID"),
            cu_top=opt_int("CU_TOP"),
            cu_bottom=opt_int("CU_BOTTOM"),
            ref=opt_int("REF"),
            start_name=fields.get("START_NAME", ""),
            end_name=fields.get("END_NAME", ""),
            add_type=fields.get("ADD_TYPE") or None,
            color=fields.get("COLOR") or None,
        ))

    logger.debug(f"parse_matrix_file: {len(steps)} step(s), {len(layers)} layer(s)")
    return Matrix(steps=steps, layers=layers)
