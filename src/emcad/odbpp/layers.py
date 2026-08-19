"""Drill tools and the per-step ``<layer_name>`` directory contents.

Reference: ODB++Design Format Specification, "<layer_name>/tools (Drill
Tools)" (p.184) and "Layer Entities" (Chapter 5, p.147).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

from .features import FeaturesFile, parse_features_file, _open_text
from .units import coord_to_meters, symbol_size_to_meters, default_symbol_size_units


@dataclass
class DrillTool:
    """One row of a ``tools`` file (ODB++ 'TOOLS' array, p.185)."""
    number: int
    type: str                      # PLATED | NON_PLATED | VIA
    type2: str = "STANDARD"        # STANDARD | PRESS_FIT | PHOTO | LASER
    min_tol: float = 0.0           # meters
    max_tol: float = 0.0           # meters
    bit: str = ""
    finish_size: Optional[float] = None   # meters, None if unset (-1 in file)
    drill_size: Optional[float] = None    # meters


@dataclass
class ToolsTable:
    """A parsed ``<layer_name>/tools`` file."""
    units: str
    thickness: Optional[float] = None   # meters (obsolete field, kept for completeness)
    user_params: str = ""
    tools: Dict[int, DrillTool] = field(default_factory=dict)


def parse_tools_file(path: Union[str, Path]) -> ToolsTable:
    text = _open_text(path).read()

    units_m = re.search(r"UNITS=(\S+)", text)
    units = units_m.group(1).strip() if units_m else "INCH"
    # Tool sizes in this file are thousandths (mil/micron), same convention as symbol names.
    size_units = default_symbol_size_units(units)

    def size(raw: str) -> Optional[float]:
        v = float(raw)
        if v < 0:
            return None
        return symbol_size_to_meters(v, size_units)

    thickness_m = re.search(r"THICKNESS=(\S*)", text)
    thickness = size(thickness_m.group(1)) if thickness_m and thickness_m.group(1) else None
    params_m = re.search(r"USER_PARAMS=(\S*)", text)
    user_params = params_m.group(1) if params_m else ""

    tools: Dict[int, DrillTool] = {}
    for block in re.finditer(r"TOOLS\s*\{(.*?)\}", text, re.DOTALL):
        fields: Dict[str, str] = {}
        for line in block.group(1).splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
        num = int(fields.get("NUM", "0"))
        tools[num] = DrillTool(
            number=num,
            type=fields.get("TYPE", "PLATED"),
            type2=fields.get("TYPE2", "STANDARD"),
            min_tol=symbol_size_to_meters(float(fields.get("MIN_TOL", "0") or 0), size_units),
            max_tol=symbol_size_to_meters(float(fields.get("MAX_TOL", "0") or 0), size_units),
            bit=fields.get("BIT", ""),
            finish_size=size(fields.get("FINISH_SIZE", "-1") or "-1"),
            drill_size=size(fields.get("DRILL_SIZE", "-1") or "-1"),
        )

    logger.debug(f"parse_tools_file: '{Path(path).name}' -- {len(tools)} tool(s)")
    return ToolsTable(units=units, thickness=thickness, user_params=user_params, tools=tools)


@dataclass
class Layer:
    """One ``steps/<step_name>/layers/<layer_name>`` directory's relevant contents.

    ``matrix_info`` (a :class:`odbpp.matrix.MatrixLayer`) carries the
    layer's role/order/type from the product model's matrix and is attached
    by :mod:`odbpp.model` when the whole product model is assembled.
    """
    name: str
    features: Optional[FeaturesFile] = None
    profile: Optional[FeaturesFile] = None
    tools: Optional[ToolsTable] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    #: the UNITS= directive from this layer's own attrlist file (INCH/MM),
    #: needed to correctly convert any distance-valued attribute (e.g.
    #: .layer_dielectric) to meters -- it can differ from the layer's
    #: features/profile file units.
    attr_units: Optional[str] = None
    matrix_info: Optional["object"] = None   # odbpp.matrix.MatrixLayer, set by ProductModel

    def _float_attr(self, key: str) -> Optional[float]:
        raw = self.attributes.get(key)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @property
    def dielectric_constant(self) -> Optional[float]:
        """``.dielectric_constant`` (Dk/Er) system attribute, if present. Unitless."""
        return self._float_attr(".dielectric_constant")

    @property
    def loss_tangent(self) -> Optional[float]:
        """``.loss_tangent`` (Df) system attribute, if present. Unitless."""
        return self._float_attr(".loss_tangent")

    @property
    def dielectric_thickness(self) -> Optional[float]:
        """``.layer_dielectric`` system attribute converted to **meters**, if
        present. Per spec this attribute is "the value is in specified
        Units of Measurement" -- i.e. this layer's own attrlist UNITS
        directive (``self.attr_units``), which may differ from its
        features/profile file units. Valid for DIELECTRIC, SOLDER_MASK
        (mask thickness) and SOLDER_PASTE (stencil thickness) layer types."""
        val = self._float_attr(".layer_dielectric")
        if val is None or self.attr_units is None:
            return None
        return coord_to_meters(val, self.attr_units)

    #: 1 oz/ft^2 of copper, spread evenly, is this many meters thick --
    #: the standard PCB-industry conversion (also commonly quoted as ~1.37
    #: mil/oz). This is a physical/industry constant, not something ODB++
    #: itself defines.
    OZ_PER_SQFT_TO_METERS = 34.79e-6

    @property
    def copper_thickness(self) -> Optional[float]:
        """``.copper_weight`` system attribute converted to **meters**, if
        present. The attribute's value is in oz/ft^2 (standard PCB copper
        weight, independent of the file's coordinate UNITS) -- see
        :attr:`OZ_PER_SQFT_TO_METERS`."""
        val = self._float_attr(".copper_weight")
        if val is None:
            return None
        return val * self.OZ_PER_SQFT_TO_METERS


def parse_layer_attrlist(path: Union[str, Path]) -> Tuple[Dict[str, str], Optional[str]]:
    """Parse a simple ``attrlist`` file (``key=value`` per attribute).

    Returns ``(attributes, units)`` where ``units`` is this file's own
    ``UNITS=`` directive (INCH/MM), needed to correctly interpret any
    distance-valued attribute -- see :attr:`Layer.dielectric_thickness`.
    """
    attrs: Dict[str, str] = {}
    units: Optional[str] = None
    try:
        for line in _open_text(path).readlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("UNITS="):
                units = line.split("=", 1)[1].strip()
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                attrs[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return attrs, units


def parse_layer(layer_dir: Union[str, Path], max_angle_deg: Optional[float] = None) -> Layer:
    """Parse everything geometry-relevant under a single layer directory.

    ``max_angle_deg`` is forwarded to :func:`odbpp.features.parse_features_file`
    for curve tessellation resolution (see its docstring).
    """
    layer_dir = Path(layer_dir)
    name = layer_dir.name

    features = None
    features_path = layer_dir / "features"
    if features_path.exists() or features_path.with_name("features.gz").exists():
        features = parse_features_file(features_path, max_angle_deg=max_angle_deg)

    profile = None
    profile_path = layer_dir / "profile"
    if profile_path.exists() or profile_path.with_name("profile.gz").exists():
        profile = parse_features_file(profile_path, max_angle_deg=max_angle_deg)

    tools = None
    tools_path = layer_dir / "tools"
    if tools_path.exists() or tools_path.with_name("tools.gz").exists():
        tools = parse_tools_file(tools_path)

    attributes, attr_units = parse_layer_attrlist(layer_dir / "attrlist")

    return Layer(name=name, features=features, profile=profile, tools=tools,
                 attributes=attributes, attr_units=attr_units)