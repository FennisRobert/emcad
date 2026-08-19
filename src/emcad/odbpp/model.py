"""Top-level containers: a ``step`` and the whole ``product_model``.

Reference: ODB++Design Format Specification, "Product Model Tree" (Chapter 2)
and "Step Entities" (Chapter 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .features import FeaturesFile, Feature, Stamp, parse_features_file
from .layers import Layer, parse_layer
from .matrix import Matrix, parse_matrix_file
from .symbols import UserDefinedSymbol


def _transform_geom(geom, rotation_deg: float, mirror: bool, dx: float, dy: float):
    """Apply rotate -> (optional) mirror -> translate to a resolved geometry
    value, which may be an Island/Surface (has .rotated/.mirrored_x/.translated)
    or a plain list of Points (a resolved Line/Arc polyline)."""
    if hasattr(geom, "rotated"):
        g = geom.rotated(rotation_deg)
        if mirror:
            g = g.mirrored_x()
        return g.translated(dx, dy)
    if isinstance(geom, list):
        pts = [p.rotated(rotation_deg) for p in geom]
        if mirror:
            pts = [p.mirrored_x() for p in pts]
        return [p.translated(dx, dy) for p in pts]
    return geom


@dataclass
class Step:
    """A ``steps/<step_name>`` directory: one PCB image, panel, or coupon."""
    name: str
    profile: Optional[FeaturesFile] = None      # board/step outline
    layers: Dict[str, Layer] = field(default_factory=dict)

    def get_layer(self, name: str) -> Optional[Layer]:
            layer = self.layers.get(name)
            if layer is not None:
                return layer
            name_lower = name.lower()
            for key, l in self.layers.items():
                if key.lower() == name_lower:
                    return l
            return None

@dataclass
class ProductModel:
    """The whole ODB++ product model: matrix (layer stack) + steps + symbol library."""
    name: str
    matrix: Matrix
    steps: Dict[str, Step] = field(default_factory=dict)
    symbol_library: Dict[str, FeaturesFile] = field(default_factory=dict)

    # ---- construction --------------------------------------------------------

    @classmethod
    def from_path(cls, root: Union[str, Path], max_angle_deg: Optional[float] = None) -> "ProductModel":
        """Parse a full ODB++ product model directory tree.

        ``root`` is the ``<product_model>`` directory itself (the one that
        directly contains ``matrix/``, ``steps/``, and ``symbols/``).
        ``max_angle_deg`` controls curve tessellation resolution -- see
        ``odbpp.parse.parse_product_model`` (this is a thin wrapper around
        it, kept here so parsing a board is one step: ``ProductModel.
        from_path(path)`` instead of a free function you have to know to
        import separately).
        """
        from .parse import parse_product_model   # deferred: parse.py imports ProductModel
        return parse_product_model(root, max_angle_deg=max_angle_deg)

    # ---- symbol resolution -------------------------------------------------

    def resolve_symbol(self, symbol: Union[UserDefinedSymbol, str]) -> FeaturesFile:
        """Return the parsed ``symbols/<name>/features`` file for a user-defined symbol."""
        name = symbol.name if isinstance(symbol, UserDefinedSymbol) else symbol
        if name not in self.symbol_library:
            raise KeyError(f"Symbol {name!r} not found in symbol library")
        return self.symbol_library[name]

    def resolve_stamp(self, stamp: Stamp, max_angle_deg: Optional[float] = None, _depth: int = 0):
        """Fully resolve a Stamp to world-space geometry.

        For standard symbols this is just ``stamp.to_island(max_angle_deg)``.
        For a user-defined symbol it recursively expands the symbol's own
        feature list (which may itself contain pads using other
        user-defined symbols), applying the stamp's rotation/mirror/
        translation to each resulting primitive, and returns a flat list of
        resolved ``(feature, geometry)`` pairs — geometry being an
        :class:`~odbpp.geometry.Island` for surfaces/stamps or a polyline
        (list of Points) for lines/arcs.

        ``max_angle_deg`` (curve resolution) is forwarded to every nested
        symbol/arc; ``None`` uses the shared :mod:`odbpp.config` default
        everywhere in the recursion, so a whole resolved symbol renders at
        one consistent resolution.
        """
        if _depth > 25:
            raise RecursionError("Symbol resolution exceeded max recursion depth (circular symbol reference?)")

        if stamp.symbol is None:
            return []
        if not isinstance(stamp.symbol, UserDefinedSymbol):
            return [(stamp, stamp.to_island(max_angle_deg))]

        sub_file = self.resolve_symbol(stamp.symbol)
        results = []
        for sub_feature in sub_file.features:
            for feat, geom in self._resolve_feature(sub_feature, max_angle_deg, _depth + 1):
                geom = _transform_geom(geom, stamp.rotation_deg, stamp.mirror,
                                        stamp.position.x, stamp.position.y)
                results.append((feat, geom))
        return results

    def _resolve_feature(self, feature: Feature, max_angle_deg: Optional[float], depth: int):
        from .features import Line, Arc, SurfaceFeature
        if isinstance(feature, Stamp):
            return self.resolve_stamp(feature, max_angle_deg, depth)
        if isinstance(feature, SurfaceFeature):
            return [(feature, feature.surface)]
        if isinstance(feature, Line):
            return [(feature, [feature.start, feature.end])]
        if isinstance(feature, Arc):
            return [(feature, feature.to_polyline(max_angle_deg))]
        return []

    # ---- convenience accessors ---------------------------------------------

    def structural_layers(self, step_name: str) -> List[Layer]:
        """Layer objects (with parsed features) for the structural (copper/
        dielectric) layers of a step, in top-to-bottom order."""
        step = self.steps[step_name]
        out = []
        for ml in self.matrix.structural_layers():
            layer = step.layers.get(ml.name)
            if layer is not None:
                out.append(layer)
        return out
