"""ODB++ symbols — the shape vocabulary used to draw pads, lines, and arcs.

Reference: ODB++Design Format Specification, "Symbols" (p.36) and Appendix A
"Standard ODB++ Symbols" (p.187).

Standard symbols are parsed straight out of their name string (e.g.
``rect100x50xr8``) into one of the dataclasses below; every numeric
parameter is converted to **meters** at parse time via
:func:`odbpp.units.symbol_size_to_meters`. User-defined symbols are only
identified by name here — resolving them to actual geometry means loading
``symbols/<name>/features`` (a features file, see :mod:`odbpp.features`),
which is done at the :class:`odbpp.model.ProductModel` level where the
symbol library is available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Set

from . import geometry as geo
from .units import symbol_size_to_meters


class Symbol:
    """Base class for every symbol type."""

    #: True for symbols whose geometry is a single simple boundary
    #: (no inherent hole). Donuts/thermals override this.
    has_hole: ClassVar[bool] = False

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        """Return this symbol's shape as an :class:`~odbpp.geometry.Island`
        (boundary + holes), centered on the origin, in meters.

        ``max_angle_deg`` controls curve resolution for any round/rounded
        part of the shape; ``None`` (the default) uses the shared
        :mod:`odbpp.config` setting so every symbol in a model renders at a
        consistent resolution unless you explicitly override it here.
        """
        raise NotImplementedError


# ---- basic filled shapes -------------------------------------------------

@dataclass
class RoundSymbol(Symbol):
    diameter: float

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.circle(self.diameter / 2.0, max_angle_deg))


@dataclass
class SquareSymbol(Symbol):
    side: float

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.rectangle(self.side, self.side))


@dataclass
class RectangleSymbol(Symbol):
    width: float
    height: float

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.rectangle(self.width, self.height))


@dataclass
class RoundedRectangleSymbol(Symbol):
    width: float
    height: float
    radius: float
    corners: Optional[Set[int]] = None

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.rounded_rect(self.width, self.height, self.radius, self.corners, max_angle_deg))


@dataclass
class ChamferedRectangleSymbol(Symbol):
    width: float
    height: float
    radius: float
    corners: Optional[Set[int]] = None

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.chamfered_rect(self.width, self.height, self.radius, self.corners))


@dataclass
class OvalSymbol(Symbol):
    width: float
    height: float

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.stadium(self.width, self.height, max_angle_deg))


@dataclass
class DiamondSymbol(Symbol):
    width: float
    height: float

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.diamond(self.width, self.height))


@dataclass
class OctagonSymbol(Symbol):
    width: float
    height: float
    corner: float

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.octagon(self.width, self.height, self.corner))


@dataclass
class HexagonSymbol(Symbol):
    width: float
    height: float
    corner: float
    orientation: str = "horizontal"   # 'horizontal' (hex_l) or 'vertical' (hex_s)

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.hexagon(self.width, self.height, self.corner, self.orientation))


@dataclass
class ButterflySymbol(Symbol):
    size: float
    shape: str = "round"   # 'round' (bfr) or 'square' (bfs) — rendered geometry is approximate

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.butterfly(self.size, self.shape))


@dataclass
class TriangleSymbol(Symbol):
    base: float
    height: float

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.triangle(self.base, self.height))


@dataclass
class HalfOvalSymbol(Symbol):
    width: float
    height: float

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.Island(geo.half_stadium(self.width, self.height, max_angle_deg))


# ---- donut (annulus) shapes ----------------------------------------------

@dataclass
class DonutSymbol(Symbol):
    """Round/square/mixed donut: donut_r, donut_s, donut_sr."""
    outer: float
    inner: float
    outer_shape: str = "round"   # 'round' or 'square'
    inner_shape: str = "round"   # 'round' or 'square'
    has_hole: ClassVar[bool] = True

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        outer_poly = (geo.circle(self.outer / 2.0, max_angle_deg) if self.outer_shape == "round"
                      else geo.rectangle(self.outer, self.outer))
        inner_poly = (geo.circle(self.inner / 2.0, max_angle_deg) if self.inner_shape == "round"
                      else geo.rectangle(self.inner, self.inner))
        return geo.donut(outer_poly, inner_poly)


@dataclass
class RoundedSquareDonutSymbol(Symbol):
    outer: float
    inner: float
    radius: float
    corners: Optional[Set[int]] = None
    has_hole: ClassVar[bool] = True

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        outer_poly = geo.rounded_rect(self.outer, self.outer, self.radius, self.corners, max_angle_deg)
        inner_poly = geo.rectangle(self.inner, self.inner)
        return geo.donut(outer_poly, inner_poly)


@dataclass
class RectDonutSymbol(Symbol):
    """A rectangular ring: donut_rc<ow>x<oh>x<lw>, optionally rounded."""
    outer_width: float
    outer_height: float
    line_width: float
    radius: float = 0.0
    corners: Optional[Set[int]] = None
    has_hole: ClassVar[bool] = True

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        if self.radius > 0:
            outer_poly = geo.rounded_rect(self.outer_width, self.outer_height, self.radius, self.corners, max_angle_deg)
        else:
            outer_poly = geo.rectangle(self.outer_width, self.outer_height)
        inner_poly = geo.rectangle(self.outer_width - 2 * self.line_width,
                                    self.outer_height - 2 * self.line_width)
        return geo.donut(outer_poly, inner_poly)


@dataclass
class OvalDonutSymbol(Symbol):
    outer_width: float
    outer_height: float
    line_width: float
    has_hole: ClassVar[bool] = True

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        outer_poly = geo.stadium(self.outer_width, self.outer_height, max_angle_deg)
        inner_poly = geo.stadium(self.outer_width - 2 * self.line_width,
                                  self.outer_height - 2 * self.line_width, max_angle_deg)
        return geo.donut(outer_poly, inner_poly)


# ---- thermal relief shapes -------------------------------------------------

@dataclass
class ThermalSymbol(Symbol):
    """Thermal relief pad: thr, ths, s_ths, s_tho."""
    outer: float
    inner: float
    angle: float
    num_spokes: int
    gap: float
    shape: str = "round"   # 'round' or 'square'
    has_hole: ClassVar[bool] = True

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        return geo.thermal(self.outer, self.inner, self.angle, self.num_spokes, self.gap,
                            self.shape, max_angle_deg)


# ---- user-defined symbols --------------------------------------------------

@dataclass
class UserDefinedSymbol(Symbol):
    """A named, non-standard symbol. Its geometry lives in
    ``symbols/<name>/features`` and must be resolved against a symbol
    library (see :class:`odbpp.model.ProductModel`); it is *not* a single
    boundary+holes shape in general (it can contain pads, lines, arcs,
    text, and nested symbols), so it has no :meth:`to_island`.
    """
    name: str

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        raise NotImplementedError(
            f"UserDefinedSymbol {self.name!r} must be resolved via its symbols/{self.name}/features "
            "file (see ProductModel.resolve_symbol) rather than rendered directly."
        )


# --------------------------------------------------------------------------
# Standard symbol name grammar -> Symbol factory
# --------------------------------------------------------------------------

_NUM = r"[\d]*\.?[\d]+"


def _corners(s: Optional[str]) -> Optional[Set[int]]:
    if not s:
        return None
    return {int(ch) for ch in s}


# Ordered (most-specific-prefix-first) list of (compiled regex, builder).
# Each builder receives a regex Match and a `conv` callable (raw units -> meters).
_PATTERNS = []


def _pattern(regex: str):
    compiled = re.compile(regex)

    def register(fn):
        _PATTERNS.append((compiled, fn))
        return fn
    return register


@_pattern(rf"^donut_sr(?P<od>{_NUM})x(?P<id>{_NUM})$")
def _p_donut_sr(m, conv):
    return DonutSymbol(conv(m["od"]), conv(m["id"]), outer_shape="square", inner_shape="round")


@_pattern(rf"^donut_s(?P<od>{_NUM})x(?P<id>{_NUM})xr(?P<rad>{_NUM})(?:x(?P<corners>\d+))?$")
def _p_donut_s_rounded(m, conv):
    return RoundedSquareDonutSymbol(conv(m["od"]), conv(m["id"]), conv(m["rad"]), _corners(m["corners"]))


@_pattern(rf"^donut_s(?P<od>{_NUM})x(?P<id>{_NUM})$")
def _p_donut_s(m, conv):
    return DonutSymbol(conv(m["od"]), conv(m["id"]), outer_shape="square", inner_shape="square")


@_pattern(rf"^donut_r(?P<od>{_NUM})x(?P<id>{_NUM})$")
def _p_donut_r(m, conv):
    return DonutSymbol(conv(m["od"]), conv(m["id"]), outer_shape="round", inner_shape="round")


@_pattern(rf"^donut_rc(?P<ow>{_NUM})x(?P<oh>{_NUM})x(?P<lw>{_NUM})xr(?P<rad>{_NUM})(?:x(?P<corners>\d+))?$")
def _p_donut_rc_rounded(m, conv):
    return RectDonutSymbol(conv(m["ow"]), conv(m["oh"]), conv(m["lw"]), conv(m["rad"]), _corners(m["corners"]))


@_pattern(rf"^donut_rc(?P<ow>{_NUM})x(?P<oh>{_NUM})x(?P<lw>{_NUM})$")
def _p_donut_rc(m, conv):
    return RectDonutSymbol(conv(m["ow"]), conv(m["oh"]), conv(m["lw"]))


@_pattern(rf"^donut_o(?P<ow>{_NUM})x(?P<oh>{_NUM})x(?P<lw>{_NUM})$")
def _p_donut_o(m, conv):
    return OvalDonutSymbol(conv(m["ow"]), conv(m["oh"]), conv(m["lw"]))


@_pattern(rf"^rect(?P<w>{_NUM})x(?P<h>{_NUM})xr(?P<rad>{_NUM})(?:x(?P<corners>\d+))?$")
def _p_rounded_rect(m, conv):
    return RoundedRectangleSymbol(conv(m["w"]), conv(m["h"]), conv(m["rad"]), _corners(m["corners"]))


@_pattern(rf"^rect(?P<w>{_NUM})x(?P<h>{_NUM})xc(?P<rad>{_NUM})(?:x(?P<corners>\d+))?$")
def _p_chamfered_rect(m, conv):
    return ChamferedRectangleSymbol(conv(m["w"]), conv(m["h"]), conv(m["rad"]), _corners(m["corners"]))


@_pattern(rf"^rect(?P<w>{_NUM})x(?P<h>{_NUM})$")
def _p_rect(m, conv):
    return RectangleSymbol(conv(m["w"]), conv(m["h"]))


@_pattern(rf"^oval_h(?P<w>{_NUM})x(?P<h>{_NUM})$")
def _p_half_oval(m, conv):
    return HalfOvalSymbol(conv(m["w"]), conv(m["h"]))


@_pattern(rf"^oval(?P<w>{_NUM})x(?P<h>{_NUM})$")
def _p_oval(m, conv):
    return OvalSymbol(conv(m["w"]), conv(m["h"]))


@_pattern(rf"^di(?P<w>{_NUM})x(?P<h>{_NUM})$")
def _p_diamond(m, conv):
    return DiamondSymbol(conv(m["w"]), conv(m["h"]))


@_pattern(rf"^oct(?P<w>{_NUM})x(?P<h>{_NUM})x(?P<r>{_NUM})$")
def _p_octagon(m, conv):
    return OctagonSymbol(conv(m["w"]), conv(m["h"]), conv(m["r"]))


@_pattern(rf"^hex_l(?P<w>{_NUM})x(?P<h>{_NUM})x(?P<r>{_NUM})$")
def _p_hex_h(m, conv):
    return HexagonSymbol(conv(m["w"]), conv(m["h"]), conv(m["r"]), orientation="horizontal")


@_pattern(rf"^hex_s(?P<w>{_NUM})x(?P<h>{_NUM})x(?P<r>{_NUM})$")
def _p_hex_v(m, conv):
    return HexagonSymbol(conv(m["w"]), conv(m["h"]), conv(m["r"]), orientation="vertical")


@_pattern(rf"^bfr(?P<d>{_NUM})$")
def _p_butterfly_round(m, conv):
    return ButterflySymbol(conv(m["d"]), shape="round")


@_pattern(rf"^bfs(?P<s>{_NUM})$")
def _p_butterfly_square(m, conv):
    return ButterflySymbol(conv(m["s"]), shape="square")


@_pattern(rf"^tri(?P<base>{_NUM})x(?P<h>{_NUM})$")
def _p_triangle(m, conv):
    return TriangleSymbol(conv(m["base"]), conv(m["h"]))


@_pattern(rf"^thr(?P<od>{_NUM})x(?P<id>{_NUM})x(?P<angle>{_NUM})x(?P<n>\d+)x(?P<gap>{_NUM})$")
def _p_thermal_round_rounded(m, conv):
    return ThermalSymbol(conv(m["od"]), conv(m["id"]), float(m["angle"]), int(m["n"]), conv(m["gap"]), shape="round")


@_pattern(rf"^ths(?P<od>{_NUM})x(?P<id>{_NUM})x(?P<angle>{_NUM})x(?P<n>\d+)x(?P<gap>{_NUM})$")
def _p_thermal_round_squared(m, conv):
    return ThermalSymbol(conv(m["od"]), conv(m["id"]), float(m["angle"]), int(m["n"]), conv(m["gap"]), shape="round")


@_pattern(rf"^s_ths(?P<os>{_NUM})x(?P<is_>{_NUM})x(?P<angle>{_NUM})x(?P<n>\d+)x(?P<gap>{_NUM})$")
def _p_thermal_square(m, conv):
    return ThermalSymbol(conv(m["os"]), conv(m["is_"]), float(m["angle"]), int(m["n"]), conv(m["gap"]), shape="square")


@_pattern(rf"^s_tho(?P<od>{_NUM})x(?P<id>{_NUM})x(?P<angle>{_NUM})x(?P<n>\d+)x(?P<gap>{_NUM})$")
def _p_thermal_square_open(m, conv):
    return ThermalSymbol(conv(m["od"]), conv(m["id"]), float(m["angle"]), int(m["n"]), conv(m["gap"]), shape="square")


@_pattern(rf"^r(?P<d>{_NUM})$")
def _p_round(m, conv):
    return RoundSymbol(conv(m["d"]))


@_pattern(rf"^s(?P<s>{_NUM})$")
def _p_square(m, conv):
    return SquareSymbol(conv(m["s"]))


def parse_symbol_name(name: str, size_units: str) -> Symbol:
    """Parse a standard symbol name into a :class:`Symbol`, converting all
    numeric parameters to meters using ``size_units`` ('mil' or 'micron').

    Falls back to :class:`UserDefinedSymbol` if the name doesn't match any
    known standard-symbol grammar (i.e. it must be resolved against the
    product model's ``symbols/`` directory).
    """
    def conv(raw: str) -> float:
        return symbol_size_to_meters(float(raw), size_units)

    for regex, builder in _PATTERNS:
        m = regex.match(name)
        if m:
            return builder(m, conv)
    return UserDefinedSymbol(name)
