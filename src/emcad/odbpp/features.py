"""ODB++ feature records — the contents of a ``features`` file.

Reference: ODB++Design Format Specification, "<layer_name>/features
(Graphic Features)" (p.164). This same file format is also used for
``symbols/<symbol_name>/features`` and for ``profile`` files (which contain
exactly one Surface feature).

All coordinates/sizes on the classes below are already normalized to
**meters**; parsing applies :mod:`odbpp.units` conversions as records are
read.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger

from . import geometry as geo
from .symbols import Symbol, UserDefinedSymbol, parse_symbol_name
from .units import coord_to_meters, default_symbol_size_units

AttrValue = Union[bool, str, int, float]


# --------------------------------------------------------------------------
# Feature record classes
# --------------------------------------------------------------------------

@dataclass
class Feature:
    """Common fields shared by every feature record."""
    polarity: str = "P"                 # 'P' positive or 'N' negative
    attributes: Dict[int, AttrValue] = field(default_factory=dict)
    id: Optional[int] = None


@dataclass
class Stamp(Feature):
    """A pad: a symbol 'stamped' at a location (ODB++ 'P' — Pad record, p.170).

    Named Stamp (rather than Pad) since that's literally the drawing
    operation: a symbol dragged to a point, rotated, optionally mirrored
    and resized.
    """
    position: geo.Point = field(default_factory=lambda: geo.Point(0.0, 0.0))
    symbol: Optional[Symbol] = None
    dcode: int = 0
    rotation_deg: float = 0.0
    mirror: bool = False
    resize_factor: Optional[float] = None   # meters, only set for '-1 sym resize' pads

    def to_island(self, max_angle_deg: Optional[float] = None) -> geo.Island:
        """Resolve to world-space geometry (translate/rotate/mirror applied).

        Only valid for standard symbols; :class:`UserDefinedSymbol` pads
        must be resolved through the product model's symbol library first
        (see :meth:`odbpp.model.ProductModel.resolve_symbol`). ``max_angle_deg``
        is forwarded to the symbol's own curve rendering (``None`` uses the
        shared :mod:`odbpp.config` default).
        """
        if self.symbol is None or isinstance(self.symbol, UserDefinedSymbol):
            raise NotImplementedError("Stamp uses a user-defined symbol; resolve it via ProductModel first.")
        island = self.symbol.to_island(max_angle_deg)
        island = island.rotated(self.rotation_deg)
        if self.mirror:
            island = island.mirrored_x()
        return island.translated(self.position.x, self.position.y)


@dataclass
class Line(Feature):
    """A line: a symbol dragged along a straight segment (ODB++ 'L' record, p.169)."""
    start: geo.Point = field(default_factory=lambda: geo.Point(0.0, 0.0))
    end: geo.Point = field(default_factory=lambda: geo.Point(0.0, 0.0))
    symbol: Optional[Symbol] = None
    dcode: int = 0


@dataclass
class Arc(Feature):
    """An arc: a symbol dragged along a circular segment (ODB++ 'A' record, p.172)."""
    start: geo.Point = field(default_factory=lambda: geo.Point(0.0, 0.0))
    end: geo.Point = field(default_factory=lambda: geo.Point(0.0, 0.0))
    center: geo.Point = field(default_factory=lambda: geo.Point(0.0, 0.0))
    symbol: Optional[Symbol] = None
    dcode: int = 0
    clockwise: bool = True

    def to_polyline(self, max_angle_deg: Optional[float] = None) -> List[geo.Point]:
        """Tessellate to a polyline (including the start point). See
        :func:`odbpp.geometry.tessellate_arc` — resolution is angle-based,
        and ``None`` uses the shared :mod:`odbpp.config` default."""
        return [self.start] + geo.tessellate_arc(self.start, self.end, self.center, self.clockwise, max_angle_deg)


@dataclass
class TextFeature(Feature):
    """Text drawn with a font (ODB++ 'T' record, p.173). Geometry of individual
    glyphs is not resolved here (see spec Appendix C for the standard font);
    this class preserves placement/sizing for anyone who needs to render it."""
    position: geo.Point = field(default_factory=lambda: geo.Point(0.0, 0.0))
    font: str = "standard"
    rotation_deg: float = 0.0
    mirror: bool = False
    char_width: float = 0.0     # meters
    char_height: float = 0.0    # meters
    width_factor: float = 0.0
    text: str = ""
    version: int = 0


@dataclass
class BarcodeFeature(Feature):
    """ODB++ 'B' — Barcode record (p.176)."""
    position: geo.Point = field(default_factory=lambda: geo.Point(0.0, 0.0))
    barcode_type: str = "UPC39"
    font: str = "standard"
    rotation_deg: float = 0.0
    mirror: bool = False
    element_width: float = 0.0   # meters
    height: float = 0.0          # meters
    full_ascii: bool = False
    checksum: bool = False
    inverted_background: bool = False
    add_text: bool = False
    text_position: str = "T"     # 'T' top or 'B' bottom
    text: str = ""


@dataclass
class SurfaceFeature(Feature):
    """A polygon feature: one or more islands, each with optional holes
    (ODB++ 'S'/'OB'/'OS'/'OC'/'OE'/'SE' records, p.178)."""
    dcode: int = 0
    surface: geo.Surface = field(default_factory=geo.Surface)


# --------------------------------------------------------------------------
# Container for a parsed features file
# --------------------------------------------------------------------------

@dataclass
class FeaturesFile:
    """A fully-parsed ``features`` (or ``profile``, or symbol ``features``) file."""
    units: str                                   # 'MM' or 'INCH', as declared in the file
    id: Optional[int] = None
    symbols: Dict[int, Symbol] = field(default_factory=dict)
    attribute_names: Dict[int, str] = field(default_factory=dict)
    attribute_values: Dict[int, str] = field(default_factory=dict)
    features: List[Feature] = field(default_factory=list)

    def resolved_attributes(self, feature: Feature, text_attrs: Optional[set] = None) -> Dict[str, AttrValue]:
        """Map a feature's raw ``{attr_index: value}`` dict to ``{attr_name: value}``.

        The features file itself does not record each attribute's *type*
        (BOOLEAN/TEXT/OPTION/FLOAT/INTEGER) — that lives in the system/user
        attribute definitions (``misc/sysattr*``, ``misc/userattr``), which
        this package does not parse. So by default this method leaves every
        value exactly as stored (a numeric string is left as a string,
        which for an OPTION attribute *is* the correct option index — do
        not assume it's a lookup-table index).

        If you know which attribute *names* are TEXT-typed (e.g. from the
        spec's system attribute list, or your own project convention), pass
        them in ``text_attrs`` and their numeric values will be resolved
        through the ``&`` text-value table instead.
        """
        text_attrs = text_attrs or set()
        out: Dict[str, AttrValue] = {}
        for idx, val in feature.attributes.items():
            name = self.attribute_names.get(idx, f"@{idx}")
            if name in text_attrs and isinstance(val, str) and val.lstrip("-").isdigit():
                out[name] = self.attribute_values.get(int(val), val)
            else:
                out[name] = val
        return out


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _open_text(path: Union[str, Path]):
    """Open a possibly gzip-compressed ODB++ text file transparently."""
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            magic = f.read(2)
        if magic == b"\x1f\x8b":
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        return open(path, "rt", encoding="utf-8", errors="replace")
    gz_path = path.with_name(path.name + ".gz")
    if gz_path.exists():
        return gzip.open(gz_path, "rt", encoding="utf-8", errors="replace")
    raise FileNotFoundError(f"No such file: {path} (also checked {gz_path})")


_TRAILER_ID_RE = re.compile(r"^ID=(-?\d+)$")


def _parse_trailer(trailer: str) -> (Dict[int, AttrValue], Optional[int]):
    """Parse the optional ``;<atr>=<value>,...;ID=<id>`` tail of a feature record."""
    attrs: Dict[int, AttrValue] = {}
    rec_id: Optional[int] = None
    trailer = trailer.strip()
    if not trailer:
        return attrs, rec_id
    parts = [p for p in trailer.split(";") if p != ""]
    for part in parts:
        m = _TRAILER_ID_RE.match(part)
        if m:
            rec_id = int(m.group(1))
            continue
        for token in part.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" in token:
                k, v = token.split("=", 1)
                attrs[int(k)] = v
            else:
                attrs[int(token)] = True
    return attrs, rec_id


def _first_semicolon_split(rest: str):
    """Split a record's remainder at the first ';' into (fixed_fields, trailer)."""
    idx = rest.find(";")
    if idx == -1:
        return rest.strip(), ""
    return rest[:idx].strip(), rest[idx:]


def _resolve_symbol(symbols: Dict[int, Symbol], sym_num: int) -> Optional[Symbol]:
    return symbols.get(sym_num)


def parse_features_file(path: Union[str, Path], max_angle_deg: Optional[float] = None) -> FeaturesFile:
    """Parse an ODB++ features/profile/symbol-features file.

    ``max_angle_deg`` controls curve tessellation resolution for arcs drawn
    as part of a surface boundary (``OC`` records — e.g. a curved board
    outline or a rounded cutout): each arc is split into enough equal
    segments that no segment sweeps more than this many degrees. ``None``
    (the default) uses the shared :mod:`odbpp.config` setting, so this
    matches the resolution used everywhere else in the package unless you
    override it here. Standalone ``A`` (Arc) feature records are
    tessellated later, on demand, via :meth:`Arc.to_polyline`, which takes
    the same kind of parameter.
    """
    lines = _open_text(path).readlines()

    units: Optional[str] = None
    file_id: Optional[int] = None
    symbols: Dict[int, Symbol] = {}
    attribute_names: Dict[int, str] = {}
    attribute_values: Dict[int, str] = {}
    feature_list: List[Feature] = []

    current_surface: Optional[SurfaceFeature] = None
    current_polygon_pts: List[geo.Point] = []
    current_poly_type: Optional[str] = None   # 'I' or 'H'

    def conv(v: str) -> float:
        return coord_to_meters(float(v), units or "INCH")

    for raw in lines:
        line = raw.rstrip("\n")
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        if s.startswith("UNITS="):
            units = s.split("=", 1)[1].strip()
            continue
        if re.fullmatch(r"ID=-?\d+", s):
            file_id = int(s.split("=", 1)[1])
            continue
        if re.fullmatch(r"F\s+\d+", s):
            continue  # feature count — informational only

        if s.startswith("$"):
            m = re.match(r"^\$(\d+)\s+(\S+)(?:\s+([IM]))?$", s)
            if m:
                idx, name, unit_flag = int(m.group(1)), m.group(2), m.group(3)
                size_units = {"I": "mil", "M": "micron"}.get(unit_flag, default_symbol_size_units(units or "INCH"))
                symbols[idx] = parse_symbol_name(name, size_units)
            continue

        if s.startswith("@"):
            m = re.match(r"^@(\d+)\s+(.+)$", s)
            if m:
                attribute_names[int(m.group(1))] = m.group(2).strip()
            continue

        if s.startswith("&"):
            m = re.match(r"^&(\d+)\s+(.*)$", s)
            if m:
                attribute_values[int(m.group(1))] = m.group(2).strip()
            continue

        tokens0 = s.split(None, 1)
        rec_type = tokens0[0]
        rest = tokens0[1] if len(tokens0) > 1 else ""

        # ---- surface polygon block ----
        if rec_type == "S":
            fixed, trailer = _first_semicolon_split(rest)
            ftoks = fixed.split()
            polarity = ftoks[0] if ftoks else "P"
            dcode = int(ftoks[1]) if len(ftoks) > 1 else 0
            attrs, rec_id = _parse_trailer(trailer)
            current_surface = SurfaceFeature(polarity=polarity, attributes=attrs, id=rec_id,
                                              dcode=dcode, surface=geo.Surface())
            continue

        if rec_type == "OB":
            ftoks = rest.split()
            xbs, ybs, poly_type = ftoks[0], ftoks[1], ftoks[2]
            current_polygon_pts = [geo.Point(conv(xbs), conv(ybs))]
            current_poly_type = poly_type
            continue

        if rec_type == "OS":
            ftoks = rest.split()
            x, y = ftoks[0], ftoks[1]
            current_polygon_pts.append(geo.Point(conv(x), conv(y)))
            continue

        if rec_type == "OC":
            ftoks = rest.split()
            xe, ye, xc, yc, cw = ftoks[0], ftoks[1], ftoks[2], ftoks[3], ftoks[4]
            start = current_polygon_pts[-1]
            end = geo.Point(conv(xe), conv(ye))
            center = geo.Point(conv(xc), conv(yc))
            current_polygon_pts.extend(geo.tessellate_arc(start, end, center, cw.upper() == "Y", max_angle_deg))
            continue

        if rec_type == "OE":
            poly = geo.Polygon(current_polygon_pts)
            if current_surface is not None:
                if current_poly_type == "H" and current_surface.surface.islands:
                    current_surface.surface.islands[-1].holes.append(poly)
                else:
                    current_surface.surface.islands.append(geo.Island(boundary=poly))
            current_polygon_pts = []
            current_poly_type = None
            continue

        if rec_type == "SE":
            if current_surface is not None:
                feature_list.append(current_surface)
            current_surface = None
            continue

        # ---- L, P, A, T, B records ----
        fixed, trailer = _first_semicolon_split(rest)
        attrs, rec_id = _parse_trailer(trailer)
        ftoks = fixed.split()

        if rec_type == "L":
            xs, ys, xe, ye, sym_num, polarity, dcode = ftoks[:7]
            feature_list.append(Line(
                polarity=polarity, attributes=attrs, id=rec_id,
                start=geo.Point(conv(xs), conv(ys)), end=geo.Point(conv(xe), conv(ye)),
                symbol=_resolve_symbol(symbols, int(sym_num)), dcode=int(dcode),
            ))
            continue

        if rec_type == "P":
            x, y = ftoks[0], ftoks[1]
            i = 2
            if ftoks[i] == "-1":
                sym_num = int(ftoks[i + 1])
                resize_raw = float(ftoks[i + 2])
                resize_factor = coord_to_meters(resize_raw / 1000.0, units or "INCH")
                i += 3
            else:
                sym_num = int(ftoks[i])
                resize_factor = None
                i += 1
            polarity = ftoks[i]; i += 1
            dcode = int(ftoks[i]); i += 1
            orient_first = ftoks[i]
            if orient_first in ("8", "9"):
                rotation_deg = float(ftoks[i + 1])
                mirror = orient_first == "9"
                i += 2
            else:
                legacy = int(orient_first)
                rotation_deg = {0: 0, 1: 90, 2: 180, 3: 270, 4: 0, 5: 90, 6: 180, 7: 270}[legacy]
                mirror = legacy >= 4
                i += 1
            feature_list.append(Stamp(
                polarity=polarity, attributes=attrs, id=rec_id,
                position=geo.Point(conv(x), conv(y)),
                symbol=_resolve_symbol(symbols, sym_num), dcode=dcode,
                rotation_deg=rotation_deg, mirror=mirror, resize_factor=resize_factor,
            ))
            continue

        if rec_type == "A":
            xs, ys, xe, ye, xc, yc, sym_num, polarity, dcode, cw = ftoks[:10]
            feature_list.append(Arc(
                polarity=polarity, attributes=attrs, id=rec_id,
                start=geo.Point(conv(xs), conv(ys)), end=geo.Point(conv(xe), conv(ye)),
                center=geo.Point(conv(xc), conv(yc)),
                symbol=_resolve_symbol(symbols, int(sym_num)), dcode=int(dcode),
                clockwise=(cw.upper() == "Y"),
            ))
            continue

        if rec_type == "T":
            # The text string is quoted (with ' or a curly “ ” pair) and may
            # itself contain spaces, so pull it out first; everything before
            # the opening quote is fixed-format fields, everything after the
            # closing quote is just the trailing `version` field. The
            # orient_def field before it is 1 or 2 tokens, so we parse the
            # fixed-field tokens from the *right* rather than the left.
            qm = re.match(r"^(?P<lead>.*?)\s*(?P<q1>['\u201c])(?P<text>.*)(?P<q2>['\u201d])\s+(?P<version>\d+)\s*$", fixed)
            if not qm:
                continue
            lead_toks = qm.group("lead").split()
            text = qm.group("text")
            version = qm.group("version")
            width_factor = lead_toks[-1]
            ysize = lead_toks[-2]
            xsize = lead_toks[-3]
            head = lead_toks[:-3]              # [x, y, font, polarity, *orient_tokens]
            x, y, font, polarity = head[0], head[1], head[2], head[3]
            orient_toks = head[4:]
            first = orient_toks[0]
            if first in ("8", "9"):
                rotation_deg = float(orient_toks[1]) if len(orient_toks) > 1 else 0.0
                mirror = first == "9"
            else:
                legacy = int(first)
                rotation_deg = {0: 0, 1: 90, 2: 180, 3: 270, 4: 0, 5: 90, 6: 180, 7: 270}[legacy]
                mirror = legacy >= 4
            feature_list.append(TextFeature(
                polarity=polarity, attributes=attrs, id=rec_id,
                position=geo.Point(conv(x), conv(y)), font=font,
                rotation_deg=rotation_deg, mirror=mirror,
                char_width=conv(xsize), char_height=conv(ysize),
                width_factor=float(width_factor), text=text, version=int(version),
            ))
            continue

        if rec_type == "B":
            # B <x> <y> <barcode> <font> <P|N> <orient...> E <w> <h> <fasc> <cs> <bg> <astr> <astr_pos> '<text>'
            m3 = re.match(
                r"^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(P|N)\s+(.+?)\s+E\s+(\S+)\s+(\S+)\s+(Y|N)\s+(Y|N)\s+(Y|N)\s+(Y|N)\s+(T|B)\s+(.*)$",
                fixed)
            if m3:
                (x, y, barcode_type, font, polarity, orient, w, h,
                 fasc, cs, bg, astr, astr_pos, text) = m3.groups()
                orient_toks = orient.split()
                first = orient_toks[0]
                if first in ("8", "9"):
                    rotation_deg = float(orient_toks[1]) if len(orient_toks) > 1 else 0.0
                    mirror = first == "9"
                else:
                    legacy = int(first)
                    rotation_deg = {0: 0, 1: 90, 2: 180, 3: 270, 4: 0, 5: 90, 6: 180, 7: 270}[legacy]
                    mirror = legacy >= 4
                feature_list.append(BarcodeFeature(
                    polarity=polarity, attributes=attrs, id=rec_id,
                    position=geo.Point(conv(x), conv(y)), barcode_type=barcode_type, font=font,
                    rotation_deg=rotation_deg, mirror=mirror,
                    element_width=conv(w), height=conv(h),
                    full_ascii=(fasc == "Y"), checksum=(cs == "Y"), inverted_background=(bg == "Y"),
                    add_text=(astr == "Y"), text_position=astr_pos, text=text.strip("'\"\u201c\u201d"),
                ))
            continue

        # Unknown/unsupported record type: ignore rather than fail the whole parse.

    logger.debug(f"parse_features_file: '{Path(path).name}' -- {len(feature_list)} feature(s), {len(symbols)} symbol(s)")
    return FeaturesFile(
        units=units or "INCH", id=file_id, symbols=symbols,
        attribute_names=attribute_names, attribute_values=attribute_values,
        features=feature_list,
    )
