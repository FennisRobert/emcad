"""PCBView — a convenience layer on top of :class:`odbpp.model.ProductModel`
for pulling out mesh-ready 2D polygons and a Z-stack.

This module does the "turn the object graph into polygons" work described in
the odbpp usage guide, so callers don't have to re-derive it: it resolves
user-defined-symbol stamps, buffers lines/arcs (centerline + symmetric
symbol width) into filled polygons, assigns each geometry layer a Z position
from the matrix layer stack, and gives every polygon its source polarity.

Everything is still in **meters**, matching the rest of the package.

Quick start
-----------
    pcbd = PCBView(pm)
    xs, ys = pcbd.get_board_polygon()
    for z1, z2, material in pcbd.iter_pcb_layers():
        ...
    for layer in pcbd.iter_geo_layers():
        for layer_poly in layer.iter_polygons():
            xs, ys = layer_poly.xys()
            z = layer_poly.z
            polarity = layer_poly.polarity
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

from loguru import logger

from .config import TESSELLATION, segments_for_sweep
from .features import Arc, BarcodeFeature, Feature, Line, Stamp, SurfaceFeature, TextFeature
from .geometry import Island, Surface
from .matrix import MatrixLayer
from .model import ProductModel
from .symbols import RoundSymbol, SquareSymbol, Symbol, UserDefinedSymbol

XY = Tuple[float, float]


# --------------------------------------------------------------------------
# Small geometry helpers specific to this module
# --------------------------------------------------------------------------

def _pts(points) -> List[XY]:
    return [(p.x, p.y) for p in points]


def _symbol_width(symbol: Optional[Symbol]) -> Optional[float]:
    """The stroke width of a symbol used to draw a Line/Arc.

    Per the spec, only *symmetric* symbols (round, or square with equal
    sides) are legal for drawing lines/arcs, so this covers every valid
    case; anything else (a non-symmetric or unresolved symbol) returns None
    and the caller should skip the feature.
    """
    if isinstance(symbol, RoundSymbol):
        return symbol.diameter
    if isinstance(symbol, SquareSymbol):
        return symbol.side
    return None


def _resized_symbol(symbol: Symbol, resize_factor: float) -> Symbol:
    """Best-effort application of a pad's ``-1 sym resize_factor`` resize.

    The spec doesn't fully pin down how resize applies to every symbol
    shape, so this grows/shrinks whichever size-like field(s) the symbol
    has (diameter, side, width/height, outer) by ``2 * resize_factor``
    (resize_factor is applied per edge). Symbols with no matching field are
    returned unchanged.
    """
    if not resize_factor:
        return symbol
    names = {f.name for f in fields(symbol)}
    delta = 2.0 * resize_factor
    kwargs = {}
    for name in ("diameter", "side", "width", "height", "outer"):
        if name in names:
            kwargs[name] = getattr(symbol, name) + delta
    return replace(symbol, **kwargs) if kwargs else symbol


def buffer_segment(p0: XY, p1: XY, width: float, max_angle_deg: Optional[float] = None) -> List[XY]:
    """Buffer a single line segment into a filled capsule polygon (round caps).

    ``p0``/``p1`` are (x, y) tuples in meters; ``width`` is the full stroke
    width (e.g. a RoundSymbol's diameter). If the segment has ~zero length,
    returns a full circle of that width instead (matching the spec's "start
    == end is a legal zero-length line" case).

    Curve resolution for the round caps uses the same shared angle-step
    budget as everything else (:mod:`odbpp.config`) — ``None`` uses the
    global default, so a trace's rounded ends match the resolution of every
    other curve in the model.
    """
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    r = width / 2.0
    if length < 1e-12:
        segments = segments_for_sweep(360, max_angle_deg, minimum=TESSELLATION.min_segments)
        return [(x0 + r * math.cos(2 * math.pi * i / segments),
                 y0 + r * math.sin(2 * math.pi * i / segments))
                for i in range(segments)]

    # Each end cap is a half-circle (180 degree sweep).
    cap_segments = segments_for_sweep(180, max_angle_deg)

    ux, uy = dx / length, dy / length
    nx, ny = -uy * r, ux * r   # perpendicular offset vector, length r

    pts: List[XY] = [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny)]
    a0 = math.atan2(ny, nx)
    for i in range(1, cap_segments + 1):
        a = a0 - math.pi * i / cap_segments
        pts.append((x1 + r * math.cos(a), y1 + r * math.sin(a)))
    pts.append((x1 - nx, y1 - ny))
    pts.append((x0 - nx, y0 - ny))
    a1 = math.atan2(-ny, -nx)
    for i in range(1, cap_segments + 1):
        a = a1 - math.pi * i / cap_segments
        pts.append((x0 + r * math.cos(a), y0 + r * math.sin(a)))
    return pts


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------

@dataclass
class LayerPolygon:
    """One resolved, world-space, fillable polygon from a geometry layer."""
    points: List[XY]
    holes: List[List[XY]]
    z: float
    polarity: str            # 'P' positive (add material) or 'N' negative (remove material)
    layer_name: str
    source: Optional[Feature] = None   # the originating Feature, for provenance/debugging

    def xys(self) -> Tuple[List[float], List[float]]:
        """Outer boundary as (xs, ys) parallel lists."""
        return [p[0] for p in self.points], [p[1] for p in self.points]

    def holes_xys(self) -> List[Tuple[List[float], List[float]]]:
        """Each hole as an (xs, ys) pair, in the same order as ``self.holes``."""
        return [([p[0] for p in h], [p[1] for p in h]) for h in self.holes]


@dataclass
class DrillHole:
    """One resolved drill/via hole."""
    x: float
    y: float
    diameter: float
    z1: float
    z2: float
    plated: bool
    layer_name: str


@dataclass
class GeoLayer:
    """One layer's worth of 2D geometry, positioned at a Z plane (or, for
    structural copper/dielectric layers, spanning a Z range)."""
    name: str
    type: str
    z1: float
    z2: float
    matrix_layer: MatrixLayer
    _designer: "PCBView" = field(repr=False)

    @property
    def z(self) -> float:
        """Representative Z for this layer (midplane of z1..z2)."""
        return (self.z1 + self.z2) / 2.0

    def iter_polygons(self) -> Iterator[LayerPolygon]:
        yield from self._designer._layer_polygons(self)


# --------------------------------------------------------------------------
# PCBView
# --------------------------------------------------------------------------

class PCBView:
    """High-level polygon/Z-stack view of one step of a :class:`ProductModel`.

    Parameters
    ----------
    source:
        Either an ODB++ product model directory path (``str``/``Path``) --
        parsed internally via :meth:`ProductModel.from_path`, so ``PCBView(
        "/path/to/Board.odb")`` is a complete, one-step way to go from a
        board on disk to a queryable view -- or an already-parsed
        :class:`ProductModel`, if you want to inspect/reuse the raw object
        graph yourself before building a view over it. Either way, the
        resolved model is available afterward as ``self.pm``.
    step_name:
        Which step to model (defaults to the first one found — pass this
        explicitly for multi-step product models, e.g. panels).
    thickness:
        Optional override of the *nominal* per-layer-type thickness
        (meters) used as a fallback in the Z stack, e.g.
        ``{"SIGNAL": 18e-6}``. Merged over :attr:`DEFAULT_THICKNESS`. This
        is only actually used when a layer doesn't have real per-layer
        thickness data of its own — see ``use_layer_thickness`` below.
    materials:
        Optional override of the per-layer-type material label, e.g.
        ``{"DIELECTRIC": "Isola 370HR"}``. Merged over
        :attr:`DEFAULT_MATERIAL`. A DIELECTRIC layer's own
        ``dielectric_name`` (from the matrix) wins over this if present.
    include_types:
        Which matrix layer TYPEs show up in :meth:`iter_geo_layers`.
        Defaults to copper + dielectric + mask/silk/paste/component.
    use_layer_thickness:
        If True (default), a structural layer's own attrlist material data
        — ``.layer_dielectric`` for DIELECTRIC layers, ``.copper_weight``
        for copper layers (see :class:`odbpp.layers.Layer`) — is used for
        its thickness in the Z stack when present, instead of the nominal
        ``thickness``/``DEFAULT_THICKNESS`` value for its type. This is
        real per-layer manufacturing data when the exporting tool wrote
        it (not every export does — check ``pcbd.step.get_layer(name).attributes``
        to see what's actually in your file); the nominal thickness is
        always the fallback for any layer that doesn't have it. Set this
        False to ignore attrlist data entirely and always use the nominal
        per-type thickness, e.g. for comparing boards on equal footing.
    strict_thickness:
        If True, disables the nominal per-type fallback entirely: a
        structural layer with no real attrlist data AND no override for it
        in ``thickness_stack`` makes construction raise a ``ValueError``
        instead of silently guessing 35 um of copper / 150 um of
        dielectric. The error message lists every structural layer in
        top-to-bottom order, which ones resolved and to what, and exactly
        which ones need a value — so you know precisely what to supply.
        Default False (silent nominal fallback, as before).
    thickness_stack:
        Optional list of per-layer thickness overrides (meters), one entry
        per structural (copper/dielectric) layer, in the *exact*
        top-to-bottom order of ``pm.matrix.structural_layers()`` (e.g.
        ``[copper, dielectric, copper, dielectric, copper, ...]`` for a
        typical multilayer board — construction will tell you the exact
        order and layer names if you get the length wrong or trigger
        ``strict_thickness``). Use ``None`` for any entry you don't want to
        override — it falls through to attrlist/nominal resolution as
        usual. A non-None entry here takes priority over everything else,
        including real attrlist data.
    thick_traces:
        Controls whether copper gets real Z-thickness in the stack.
        Dielectric layers are ALWAYS perfectly contiguous regardless of
        this setting — the running z accumulator only ever advances by
        DIELECTRIC thicknesses, so one dielectric's z2 always equals the
        next dielectric's z1, as if copper weren't there at all. Copper's
        own [z1, z2] is then derived from that shared seam:

        - ``True`` (default): copper gets its nominal thickness, placed
          relative to the seam per ``copper_placement``.
        - ``False``: copper is a flat, zero-thickness marker exactly at
          the seam (z1 == z2). Simplest option — no copper Z-extent to
          worry about at all, total board thickness is exactly the sum of
          dielectric thicknesses.
    copper_placement:
        Only used when ``thick_traces=True``. Where a copper layer's
        thickness sits relative to the seam between its neighboring
        dielectrics:

        - ``"top"``: embedded in the tail end of the dielectric ABOVE the
          seam — ``[seam - t_cu, seam]`` ("sits on top of the previous
          dielectric").
        - ``"bottom"`` (default): embedded in the leading edge of the
          dielectric BELOW the seam — ``[seam, seam + t_cu]`` (copper
          laminated onto the surface of the next dielectric).
        - ``"center"``: straddles the seam symmetrically —
          ``[seam - t_cu/2, seam + t_cu/2]``.

        Either way, the dielectric on the *other* side of that copper
        layer is untouched and still starts/ends exactly at the seam — so
        the copper slab overlaps into whichever one neighboring dielectric
        the placement says, rather than pushing anything apart. If the
        dielectric neighbor a placement needs doesn't exist (e.g. an outer
        copper layer with nothing above/below it in the matrix), this
        falls back to whichever side does have a dielectric neighbor, or
        to the flat/zero-thickness behavior if neither side does.
    reverse:
        If True, Z-flips the whole board: every Z value this designer
        emits (``iter_pcb_layers``, ``GeoLayer.z1``/``z2``/``z``,
        ``LayerPolygon.z``, ``DrillHole.z1``/``z2``) is mirrored about the
        board's total thickness (``z -> total_thickness - z``), and the
        top-to-bottom iteration order of ``iter_pcb_layers``/
        ``iter_geo_layers`` is reversed to match, so layer order still
        reads top-to-bottom (increasing Z) in the flipped orientation —
        i.e. what was physically the top copper now sits at the bottom of
        the Z range and is yielded last, and vice versa. Only Z is
        affected; X/Y geometry (including :meth:`centralize`) is
        untouched. The internal stack-building logic (dielectric
        contiguity, copper placement, mask/silk REF resolution) is always
        computed in the original, unflipped orientation first — reverse is
        applied only at the point each value is handed back to you, so
        turning it on/off doesn't change anything else about the model.
    max_angle_deg:
        Curve tessellation resolution for everything this designer renders
        (standalone Arc features, round buffer caps on Line/Arc strokes,
        and every symbol shape rendered via ``Stamp.to_island``/resolved
        user-defined symbols): each curve is split into enough equal
        segments that no segment sweeps more than this many degrees.
        ``None`` (the default) uses the shared, package-wide
        :mod:`odbpp.config` setting — so this designer's curves match the
        resolution used when the product model was parsed (board/layer
        outlines) unless you deliberately override it here. Set this
        explicitly if you want this particular designer's pad/trace
        rendering to differ from the outline resolution used at parse time.
    """

    #: nominal thickness (meters) per matrix LAYER TYPE, used for the Z stack
    DEFAULT_THICKNESS: Dict[str, float] = {
        "SIGNAL": 35e-6,
        "POWER_GROUND": 35e-6,
        "MIXED": 35e-6,
        "DIELECTRIC": 150e-6,
    }

    #: material label per matrix LAYER TYPE
    DEFAULT_MATERIAL: Dict[str, str] = {
        "SIGNAL": "COPPER",
        "POWER_GROUND": "COPPER",
        "MIXED": "COPPER",
        "DIELECTRIC": "DIELECTRIC",
        "SOLDER_MASK": "SOLDER_MASK",
        "SILK_SCREEN": "SILKSCREEN",
        "SOLDER_PASTE": "SOLDER_PASTE",
        "COMPONENT": "COMPONENT",
        "MASK": "MASK",
        "CONDUCTIVE_PASTE": "CONDUCTIVE_PASTE",
    }

    #: matrix LAYER TYPEs treated as flat, single-Z-plane geometry layers
    #: (i.e. everything geometrically meaningful except DRILL/ROUT, which
    #: are handled separately via iter_drill_holes())
    DEFAULT_GEO_TYPES: Set[str] = {
        "SIGNAL", "POWER_GROUND", "MIXED", "DIELECTRIC",
        "SOLDER_MASK", "SILK_SCREEN", "SOLDER_PASTE", "COMPONENT",
        "MASK", "CONDUCTIVE_PASTE",
    }

    #: matrix LAYER TYPEs treated as copper for z-stack purposes
    COPPER_TYPES: Set[str] = {"SIGNAL", "POWER_GROUND", "MIXED"}

    _VALID_PLACEMENTS = ("top", "center", "bottom")

    def __init__(self, source: Union[str, Path, ProductModel], step_name: Optional[str] = None,
                 thickness: Optional[Dict[str, float]] = None,
                 materials: Optional[Dict[str, str]] = None,
                 include_types: Optional[Set[str]] = None,
                 thick_traces: bool = True,
                 copper_placement: str = "bottom",
                 reverse: bool = False,
                 use_layer_thickness: bool = True,
                 strict_thickness: bool = True,
                 thickness_stack: Optional[List[Optional[float]]] = None,
                 max_angle_deg: Optional[float] = None):
        if isinstance(source, ProductModel):
            pm = source
        else:
            pm = ProductModel.from_path(source, max_angle_deg=max_angle_deg)
        self.pm = pm
        if step_name is None:
            if not pm.steps:
                raise ValueError("Product model has no steps to model")
            step_name = next(iter(pm.steps))
        if step_name not in pm.steps:
            raise KeyError(f"No such step: {step_name!r}")
        self.step_name = step_name
        self.step = pm.steps[step_name]

        if copper_placement not in self._VALID_PLACEMENTS:
            raise ValueError(
                f"copper_placement must be one of {self._VALID_PLACEMENTS}, got {copper_placement!r}")

        self.thickness = {**self.DEFAULT_THICKNESS, **(thickness or {})}
        self.materials = {**self.DEFAULT_MATERIAL, **(materials or {})}
        self.include_types = include_types or set(self.DEFAULT_GEO_TYPES)
        self.thick_traces = thick_traces
        self.copper_placement = copper_placement
        self.reverse = reverse
        self.use_layer_thickness = use_layer_thickness
        self.strict_thickness = strict_thickness
        self.thickness_stack = thickness_stack
        self.max_angle_deg = max_angle_deg

        #: features this designer couldn't turn into geometry (text/barcode,
        #: lines/arcs with a non-symmetric or unresolved symbol, etc.) —
        #: inspect this if polygons look like they're missing something.
        self.skipped_features: List[Feature] = []

        #: running X/Y translation applied to every coordinate this
        #: designer emits (board outline, layer polygons, drill holes).
        #: Z is never affected. Set via :meth:`centralize`.
        self._offset_x = 0.0
        self._offset_y = 0.0

        self._build_z_stack()

    # ---- board outline -----------------------------------------------------

    def get_board_polygon(self) -> Tuple[List[float], List[float]]:
        """The step's board outline as (xs, ys). Uses the first (outer)
        island of the step profile; see :meth:`get_board_holes` for cutouts."""
        island = self._board_island()
        return ([p.x + self._offset_x for p in island.boundary.points],
                [p.y + self._offset_y for p in island.boundary.points])

    def get_board_holes(self) -> List[Tuple[List[float], List[float]]]:
        """Cutouts/holes in the board outline, each as an (xs, ys) pair."""
        island = self._board_island()
        return [([p.x + self._offset_x for p in h.points],
                  [p.y + self._offset_y for p in h.points]) for h in island.holes]

    def centralize(self) -> None:
        """Shift all X/Y geometry (board outline, layer polygons, drill
        holes) so the board outline's bounding-box center sits at the
        origin (0, 0). Only X/Y are touched -- Z is untouched. Idempotent:
        calling it again after it's already centered is a no-op."""
        xs, ys = self.get_board_polygon()   # already includes any prior offset
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        self._offset_x -= cx
        self._offset_y -= cy

    def _board_island(self) -> Island:
        if self.step.profile is None or not self.step.profile.features:
            raise ValueError(f"Step {self.step_name!r} has no profile (board outline) data")
        surface_feature = self.step.profile.features[0]
        if not isinstance(surface_feature, SurfaceFeature) or not surface_feature.surface.islands:
            raise ValueError(f"Step {self.step_name!r} profile does not contain a surface")
        return surface_feature.surface.islands[0]

    # ---- Z stack -------------------------------------------------------------

    def _thickness_for(self, ml: MatrixLayer, index: int) -> Optional[float]:
        """Resolve the effective thickness (meters) for structural layer
        ``ml`` at position ``index`` in ``matrix.structural_layers()``.
        Priority: (1) ``self.thickness_stack[index]`` if given and not
        None, (2) real per-layer attrlist data if
        ``self.use_layer_thickness`` (see :class:`odbpp.layers.Layer`),
        (3) the nominal ``self.thickness[ml.type]`` default -- UNLESS
        ``self.strict_thickness`` is True, in which case this returns None
        instead of falling back, and the caller is responsible for turning
        any None into a clear error."""
        if self.thickness_stack is not None:
            override = self.thickness_stack[index]
            if override is not None:
                return override
        if self.use_layer_thickness:
            layer = self.step.get_layer(ml.name)
            if layer is not None:
                t = layer.dielectric_thickness if ml.type == "DIELECTRIC" else (
                    layer.copper_thickness if ml.type in self.COPPER_TYPES else None)
                if t is not None and t > 0:
                    return t
        if self.strict_thickness:
            return None
        return self.thickness.get(ml.type, 0.0)

    def _structural_layer_listing(self, structural: List[MatrixLayer]) -> str:
        return "\n".join(f"  [{i}] {ml.name} ({ml.type})" for i, ml in enumerate(structural))

    def _missing_thickness_error(self, structural: List[MatrixLayer],
                                  resolved: List[Optional[float]], missing_idx: List[int]) -> ValueError:
        n = len(structural)
        lines = [
            f"PCBView needs an explicit thickness for {len(missing_idx)} of {n} structural "
            "layer(s): they have no real per-layer material data (attrlist), and "
            "strict_thickness=True disables the nominal per-type fallback.",
            "",
            "You need to define a vector of thicknesses for these layers, in meters, one entry per "
            f"structural (copper/dielectric) layer ({n} entries total), in this exact top-to-bottom "
            "order (use None for a layer you don't need to override -- it'll use attrlist/nominal "
            "as normal):",
            "",
        ]
        for i, ml in enumerate(structural):
            if i in missing_idx:
                lines.append(f"  [{i}] {ml.name:20s} ({ml.type:12s}) <- MISSING, needs a value")
            else:
                lines.append(f"  [{i}] {ml.name:20s} ({ml.type:12s}) resolved: {resolved[i] * 1e6:.3f} um")
        lines += ["", "Example:", f"  thickness_stack = [None] * {n}"]
        for i in missing_idx:
            lines.append(f"  thickness_stack[{i}] = 35e-6   # {structural[i].name}")
        lines.append("  pcbd = odbpp.PCBView(pm, strict_thickness=True, thickness_stack=thickness_stack)")
        return ValueError("\n".join(lines))

    def _build_z_stack(self) -> None:
        """Build the copper/dielectric Z-stack. See the class docstring's
        ``thick_traces``/``copper_placement`` entries for the exact rules —
        in short: dielectrics alone define contiguous "seams" (the z
        accumulator only ever advances by DIELECTRIC thickness), and
        copper's own [z1, z2] is derived from the seam without ever moving
        it, so dielectric-to-dielectric contiguity is unconditional."""
        structural = self.pm.matrix.structural_layers()   # top -> bottom
        n = len(structural)

        if self.thickness_stack is not None and len(self.thickness_stack) != n:
            raise ValueError(
                f"thickness_stack has {len(self.thickness_stack)} entries but this board has {n} "
                "structural (copper/dielectric) layers. It must have exactly one entry per "
                "structural layer, in this exact top-to-bottom order:\n"
                + self._structural_layer_listing(structural)
            )

        resolved: List[Optional[float]] = [None] * n
        missing_idx: List[int] = []
        for i, ml in enumerate(structural):
            t = self._thickness_for(ml, i)
            resolved[i] = t
            if t is None:
                missing_idx.append(i)
        if missing_idx:
            raise self._missing_thickness_error(structural, resolved, missing_idx)

        # seams[i] = z immediately before structural[i]; only DIELECTRIC
        # entries advance it, so a copper layer's seams[i] == seams[i+1].
        seams = [0.0] * (n + 1)
        z = 0.0
        for i, ml in enumerate(structural):
            seams[i] = z
            if ml.type == "DIELECTRIC":
                z += resolved[i]
        seams[n] = z

        ranges: Dict[str, Tuple[float, float]] = {}
        info: List[Tuple[MatrixLayer, float, float, str]] = []
        for i, ml in enumerate(structural):
            if ml.type == "DIELECTRIC":
                z1, z2 = seams[i], seams[i + 1]
            else:
                z1, z2 = self._copper_z_range(structural, i, seams[i], resolved[i])
            ranges[ml.name] = (z1, z2)
            info.append((ml, z1, z2, self._material_for(ml)))

        self._structural = structural
        self._structural_ranges = ranges
        self._pcb_layers = info
        self._z_top = 0.0
        self._z_bottom = z
        self._structural_rows = [ml.row for ml in structural]

        logger.debug(
            f"PCBView: step '{self.step_name}' -- {n} structural layer(s), "
            f"total thickness {z * 1e6:.1f}um"
        )

    def _copper_z_range(self, structural: List[MatrixLayer], i: int, seam: float, t_cu: float) -> Tuple[float, float]:
        ml = structural[i]
        if not self.thick_traces or t_cu <= 0:
            return (seam, seam)

        n = len(structural)
        has_prev_diel = i > 0 and structural[i - 1].type == "DIELECTRIC"
        has_next_diel = i + 1 < n and structural[i + 1].type == "DIELECTRIC"

        placement = self.copper_placement
        if placement == "top" and not has_prev_diel:
            placement = "bottom" if has_next_diel else "flat"
        elif placement == "bottom" and not has_next_diel:
            placement = "top" if has_prev_diel else "flat"
        elif placement == "center" and not (has_prev_diel and has_next_diel):
            placement = "top" if has_prev_diel else ("bottom" if has_next_diel else "flat")

        if placement == "top":
            return (seam - t_cu, seam)
        if placement == "bottom":
            return (seam, seam + t_cu)
        if placement == "center":
            return (seam - t_cu / 2.0, seam + t_cu / 2.0)
        return (seam, seam)   # "flat" fallback: no dielectric neighbor on either side

    def _flip_z_range(self, z1: float, z2: float) -> Tuple[float, float]:
        """Mirror a (z1, z2) pair about the board's total thickness if
        ``self.reverse`` is set (z -> total_thickness - z, with z1/z2
        swapped so the result is still ordered z1 <= z2); passthrough
        otherwise. This is the single point where the Z-flip is actually
        applied — see the ``reverse`` parameter docs on the class."""
        if not self.reverse:
            return (z1, z2)
        total = self._z_bottom
        return (total - z2, total - z1)

    def _material_for(self, ml: MatrixLayer) -> str:
        if ml.type == "DIELECTRIC" and ml.dielectric_name:
            return ml.dielectric_name
        return self.materials.get(ml.type, ml.type)

    def iter_pcb_layers(self) -> Iterator[Tuple[float, float, str]]:
        """Yield ``(z1, z2, material)`` for each structural (copper/
        dielectric) layer, top to bottom (i.e. increasing Z, honoring
        ``self.reverse`` if set), in meters."""
        items = reversed(self._pcb_layers) if self.reverse else self._pcb_layers
        for ml, z1, z2, material in items:
            z1, z2 = self._flip_z_range(z1, z2)
            yield z1, z2, material

    def _z_range_for(self, ml: MatrixLayer) -> Tuple[float, float]:
        """Z (as a zero-thickness plane z1==z2) for a non-structural layer
        (mask/silk/paste/component/...). Prefers the copper layer it's
        REF-linked to (see spec p.67); otherwise falls back to the top or
        bottom surface of the board depending on where it sits in the
        matrix row order. This is a reasonable heuristic, not exact physics
        — override via a custom subclass if you need more precision."""
        if ml.name in self._structural_ranges:
            return self._structural_ranges[ml.name]
        if ml.ref is not None:
            ref_layer = next((m for m in self.pm.matrix.layers if m.id == ml.ref), None)
            if ref_layer is not None and ref_layer.name in self._structural_ranges:
                r1, r2 = self._structural_ranges[ref_layer.name]
                z = r1 if ml.row < ref_layer.row else r2
                return (z, z)
        if not self._structural_rows or ml.row < self._structural_rows[0]:
            return (self._z_top, self._z_top)
        return (self._z_bottom, self._z_bottom)

    # ---- geometry layers -------------------------------------------------

    def iter_geo_layers(self) -> Iterator[GeoLayer]:
        """Yield one :class:`GeoLayer` per matrix layer whose TYPE is in
        ``self.include_types`` and that has a parsed ``features`` file, top
        to bottom (i.e. increasing Z, honoring ``self.reverse`` if set)."""
        matrix_layers = self.pm.matrix.layers_top_to_bottom()
        if self.reverse:
            matrix_layers = list(reversed(matrix_layers))
        for ml in matrix_layers:
            if ml.type not in self.include_types:
                continue
            layer = self.step.get_layer(ml.name)
            if layer is None or layer.features is None:
                continue
            z1, z2 = self._flip_z_range(*self._z_range_for(ml))
            logger.debug(f"iter_geo_layers: '{ml.name}' ({ml.type}) z=[{z1 * 1e6:.1f}, {z2 * 1e6:.1f}]um")
            yield GeoLayer(name=ml.name, type=ml.type, z1=z1, z2=z2, matrix_layer=ml, _designer=self)

    def _layer_polygons(self, geo_layer: GeoLayer) -> Iterator[LayerPolygon]:
        layer = self.step.get_layer(geo_layer.name)
        if layer is None or layer.features is None:
            return
        z = geo_layer.z
        ox, oy = self._offset_x, self._offset_y
        for feature in layer.features.features:
            for pts, holes, polarity, source in self._feature_polygons(feature):
                if ox or oy:
                    pts = [(x + ox, y + oy) for x, y in pts]
                    holes = [[(x + ox, y + oy) for x, y in h] for h in holes]
                yield LayerPolygon(points=pts, holes=holes, z=z, polarity=polarity,
                                    layer_name=geo_layer.name, source=source)

    # ---- feature -> polygon resolution ------------------------------------

    def _feature_polygons(self, feature: Feature):
        """Yield ``(points, holes, polarity, source_feature)`` tuples for one
        top-level feature, in world-space meters."""
        if isinstance(feature, SurfaceFeature):
            for island in feature.surface.islands:
                yield (_pts(island.boundary.points), [_pts(h.points) for h in island.holes],
                       feature.polarity, feature)
            return

        if isinstance(feature, Stamp):
            if feature.symbol is None:
                self.skipped_features.append(feature)
                return
            if isinstance(feature.symbol, UserDefinedSymbol):
                for sub_feat, geom in self.pm.resolve_stamp(feature, self.max_angle_deg):
                    yield from self._resolved_geom_polygons(geom, sub_feat)
                return
            island = self._stamp_island(feature)
            yield (_pts(island.boundary.points), [_pts(h.points) for h in island.holes],
                   feature.polarity, feature)
            return

        if isinstance(feature, Line):
            width = _symbol_width(feature.symbol)
            if width is None:
                self.skipped_features.append(feature)
                return
            poly = buffer_segment((feature.start.x, feature.start.y),
                                   (feature.end.x, feature.end.y), width, self.max_angle_deg)
            yield (poly, [], feature.polarity, feature)
            return

        if isinstance(feature, Arc):
            width = _symbol_width(feature.symbol)
            if width is None:
                self.skipped_features.append(feature)
                return
            polyline = feature.to_polyline(max_angle_deg=self.max_angle_deg)
            for p0, p1 in zip(polyline[:-1], polyline[1:]):
                seg = buffer_segment((p0.x, p0.y), (p1.x, p1.y), width, self.max_angle_deg)
                yield (seg, [], feature.polarity, feature)
            return

        if isinstance(feature, (TextFeature, BarcodeFeature)):
            # No fillable geometry modeled for text/barcodes (see package docs).
            self.skipped_features.append(feature)
            return

    def _resolved_geom_polygons(self, geom, sub_feature: Feature):
        if isinstance(geom, Island):
            yield (_pts(geom.boundary.points), [_pts(h.points) for h in geom.holes],
                   sub_feature.polarity, sub_feature)
            return
        if isinstance(geom, Surface):
            for island in geom.islands:
                yield (_pts(island.boundary.points), [_pts(h.points) for h in island.holes],
                       sub_feature.polarity, sub_feature)
            return
        if isinstance(geom, list):   # a resolved Line/Arc polyline (list[Point])
            width = _symbol_width(getattr(sub_feature, "symbol", None))
            if width is None or len(geom) < 2:
                self.skipped_features.append(sub_feature)
                return
            for p0, p1 in zip(geom[:-1], geom[1:]):
                seg = buffer_segment((p0.x, p0.y), (p1.x, p1.y), width, self.max_angle_deg)
                yield (seg, [], sub_feature.polarity, sub_feature)
            return
        self.skipped_features.append(sub_feature)

    def _stamp_island(self, stamp: Stamp) -> Island:
        symbol = stamp.symbol
        if stamp.resize_factor:
            symbol = _resized_symbol(symbol, stamp.resize_factor)
        island = symbol.to_island(self.max_angle_deg).rotated(stamp.rotation_deg)
        if stamp.mirror:
            island = island.mirrored_x()
        return island.translated(stamp.position.x, stamp.position.y)

    # ---- drill holes (bonus: not required by the polygon-only API above) ---

    def iter_drill_holes(self) -> Iterator[DrillHole]:
        """Yield resolved through/blind/buried holes from every DRILL-type
        matrix layer, with the Z span resolved against the copper stack."""
        for ml in self.pm.matrix.layers_top_to_bottom():
            if ml.type != "DRILL":
                continue
            layer = self.step.get_layer(ml.name)
            if layer is None or layer.features is None:
                continue
            logger.debug(f"iter_drill_holes: resolving drill layer '{ml.name}' ({len(layer.features.features)} feature(s))")
            start_name = ml.start_name or (self._structural[0].name if self._structural else None)
            end_name = ml.end_name or (self._structural[-1].name if self._structural else None)
            z1 = self._structural_ranges.get(start_name, (self._z_top, self._z_top))[0]
            z2 = self._structural_ranges.get(end_name, (self._z_bottom, self._z_bottom))[1]
            z1, z2 = self._flip_z_range(z1, z2)

            tools = layer.tools.tools if layer.tools else {}
            for feature in layer.features.features:
                if not isinstance(feature, Stamp):
                    continue
                diameter = None
                plated = True
                if isinstance(feature.symbol, RoundSymbol):
                    diameter = feature.symbol.diameter
                if feature.dcode in tools:
                    tool = tools[feature.dcode]
                    if tool.finish_size:
                        diameter = tool.finish_size
                    plated = tool.type != "NON_PLATED"
                if diameter is None:
                    self.skipped_features.append(feature)
                    continue
                yield DrillHole(x=feature.position.x + self._offset_x, y=feature.position.y + self._offset_y,
                                 diameter=diameter,
                                 z1=z1, z2=z2, plated=plated, layer_name=ml.name)

    # ---- boolean-unified polygons (reaches into emcad's own kernel) --------

    def resolve_layer_polygons(
        self,
        layer: "GeoLayer",
        simplify_delta: float = 1e-6,
        dezigzag: bool = True,
        dezigzag_max_kink_length: Optional[float] = None,
        dezigzag_max_angle_deg: float = 20.0,
        dezigzag_min_neighbor_factor: float = 3.0,
        post_simplify: bool = True,
        post_simplify_delta: Optional[float] = None,
        merge_tol: Optional[float] = None,
        regularize: bool = False,
        regularize_tol: Optional[float] = None,
        regularize_dangle_deg: float = 5.0,
    ) -> List["cad.Polygon"]:
        """See the module-level `resolve_layer_polygons` function --
        this is a thin method wrapper so existing call sites keep
        working. Pulled out to module level so `cache.CachedPCBView`
        (a `PCBView`-compatible stand-in loaded from a saved cache,
        see `odbpp/cache.py`) can share the exact same boolean-unify
        recipe instead of duplicating it.
        """
        return resolve_layer_polygons(
            layer, simplify_delta, dezigzag, dezigzag_max_kink_length,
            dezigzag_max_angle_deg, dezigzag_min_neighbor_factor,
            post_simplify, post_simplify_delta, merge_tol,
            regularize, regularize_tol, regularize_dangle_deg,
        )

    # ---- plotting (bonus: a quick visual sanity check) ---------------------

    def plot_layers(
        self,
        board_color: str = "seagreen",
        copper_color: str = "#B87333",
        figsize: Tuple[float, float] = (8.0, 8.0),
        layer_names: Optional[Iterable[str]] = None,
        simplify_delta: float = 1e-6,
    ) -> None:
        """See the module-level `plot_layers_for` function -- this is a
        thin method wrapper, pulled out for the same sharing reason as
        `resolve_layer_polygons` above.
        """
        plot_layers_for(self, board_color, copper_color, figsize, layer_names, simplify_delta)


# --------------------------------------------------------------------------
# Module-level, designer-agnostic helpers -- these only ever call the
# PUBLIC methods on whatever `designer` they're given (`get_board_polygon`,
# `get_board_holes`, `iter_geo_layers`, `resolve_layer_polygons` /
# `layer.iter_polygons()`), so they work identically whether `designer`
# is a real `PCBView` (backed by a freshly-parsed ProductModel) or a
# `cache.CachedPCBView` (backed by a saved snapshot, skipping ODB++
# parsing and feature resolution entirely) -- see `odbpp/cache.py`.
# --------------------------------------------------------------------------

#: Default tolerance for `dezigzag_max_kink_length`/`post_simplify_delta`
#: when left as `None` -- deliberately NOT tied to `simplify_delta` (see
#: `resolve_layer_polygons`'s docstring for why those two scales must
#: stay independent).
_DEFAULT_POST_UNION_TOLERANCE = 10e-6

#: Default `regularize_tol` (meters): the largest protrusion
#: `Polygon.regularize` flattens back onto a straight edge -- sized for
#: via pads poking ~0.1-0.2 mm out of a copper edge. Must stay below the
#: board's smallest real feature (slot/gap width) you want to keep.
_DEFAULT_REGULARIZE_TOL = 0.25e-3


def resolve_layer_polygons(
    layer: "GeoLayer",
    simplify_delta: float = 1e-6,
    dezigzag: bool = True,
    dezigzag_max_kink_length: Optional[float] = None,
    dezigzag_max_angle_deg: float = 20.0,
    dezigzag_min_neighbor_factor: float = 3.0,
    post_simplify: bool = True,
    post_simplify_delta: Optional[float] = None,
    merge_tol: Optional[float] = None,
    regularize: bool = False,
    regularize_tol: Optional[float] = None,
    regularize_dangle_deg: float = 5.0,
) -> List["cad.Polygon"]:
    """Resolve one geometry layer's raw features into the SAME
    boolean-unified polygons `emerge_interface.ODBImport.
    generate_traces()` builds CAD solids from: positive-polarity
    ("P") features unioned together, negative-polarity ("N")
    features unioned together, and the result of add-minus-remove
    via `emcad`'s own 2D kernel (`subtract_polygons` computes
    "union(add) minus union(remove)" in one pass -- no need to
    union each side separately first).

    This is the one place in `odbpp` that reaches outside the
    ODB++-parsing-only boundary the rest of this package keeps to
    (see the package docstring) -- specifically so this recipe
    lives in exactly one place, used by `plot_layers_for` below,
    `generate_traces()`, AND `cache.CachedPCBView`, instead of
    being duplicated (and silently drifting out of sync) wherever
    it's needed. The import is deferred for that same reason: it
    doesn't cost anything unless you actually call this.

    IMPORTANT -- why `simplify_delta` defaults small (1um), and why it
    is NOT reused as the default for `dezigzag_max_kink_length`/
    `post_simplify_delta` the way it used to be: `simplify_delta` runs
    on each RAW feature INDEPENDENTLY, before the union -- two features
    that were meant to share an exact boundary can each get RDP-nudged
    a little differently, opening a gap between them up to roughly
    `2 * simplify_delta` wide. The union's own vertex-merge tolerance
    (`merge_tol`, exact-grid-snapping in the arrangement engine -- see
    `kernel/_exact.py`) then has to be at least that big to re-snap them
    back together, or the two pieces come out as separate,
    disconnected polygons with a hairline gap between them instead of
    one fused shape -- exactly the failure mode that motivated this
    default change (a dense, touching trace/divider network coming out
    as dozens of disconnected fragments and zero-area slivers instead
    of one connected polygon). Bumping `merge_tol` up to compensate is
    NOT a reliable fix on its own -- it trades "gaps where things
    should touch" for "spurious merges where things shouldn't", and
    empirically that tradeoff is NOT monotonic in `merge_tol` (testing
    against a real board found nearby multiples of `simplify_delta`
    giving wildly different fragment counts). The robust fix is instead
    to keep the PRE-union tolerance small (this default) and do the
    real, aggressive simplification AFTER the union via `post_simplify`/
    `dezigzag`, which operate on the already-correctly-fused topology
    and can't introduce this kind of independent drift.

    Args:
        simplify_delta: RDP simplification tolerance for every raw
            feature polygon before it's unioned (see
            `kernel.api.simplify_polyline`). Keep this small (well under
            your board's smallest real feature-to-feature clearance) --
            it exists to strip degenerate/tessellation-noise points and
            reduce the union's own input size for performance, not to
            do meaningful shape smoothing; that's `post_simplify`'s job.
        dezigzag: if True (the default), run `Polygon.dezigzag()` on
            every OUTPUT polygon (recursing into its holes) before
            returning -- cleans up the short zigzag/step artifacts a
            union of many round-capped trace segments tends to leave
            behind (a tessellated circle's polygon approximation
            slightly missing the tangent point where it should cleanly
            meet a straight segment). See `kernel.api.dezigzag_polyline`
            for the exact algorithm.
        dezigzag_max_kink_length: forwarded to `Polygon.dezigzag()`.
            `None` (the default) uses `_DEFAULT_POST_UNION_TOLERANCE`
            (10um) -- deliberately independent of `simplify_delta`, see
            above.
        dezigzag_max_angle_deg, dezigzag_min_neighbor_factor: forwarded
            to `Polygon.dezigzag()` as-is.
        post_simplify: if True (the default), run `Polygon.simplify()`
            on every OUTPUT polygon (recursing into its holes) AFTER
            dezigzag -- this is where the REAL point-count reduction
            happens, safely, since it runs on the union's own output
            (already correctly fused) rather than independently on
            each raw input feature.
        post_simplify_delta: RDP tolerance for that pass. `None` (the
            default) uses `_DEFAULT_POST_UNION_TOLERANCE` (10um).
        merge_tol: exact-grid vertex-merge tolerance for the union
            itself (forwarded to `subtract_polygons`). `None` (the
            default) uses the kernel's own default
            (`kernel._constants.DEFAULT_MERGE_TOL`, 0.1um). Only worth
            raising as a targeted escape hatch for a specific board
            that still shows gaps with the defaults above -- per the
            note above, this is a blunter, less reliable knob than
            keeping `simplify_delta` small in the first place.

        regularize: if True, run `Polygon.regularize()` on every
            output polygon (recursing into holes) after dezigzag and
            before `post_simplify` -- map-making style outline
            regularization that snaps edges back onto their dominant
            straight lines and drops small protrusions such as via pads
            sticking out of a copper edge, without the edge skew a large
            `post_simplify_delta` causes. Off by default.
        regularize_tol: largest protrusion (meters) flattened; `None`
            uses `_DEFAULT_REGULARIZE_TOL` (0.25 mm).
        regularize_dangle_deg: snap-angle step for straight edges.

    Returns:
        List of `emcad.Polygon` -- the final, fully-resolved shapes
        for this layer, each possibly carrying `.holes` (at any
        nesting depth).
    """
    import time

    import emcad as cad

    to_add: List["cad.Polygon"] = []
    to_remove: List["cad.Polygon"] = []
    for lp in layer.iter_polygons():
        xs, ys = cad.simplify_polyline(*lp.xys(), simplify_delta)
        polygon = cad.Polygon(xs, ys)
        polygon.remove_keyholes()
        (to_add if lp.polarity == "P" else to_remove).append(polygon)

    if not to_add:
        return []
    t0 = time.time()
    subtract_kwargs = {} if merge_tol is None else {"merge_tol": merge_tol}
    result = cad.subtract_polygons(tuple(to_add), tuple(to_remove), **subtract_kwargs)
    t1 = time.time()
    logger.debug(
        f"resolve_layer_polygons: '{layer.name}' -- {len(to_add)} add + {len(to_remove)} remove "
        f"-> {len(result)} polygon(s) [{t1 - t0:.3f}s]"
    )

    if dezigzag:
        kink_length = (
            dezigzag_max_kink_length if dezigzag_max_kink_length is not None else _DEFAULT_POST_UNION_TOLERANCE
        )
        for poly in result:
            poly.dezigzag(kink_length, dezigzag_max_angle_deg, dezigzag_min_neighbor_factor)

    if regularize:
        tol = regularize_tol if regularize_tol is not None else _DEFAULT_REGULARIZE_TOL
        for poly in result:
            poly.regularize(tol, regularize_dangle_deg)

    if post_simplify:
        delta = post_simplify_delta if post_simplify_delta is not None else _DEFAULT_POST_UNION_TOLERANCE
        for poly in result:
            poly.simplify(delta)

    return result


def plot_layers_for(
    designer,
    board_color: str = "seagreen",
    copper_color: str = "#B87333",
    figsize: Tuple[float, float] = (8.0, 8.0),
    layer_names: Optional[Iterable[str]] = None,
    simplify_delta: float = 1e-6,
) -> None:
    """Cycle through `designer`'s geometry layers (`iter_geo_layers`),
    popping up one matplotlib figure per layer: the board outline as
    translucent context underneath, that layer's own boolean-unified
    polygons on top in `copper_color` (see `resolve_layer_polygons`
    -- these are the SAME final shapes `generate_traces()` builds
    CAD solids from, not the raw per-feature input polygons: pads,
    traces, and their antipads/cutouts are already merged and
    subtracted, so what you see here is what actually goes into the
    simulation). Each figure blocks on `plt.show()` until closed, so
    closing one window advances to the next layer.

    Args:
        designer: a `PCBView` or `cache.CachedPCBView`.
        board_color: fill color for the board outline shown as
            context on every layer.
        copper_color: fill color for each layer's unified polygons.
        figsize: matplotlib figure size, one figure per layer.
        layer_names: optional iterable restricting which layers to
            show (matched against `GeoLayer.name`) -- default is
            every layer `iter_geo_layers()` yields.
        simplify_delta: forwarded to `resolve_layer_polygons`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    def compound_path(outer: Tuple[List[float], List[float]],
                       holes: List[Tuple[List[float], List[float]]]) -> Path:
        vertices: List[XY] = []
        codes: List[int] = []

        def add_ring(xs: List[float], ys: List[float]) -> None:
            if not xs:
                return
            vertices.append((xs[0], ys[0]))
            codes.append(Path.MOVETO)
            for x, y in zip(xs[1:], ys[1:]):
                vertices.append((x, y))
                codes.append(Path.LINETO)
            vertices.append((xs[0], ys[0]))
            codes.append(Path.CLOSEPOLY)

        add_ring(*outer)
        for h in holes:
            add_ring(*h)
        return Path(vertices, codes)

    def polygon_rings(polygon) -> List[Tuple[List[float], List[float]]]:
        """Every ring of an emcad.Polygon, at every nesting depth
        (a hole's own nested holes are islands, and so on) -- even-odd
        fill handles arbitrary depth correctly as long as every ring
        is in the same compound path."""
        rings = [(list(polygon.xs), list(polygon.ys))]
        for hole in polygon.holes:
            rings.extend(polygon_rings(hole))
        return rings

    board_path = compound_path(designer.get_board_polygon(), designer.get_board_holes())
    names = set(layer_names) if layer_names is not None else None

    for layer in designer.iter_geo_layers():
        if names is not None and layer.name not in names:
            continue

        polies = designer.resolve_layer_polygons(layer, simplify_delta)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.set_title(f"{layer.name} ({layer.type})")
        ax.add_patch(PathPatch(board_path, facecolor=board_color, edgecolor="none", alpha=0.4, zorder=0))

        for poly in polies:
            rings = polygon_rings(poly)
            path = compound_path(rings[0], rings[1:])
            ax.add_patch(PathPatch(path, facecolor=copper_color, edgecolor="black", linewidth=0.2, alpha=0.9, zorder=1))

        ax.relim()
        ax.autoscale_view()
        plt.show()