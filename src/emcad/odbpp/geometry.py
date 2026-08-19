"""SI-normalized geometric primitives shared across the ODB++ object model.

Every coordinate in this module is in **meters**. These classes are plain
geometry containers — they know nothing about ODB++ file syntax; that
lives in :mod:`odbpp.symbols` and :mod:`odbpp.features`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, degrees, hypot, pi, radians, sin
from typing import List, Optional, Tuple

from .config import TESSELLATION, segments_for_sweep


@dataclass(frozen=True)
class Point:
    """A single 2D point, in meters."""

    x: float
    y: float

    def translated(self, dx: float, dy: float) -> "Point":
        return Point(self.x + dx, self.y + dy)

    def rotated(self, angle_deg: float, about: Optional["Point"] = None) -> "Point":
        """Rotate clockwise by ``angle_deg`` (matching ODB++'s rotation convention)."""
        about = about or Point(0.0, 0.0)
        a = radians(-angle_deg)  # negative => clockwise in a standard math (CCW-positive) frame
        dx, dy = self.x - about.x, self.y - about.y
        x = dx * cos(a) - dy * sin(a)
        y = dx * sin(a) + dy * cos(a)
        return Point(x + about.x, y + about.y)

    def mirrored_x(self, about_x: float = 0.0) -> "Point":
        """Mirror across a vertical line at ``about_x`` (ODB++ mirroring is about the X axis)."""
        return Point(2 * about_x - self.x, self.y)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Polygon:
    """A single closed contour (list of vertices), in meters.

    No winding order is enforced by this class itself; ODB++ convention
    (see the spec's "Surfaces" section) is that island boundaries wind
    clockwise and hole boundaries wind counter-clockwise when the polygon
    is used inside a :class:`Island`.
    """

    points: List[Point] = field(default_factory=list)

    def translated(self, dx: float, dy: float) -> "Polygon":
        return Polygon([p.translated(dx, dy) for p in self.points])

    def rotated(self, angle_deg: float, about: Optional[Point] = None) -> "Polygon":
        return Polygon([p.rotated(angle_deg, about) for p in self.points])

    def mirrored_x(self, about_x: float = 0.0) -> "Polygon":
        return Polygon([p.mirrored_x(about_x) for p in self.points])

    def as_tuples(self) -> List[Tuple[float, float]]:
        return [p.as_tuple() for p in self.points]

    def __iter__(self):
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class Island:
    """A surface island: one outer boundary plus zero or more holes.

    This is the ODB++ "contour" concept (spec: Surfaces / Geometric
    Entities) — the natural container for anything that isn't a single
    convex shape: donuts, thermal reliefs, board outlines with cutouts,
    copper pours with anti-pads, etc.
    """

    boundary: Polygon
    holes: List[Polygon] = field(default_factory=list)

    def translated(self, dx: float, dy: float) -> "Island":
        return Island(self.boundary.translated(dx, dy), [h.translated(dx, dy) for h in self.holes])

    def rotated(self, angle_deg: float, about: Optional[Point] = None) -> "Island":
        return Island(self.boundary.rotated(angle_deg, about), [h.rotated(angle_deg, about) for h in self.holes])

    def mirrored_x(self, about_x: float = 0.0) -> "Island":
        return Island(self.boundary.mirrored_x(about_x), [h.mirrored_x(about_x) for h in self.holes])


@dataclass
class Surface:
    """A full ODB++ surface feature: an ordered list of islands (with their holes)."""

    islands: List[Island] = field(default_factory=list)

    def translated(self, dx: float, dy: float) -> "Surface":
        return Surface([isl.translated(dx, dy) for isl in self.islands])

    def rotated(self, angle_deg: float, about: Optional[Point] = None) -> "Surface":
        return Surface([isl.rotated(angle_deg, about) for isl in self.islands])

    def mirrored_x(self, about_x: float = 0.0) -> "Surface":
        return Surface([isl.mirrored_x(about_x) for isl in self.islands])


# --------------------------------------------------------------------------
# Shape-construction helpers.
# All return a Polygon (or Island, where the shape has an inherent hole)
# centered on the origin, in meters. Used by odbpp.symbols to render
# standard symbol names into concrete geometry.
#
# Every curve here is discretized using the shared angle-step budget from
# odbpp.config (see that module's docstring) — pass max_angle_deg=None
# (the default) to use whatever odbpp.config.TESSELLATION.max_angle_deg is
# currently set to, or pass an explicit value to override it just for this
# call.
# --------------------------------------------------------------------------

def circle(radius: float, max_angle_deg: Optional[float] = None) -> Polygon:
    segments = segments_for_sweep(360, max_angle_deg, minimum=TESSELLATION.min_segments)
    return Polygon([
        Point(radius * cos(2 * pi * i / segments), radius * sin(2 * pi * i / segments))
        for i in range(segments)
    ])


def rectangle(width: float, height: float) -> Polygon:
    w, h = width / 2.0, height / 2.0
    return Polygon([Point(w, h), Point(-w, h), Point(-w, -h), Point(w, -h)])


def regular_polygon(radius: float, sides: int, rotation_deg: float = 0.0) -> Polygon:
    a0 = radians(rotation_deg)
    return Polygon([
        Point(radius * cos(a0 + 2 * pi * i / sides), radius * sin(a0 + 2 * pi * i / sides))
        for i in range(sides)
    ])


# Corner numbering per the ODB++ spec: ascending, counter-clockwise, starting
# at the top-right corner. 1=top-right, 2=top-left, 3=bottom-left, 4=bottom-right.
_CORNER_ORDER = (1, 2, 3, 4)


def rounded_rect(width: float, height: float, radius: float,
                  corners: Optional[set] = None, max_angle_deg: Optional[float] = None) -> Polygon:
    corners = corners if corners is not None else {1, 2, 3, 4}
    segments_per_corner = segments_for_sweep(90, max_angle_deg)
    hw, hh = width / 2.0, height / 2.0
    arc_defs = {
        1: (hw - radius, hh - radius, 0, 90),
        2: (-hw + radius, hh - radius, 90, 180),
        3: (-hw + radius, -hh + radius, 180, 270),
        4: (hw - radius, -hh + radius, 270, 360),
    }
    sharp_pts = {1: Point(hw, hh), 2: Point(-hw, hh), 3: Point(-hw, -hh), 4: Point(hw, -hh)}
    pts: List[Point] = []
    for c in _CORNER_ORDER:
        if c in corners and radius > 0:
            cx, cy, a0, a1 = arc_defs[c]
            for i in range(segments_per_corner + 1):
                a = radians(a0 + (a1 - a0) * i / segments_per_corner)
                pts.append(Point(cx + radius * cos(a), cy + radius * sin(a)))
        else:
            pts.append(sharp_pts[c])
    return Polygon(pts)


def chamfered_rect(width: float, height: float, chamfer: float,
                    corners: Optional[set] = None) -> Polygon:
    corners = corners if corners is not None else {1, 2, 3, 4}
    hw, hh = width / 2.0, height / 2.0
    sharp = {1: [Point(hw, hh)], 2: [Point(-hw, hh)], 3: [Point(-hw, -hh)], 4: [Point(hw, -hh)]}
    chamfered = {
        1: [Point(hw - chamfer, hh), Point(hw, hh - chamfer)],
        2: [Point(-hw + chamfer, hh), Point(-hw, hh - chamfer)],
        3: [Point(-hw, -hh + chamfer), Point(-hw + chamfer, -hh)],
        4: [Point(hw, -hh + chamfer), Point(hw - chamfer, -hh)],
    }
    pts: List[Point] = []
    for c in _CORNER_ORDER:
        if c in corners and chamfer > 0:
            pts.extend(chamfered[c])
        else:
            pts.extend(sharp[c])
    return Polygon(pts)


def diamond(width: float, height: float) -> Polygon:
    w, h = width / 2.0, height / 2.0
    return Polygon([Point(0, h), Point(-w, 0), Point(0, -h), Point(w, 0)])


def stadium(width: float, height: float, max_angle_deg: Optional[float] = None) -> Polygon:
    """Oval / obround: a rectangle fully rounded on its two short ends."""
    radius = min(width, height) / 2.0
    return rounded_rect(width, height, radius, corners={1, 2, 3, 4}, max_angle_deg=max_angle_deg)


def half_stadium(width: float, height: float, max_angle_deg: Optional[float] = None) -> Polygon:
    """Half oval: approximated as a 'D' shape (flat bottom, rounded top)."""
    radius = min(width, height) / 2.0
    return rounded_rect(width, height, radius, corners={1, 2}, max_angle_deg=max_angle_deg)


def hexagon(width: float, height: float, corner: float, orientation: str = "horizontal") -> Polygon:
    """Hexagon inscribed in a width x height box, with `corner` sized end-chamfers.

    orientation='horizontal' chamfers the left/right tips; 'vertical' chamfers
    the top/bottom tips.
    """
    w, h = width / 2.0, height / 2.0
    if orientation == "horizontal":
        pts = [Point(w - corner, h), Point(w, 0), Point(w - corner, -h),
                Point(-w + corner, -h), Point(-w, 0), Point(-w + corner, h)]
    else:
        pts = [Point(w, h - corner), Point(0, h), Point(-w, h - corner),
                Point(-w, -h + corner), Point(0, -h), Point(w, -h + corner)]
    return Polygon(pts)


def triangle(base: float, height: float) -> Polygon:
    return Polygon([Point(0, 2 * height / 3), Point(-base / 2, -height / 3), Point(base / 2, -height / 3)])


def butterfly(size: float, shape: str = "round") -> Polygon:
    """Bowtie / butterfly shape (approximate): two triangles touching at the origin."""
    r = size / 2.0
    return Polygon([
        Point(-r, r), Point(-r, -r), Point(0, 0),
        Point(r, -r), Point(r, r), Point(0, 0),
    ])


def octagon(width: float, height: float, corner: float) -> Polygon:
    return chamfered_rect(width, height, corner, corners={1, 2, 3, 4})


def donut(outer: Polygon, inner: Polygon) -> Island:
    return Island(boundary=outer, holes=[inner])


def thermal(outer: float, inner: float, angle_deg: float, num_spokes: int, gap: float,
            shape: str = "round", max_angle_deg: Optional[float] = None) -> Island:
    """Thermal-relief pad: an annulus (or square-annulus) with radial spoke gaps cut out.

    `outer`/`inner` are diameters for shape='round', side lengths for shape='square'.
    The result is expressed as an Island: the annulus boundary, with the inner
    clearance hole plus one narrow rectangular "slit" hole per spoke gap — this
    matches ODB++'s own island/hole modeling instead of needing boolean geometry.
    """
    if shape == "round":
        boundary = circle(outer / 2.0, max_angle_deg)
        inner_hole = circle(inner / 2.0, max_angle_deg)
    else:
        boundary = rectangle(outer, outer)
        inner_hole = rectangle(inner, inner)

    holes = [inner_hole]
    if num_spokes > 0 and gap > 0:
        r_in = inner / 2.0 * 0.9   # start slightly inside the inner hole
        r_out = outer / 2.0 * 1.1  # end slightly outside the boundary
        slit_len = r_out - r_in
        slit_mid = (r_out + r_in) / 2.0
        spoke_step = 360.0 / num_spokes
        for k in range(num_spokes):
            theta = angle_deg + k * spoke_step
            slit = rectangle(slit_len, gap).translated(slit_mid, 0.0).rotated(theta)
            holes.append(slit)
    return Island(boundary=boundary, holes=holes)


def tessellate_arc(start: Point, end: Point, center: Point, clockwise: bool,
                    max_angle_deg: Optional[float] = None) -> List[Point]:
    """Approximate an ODB++ arc (start/end/center/direction) as line points.

    Resolution is **angle-based**, not a fixed point count: the arc is split
    into as many equal segments as needed so that no single segment sweeps
    more than ``max_angle_deg`` degrees (falls back to the shared
    :mod:`odbpp.config` default if not given). A tight fillet gets few
    points; a long, sweeping curve (or a full 360 degree arc) automatically
    gets more — so fidelity scales with how much the arc actually curves,
    not with an arbitrary fixed count.

    Returns the intermediate + end points (NOT including ``start``, which the
    caller already has). Handles the ODB++ "start == end means a 360 degree
    arc" convention.
    """
    r0 = hypot(start.x - center.x, start.y - center.y)
    a0 = atan2(start.y - center.y, start.x - center.x)
    a1 = atan2(end.y - center.y, end.x - center.x)
    full_circle = abs(start.x - end.x) < 1e-12 and abs(start.y - end.y) < 1e-12

    if clockwise:
        while a1 > a0:
            a1 -= 2 * pi
        if full_circle:
            a1 = a0 - 2 * pi
    else:
        while a1 < a0:
            a1 += 2 * pi
        if full_circle:
            a1 = a0 + 2 * pi

    sweep_deg = abs(degrees(a1 - a0))
    segments = segments_for_sweep(sweep_deg, max_angle_deg)

    pts = []
    for i in range(1, segments + 1):
        a = a0 + (a1 - a0) * i / segments
        pts.append(Point(center.x + r0 * cos(a), center.y + r0 * sin(a)))
    return pts
