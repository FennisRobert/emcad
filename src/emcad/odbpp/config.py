"""Central configuration for geometry discretization (tessellation).

Previously, different parts of this package picked curve resolution two
different, uncoordinated ways: circles/donuts/thermal reliefs used a fixed
**segment count** (32, i.e. ~11.25 degrees/segment for a full circle),
while outline/profile arcs used a fixed **angle step** (4 degrees/segment).
Mixing a count-based scheme with an angle-based scheme means the *effective*
angular resolution differs between shapes for no principled reason — pads
render coarser than the board outline even though nothing about the request
asked for that.

Everything in this package now derives its curve resolution from a single
angle-step budget, read from one shared object: **no function has its own
independent hardcoded default anymore**. Change it once, globally:

    import odbpp
    odbpp.config.TESSELLATION.max_angle_deg = 2.0   # finer, everywhere

...or override it for a single parse / a single PCBView / a single
call, which takes precedence over the global setting without changing it
for anyone else:

    odbpp.parse_product_model(root, max_angle_deg=1.0)
    pcbd = odbpp.PCBView(pm, max_angle_deg=1.0)
    island = some_symbol.to_island(max_angle_deg=8.0)

Every ``max_angle_deg`` parameter throughout the package defaults to
``None``, meaning "use whatever ``TESSELLATION.max_angle_deg`` is right
now" — so changing the global also changes the default for code that
already imported these functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Optional


@dataclass
class TessellationConfig:
    """Shared discretization settings. Mutate the module-level ``TESSELLATION``
    instance directly to change defaults package-wide."""

    max_angle_deg: float = 15
    """Maximum angle (degrees) any single straight-line segment is allowed
    to sweep when approximating a curve. This is the one knob that governs
    circles, donuts, thermal-relief spokes, rounded-rectangle/oval corners,
    standalone Arc features, outline/surface OC arcs, and the round end-caps
    used when buffering a Line/Arc into a filled stroke. Smaller = smoother
    curves and more points; larger = coarser and fewer points."""

    min_segments: int = 3
    """Never render a *closed* curve (a full circle, donut, or thermal
    boundary) with fewer than this many segments, even if ``max_angle_deg``
    is set very coarse (e.g. 200) — avoids degenerate shapes. Does not apply
    to partial arcs/fillets, where fewer segments is a legitimate outcome
    of a small sweep angle."""


#: The shared, mutable configuration instance every function in this
#: package falls back to when not given an explicit ``max_angle_deg``.
TESSELLATION = TessellationConfig()


def resolve_max_angle_deg(explicit: Optional[float]) -> float:
    """Return ``explicit`` if given, otherwise the current global default."""
    value = explicit if explicit is not None else TESSELLATION.max_angle_deg
    if value <= 0:
        raise ValueError(f"max_angle_deg must be > 0 (got {value!r})")
    return value


def segments_for_sweep(sweep_deg: float, max_angle_deg: Optional[float] = None, minimum: int = 1) -> int:
    """Number of equal-length segments to split a ``sweep_deg`` curve into,
    given an angular-step budget (falls back to the global config if
    ``max_angle_deg`` is None)."""
    step = resolve_max_angle_deg(max_angle_deg)
    return max(minimum, ceil(abs(sweep_deg) / step))
