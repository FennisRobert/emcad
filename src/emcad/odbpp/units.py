"""Unit conversion helpers.

ODB++ uses two distinct kinds of "units" and it's easy to mix them up:

* **Coordinate units** — set by a file's ``UNITS=MM`` or ``UNITS=INCH``
  directive. Coordinates, distances, and other measurements in the body of
  the file are expressed directly in millimeters or inches.

* **Symbol-size units** — the numeric parameters embedded in a *symbol
  name* (e.g. the ``120`` in ``r120``) are always expressed in thousandths
  of an inch (mils) or thousandths of a millimeter (microns) — never
  directly in mm/inch. Which of the two applies is selected either by an
  explicit ``I`` (mil) / ``M`` (micron) suffix on the symbol's ``$`` table
  line, or, if absent, by the file's coordinate UNITS directive
  (INCH -> mil, MM -> micron).

Everything this package produces is normalized to **meters (SI)**.
"""
from __future__ import annotations

MM_TO_M = 1e-3
INCH_TO_M = 0.0254
MIL_TO_M = INCH_TO_M / 1000.0   # 1 mil = 1/1000 inch
MICRON_TO_M = 1e-6              # 1 micron = 1/1000 mm


def coord_to_meters(value: float, units: str) -> float:
    """Convert a coordinate/distance value to meters given a UNITS directive."""
    units = units.upper()
    if units == "MM":
        return value * MM_TO_M
    if units == "INCH":
        return value * INCH_TO_M
    raise ValueError(f"Unknown coordinate UNITS value: {units!r}")


def symbol_size_to_meters(value: float, size_units: str) -> float:
    """Convert a symbol-name numeric parameter to meters.

    ``size_units`` must be ``'mil'`` or ``'micron'`` (see module docstring).
    """
    if size_units == "mil":
        return value * MIL_TO_M
    if size_units == "micron":
        return value * MICRON_TO_M
    raise ValueError(f"Unknown symbol size unit system: {size_units!r}")


def default_symbol_size_units(coord_units: str) -> str:
    """The default symbol-size unit system implied by a file's coordinate UNITS."""
    coord_units = coord_units.upper()
    if coord_units == "MM":
        return "micron"
    if coord_units == "INCH":
        return "mil"
    raise ValueError(f"Unknown coordinate UNITS value: {coord_units!r}")
