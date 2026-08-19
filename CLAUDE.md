# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`emcad` is a pure-Python 2D polygon/CAD geometry kernel plus a native-Python ODB++ PCB-format parser. The package lives entirely under `src/emcad/`. It has two largely independent subsystems that meet in `src/emcad/emerge.py`:

1. **`emcad.kernel` / `emcad.poly`** — a general-purpose 2D polygon boolean/geometry kernel (union, intersect, subtract, fragment, simplify, hole handling).
2. **`emcad.odbpp`** — a from-scratch ODB++ PCB design-format reader (matrix/layers/features/symbols) that resolves into the same `Polygon`-shaped geometry.

`emcad.emerge.ODBppPCB` (in `src/emcad/emerge.py`) is the bridge: it drives `odbpp.PCBDesigner` to build per-layer polygons and a Z-stack, then emits them as solids/surfaces for the sibling `emerge` FEM electromagnetics simulator (an external, editable-installed dependency — not declared in `pyproject.toml`, and its source is outside this repo). Treat any `import emerge` / `em.*` usage in this repo as calling out to that external package; don't try to modify it from here.

## Commands

- Install in editable mode: `uv sync` (project uses `uv.lock`) or `pip install -e .`
- Run the interactive/manual test script: `python main.py` (see caveat below)
- No test suite, linter, or formatter is configured yet — `tests/` is empty and `pyproject.toml` has no pytest/ruff/mypy config. Don't assume `pytest` or `ruff` exist as project commands until such config is added.

**`main.py` import quirk**: it imports via `from src.emcad import ...` (path-relative to the repo root) rather than `from emcad import ...` (the installed package name declared in `pyproject.toml`). It must be run from the repo root as a script, not as an installed-package example. New example/demo code should generally prefer `import emcad as cad` (matches how `emerge.py` and everything under `src/` itself import the package) unless matching `main.py`'s existing style.

Root-level `odb_parse.py`, "`odb_parse copy.py`", `odb_parse_post.py`, and `mathexp.py` are standalone scratch/experiment scripts (ODB++ → EM-simulation pipeline runs, and a sympy scratchpad for the edge-intersection math) — not part of the installable package and not imported by anything under `src/`.

## Architecture

### Geometry kernel (`src/emcad/kernel/`)

`kernel/api.py` is the **only** public interface — it's a thin, fully-typed pass-through with no logic of its own. Every real implementation lives in an underscore-prefixed module, one topic per file:

| Module | Responsibility |
|---|---|
| `_ch.py` | convex hull |
| `_primitives.py` + `_intersect_kernels.py` + `_inside.py` | edge-intersection / point-in-polygon primitives (numba-jitted); never import `Polygon` |
| `_simplify.py` | Ramer-Douglas-Peucker polyline simplification, polygon sanitization (numba-jitted) |
| `_fragment.py` + `fragment_tools.py` + `fragment_kernels.py` | `poly_fragment`: splits N (possibly overlapping/self-intersecting) polygons into a mutually-disjoint planar arrangement via a half-edge structure; this is the core primitive everything else composes |
| `_boolean_ops.py` | `add_polygons`/`intersect_polygons`/`subtract_polygons`, built on top of `poly_fragment` + a sign/containment filter |
| `_join.py` | `join_polygons`: fuses already-disjoint, edge-touching polygons back together (e.g. after a boolean-select step) — explicitly *not* a general union; raises on genuine crossings or fully-enclosed non-touching input |
| `_keyhole.py` | converts keyhole-encoded rings (self-touching bridge/slit hole encoding, common in ODB++/Gerber) into proper `Polygon.holes` |
| `_constants.py` | shared tolerances (`DEFAULT_MERGE_TOL`, `DEFAULT_T_TOL`, `DEFAULT_AREA_TOL`) — single source of truth, no deps |

**Import-order constraint you must preserve**: `_join.py` constructs `Polygon` at its own module top level, so `api.py` imports it *lazily* (inside `join_polygons()`, not at module top) to avoid a circular import (`poly.py` → `kernel/api.py` → `_join.py` → `poly.py`). `_boolean_ops.py` does the same for the same reason. Every other kernel module avoids importing `Polygon` at runtime entirely (only under `TYPE_CHECKING`), which is what lets `api.py` import them eagerly. Read `kernel/api.py`'s module docstring before changing any kernel module's import structure.

Performance-critical inner loops (proximity/intersection counting, convex hull, RDP simplification) are `numba`-`@njit`-compiled; keep those functions free of Python-object types (dataclasses, `Polygon`, etc.) — they take/return plain `np.ndarray`.

### `Polygon` (`src/emcad/poly.py`)

The core data type: `xs`/`ys` open-ring coordinate lists plus a recursive `holes: list[Polygon]` (holes may themselves carry holes, representing islands-inside-holes to arbitrary depth). Containment/`is_inside`/`point_inside` use the even-odd rule across every ring at once — no explicit depth bookkeeping. `Polygon.__init__` validates hole nesting via exact edge-crossing checks; `Polygon._construct_verified` (used internally by `dekeyhole_polygon`) skips that specific check for algorithm-derived geometry that's already been containment-verified another way, since a keyhole junction is *expected* to touch its parent at the extraction point.

### ODB++ parser (`src/emcad/odbpp/`)

A native-Python object model for the ODB++ PCB interchange format. **Every coordinate everywhere in this package is a plain float in meters** — no exceptions; `units.py` is the only place raw mm/inch/mil/micron conversion happens. Curve/arc tessellation resolution is controlled from one place, `odbpp.config.TESSELLATION` (settable globally or per-call via `max_angle_deg=`).

Layering (see `odbpp/__init__.py`'s docstring for the exact map): `config`/`units` (shared settings) → `geometry` (`Point`/`Polygon`/`Island`/`Surface`) → `symbols` (standard + user-defined pad shapes) → `features` (`Stamp`/`Line`/`Arc`/text/barcode/surface + the features-file parser) → `layers` (drill tools, layer parser) → `matrix` (layer-stack parser) → `model` (`Step`/`ProductModel`, ties it together — including recursive `resolve_stamp` for user-defined symbols, capped at depth 25) → `parse.parse_product_model()` (the entry point) → `design.PCBDesigner` (the high-level bridge that turns a parsed `ProductModel` into Z-stacked layer polygons + drill holes ready for `emcad.emerge.ODBppPCB` to consume).

### `emcad.emerge` / `viaconnect.py`

`emerge.py`'s `ODBppPCB` wraps a `PCBDesigner` and exposes `generate_dielectric()` / `generate_traces()` / `generate_vias()`, each producing `emerge`-library geometry (`em.geo.*`) from the parsed PCB — traces run through `cad.simplify_polyline` + `Polygon.remove_keyholes()` + `cad.add_polygons` before being handed to `emerge`.

`viaconnect.py`'s `via_wall_polygons()` is a standalone algorithm (documented in detail in its own module docstring) for turning a dense cloud of via coordinates into a thickened "wall" polygon with correctly-formed holes where the via network loops — an alternative to meshing every individual via, used by `ODBppPCB.generate_vias(autojoin_limit=...)`. It only depends on `emcad.poly`/`emcad.kernel.api`/`emcad.plot`, not on the ODB++ subsystem.

### Plotting

`src/emcad/plot.py`'s `GeometryPlotter` is a thin matplotlib wrapper for visually debugging `Vertex`/`Edge`/`Polygon` (holes included, via a compound even-odd path) — used throughout the kernel's `debug=True` paths and ad hoc during development, not part of the production pipeline.
