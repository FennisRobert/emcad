# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`emcad` is a 2D polygon/CAD geometry kernel (Rust core, Python API) plus a native-Python ODB++ PCB-format parser. The package lives entirely under `src/emcad/`. It has two largely independent subsystems that meet in `src/emcad/emerge.py`:

1. **`emcad.kernel` / `emcad.poly`** — a general-purpose 2D polygon boolean/geometry kernel (union, intersect, subtract, fragment, simplify, hole handling).
2. **`emcad.odbpp`** — a from-scratch ODB++ PCB design-format reader (matrix/layers/features/symbols) that resolves into the same `Polygon`-shaped geometry.

`emcad.emerge.ODBppPCB` (in `src/emcad/emerge.py`) is the bridge: it drives `odbpp.PCBDesigner` to build per-layer polygons and a Z-stack, then emits them as solids/surfaces for the sibling `emerge` FEM electromagnetics simulator (an external, editable-installed dependency — not declared in `pyproject.toml`, and its source is outside this repo). Treat any `import emerge` / `em.*` usage in this repo as calling out to that external package; don't try to modify it from here.

## Commands

- Install / build: `uv sync --group dev`. The build backend is **maturin**: this compiles the Rust kernel in `rust/` into the `emcad._rs` extension module (needs a Rust toolchain). `[tool.uv] cache-keys` makes `uv sync`/`uv run` rebuild automatically whenever `rust/src/**/*.rs` or `rust/Cargo.toml` change.
- Tests: `uv run pytest`. `tests/test_09_golden.py` pins every kernel op to golden fingerprints (`tests/golden/kernel_fingerprints.json`) recorded from the original numba implementation, which the Rust port reproduced bit-for-bit before numba was removed. Regenerate only after an *intended* behavior change: `uv run python tests/test_09_golden.py --regenerate`.
- Benchmark: `uv run python benchmarks/bench.py [--quick]` -- startup, the test suite, scaled PCB-like workloads (`tests/workloads.py`), boundary-crossing cost, per-call overhead. `benchmarks/RESULTS.md` is the historical Rust-vs-numba write-up (raw data `results_vs_numba.json`).
- Release: see "Releasing" below. No linter/formatter is configured.

## Architecture

### Geometry kernel: Rust (`rust/src/`), Python API (`src/emcad/kernel/api.py`)

`kernel/api.py` is the **only** public kernel interface -- thin, fully-typed wrappers over `emcad._rs`, and the place each operation's contract is documented. The other modules in `src/emcad/kernel/` are dependency-free helpers: `_constants.py` (shared tolerances `DEFAULT_MERGE_TOL`/`DEFAULT_T_TOL`/`DEFAULT_AREA_TOL`), `_errors.py` (`ArrangementError`, raised from Rust), `_profiling.py`.

| Rust module | Responsibility |
|---|---|
| `exact.rs` | grid-snapped exact integer predicates + the x-sorted sweep split finder (every crossing / T-junction / collinear overlap) |
| `halfedge.rs` | half-edge rotation system, face tracing, containment queries, union-find |
| `arrangement.rs` | the arrangement engine: split, chain, dedupe, trace faces, winding-parity membership propagation, label + assemble; `is_simple_ring` |
| `boolean.rs` | bbox clustering + `add`/`intersect`/`subtract`/`join` as membership label functions over one arrangement (join additionally rejects any face claimed by two operands) |
| `fragment.rs` | `poly_fragment` (float-tolerance arrangement, containment-forest nesting) |
| `keyhole.rs` | keyhole-encoded rings (ODB++/Gerber bridge/slit holes) -> real holes |
| `regularize.rs` | map-making style outline regularization (`Polygon.regularize`): snap edges onto a `dangle` direction grid, drop small protrusions (via pads on copper edges), rebuild corners |
| `primitives.rs` | edge intersections, convex hull, RDP / sanitize / dezigzag |
| `via.rs` | grid-bucketed via proximity graph + RNG pruning for `viaconnect.py` |
| `geom.rs` | `PolyTree` (Rust mirror of `Polygon`), `point_inside`, even-odd `is_inside` |
| `lib.rs` | PyO3 bindings |

Boundary design (see `benchmarks/RESULTS.md` for measurements): one Python->Rust crossing per *operation*. Rust reads `Polygon.xs`/`.ys`/`.holes` directly and builds result `Polygon`s itself via `Polygon.__new__` + attribute assignment (the same object `_construct_verified` produces -- its validation is replicated in `geom.rs::PolyTree::verified`, reading `GlobalSettings.EPS` per call). Don't add per-ring or per-primitive calls into Rust from Python loops. Rust raises `emcad.poly.GeometryException` / `ValueError` / `emcad.kernel._errors.ArrangementError` by importing them at call time.

Determinism matters: outputs (vertex numbering, face order, output order) are deliberately reproducible -- parallel stages re-sort into a fixed order -- and pinned by the golden test.

### Releasing (wheels + PyPI)

`.github/workflows/wheels.yml` builds abi3 wheels (one per platform, covering every CPython >= 3.10) for Linux (manylinux x86_64/aarch64, musllinux x86_64), macOS (x86_64, arm64) and Windows (x64), plus the sdist, and runs the test suite against each wheel. It runs on every push/PR. Publishing to PyPI happens ONLY from a manual `workflow_dispatch` run with `publish: true` (or a published GitHub release), via PyPI Trusted Publishing (no API token), gated by the `pypi` GitHub environment. Bump `version` in `pyproject.toml` (and `rust/Cargo.toml`) before releasing -- PyPI never accepts the same version twice.

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
