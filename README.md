# emcad

A 2D polygon/CAD geometry kernel with a Rust core, plus a from-scratch
[ODB++](https://en.wikipedia.org/wiki/ODB%2B%2B) PCB design-format parser.

> **Written with AI assistance.** The vast majority of this codebase was
> written by [Claude Code](https://claude.com/claude-code), under human
> direction and review. See [`LICENSE`](./LICENSE) for the license and an
> explicit note on what that means for reuse.

## What this is

`emcad` is two largely independent things that happen to share one package:

1. **A 2D polygon boolean/geometry kernel** (`emcad.kernel` / `emcad.poly`) --
   exact-predicate union / intersect / subtract / fragment, robust hole and
   nested-island handling, RDP simplification, and a "de-zigzag" cleanup pass
   for the tiny step artifacts that round-capped, tessellated-circle unions
   tend to leave behind. Built around an exact-predicate half-edge
   arrangement engine (grid-snapped integer coordinates, no epsilon-tuning),
   so degenerate cases -- T-junctions, same-side overlaps, islands nested
   inside dynamically-formed holes -- resolve correctly instead of needing
   ad-hoc tolerance fixes.
2. **A native-Python ODB++ reader** (`emcad.odbpp`) -- parses the ODB++
   product-model tree (matrix, layers, features, symbols, drill tools) into a
   typed object graph, then resolves it into per-layer, boolean-unified 2D
   polygons and a Z-stack, ready to hand to a mesher or FEM tool.

The kernel is a compiled Rust extension (`emcad._rs`, built from `rust/` with
PyO3). It's a port of the original Numba/Python implementation (emcad 0.1)
that reproduced its results bit-for-bit while running typically 20-100x
faster; see [`benchmarks/RESULTS.md`](./benchmarks/RESULTS.md). There's no
external geometry library dependency (no Shapely/CGAL/etc).

## Installation

```bash
pip install emcad
```

Prebuilt wheels cover Linux (x86_64, aarch64), macOS (Intel, Apple
Silicon) and Windows (x64) for every CPython >= 3.10. Building from source
needs a Rust toolchain (`rustup`), since the package is built with
[maturin](https://www.maturin.rs/). Core dependencies: `numpy`, `loguru`,
`matplotlib` (used by the debug plotting helpers), `msgpack` (used by the
optional parsed-geometry cache).

See [`DEMO.md`](./DEMO.md) for a visual walkthrough of all of this rendered
with `emcad.plot.GeometryPlotter`, including holes/nested islands, the
`dezigzag()` cleanup pass, and `via_wall_polygons`.

## Quick start: the polygon kernel

```python
import emcad as cad

a = cad.Polygon([0, 10, 10, 0], [0, 0, 10, 10])
b = cad.Polygon([5, 15, 15, 5], [5, 5, 15, 15])

union = cad.add_polygons(a, b)          # [Polygon] -- fused into one shape
intersection = cad.intersect_polygons(a, b)
difference = cad.subtract_polygons((a,), (b,))   # a minus b

# In-place cleanup, both fail-safe (never turn a valid polygon invalid --
# see Polygon.simplify()/.dezigzag()'s own docstrings for why):
union[0].simplify(1e-6)     # Ramer-Douglas-Peucker point reduction
union[0].dezigzag(1e-5)     # collapse tessellated-circle "step" artifacts
union[0].regularize(0.25e-3)  # snap edges back onto straight lines, drop small bumps
```

`regularize(tol, dangle_deg=5.0)` is map-making style outline regularization,
meant for copper outlines whose straight edges are littered with via pads
poking out after a union. RDP can only remove those bumps by keeping their
tips, which tilts the long edges; `regularize` instead anchors on edges lying
on a `dangle_deg` direction grid, flattens protrusions smaller than `tol`,
and rebuilds corners hidden under pads, keeping real arcs. In the ODB++
pipeline it's the `regularize=True` option of `resolve_layer_polygons` /
`generate_traces` (`ODBImportConfig.trace_regularize`).

`Polygon.holes` nests arbitrarily deep (a hole can carry its own holes, i.e.
islands, to any depth), and every boolean op / cleanup pass handles that
nesting automatically via the even-odd rule -- no manual depth bookkeeping.

## Quick start: parsing an ODB++ board

```python
from emcad.odbpp import PCBView

pcb = PCBView("/path/to/MyBoard.odb", thickness_stack=[35e-6, 508e-6, 35e-6])

xs, ys = pcb.get_board_polygon()          # board outline, in meters
for z1, z2, material in pcb.iter_pcb_layers():
    ...                                      # physical copper/dielectric stack

for layer in pcb.iter_geo_layers():
    polygons = pcb.resolve_layer_polygons(layer)   # boolean-unified per layer
    for poly in polygons:
        xs, ys = poly.xs, poly.ys            # ready to feed to a mesher

for hole in pcb.iter_drill_holes():
    ...                                      # plated + non-plated drill holes
```

`PCBView` parses the board itself (via `ProductModel.from_path`, its
one-step wrapper around the lower-level parser) -- the raw parsed object
graph is still available afterward as `pcb.pm`, if you want to walk it
yourself. If you already have a `ProductModel` (e.g. from your own call to
`ProductModel.from_path(...)`), pass that instead and `PCBView` reuses it
rather than parsing twice:

```python
from emcad.odbpp import ProductModel, PCBView

pm = ProductModel.from_path("/path/to/MyBoard.odb")
pcb = PCBView(pm, thickness_stack=[35e-6, 508e-6, 35e-6])
```

Every coordinate everywhere in `emcad.odbpp` is a plain float in **meters** --
`emcad.odbpp.units` is the only place raw mm/inch/mil/micron conversion
happens, so nothing downstream needs to think about units.

Re-parsing and re-resolving a large board on every run gets expensive; if
you're iterating on downstream code against the same board repeatedly, cache
the resolved geometry once:

```python
from emcad.odbpp import save_pcb_cache, load_pcb_cache

save_pcb_cache(pcb, "board.pcbcache")
pcb_fast = load_pcb_cache("board.pcbcache")   # no re-parsing, no re-resolving
```

`load_pcb_cache` returns a drop-in stand-in for `PCBView` -- same public
methods, so existing code doesn't need to change.

## Architecture

See [`CLAUDE.md`](./CLAUDE.md) for a detailed map of both subsystems --
module-by-module responsibilities, the exact-predicate arrangement engine's
design, and the ODB++ object-graph layering. It's written for an AI coding
assistant working in this repo, but it's an accurate, up-to-date architecture
reference for a human reader too.

## Testing

```bash
uv sync --group dev                 # builds the Rust extension
uv run pytest
uv run python benchmarks/bench.py   # kernel benchmark
```

`uv sync`/`uv run` rebuild the extension automatically when anything under
`rust/` changes. `tests/test_09_golden.py` pins every kernel operation to
golden results recorded from the original implementation.

The suite is almost entirely regression coverage for the boolean kernel
(basic ops, touching polygons, holes, islands nested in holes, bounding-box
clustering, de-zigzag, ring self-intersection safety) plus a round-trip test
for the ODB++ geometry cache. It doesn't depend on any ODB++
board files being present.

## License

MIT -- see [`LICENSE`](./LICENSE), which also explains what the AI-authorship
note above means for reuse in more detail.
