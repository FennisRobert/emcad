# Rust vs numba kernel benchmark

> **Historical comparison.** These numbers were measured when the kernel was
> ported, against the original numba/Python implementation (emcad 0.1.0).
> The numba backend was removed in 0.2.0, after the Rust port was verified to
> give the same results. That check now lives on as the golden-fingerprint
> test, `tests/test_09_golden.py`. `benchmarks/bench.py` now measures the Rust
> kernel only, on the same workloads.

Apple M4 Pro (14 cores), macOS 26.6, Python 3.10.18, numpy 2.2.6, numba 0.66.0, rustc 1.91.1.
Raw numbers: `results_vs_numba.json`.

All timings are best-of-N after a warm-up call. The numba backend's loguru
debug logging is disabled during timing, since the Rust backend doesn't log.
Both backends produced bit-identical results at the time.

## 1. Startup (fresh process: `import emcad` + first `add_polygons` call)

| backend | import | import + first call |
|---|---|---|
| rust | 170 ms | **170 ms** |
| numba, warm JIT cache | 170 ms | 306 ms |
| numba, cold JIT cache (first run) | 173 ms | 2.18 s |

About 137 ms of the 170 ms import is `matplotlib.pyplot`, pulled in by
`emcad.plot` at package import. That's the next startup cost to remove,
and it's independent of the backend.

## 2. The test suite (tests/test_01..08, sum of best per-test call time)

| test file | tests | rust | numba | speed-up |
|---|---|---|---|---|
| test_01_add_subtract_basic | 13 | 141 µs | 3.91 ms | 28× |
| test_02_touching | 9 | 84 µs | 1.64 ms | 19× |
| test_03_holes | 9 | 235 µs | 2.87 ms | 12× |
| test_04_islands_in_holes | 8 | 226 µs | 5.63 ms | 25× |
| test_05_bbox_clustering | 10 | 4.13 ms | 466 ms | 113× |
| test_06_dezigzag | 7 | 45 µs | 125 µs | 2.8× |
| test_08_ring_safety | 8 | 56 µs | 236 µs | 4.2× |
| **total (64 tests)** | | **4.9 ms** | **480 ms** | **98×** |

The test geometry is tiny (a few squares each), so the totals mostly
measure per-call overhead. The one test with real work in it,
`test_add_many_disjoint_squares_is_fast` (1,000 squares), runs in 447 ms on
numba and 3.8 ms on Rust (119×). On the numba backend, a pytest run of the
suite also pays the JIT startup from section 1.

## 3. Scaling workloads (PCB-like geometry, `tests/workloads.py`)

| operation | size | edges | rust | numba | speed-up |
|---|---|---|---|---|---|
| union: overlapping pad grid | 4×4 pads | 512 | 173 µs | 3.19 ms | 18× |
| | 16×16 | 8,192 | 3.53 ms | 101 ms | 29× |
| | 32×32 | 32,768 | 14.5 ms | 1.01 s | 70× |
| union: routed traces + pads | 25 traces | 1,250 | 481 µs | 9.43 ms | 20× |
| | 200 | 10,000 | 6.68 ms | 171 ms | 26× |
| | 800 | 40,000 | 59.5 ms | 2.16 s | 36× |
| subtract: pour − via clearances | 25 vias | 640 | 222 µs | 4.16 ms | 19× |
| | 400 | 10,324 | 4.89 ms | 153 ms | 31× |
| union: disjoint pads (clustered) | 1,000 pads | 24,000 | 3.41 ms | 400 ms | 117× |
| | 10,000 | 240,000 | 36.2 ms | 4.02 s | 111× |
| join: touching tiles | 10×10 | 320 | 115 µs | 2.21 ms | 19× |
| | 60×60 | 11,520 | 4.58 ms | 196 ms | 43× |
| poly_fragment: pad grid | 6×6 | 576 | 423 µs | 11.5 ms | 27× |
| | 24×24 | 9,216 | 23.7 ms | 1.73 s | 73× |
| dekeyhole | 60 holes | 1,144 | 190 µs | 5.08 ms | 27× |
| is_simple_ring | 1,000 pts | | 48 µs | 924 µs | 19× |
| | 20,000 pts | | 528 µs | 305 ms | 579× |
| simplify_polyline (RDP) | 1,000 pts | | 38 µs | 178 µs | 4.7× |
| | 10,000 | | 330 µs | 441 µs | 1.3× |
| | 100,000 | | 3.37 ms | 3.09 ms | 0.9× |
| via_wall_polygons | 190 vias | | 2.41 ms | 22.7 ms | 9.4× |
| | 1,910 vias | | 49.5 ms | 437 ms | 8.8× |

Where the speed-up comes from:

- **The Python around the numba kernels, not the kernels.** In the numba
  backend only the innermost loops are compiled. Vertex merging, segment
  chaining, membership propagation (dicts of frozensets), clustering, and
  Polygon assembly are interpreted Python, and that's where most of the time
  went. Porting the whole pipeline removes it, which is why even 4-vertex
  inputs are 20–45× faster.
- **Algorithmic fixes the port made cheap.** The split finder now uses an
  x-sorted sweep instead of all-pairs O(E²), and the via proximity search is
  grid-bucketed. This is the growing speed-up with size: `is_simple_ring`
  reaches 579× at 20k points, the pad grid 70× at 32×32. Results are
  re-sorted into the original discovery order, so output stays bit-identical.
- **Parallelism.** Large split searches, containment queries, and
  independent bbox clusters run on all cores (rayon).
- **RDP is a tie.** It was already a tight compiled loop under numba, with
  the same per-point arithmetic in both. numba keeps about a 10% edge at
  100k points, probably because it compiles for this exact CPU; that's not
  verified.
- **`via_wall_polygons` is capped at about 9×.** Only 43% of its remaining
  time is kernel calls. The rest is viaconnect's own Python logic (building
  thousands of circle/rectangle `Polygon`s, chain merging), which wasn't
  ported. See below.

## 4. Where to draw the Python ↔ Rust line

`emcad._rs.profile_boolean_op` times the three phases of one boolean op:
reading the input `Polygon` objects in Rust, computing, and building the
output `Polygon` objects.

| workload | extract | compute | build | boundary share |
|---|---|---|---|---|
| pad grid 12×12 (4.6k edges, 1 output) | 87 µs | 1.86 ms | 59 µs | 7% |
| routed 200 traces (10k edges) | 225 µs | 6.52 ms | 113 µs | 5% |
| pour − 200 vias (5.2k edges) | 88 µs | 2.38 ms | 70 µs | 6% |
| 1,000 disjoint pads (24k edges, 1,000 outputs) | 431 µs | 2.21 ms | 614 µs | 32% |
| join 20×20 tiles | 71 µs | 478 µs | 26 µs | 17% |

Crossing the boundary costs about 20 ns per coordinate each way. Rust reads
`Polygon.xs`/`.ys` lists directly and creates the result `Polygon`s itself,
so there are no intermediate numpy arrays. That's 5–7% of the runtime on
realistic connected geometry. The worst case is many small, independent
outputs, where per-object compute is tiny. It reaches about 30% but still
leaves the op 117× faster than numba.

The line drawn: **one crossing per operation.** Every polygon-level
operation (boolean ops, join, fragment, dekeyhole, point_inside,
is_inside) is a single Rust call over the whole input. Nothing calls into
Rust per ring or per primitive from a Python loop. Per-call overhead for
tiny calls is 0.4–7 µs:

| call | rust | numba |
|---|---|---|
| Polygon.is_inside (4-gon) | 0.4 µs | 1.2 µs |
| Polygon.is_inside (with hole) | 1.0 µs | 5.0 µs |
| Polygon.point_inside (with hole) | 1.1 µs | 21.5 µs |
| add_polygons (2 squares) | 6.9 µs | 305 µs |
| Polygon(..., holes=[...]) constructor | 11.3 µs | 27.1 µs |

Not ported, on purpose:

- **`viaconnect.py`'s orchestration** (graph building, chain merging,
  primitive construction). It's application logic around the kernels, and
  porting it would make `via_wall_polygons` at most about 2× faster.
- **`odbpp/`.** A text-format parser, I/O and string bound, with nothing to
  benchmark here: the example board the PCB tests need isn't in the repo.
- **`Polygon` itself** stays a Python class with Python-list coordinates.
  Making it a Rust-owned type would remove the remaining ~5% boundary cost,
  but it would change the public API (`poly.xs` is a list that callers
  mutate).
