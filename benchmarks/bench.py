"""Kernel benchmark (Rust kernel only).

    uv run python benchmarks/bench.py            # full run, writes benchmarks/results.json
    uv run python benchmarks/bench.py --quick    # smaller sizes

The original Rust-vs-numba comparison (made when the kernel was ported,
before the numba backend was removed) is written up in
`benchmarks/RESULTS.md`; this script keeps measuring the same workloads
so regressions show up against those numbers.

Sections:
  1. startup: import + first boolean op in a fresh process
  2. the pytest suite, every test re-run until ~0.3 s has elapsed, best-of
  3. PCB-like workloads at increasing size (best of N after a warm-up)
  4. boundary breakdown: extract (Python -> Rust) / compute / build
     (Rust -> Python) for the boolean-op workloads
  5. per-call micro-benchmarks (tiny inputs -- pure call overhead)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(TESTS))

from loguru import logger  # noqa: E402

logger.disable("emcad")


def best_of(fn, min_time: float = 0.3, max_reps: int = 50, budget: float = 60.0) -> tuple[float, int]:
    """Warm-up call, then repeat until min_time elapsed; returns (best, reps)."""
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first > budget / 3:
        return first, 1
    best, total, reps = first, 0.0, 0
    while (total < min_time and reps < max_reps) or reps < 3:
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        best = min(best, dt)
        total += dt
        reps += 1
        if total > budget:
            break
    return best, reps


def fmt(t: float | None) -> str:
    if t is None:
        return "—"
    if t >= 1:
        return f"{t:.2f} s"
    if t >= 1e-3:
        return f"{t * 1e3:.2f} ms"
    return f"{t * 1e6:.1f} µs"


# --------------------------------------------------------------------------
# 1 + 2: subprocess workers
# --------------------------------------------------------------------------

STARTUP_SNIPPET = """
import time
t0 = time.perf_counter()
import emcad
from emcad.kernel.api import add_polygons
from emcad.poly import Polygon
t1 = time.perf_counter()
add_polygons(Polygon([0, 2, 2, 0], [0, 0, 2, 2]), Polygon([1, 3, 3, 1], [1, 1, 3, 3]))
t2 = time.perf_counter()
print(t1 - t0, t2 - t0)
"""


def startup() -> tuple[float, float]:
    """(import emcad, import + first boolean op) in a fresh process; best of 5."""
    runs = []
    for _ in range(5):
        out = subprocess.run([sys.executable, "-c", STARTUP_SNIPPET], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
        runs.append((float(out[0]), float(out[1])))
    return min(runs, key=lambda r: r[1])


def suite_worker() -> None:
    import pytest

    class Repeat:
        def __init__(self):
            self.times: dict[str, tuple[float, float, int]] = {}

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            t0 = time.perf_counter()
            outcome = yield
            first = time.perf_counter() - t0
            best, total, reps = first, first, 1
            if outcome.excinfo is None:
                while total < 0.3 and reps < 500:
                    t = time.perf_counter()
                    item.runtest()
                    dt = time.perf_counter() - t
                    best = min(best, dt)
                    total += dt
                    reps += 1
            self.times[item.nodeid] = (first, best, reps)

    plugin = Repeat()
    files = sorted(str(p) for p in TESTS.glob("test_*.py") if "golden" not in p.name)
    t0 = time.perf_counter()
    code = pytest.main(["-q", "-p", "no:cacheprovider", *files], plugins=[plugin])
    wall = time.perf_counter() - t0
    print("@@JSON@@" + json.dumps({"exit": int(code), "wall": wall, "times": plugin.times}))


def suite() -> dict:
    out = subprocess.run([sys.executable, __file__, "--suite-worker"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    return json.loads(out.split("@@JSON@@", 1)[1])


# --------------------------------------------------------------------------
# 3 + 4 + 5: in-process
# --------------------------------------------------------------------------


def workloads(quick: bool) -> list[dict]:
    import workloads as wl
    from emcad import _rs
    from emcad.kernel import api
    from emcad.viaconnect import via_wall_polygons

    def edges(polys) -> int:
        def n(p):
            return len(p.xs) + sum(n(h) for h in p.holes)

        return sum(n(p) for p in polys)

    cases = []

    def add(name, size, fn, n_edges, profile=None):
        cases.append(dict(name=name, size=size, fn=fn, edges=n_edges, profile=profile))

    for n in ([4, 8, 12] if quick else [4, 8, 16, 24, 32]):
        polys = wl.pad_grid(n)
        add("union: overlapping pad grid", f"{n}x{n} pads", lambda p=polys: api.add_polygons(*p), edges(polys),
            (polys, "add", 0))
    for n in ([25, 100, 200] if quick else [25, 100, 200, 400, 800]):
        polys = wl.routed_layer(n)
        add("union: routed traces + pads", f"{n} traces", lambda p=polys: api.add_polygons(*p), edges(polys),
            (polys, "add", 0))
    for n in ([25, 100, 200] if quick else [25, 100, 200, 400, 800]):
        a, s = wl.pour_with_clearances(n)
        add("subtract: pour - clearances", f"{n} vias", lambda a=a, s=s: api.subtract_polygons(tuple(a), tuple(s)),
            edges(a + s), (a + s, "subtract", len(a)))
    for n in ([100, 1000] if quick else [100, 1000, 4000, 10000]):
        polys = wl.disjoint_pads(n)
        add("union: disjoint pads (clustered)", f"{n} pads", lambda p=polys: api.add_polygons(*p), edges(polys),
            (polys, "add", 0))
    for n in ([10, 20] if quick else [10, 20, 40, 60]):
        polys = wl.tiles(n)
        add("join: touching tiles", f"{n}x{n} tiles", lambda p=polys: api.join_polygons(p), edges(polys),
            (polys, "join", 0))
    for n in ([6, 10] if quick else [6, 10, 16, 24]):
        polys = wl.pad_grid(n, seg=16)
        add("poly_fragment: pad grid", f"{n}x{n} pads", lambda p=polys: api.poly_fragment(p), edges(polys))
    for n in ([5, 20] if quick else [5, 20, 60]):
        p = wl.keyhole_polygon(n)
        add("dekeyhole", f"{n} holes", lambda p=p: api.dekeyhole_polygon(p), len(p.xs))
    for n in ([1000, 10000] if quick else [1000, 10000, 100000]):
        xs, ys = wl.wiggly_polyline(n)
        add("simplify_polyline (RDP)", f"{n} pts", lambda xs=xs, ys=ys: api.simplify_polyline(xs, ys, 1e-5), n)
    for n in ([1000, 5000] if quick else [1000, 5000, 20000]):
        xs, ys = wl.wiggly_polyline(n)
        add("is_simple_ring", f"{n} pts", lambda xs=xs, ys=ys: api.is_simple_ring(xs, ys), n)
    for n in ([10, 40] if quick else [10, 40, 160]):
        (bar,) = api.add_polygons(wl.rect(0, 0, 6e-3, n * 2.8e-3),
                                  *[wl.circle(x, (k + 0.5) * 2.8e-3, 0.4e-3, 24)
                                    for k in range(n) for x in (0.25e-3, 5.75e-3)])
        add("regularize_polyline (via-lined bar)", f"{2 * n} pads",
            lambda b=bar: api.regularize_polyline(b.cxs, b.cys, 0.25e-3), len(bar.xs))
    for n in ([200, 800] if quick else [200, 800, 2000]):
        vias = wl.via_cloud(n)
        add("via_wall_polygons", f"{len(vias)} vias",
            lambda v=vias: via_wall_polygons(v, max_dist=0.6e-3, thickness=0.3e-3), len(vias))

    results = []
    for c in cases:
        row = dict(name=c["name"], size=c["size"], edges=c["edges"])
        row["rust"], _ = best_of(c["fn"])
        share = ""
        if c["profile"] is not None:
            polys, op, n_add = c["profile"]
            best = None
            for _ in range(5):
                ph = _rs.profile_boolean_op(polys, op, n_add, 1e-7, 1e-10)
                if best is None or sum(ph[:3]) < sum(best[:3]):
                    best = ph
            row["phases"] = dict(extract=best[0], compute=best[1], build=best[2], n_out=best[3])
            share = f"  boundary {100 * (best[0] + best[2]) / sum(best[:3]):4.1f}%"
        results.append(row)
        print(f"  {c['name']:<38} {c['size']:>14}  {fmt(row['rust']):>10}{share}", flush=True)
    return results


def micro() -> list[dict]:
    import workloads as wl
    from emcad.kernel import api
    from emcad.poly import Polygon

    sq = wl.rect(0, 0, 1, 1)
    holed = Polygon(wl.circle(0, 0, 3, 32).xs, wl.circle(0, 0, 3, 32).ys, holes=[wl.circle(0, 0, 1, 16)])
    small_xs, small_ys = [0, 1, 1, 0], [0, 0, 1, 1]
    a, b = wl.rect(0, 0, 2, 2), wl.rect(1, 1, 3, 3)
    cases = [
        ("Polygon.is_inside (4-gon)", lambda: sq.is_inside(0.5, 0.5)),
        ("Polygon.is_inside (with hole)", lambda: holed.is_inside(2.0, 0.1)),
        ("Polygon.point_inside (with hole)", lambda: holed.point_inside()),
        ("is_simple_ring (4 pts)", lambda: api.is_simple_ring(small_xs, small_ys)),
        ("add_polygons (2 squares)", lambda: api.add_polygons(a, b)),
        ("subtract_polygons (2 squares)", lambda: api.subtract_polygons((a,), (b,))),
        ("Polygon(..., holes=[...]) ctor", lambda: Polygon(holed.xs, holed.ys, holes=[holed.holes[0]])),
    ]
    out = []
    for name, fn in cases:
        t, _ = best_of(fn, min_time=0.2, max_reps=20000)
        out.append({"name": name, "rust": t})
        print(f"  {name:<38} {fmt(t):>10}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--suite-worker", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results.json"))
    args = ap.parse_args()
    if args.suite_worker:
        suite_worker()
        return

    import platform
    from importlib.metadata import version

    import numpy

    res: dict = {"meta": dict(python=sys.version.split()[0], numpy=numpy.__version__, emcad=version("emcad"),
                              machine=platform.platform(), cpus=os.cpu_count())}
    print("== 1. startup ==")
    res["startup"] = startup()
    print(f"  import {fmt(res['startup'][0])}, import + first call {fmt(res['startup'][1])}")
    print("== 2. test suite ==")
    res["suite"] = suite()
    s = res["suite"]
    print(f"  pytest wall {fmt(s['wall'])}, sum of best per-test call times "
          f"{fmt(sum(v[1] for v in s['times'].values()))}, {len(s['times'])} tests, exit={s['exit']}")
    print("== 3/4. workloads ==")
    res["workloads"] = workloads(args.quick)
    print("== 5. per-call micro-benchmarks ==")
    res["micro"] = micro()
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
