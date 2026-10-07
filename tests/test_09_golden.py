"""Layer 9: golden-result regression for the whole kernel.

Every case below was run on the original numba/Python reference
implementation and on the Rust port (`emcad._rs`), which produced
bit-identical results; the numba backend was then removed. This test
pins the Rust kernel to that reference via fingerprints stored in
`golden/kernel_fingerprints.json`: full result STRUCTURE (number of
output polygons, hole nesting, vertex counts, exception types, integer
arrays, booleans) must match exactly, and the floating-point values
(ring areas, bounding boxes, coordinate sums) must match to 1e-9
relative. Exact float hashes aren't used on purpose: inputs are built
with numpy's sin/cos, which can differ by an ulp across platforms, and
the fingerprints have to hold on every OS the wheels are built for.

Regenerate (only after an INTENDED kernel behavior change):

    uv run python tests/test_09_golden.py --regenerate
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from emcad.kernel import api
from emcad.poly import GeometryException, Polygon

sys.path.insert(0, str(Path(__file__).parent))
import workloads as wl  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "kernel_fingerprints.json"
REL, ABS = 1e-9, 1e-15


# --------------------------------------------------------------------------
# fingerprints
# --------------------------------------------------------------------------

def _ring_area(xs, ys) -> float:
    n = len(xs)
    return sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n)) / 2.0


def _poly_fp(p: Polygon) -> dict:
    xs = [float(v) for v in p.xs]
    ys = [float(v) for v in p.ys]
    return {
        "n": len(xs),
        "area": _ring_area(xs, ys),
        "bbox": [min(xs), max(xs), min(ys), max(ys)],
        "sx": math.fsum(xs),
        "sy": math.fsum(ys),
        "holes": [_poly_fp(h) for h in p.holes],
    }


def fingerprint(result):
    if isinstance(result, Polygon):
        return {"polygon": _poly_fp(result)}
    if isinstance(result, list) and all(isinstance(p, Polygon) for p in result):
        return {"polygons": [_poly_fp(p) for p in result]}
    if isinstance(result, tuple):
        return {"tuple": [fingerprint(r) for r in result]}
    if isinstance(result, np.ndarray):
        if result.dtype.kind in "iub":
            return {"array": result.dtype.kind, "shape": list(result.shape), "values": result.tolist()}
        flat = result.astype(np.float64).ravel()
        return {
            "array": "f",
            "shape": list(result.shape),
            "sum": math.fsum(flat.tolist()),
            "sumsq": math.fsum((flat * flat).tolist()),
            "head": flat[:8].tolist(),
            "tail": flat[-8:].tolist(),
        }
    if isinstance(result, (bool, np.bool_)):
        return {"bool": bool(result)}
    if isinstance(result, (float, int, np.floating, np.integer)):
        return {"float": float(result)}
    raise TypeError(f"no fingerprint for {type(result)}")


def run_case(fn):
    try:
        return {"ok": fingerprint(fn())}
    except (GeometryException, ValueError) as e:
        return {"raise": type(e).__name__}


def assert_close(actual, expected, path="$"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), f"{path}: keys {actual} != {expected}"
        for k in expected:
            assert_close(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=REL, abs_tol=ABS), f"{path}: {actual!r} != {expected!r}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

def _random_polys(seed: int, n: int, with_holes: bool) -> list[Polygon]:
    rng = random.Random(seed)
    polys = []
    for _ in range(n):
        cx, cy = rng.uniform(0, 10), rng.uniform(0, 10)
        kind = rng.random()
        if kind < 0.4:
            p = wl.star(cx, cy, 1.0, 3.0, rng.randint(5, 30), rng)
        elif kind < 0.7:
            w, h = rng.uniform(0.5, 4), rng.uniform(0.5, 4)
            p = wl.rect(cx, cy, cx + w, cy + h)
        else:
            p = wl.circle(cx, cy, rng.uniform(0.5, 3), rng.choice([6, 8, 16, 32]), rng.uniform(0, 1))
        if with_holes and rng.random() < 0.4:
            hole = wl.circle(cx, cy, 0.2, 8)
            if p.is_inside(cx, cy) and p.is_inside(cx + 0.2, cy) and p.is_inside(cx - 0.2, cy):
                try:
                    p = Polygon(p.xs, p.ys, holes=[hole])
                except GeometryException:
                    pass
        polys.append(p)
    return polys


def build_cases() -> dict:
    cases: dict = {}
    for seed in range(12):
        polys = _random_polys(seed, 6 + seed, with_holes=True)
        cases[f"add_random_{seed}"] = lambda p=polys: api.add_polygons(*p)
        polys = _random_polys(100 + seed, 2 + seed % 3, with_holes=True)
        cases[f"intersect_random_{seed}"] = lambda p=polys: api.intersect_polygons(*p)
        polys = _random_polys(200 + seed, 8, with_holes=True)
        cases[f"subtract_random_{seed}"] = lambda p=polys: api.subtract_polygons(tuple(p[:3]), tuple(p[3:]))
        polys = _random_polys(300 + seed, 4, with_holes=seed % 2 == 0)
        for keep in ("positive", "negative", "all"):
            cases[f"fragment_{keep}_{seed}"] = lambda p=polys, k=keep: api.poly_fragment(p, keep=k)
        cases[f"fragment_unfiltered_{seed}"] = lambda p=polys: api.poly_fragment(p, filter_to_originals=False)
        polys = _random_polys(400 + seed, 5, with_holes=False)
        cases[f"join_random_{seed}"] = lambda p=polys: api.join_polygons(p)

        rng = random.Random(500 + seed)
        star = wl.star(0, 0, 1, 3, 40, rng)
        xs, ys = list(star.xs), list(star.ys)
        cases[f"simple_ring_{seed}"] = lambda xs=xs, ys=ys: api.is_simple_ring(xs, ys)
        xs2 = list(xs)
        xs2[3], xs2[20] = xs2[20], xs2[3]
        cases[f"simple_ring_swapped_{seed}"] = lambda xs=xs2, ys=ys: api.is_simple_ring(xs, ys)
        pts = [(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(50)]
        for qi, q in enumerate(_random_polys(600 + seed, 3, with_holes=True)):
            cases[f"point_inside_{seed}_{qi}"] = lambda q=q: q.point_inside()
            cases[f"is_inside_{seed}_{qi}"] = lambda q=q, pts=pts: np.array(
                [[q.is_inside(x, y, True), q.is_inside(x, y, False)] for x, y in pts])

    cases["join_tiles"] = lambda: api.join_polygons(wl.tiles(12))
    cases["pcb_pad_grid"] = lambda: api.add_polygons(*wl.pad_grid(6))
    cases["pcb_routed"] = lambda: api.add_polygons(*wl.routed_layer(40))
    add, sub = wl.pour_with_clearances(40)
    cases["pcb_pour"] = lambda: api.subtract_polygons(tuple(add), tuple(sub))
    cases["pcb_disjoint"] = lambda: api.add_polygons(*wl.disjoint_pads(60))
    add5, sub5 = wl.pour_with_clearances(15, seed=5)
    extra = wl.routed_layer(10, seed=6)

    def feed_back():
        carved = api.subtract_polygons(tuple(add5), tuple(sub5))
        return api.add_polygons(*carved, *extra) + api.intersect_polygons(carved[0], wl.circle(15e-3, 15e-3, 8e-3, 64))

    cases["pcb_feed_back"] = feed_back
    for n_holes in (0, 1, 3, 6):
        p = wl.keyhole_polygon(n_holes)
        cases[f"dekeyhole_{n_holes}"] = lambda p=p: api.dekeyhole_polygon(p)
        cases[f"dekeyhole_batch_{n_holes}"] = lambda p=p: api.dekeyhole_polygons([p, p])

    xs, ys = wl.wiggly_polyline(3000)
    for delta in (0.0, 1e-6, 1e-5, 1e-4):
        cases[f"simplify_{delta:g}"] = lambda d=delta: api.simplify_polyline(xs, ys, d)
    zx, zy = wl.zigzag_ring(40)
    cases["dezigzag_default"] = lambda: api.dezigzag_polyline(zx, zy, 0.01e-3)
    cases["dezigzag_tight"] = lambda: api.dezigzag_polyline(zx, zy, 0.01e-3, 5.0, 10.0)
    cases["sanitize_dupes"] = lambda: api.sanitize_polygon([0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0])
    cases["sanitize_degenerate"] = lambda: api.sanitize_polygon([0, 1e-12, 0], [0, 0, 1e-12])
    cases["sanitize_open"] = lambda: api.sanitize_polygon([0, 1, 2], [0, 0, 0], closed=False)

    e0, e1 = _random_polys(700, 2, with_holes=False)
    s0, t0 = e0._edge_points()
    s1, t1 = e1._edge_points()
    cases["edge_self_intersections"] = lambda: api.edge_self_intersections(s0, t0)
    cases["edge_cross_intersections"] = lambda: api.edge_cross_intersections(s0, t0, s1, t1)

    vias = wl.via_cloud(300)

    def via_walls():
        from emcad.viaconnect import via_wall_polygons

        return via_wall_polygons(vias, max_dist=0.6e-3, thickness=0.3e-3)

    cases["via_walls"] = via_walls
    return cases


CASES = build_cases()


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_golden(name, golden):
    assert name in golden, f"no golden fingerprint for {name!r} -- regenerate (see module docstring)"
    assert_close(run_case(CASES[name]), golden[name])


def test_golden_has_no_stale_cases(golden):
    assert set(golden) == set(CASES)


def test_convex_hull_is_hull():
    # (not golden: the original numba `_clean_loop` read uninitialized
    # memory, so its index output was never well-defined to pin against)
    rng = np.random.default_rng(0)
    xs, ys = rng.normal(size=200), rng.normal(size=200)
    ids = api.convex_hull(xs, ys)
    assert len(set(ids.tolist())) == len(ids)
    hx, hy = xs[ids], ys[ids]
    n = len(ids)
    for i in range(n):
        ax, ay, bx, by = hx[i], hy[i], hx[(i + 1) % n], hy[(i + 1) % n]
        cross = (bx - ax) * (ys - ay) - (by - ay) * (xs - ax)
        assert np.all(cross <= 1e-12) or np.all(cross >= -1e-12)


if __name__ == "__main__":
    if "--regenerate" not in sys.argv:
        sys.exit(__doc__)
    from loguru import logger

    logger.disable("emcad")
    GOLDEN.parent.mkdir(exist_ok=True)
    out = {name: run_case(fn) for name, fn in sorted(CASES.items())}
    GOLDEN.write_text(json.dumps(out, indent=0, sort_keys=True) + "\n")
    print(f"wrote {len(out)} fingerprints to {GOLDEN}")
