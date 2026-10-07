"""Deterministic geometry generators shared by the backend-parity test
(`test_09_golden.py`) and the benchmark (`benchmarks/bench.py`).

Shapes are PCB-like (every coordinate in meters, mm-scale features):
tessellated pads/vias, thick trace segments, copper pours, keyhole
outlines -- plus randomized adversarial cases (star polygons, random
overlap) for parity coverage.
"""
from __future__ import annotations

import math
import random

import numpy as np

from emcad.poly import Polygon

MM = 1e-3


def circle(cx: float, cy: float, r: float, n: int = 32, phase: float = 0.0) -> Polygon:
    t = phase + 2 * math.pi * np.arange(n) / n
    return Polygon(cx + r * np.cos(t), cy + r * np.sin(t))


def rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return Polygon([x0, x1, x1, x0], [y0, y0, y1, y1])


def trace(x0: float, y0: float, x1: float, y1: float, w: float, cap_n: int = 8) -> Polygon:
    """A thick line segment with round end caps (a stadium)."""
    ang = math.atan2(y1 - y0, x1 - x0)
    pts = []
    for k in range(cap_n + 1):
        a = ang + math.pi / 2 + math.pi * k / cap_n
        pts.append((x0 + w / 2 * math.cos(a), y0 + w / 2 * math.sin(a)))
    for k in range(cap_n + 1):
        a = ang - math.pi / 2 + math.pi * k / cap_n
        pts.append((x1 + w / 2 * math.cos(a), y1 + w / 2 * math.sin(a)))
    return Polygon([p[0] for p in pts], [p[1] for p in pts])


def star(cx: float, cy: float, r0: float, r1: float, n: int, rng: random.Random) -> Polygon:
    """Simple (non-self-intersecting) star-shaped polygon with jittered radii."""
    t = np.sort(np.array([rng.uniform(0, 2 * math.pi) for _ in range(n)]))
    t = np.unique(np.round(t, 9))
    r = np.array([rng.uniform(r0, r1) for _ in range(len(t))])
    return Polygon(cx + r * np.cos(t), cy + r * np.sin(t))


# --------------------------------------------------------------------------
# PCB-like workloads
# --------------------------------------------------------------------------

def pad_grid(n: int, pitch: float = 1.0 * MM, r: float = 0.6 * MM, seg: int = 32) -> list[Polygon]:
    """n x n grid of overlapping round pads -- one big connected cluster."""
    return [circle(i * pitch, j * pitch, r, seg) for i in range(n) for j in range(n)]


def routed_layer(n_traces: int, seed: int = 0, extent: float = 20 * MM) -> list[Polygon]:
    """Random thick traces with pads at both ends, heavily overlapping."""
    rng = random.Random(seed)
    polys = []
    for _ in range(n_traces):
        x0, y0 = rng.uniform(0, extent), rng.uniform(0, extent)
        ang = rng.choice([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]) + rng.choice([0, math.pi])
        ln = rng.uniform(1, 5) * MM
        x1, y1 = x0 + ln * math.cos(ang), y0 + ln * math.sin(ang)
        polys.append(trace(x0, y0, x1, y1, 0.2 * MM))
        polys.append(circle(x0, y0, 0.3 * MM, 16))
        polys.append(circle(x1, y1, 0.3 * MM, 16))
    return polys


def pour_with_clearances(n_vias: int, seed: int = 1, extent: float = 30 * MM):
    """(add, subtract): a board-sized pour minus via clearances + a few traces."""
    rng = random.Random(seed)
    pour = rect(0, 0, extent, extent)
    cuts = [circle(rng.uniform(1 * MM, extent - 1 * MM), rng.uniform(1 * MM, extent - 1 * MM), 0.4 * MM, 24)
            for _ in range(n_vias)]
    for _ in range(max(1, n_vias // 10)):
        x0, y0 = rng.uniform(2 * MM, extent - 2 * MM), rng.uniform(2 * MM, extent - 2 * MM)
        cuts.append(trace(x0, y0, x0 + rng.uniform(-8, 8) * MM * 0.5, y0 + rng.uniform(-8, 8) * MM * 0.5, 0.5 * MM))
    return [pour], cuts


def disjoint_pads(n: int, pitch: float = 3.0 * MM) -> list[Polygon]:
    """Many non-touching pads (exercises bbox clustering)."""
    side = int(math.ceil(math.sqrt(n)))
    return [circle((k % side) * pitch, (k // side) * pitch, 0.5 * MM, 24) for k in range(n)]


def tiles(n: int, size: float = 1.0 * MM) -> list[Polygon]:
    """n x n edge-touching square tiles with every 3rd one missing (holes) -- join input."""
    return [rect(i * size, j * size, (i + 1) * size, (j + 1) * size)
            for i in range(n) for j in range(n) if (i * 7 + j * 3) % 5 != 0]


def keyhole_polygon(n_holes: int, seg: int = 16) -> Polygon:
    """Outer square traced with n_holes circular holes bridged in via slits."""
    size = (n_holes + 1) * 2 * MM
    xs, ys = [0.0], [0.0]
    for k in range(n_holes):
        cx = (k + 1) * 2 * MM
        cy = size / 2
        r = 0.5 * MM
        xs.append(cx); ys.append(0.0)
        # slit up to the hole, trace it, come back down
        hx, hy = cx, cy - r
        xs.append(hx); ys.append(hy)
        for s in range(1, seg):
            a = -math.pi / 2 - 2 * math.pi * s / seg
            xs.append(cx + r * math.cos(a)); ys.append(cy + r * math.sin(a))
        xs.append(hx); ys.append(hy)
        xs.append(cx); ys.append(0.0)
    xs += [size, size, 0.0]
    ys += [0.0, size, size]
    return Polygon(xs, ys)


def wiggly_polyline(n: int, seed: int = 2):
    """Long noisy polyline (simplify input)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 20 * math.pi, n)
    return (np.cos(t) * (1 + 0.1 * t) + rng.normal(0, 1e-3, n)) * MM, (np.sin(t) * (1 + 0.1 * t) + rng.normal(0, 1e-3, n)) * MM


def zigzag_ring(n_kinks: int):
    """Closed rectangle-ish ring whose long edges carry tiny sideways steps."""
    xs, ys = [], []
    step = 1.0 * MM
    for k in range(n_kinks):
        xs += [k * step, k * step + 0.9 * step]
        ys += [0.0, 0.0 + (0.002 * MM if k % 2 else 0.0)]
    xs += [n_kinks * step, n_kinks * step, 0.0]
    ys += [0.0, 5 * MM, 5 * MM]
    xs.append(xs[0]); ys.append(ys[0])
    return np.array(xs), np.array(ys)


def via_cloud(n: int, seed: int = 3) -> np.ndarray:
    """Via fences: a few closed rectangular loops + straight runs, 0.5mm pitch."""
    rng = random.Random(seed)
    pts = []
    pitch = 0.5 * MM
    while len(pts) < n:
        x0, y0 = rng.uniform(0, 40) * MM, rng.uniform(0, 40) * MM
        w, h = rng.randint(4, 16), rng.randint(4, 16)
        for i in range(w):
            pts.append((x0 + i * pitch, y0))
            pts.append((x0 + i * pitch, y0 + h * pitch))
        for j in range(h + 1):
            pts.append((x0, y0 + j * pitch))
            pts.append((x0 + w * pitch, y0 + j * pitch))
    arr = np.unique(np.round(np.array(pts[:n]), 9), axis=0)
    return arr
