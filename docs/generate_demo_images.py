"""Generate the images used in DEMO.md. Run once, from the repo root:

    python generate_demo_images.py

Writes PNGs into docs/images/. Not part of the installed package -- a
one-off content-generation script for the docs.
"""
import math

import matplotlib
matplotlib.use("Agg")

from loguru import logger
logger.remove()

import emcad as cad
from emcad.plot import GeometryPlotter
from emcad.viaconnect import via_wall_polygons

OUT = "docs/images"


def save(plotter: GeometryPlotter, name: str) -> None:
    plotter.save(f"{OUT}/{name}.png", dpi=160)
    print(f"wrote {OUT}/{name}.png")


# --------------------------------------------------------------------------
# 1. Union / intersect / subtract of two overlapping squares
# --------------------------------------------------------------------------

a = cad.Polygon([0, 10, 10, 0], [0, 0, 10, 10])
b = cad.Polygon([5, 15, 15, 5], [5, 5, 15, 15])

p = GeometryPlotter(figsize=(6, 6), title="Two input polygons")
p.add_polygon([a, b], alpha=0.55)
save(p, "01_inputs")

union = cad.add_polygons(a, b)
p = GeometryPlotter(figsize=(6, 6), title="add_polygons(a, b) -- union")
p.add_polygon(union, facecolor="#4C72B0")
save(p, "02_union")

inter = cad.intersect_polygons(a, b)
p = GeometryPlotter(figsize=(6, 6), title="intersect_polygons(a, b)")
p.add_polygon(a, facecolor="lightgray", alpha=0.3, edgecolor="gray")
p.add_polygon(b, facecolor="lightgray", alpha=0.3, edgecolor="gray")
p.add_polygon(inter, facecolor="#DD8452")
save(p, "03_intersect")

diff = cad.subtract_polygons((a,), (b,))
p = GeometryPlotter(figsize=(6, 6), title="subtract_polygons((a,), (b,)) -- a minus b")
p.add_polygon(b, facecolor="lightgray", alpha=0.3, edgecolor="gray")
p.add_polygon(diff, facecolor="#55A868")
save(p, "04_subtract")

# --------------------------------------------------------------------------
# 2. Holes and islands nested inside holes
# --------------------------------------------------------------------------


def square(cx, cy, size):
    h = size / 2
    return cad.Polygon([cx - h, cx + h, cx + h, cx - h], [cy - h, cy - h, cy + h, cy + h])


outer = square(0, 0, 20)
hole = square(0, 0, 10)
island = square(0, 0, 4)

plane = cad.subtract_polygons((outer,), (hole,))[0]
with_island = cad.add_polygons(plane, island)

p = GeometryPlotter(figsize=(6, 6), title="Island nested inside a hole")
p.add_polygon(with_island, facecolor="#8172B2")
save(p, "05_island_in_hole")

# --------------------------------------------------------------------------
# 3. dezigzag(): before/after on a trace with a small sideways step
#    artifact -- the kind a tessellated circle undercutting a straight
#    segment's tangent point leaves behind (see kernel.api.dezigzag_polyline).
# --------------------------------------------------------------------------

# top edge has a short "kink" (a 0.4-unit sideways step) sandwiched
# between two long (~13-unit) straight runs; bottom edge is flat.
kinked = cad.Polygon(
    [0, 12, 12, 25, 38, 38, 0],
    [0, 0, 0.4, 0.4, 0.4, -4, -4],
)
before_xs, before_ys = list(kinked.xs), list(kinked.ys)
n_before = len(before_xs)

kinked.dezigzag(1.0, max_angle_deg=25.0, min_neighbor_factor=2.0)
n_after = len(kinked.xs)

p = GeometryPlotter(figsize=(8, 4), equal_aspect=False,
                     title=f"Polygon.dezigzag(): {n_before} -> {n_after} vertices (zoomed on the kink)")
p.ax.plot(before_xs + [before_xs[0]], before_ys + [before_ys[0]],
          color="#C44E52", linewidth=2.2, linestyle="--", label=f"before ({n_before} verts)", marker="o")
p.ax.plot(kinked.xs + [kinked.xs[0]], kinked.ys + [kinked.ys[0]],
          color="#4C72B0", linewidth=2.2, label=f"after ({n_after} verts)", marker="o")
p.ax.set_xlim(6, 30)
p.ax.set_ylim(-0.15, 0.65)
p.legend(loc="lower right")
save(p, "06_dezigzag")

# --------------------------------------------------------------------------
# 4. via_wall_polygons(): a dense via cloud fused into a thickened wall
# --------------------------------------------------------------------------

import numpy as np

# a closed ring (fuses into an annulus with a hole), a straight run well
# clear of it (fuses into one connecting segment), and two vias each far
# enough from everything to stay isolated.
n_loop = 20
radius = 10.0
loop_pts = [
    (radius * math.cos(2 * math.pi * k / n_loop), radius * math.sin(2 * math.pi * k / n_loop))
    for k in range(n_loop)
]
straight_run = [(16.0 + i * 1.0, 0.0) for i in range(8)]
isolated = [(-22.0, 18.0), (-22.0, 12.0)]
via_xy = np.array(loop_pts + straight_run + isolated)

max_dist = 3.5   # > ring spacing (~3.14) and straight-run spacing (1.0),
                  # but well under the ~6-unit ring-to-run gap and the
                  # isolated points' own distance from everything else.
walls = via_wall_polygons(via_xy, max_dist=max_dist, thickness=1.2, circle_segments=16,
                           include_isolated_vias=True)

p = GeometryPlotter(figsize=(7, 7), title=f"via_wall_polygons(): {len(via_xy)} vias -> {len(walls)} polygon(s)")
p.add_scatter(via_xy[:, 0], via_xy[:, 1], s=10, zorder=5)
p.add_polygon(walls, facecolor="#B87333", alpha=0.7)
save(p, "07_via_walls")

print("done")
