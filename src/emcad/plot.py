from __future__ import annotations

import itertools
from typing import Iterable, Union, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

# A palette of visually distinct colors used to auto-color successive
# polygons when no explicit facecolor is given, so multi-polygon plots
# don't all default to the same blue.
_DISTINCT_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray",
]

# Import your geometry primitives. Adjust the module name below to match
# wherever Vertex / Edge / Polygon actually live in your project.
from .poly import Vertex, Edge, Polygon


# Convenience type aliases: every "add_*" method accepts either a single
# object or an iterable of objects.
VertexLike = Union[Vertex, Iterable[Vertex]]
EdgeLike = Union[Edge, Iterable[Edge]]
PolygonLike = Union[Polygon, Iterable[Polygon]]


class GeometryPlotter:
    """A small matplotlib wrapper for visualizing Vertex / Edge / Polygon
    objects (with optional holes) plus generic scatter data.

    Usage
    -----
    >>> plotter = GeometryPlotter()
    >>> plotter.add_polygon(poly)  # auto-colored; pass facecolor=... to override
    >>> plotter.add_edge(edge, color="red")
    >>> plotter.add_vertex(vertex, label=True)
    >>> plotter.add_scatter(xs, ys, color="green", marker="x")
    >>> plotter.show()
    """

    def __init__(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple[float, float] = (8, 8),
        equal_aspect: bool = True,
        title: Optional[str] = None,
    ):
        if ax is not None:
            self.fig = ax.figure
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(figsize=figsize)

        if equal_aspect:
            self.ax.set_aspect("equal", adjustable="datalim")

        if title:
            self.ax.set_title(title)

        self.ax.grid(True, linestyle=":", alpha=0.4)

        # Keep track of artists in case the caller wants to inspect/remove them.
        self.artists: list[Any] = []

        # Cycles through _DISTINCT_COLORS so successive polygons (across
        # separate add_polygon calls too) get visibly different fill
        # colors when the caller doesn't specify one explicitly.
        self._polygon_color_cycle = itertools.cycle(_DISTINCT_COLORS)

    # ------------------------------------------------------------------ #
    #                             INTERNAL                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_list(obj, single_type) -> list:
        """Normalize `obj` (single instance or iterable) into a list."""
        if isinstance(obj, single_type):
            return [obj]
        return list(obj)

    @staticmethod
    def _ring_signed_area(xs: list[float], ys: list[float]) -> float:
        n = len(xs)
        a = 0.0
        for i in range(n):
            j = (i + 1) % n
            a += xs[i] * ys[j] - xs[j] * ys[i]
        return a * 0.5

    @classmethod
    def _polygon_to_path(cls, polygon: Polygon) -> mpath.Path:
        """Build a single compound matplotlib Path for a polygon and
        every hole it carries, at every nesting depth (a hole's own
        nested holes are islands, and so on), so the fill is correctly
        punched out where holes are and filled back in for islands.

        matplotlib's `PathPatch` fills using the NONZERO winding rule,
        not even-odd -- unlike the even-odd rule this package's own
        containment tests use everywhere else (`Polygon.is_inside` etc,
        which don't care about winding direction at all), getting a
        hole to actually punch out here requires each ring's winding to
        alternate with its nesting depth (solid, hole, island, hole-of-
        island, ...) -- so that's enforced explicitly below, regardless
        of whatever winding direction a ring already happens to have.
        """
        vertices: list[tuple[float, float]] = []
        codes: list[int] = []

        def add_ring(xs: list[float], ys: list[float], want_ccw: bool) -> None:
            n = len(xs)
            is_ccw = cls._ring_signed_area(xs, ys) > 0
            if is_ccw != want_ccw:
                xs = list(reversed(xs))
                ys = list(reversed(ys))
            vertices.append((xs[0], ys[0]))
            codes.append(mpath.Path.MOVETO)
            for i in range(1, n):
                vertices.append((xs[i], ys[i]))
                codes.append(mpath.Path.LINETO)
            # close back to the first point of this ring
            vertices.append((xs[0], ys[0]))
            codes.append(mpath.Path.CLOSEPOLY)

        def walk(poly: Polygon, depth: int) -> None:
            add_ring(poly.xs, poly.ys, want_ccw=(depth % 2 == 0))
            for hole in poly.holes:
                walk(hole, depth + 1)

        walk(polygon, 0)
        return mpath.Path(vertices, codes)

    # ------------------------------------------------------------------ #
    #                              SCATTER                              #
    # ------------------------------------------------------------------ #

    def add_scatter(
        self,
        xs: Iterable[float],
        ys: Iterable[float],
        zs: Optional[Iterable[float]] = None,
        **kwargs,
    ) -> Any:
        """Add a scatter plot.

        If `zs` is given, it is used to color the points via a colormap
        (since this plotter is 2D by default and doesn't project a third
        spatial axis). Pass e.g. `cmap="viridis"` to customize.
        """
        xs = list(xs)
        ys = list(ys)

        if zs is not None:
            zs = list(zs)
            kwargs.setdefault("cmap", "viridis")
            sc = self.ax.scatter(xs, ys, c=zs, **kwargs)
            self.fig.colorbar(sc, ax=self.ax, shrink=0.8, label=kwargs.get("zlabel", "z"))
        else:
            kwargs.setdefault("color", "red")
            kwargs.setdefault("s", 10)
            sc = self.ax.scatter(xs, ys, **kwargs)

        self.artists.append(sc)
        return sc

    # ------------------------------------------------------------------ #
    #                              VERTEX                                #
    # ------------------------------------------------------------------ #

    def add_vertex(
        self,
        vertex: VertexLike,
        label: bool = False,
        color: str = "black",
        marker: str = "o",
        size: float = 14.0,
        **kwargs,
    ) -> Any:
        """Add one Vertex or a list of Vertex objects as points."""
        vertices = self._as_list(vertex, Vertex)
        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]

        sc = self.ax.scatter(xs, ys, color=color, marker=marker, s=size, **kwargs)
        self.artists.append(sc)

        if label:
            for i, v in enumerate(vertices):
                txt = self.ax.annotate(
                    str(i),
                    (v.x, v.y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color=color,
                )
                self.artists.append(txt)

        return sc

    # ------------------------------------------------------------------ #
    #                               EDGE                                 #
    # ------------------------------------------------------------------ #

    def add_edge(
        self,
        edge: EdgeLike,
        color: str = "black",
        linewidth: float = 1.5,
        linestyle: str = "-",
        **kwargs,
    ) -> list:
        """Add one Edge or a list of Edge objects as line segments."""
        edges = self._as_list(edge, Edge)
        lines = []
        for e in edges:
            (line,) = self.ax.plot(
                [e.v1.x, e.v2.x],
                [e.v1.y, e.v2.y],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                **kwargs,
            )
            lines.append(line)
            self.artists.append(line)
        return lines

    # ------------------------------------------------------------------ #
    #                              POLYGON                               #
    # ------------------------------------------------------------------ #

    def add_polygon(
        self,
        polygon: PolygonLike,
        facecolor: Optional[str] = None,
        edgecolor: str = "black",
        alpha: float = 0.5,
        linewidth: float = 1.5,
        hatch: Optional[str] = None,
        hole_edgecolor: str = "red",
        hole_linestyle: str = "--",
        show_hole_outline: bool = True,
        **kwargs,
    ) -> list:
        """Add one Polygon or a list of Polygon objects.

        Holes are handled by building a single compound matplotlib Path
        (outer ring + hole rings) so the fill is correctly punched out --
        matplotlib has no native "hole" concept for filled patches, but
        even-odd filling of a compound path achieves the same visual
        result. To make holes unambiguous at a glance (since a "punched
        out" region can otherwise be confused with plain background),
        each hole's boundary is additionally drawn with a dashed outline
        in `hole_edgecolor`.

        If `facecolor` is left as None, each polygon (including across
        separate calls to this method) is assigned the next color from a
        distinct color palette, so multiple polygons are easy to tell
        apart. Pass an explicit `facecolor` to override this for all
        polygons in this call.
        """
        polygons = self._as_list(polygon, Polygon)
        patches = []

        for poly in polygons:
            this_facecolor = facecolor if facecolor is not None else next(self._polygon_color_cycle)
            path = self._polygon_to_path(poly)
            patch = mpatches.PathPatch(
                path,
                facecolor=this_facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
                hatch=hatch,
                **kwargs,
            )
            self.ax.add_patch(patch)
            patches.append(patch)
            self.artists.append(patch)

            if show_hole_outline:
                def draw_hole_outlines(p: Polygon) -> None:
                    for hole in p.holes:
                        (hole_line,) = self.ax.plot(
                            hole.cxs,
                            hole.cys,
                            color=hole_edgecolor,
                            linestyle=hole_linestyle,
                            linewidth=linewidth,
                        )
                        self.artists.append(hole_line)
                        draw_hole_outlines(hole)   # islands can carry their own holes too

                draw_hole_outlines(poly)

        self.ax.autoscale_view()
        return patches

    # ------------------------------------------------------------------ #
    #                              DISPLAY                               #
    # ------------------------------------------------------------------ #

    def legend(self, **kwargs) -> None:
        self.ax.legend(**kwargs)

    def show(self) -> None:
        self.ax.relim()
        self.ax.autoscale_view()
        plt.show()

    def save(self, path: str, dpi: int = 200, **kwargs) -> None:
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)