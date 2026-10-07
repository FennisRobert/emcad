from __future__ import annotations
from .glob import GlobalSettings
from dataclasses import dataclass
from typing import Iterable, Generator
import numpy as np
from .kernel.api import edge_self_intersections, edge_cross_intersections, is_inside, dekeyhole_polygon, simplify_polyline, dezigzag_polyline, is_simple_ring, regularize_polyline
from . import _rs

class GeometryException(Exception):
    pass

@dataclass
class Vertex:
    x: float
    y: float
    
    @staticmethod
    def parse(xy: Iterable[float]) -> Vertex:
        return Vertex(*xy)
    
@dataclass
class Edge:
    v1: Vertex
    v2: Vertex

    @property
    def complex(self) -> complex:
        """ Complex number representing the edge """
        return (self.v2.x-self.v1.x) + 1j * (self.v2.y - self.v1.y)
    
class Polygon:
    """Generic polygon class as list of points
    End of polygon != begin of polygon

    A polygon may carry `holes`: other Polygon instances (which may
    themselves carry holes, representing islands-inside-holes, and so
    on to arbitrary depth) nested inside this one's outer boundary.
    Containment/inclusion tests use the even-odd rule across every
    ring (this polygon's own boundary plus every hole at every depth)
    at once, which handles that nesting correctly without any explicit
    depth bookkeeping.
    """
    def __init__(self, xs: Iterable[float], ys: Iterable[float], holes: list[Polygon] | None = None):
        self.xs: list[float] = list(xs)
        self.ys: list[float] = list(ys)
        
        if holes is None:
            holes = []
        if isinstance(holes, Polygon):
            holes = [holes,]
        self.holes: list[Polygon] = holes

        self._validate()
        self._check()

    @classmethod
    def _construct_verified(cls, xs: Iterable[float], ys: Iterable[float],
                             holes: list[Polygon] | None = None) -> Polygon:
        """Build a Polygon skipping `_check_holes`'s edge-crossing
        nesting validation, for internal callers (e.g. `dekeyhole_polygon`)
        that have already verified correct nesting themselves via
        containment testing.

        Containment testing is actually the *more* appropriate check
        for geometry an algorithm just derived from real input: it can
        legitimately touch its parent at exactly the point or edge it
        was extracted from (that's the whole nature of a keyhole
        junction), which `_check_holes`'s exact edge-crossing detection
        would reject -- appropriately, when it's guarding against
        mistakes in a user's own manually-constructed `holes=`, not
        when the touch is expected and already independently verified.

        Every other validation still runs (`xs`/`ys` length match,
        closing-point stripping, minimum 3 points) -- only the nesting
        check on `holes` is skipped.
        """
        obj = cls.__new__(cls)
        obj.xs = list(xs)
        obj.ys = list(ys)

        if holes is None:
            holes = []
        if isinstance(holes, Polygon):
            holes = [holes]
        obj.holes = holes

        obj._validate()
        if len(obj.xs) != len(obj.ys):
            raise GeometryException(f"Length of xs({len(obj.xs)}) and ys({len(obj.ys)}) is not equal.")
        if len(obj.xs) < 3:
            raise GeometryException(f"Number of points must be 3 or larger. Not {len(obj.xs)}.")
        return obj
    

    ############################################################
    #                          PROPERTIES                     #
    ############################################################
    @property
    def cxs(self) -> list[float]:
        return self.xs + [self.xs[0],]
    
    @property
    def cys(self) -> list[float]:
        return self.ys + [self.ys[0],]
    
    @property
    def axs(self) -> np.ndarray:
        return np.array(self.xs)
    
    @property
    def ays(self) -> np.ndarray:
        return np.array(self.ys)
    
    @property
    def acxs(self) -> np.ndarray:
        return np.array(self.cxs)
    
    @property
    def acys(self) -> np.ndarray:
        return np.array(self.cys)

    @property
    def has_holes(self) -> bool:
        return len(self.holes) > 0

    @property
    def n_points(self) -> int:
        return len(self.xs)

    @property
    def n_edges(self) -> int:
        return len(self.xs)

    def point_inside(self) -> tuple[float, float]:
        """Find a point guaranteed to be inside this polygon's solid
        material -- i.e. inside the outer boundary and outside every
        hole (at every nesting depth).

        Same "ear under the topmost vertex" trick as the hole-free
        version, generalized to look at every ring at once: take the
        globally topmost vertex across *all* rings, test a horizontal
        line just below it, and gather x-crossings from every ring. The
        region between the two leftmost crossings is always solid under
        the even-odd rule, regardless of which ring each crossing came
        from or how deep the nesting goes -- far enough to the left of
        everything is unambiguously outside, and the first crossing
        flips that to "inside" no matter what.
        """
        return _rs.polygon_point_inside(self)

    def remove_keyholes(self) -> None:
            new_poly = dekeyhole_polygon(self)
            self.xs = new_poly.xs
            self.ys = new_poly.ys
            self.holes = new_poly.holes
            self._validate()
            self._check()

    def simplify(self, delta: float) -> None:
        """Re-run Ramer-Douglas-Peucker simplification (see
        `kernel.api.simplify_polyline`) on this polygon's own boundary,
        in place -- and, recursively, on every hole it carries, at
        every nesting depth.

        Best-effort, same guarantee as `.dezigzag()` below: even though
        RDP only ever REMOVES points, cutting straight from one kept
        point to the next can still cut a corner that was deliberately
        routing around a nearby hole or sibling ring closer than
        `delta`, introducing a crossing that didn't exist before. Every
        ring is simplified and validated as a candidate first; any ring
        whose change would break validity is silently kept at its
        ORIGINAL, untouched shape instead.

        Args:
            delta: maximum perpendicular deviation (same units as
                xs/ys) a dropped point may have from the simplified
                line -- larger delta simplifies more aggressively.
        """
        self._apply_ring_transform(lambda xs, ys: simplify_polyline(xs, ys, delta))

    def dezigzag(self, max_kink_length: float, max_angle_deg: float = 20.0,
                 min_neighbor_factor: float = 3.0) -> None:
        """Remove short zigzag/step artifacts from this polygon's own
        boundary, in place -- and, recursively, from every hole it
        carries, at every nesting depth. See
        `kernel.api.dezigzag_polyline` for the exact algorithm and
        parameter meanings: collapses a short "kink" segment sandwiched
        between two much longer, near-parallel segments (e.g. a round
        trace end-cap's polygon approximation slightly missing the
        tangent point where it should cleanly meet a straight segment)
        by extending those two long segments until they meet.

        Best-effort, same guarantee as `.simplify()` above: when a hole
        or sibling ring sits closer to the kink than `max_kink_length`,
        collapsing it can occasionally make the cleaned-up ring cross
        another ring that it didn't cross before. Every ring is
        dezigzagged and validated as a candidate first; any ring whose
        change would break validity (on its own, against a sibling
        hole, or against the parent boundary) is silently kept at its
        ORIGINAL, untouched shape instead -- this is a cosmetic cleanup
        pass and must never be able to turn a valid design invalid.

        Args:
            max_kink_length: a segment shorter than this (same units as
                xs/ys) is a kink candidate -- should be well under your
                smallest genuine feature size.
            max_angle_deg: maximum direction difference allowed between
                a kink's two neighboring segments for it to still count
                as "basically straight".
            min_neighbor_factor: each neighboring segment must be at
                least `max_kink_length * min_neighbor_factor` long.
        """
        self._apply_ring_transform(
            lambda xs, ys: dezigzag_polyline(xs, ys, max_kink_length, max_angle_deg, min_neighbor_factor)
        )

    def regularize(self, tol: float, dangle_deg: float = 5.0, angle_tol_deg: float = 0.05,
                   min_anchor_len: float | None = None, offset_tol: float = 5e-6,
                   vw_area: float | None = None) -> None:
        """Map-making style outline regularization, in place -- and,
        recursively, on every hole at every nesting depth. Snaps the
        outline back onto its dominant straight lines (directions on a
        `dangle_deg` grid), drops protrusions smaller than `tol` (e.g.
        via pads poking out of a copper edge) and rebuilds corners
        hidden under them, without the edge skew RDP (`.simplify()`)
        produces on that kind of outline. Real arcs are kept and lightly
        simplified. See `kernel.api.regularize_polyline` for the exact
        algorithm and every parameter.

        Best-effort with the same fail-safe as `.simplify()`/
        `.dezigzag()`: any ring whose regularized shape would
        self-intersect, or break hole nesting, keeps its original shape.

        Args:
            tol: largest detour (same units as xs/ys) that may be
                flattened -- bigger than the bumps to remove, smaller
                than the smallest real feature to keep.
            dangle_deg: snap-angle step for straight edges.
        """
        self._apply_ring_transform(
            lambda xs, ys: regularize_polyline(xs, ys, tol, dangle_deg, angle_tol_deg, min_anchor_len,
                                               offset_tol, vw_area)
        )

    def _apply_ring_transform(self, ring_fn) -> None:
        """Shared fail-safe machinery for `.simplify()`/`.dezigzag()`:
        build a fully-transformed, freshly-validated candidate (this
        polygon's own boundary, and recursively every hole, each run
        through `ring_fn(xs, ys) -> (xs, ys)`), and commit it in place
        only if the WHOLE result validates. Left completely untouched
        if not, since either caller is a cosmetic cleanup pass that
        must never be able to turn a valid design invalid.

        Catches both `GeometryException` (invalid nesting/crossing,
        from the `Polygon(...)` constructor below) and `ValueError`
        (from `simplify_polyline`/`sanitize_polygon`, raised when a
        ring is small enough relative to the transform's own tolerance
        -- e.g. `delta`/`max_kink_length` -- that it collapses to too
        few distinct points to remain a valid ring) -- both mean the
        same thing here: this specific tolerance is too coarse for
        this specific ring, so leave it alone rather than propagate
        the error.
        """
        try:
            candidate = self._ring_transform_candidate(ring_fn)
        except (GeometryException, ValueError):
            return
        self.xs = candidate.xs
        self.ys = candidate.ys
        self.holes = candidate.holes

    def _ring_transform_candidate(self, ring_fn) -> Polygon:
        """Build a candidate copy of this polygon with `ring_fn` applied
        to its own closed ring and, recursively, to every hole -- raises
        `GeometryException` or `ValueError` (see `_apply_ring_transform`)
        if the combined result isn't valid, which `_apply_ring_transform`
        catches to fall back to the untouched original. A hole whose OWN
        candidate fails is kept at its original shape rather than
        aborting the whole polygon, so one problematic ring doesn't
        block cleanup of its unrelated siblings.

        Checks the transformed ring's own SIMPLICITY explicitly (no
        self-crossing), not just its nesting relative to other rings --
        `Polygon.__init__`'s own validation never did that (it only
        checks a hole/parent relationship between DIFFERENT rings), so
        without this a transform that moves a vertex enough to fold a
        ring back over its own tight corners (a real risk on
        intricate, thin geometry like text-trace glyph strokes) would
        silently produce a self-intersecting sliver instead of being
        caught and reverted like every other invalid case here.
        """
        xs, ys = ring_fn(self.cxs, self.cys)
        if not is_simple_ring(xs, ys):
            raise GeometryException("dezigzag/simplify candidate ring self-intersects")
        new_holes: list[Polygon] = []
        for hole in self.holes:
            try:
                new_holes.append(hole._ring_transform_candidate(ring_fn))
            except (GeometryException, ValueError):
                new_holes.append(hole)
        return Polygon(list(xs), list(ys), holes=new_holes)
    ############################################################
    #                       PRIVATE FUNCTIONS                  #
    ############################################################

    def _validate(self) -> None:
        
        # Fix last point != first point
        if abs(self.xs[-1] - self.xs[0]) < GlobalSettings.EPS and abs(self.ys[-1] - self.ys[0]) < GlobalSettings.EPS:
            self.xs.pop(-1)
            self.ys.pop(-1)
        
    def _check(self) -> None:
        
        if len(self.xs) != len(self.ys):
            raise GeometryException(f"Length of xs({len(self.xs)}) and ys({len(self.ys)}) is not equal.")
        
        if len(self.xs) < 3:
            raise GeometryException(f"Number of points must be 3 or larger. Not {len(self.xs)}.")

        if self.has_holes:
            self._check_holes()

    def _check_holes(self) -> None:
        """Sanity-check that every hole is actually a hole: a Polygon,
        fully nested inside this polygon's own outer boundary, and not
        crossing (or coinciding with the nesting of) any sibling hole.

        This only checks *this* polygon's own outer ring against each
        hole's outer ring -- a hole's own nested holes (islands, etc.)
        are validated independently when that hole itself was
        constructed, since `_check` runs in every Polygon's `__init__`.

        Note: uses exact edge-crossing detection, so a hole that merely
        *touches* this polygon's boundary (shares a vertex or lies
        tangent along an edge) will also raise -- there's no tolerance
        band here like `poly_fragment`'s `t_tol` classification has.
        """
        for hole in self.holes:
            if not isinstance(hole, Polygon):
                raise GeometryException("Every entry in `holes` must be a Polygon instance.")

            self._check_ring_nested_in_self(hole, label="hole")

        for i, hole_a in enumerate(self.holes):
            for hole_b in self.holes[i + 1:]:
                ids, _ = edge_cross_intersections(*hole_a._edge_points(), *hole_b._edge_points())
                if ids.shape[1] > 0:
                    raise GeometryException("Two holes of the same polygon cross each other.")
                if is_inside(hole_a.axs, hole_a.ays, *hole_b.point_inside(), True):
                    raise GeometryException(
                        "One hole is nested inside another sibling hole -- express that as "
                        "a hole-of-a-hole (i.e. an island), not as two entries in the same "
                        "`holes` list."
                    )

    def _check_ring_nested_in_self(self, other: Polygon, label: str) -> None:
        ids, _ = edge_cross_intersections(*self._edge_points(), *other._edge_points())
        if ids.shape[1] > 0:
            raise GeometryException(f"A {label}'s boundary crosses this polygon's own boundary.")

        ox, oy = other.point_inside()
        if not is_inside(self.axs, self.ays, ox, oy, True):
            raise GeometryException(f"A {label} is not nested inside this polygon's own boundary.")

    def _all_loops(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Flatten this polygon and every (recursively) nested hole into
        a flat list of (xs, ys) rings, for the even-odd tests in
        `point_inside`/`is_inside` that need to see every boundary at
        once. Each ring uses the open (non-closed) coordinate form --
        same convention as `self.axs`/`self.ays`.
        """
        loops = [(self.axs, self.ays)]
        for hole in self.holes:
            loops.extend(hole._all_loops())
        return loops

    def _iter_iivve(self) -> Generator[tuple[int, int, Vertex, Vertex, Edge], None, None]:
        n = self.n_points
        for iv1 in range(n):
            v1 = Vertex(self.xs[iv1], self.ys[iv1])
            v2 = Vertex(self.xs[(iv1+1)%n], self.ys[(iv1+1)%n])

            yield (iv1, (iv1+1)%n,  v1, v2, Edge(v1,v2))

    def _edge_points(self) -> tuple[np.ndarray, np.ndarray]:
        x1 = self.acxs
        y1 = self.acys

        pts1s = np.vstack([x1[:-1], y1[:-1]])
        pts1e = np.vstack([x1[1:], y1[1:]])
        return (pts1s, pts1e)
    ############################################################
    #                          OPERATIONS                     #
    ############################################################

    def intersections(self, other: Polygon) -> tuple[np.ndarray, np.ndarray]:
        """Returns the intersection coordinates between two polygons

        Args:
            other (Polygon): The other polygon

        Returns:
            np.ndarray: A list of intersection cooridinates
            np.ndarray: A (2,N) list of edge ids who have been crossed
        """
        pts1s, pts1e = self._edge_points()
        pts2s, pts2e = other._edge_points()
        ids, coords = edge_cross_intersections(pts1s, pts1e, pts2s, pts2e)
        return coords, ids

    def is_inside(self, x: float, y: float, include_boundary: bool = True) -> bool:
        """Even-odd point-in-polygon-with-holes test.

        XORs the single-ring inclusion result across this polygon's own
        boundary and every (recursively) nested hole boundary. This
        handles holes, islands-inside-holes, holes-inside-those-islands,
        etc. automatically -- no explicit depth tracking needed, same
        reasoning as `point_inside`.

        Boundary handling: if the point lies exactly on *any* ring's
        edge (this polygon's own, or any hole's, at any depth), the
        point counts as touching the material's boundary and
        `include_boundary` decides the result -- same convention as a
        simple polygon.
        """
        return _rs.polygon_is_inside(self, x, y, include_boundary)
