"""Save/load a `PCBView`'s resolved world-space geometry to a single
binary file, so a script that runs against the same board repeatedly
(e.g. iterating on downstream `emerge_interface`/simulation code) can
skip re-parsing the ODB++ file tree and re-resolving every feature
(symbol expansion, arc tessellation, line/arc buffering) on every run --
by far the two most expensive steps before `resolve_layer_polygons`'s
own boolean-kernel work even starts.

What's cached is the OUTPUT of feature resolution -- flat, already
world-space `LayerPolygon`/`DrillHole` data per geometry layer, plus the
board outline and Z-stack -- not the raw `ProductModel` object graph and
not the boolean-unified result (that stays a fresh, parameter-dependent
`resolve_layer_polygons()` call, same as it always was, just now running
against cached input instead of freshly-parsed input).

`CachedPCBView` is a drop-in, duck-typed stand-in for `PCBView`:
it implements the exact same public surface (`get_board_polygon`,
`get_board_holes`, `centralize`, `iter_pcb_layers`, `iter_geo_layers`,
`iter_drill_holes`, `resolve_layer_polygons`, `plot_layers`), sharing
`design.py`'s own `resolve_layer_polygons`/`plot_layers_for` functions
rather than reimplementing them -- so any code written against
`PCBView` (including `emerge_interface.ODBImport`) works unmodified
against a loaded cache.

File format: msgpack (binary, fast to encode/decode, no XML/JSON text
overhead), with every polygon ring's (x, y) coordinates stored as a
raw float64 buffer (`ring.tobytes()` / `np.frombuffer(...)`) inside a
msgpack "bin" blob rather than as a nested list of numbers -- a plain
list-of-floats has real Python-level per-element overhead both to
encode and decode at these point counts (a single dense layer can have
hundreds of thousands of vertices total, see the module's own
motivating case), while a raw buffer is a single memcpy each way.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import msgpack
import numpy as np
from loguru import logger

from .design import DrillHole, GeoLayer, LayerPolygon, XY, plot_layers_for
from .design import resolve_layer_polygons as _resolve_layer_polygons

FORMAT_VERSION = 1


# --------------------------------------------------------------------------
# Ring encode/decode
# --------------------------------------------------------------------------

def _encode_ring(xs: Iterable[float], ys: Iterable[float]) -> dict:
    arr = np.empty((len(xs) if hasattr(xs, "__len__") else len(list(xs)), 2), dtype=np.float64)
    arr[:, 0] = np.asarray(xs, dtype=np.float64)
    arr[:, 1] = np.asarray(ys, dtype=np.float64)
    return {"n": arr.shape[0], "data": arr.tobytes()}


def _decode_ring(d: dict) -> Tuple[List[float], List[float]]:
    arr = np.frombuffer(d["data"], dtype=np.float64).reshape(d["n"], 2)
    return arr[:, 0].tolist(), arr[:, 1].tolist()


# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------

def save_pcb_cache(pcbd, path: Union[str, Path]) -> None:
    """Snapshot everything `resolve_layer_polygons`/`plot_layers`/
    `emerge_interface.ODBImport` need from a `PCBView` (real or
    already-cached -- both implement the same interface) into `path`.

    Args:
        pcbd: a `PCBView` (or `CachedPCBView`) instance.
        path: output file path. Convention: `.pcbcache`.
    """
    path = Path(path)
    logger.debug(f"save_pcb_cache: snapshotting '{pcbd.step_name}' -> '{path}'")

    geo_layers = []
    for layer in pcbd.iter_geo_layers():
        polygons = []
        for lp in layer.iter_polygons():
            polygons.append({
                "outer": _encode_ring(*lp.xys()),
                "holes": [_encode_ring(*h) for h in lp.holes_xys()],
                "polarity": lp.polarity,
            })
        geo_layers.append({
            "name": layer.name, "type": layer.type, "z1": layer.z1, "z2": layer.z2,
            "polygons": polygons,
        })

    payload = {
        "format_version": FORMAT_VERSION,
        "step_name": pcbd.step_name,
        "board_outline": _encode_ring(*pcbd.get_board_polygon()),
        "board_holes": [_encode_ring(xs, ys) for xs, ys in pcbd.get_board_holes()],
        "pcb_layers": [{"z1": z1, "z2": z2, "material": material} for z1, z2, material in pcbd.iter_pcb_layers()],
        "geo_layers": geo_layers,
        "drill_holes": [
            {"x": h.x, "y": h.y, "diameter": h.diameter, "z1": h.z1, "z2": h.z2,
             "plated": h.plated, "layer_name": h.layer_name}
            for h in pcbd.iter_drill_holes()
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(msgpack.packb(payload, use_bin_type=True))

    n_polys = sum(len(gl["polygons"]) for gl in geo_layers)
    logger.debug(
        f"save_pcb_cache: {len(geo_layers)} geo layer(s), {n_polys} raw polygon(s), "
        f"{len(payload['drill_holes'])} drill hole(s) -> {path.stat().st_size / 1e6:.2f}MB"
    )


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_pcb_cache(path: Union[str, Path]) -> "CachedPCBView":
    """Load a cache written by `save_pcb_cache` -- no ODB++ parsing, no
    feature resolution, just deserializing already-resolved geometry.
    """
    path = Path(path)
    with open(path, "rb") as f:
        payload = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)

    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported PCB cache format_version={payload.get('format_version')!r} "
            f"(expected {FORMAT_VERSION}) in '{path}' -- re-save with the current emcad version"
        )

    designer = CachedPCBView(payload)
    logger.debug(
        f"load_pcb_cache: '{path}' -- step '{designer.step_name}', "
        f"{len(designer._geo_layers)} geo layer(s), {len(designer._drill_holes)} drill hole(s)"
    )
    return designer


# --------------------------------------------------------------------------
# CachedPCBView -- duck-typed PCBView stand-in
# --------------------------------------------------------------------------

class CachedPCBView:
    """A `PCBView`-compatible view over a loaded cache payload. Not
    constructed directly -- use `load_pcb_cache`.

    Implements the same public surface `PCBView` does (see the
    module docstring), including the same `_offset_x`/`_offset_y`
    `centralize()` pattern -- coordinates in the cache are whatever
    offset state the source `PCBView` was in when it was saved
    (baseline `_offset_x=0, _offset_y=0` from this instance's own point
    of view), and `centralize()`/further offsetting still works exactly
    like the real class.
    """

    def __init__(self, payload: dict):
        self._payload = payload
        self.step_name: str = payload["step_name"]
        #: always None -- a cache stores already-resolved geometry, not
        #: the raw ProductModel object graph. Present (rather than just
        #: absent) so code reading `.pm` off either a PCBView or a
        #: CachedPCBView doesn't need an isinstance check first.
        self.pm = None
        self._offset_x = 0.0
        self._offset_y = 0.0

        self._board_outline = _decode_ring(payload["board_outline"])
        self._board_holes = [_decode_ring(h) for h in payload["board_holes"]]
        self._pcb_layers_raw = [(pl["z1"], pl["z2"], pl["material"]) for pl in payload["pcb_layers"]]

        self._geo_layers = payload["geo_layers"]
        self._polygons_by_layer: Dict[str, List[dict]] = {
            gl["name"]: gl["polygons"] for gl in self._geo_layers
        }

        self._drill_holes = [
            DrillHole(x=d["x"], y=d["y"], diameter=d["diameter"], z1=d["z1"], z2=d["z2"],
                      plated=d["plated"], layer_name=d["layer_name"])
            for d in payload["drill_holes"]
        ]

        #: mirrors PCBView.skipped_features -- always empty here,
        #: since a cache only ever stores already-successfully-resolved
        #: geometry; nothing to inspect after the fact.
        self.skipped_features: list = []

    # ---- board outline -----------------------------------------------------

    def get_board_polygon(self) -> Tuple[List[float], List[float]]:
        xs, ys = self._board_outline
        return [x + self._offset_x for x in xs], [y + self._offset_y for y in ys]

    def get_board_holes(self) -> List[Tuple[List[float], List[float]]]:
        return [
            ([x + self._offset_x for x in xs], [y + self._offset_y for y in ys])
            for xs, ys in self._board_holes
        ]

    def centralize(self) -> None:
        xs, ys = self.get_board_polygon()
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        self._offset_x -= cx
        self._offset_y -= cy

    # ---- Z stack -------------------------------------------------------------

    def iter_pcb_layers(self) -> Iterator[Tuple[float, float, str]]:
        for z1, z2, material in self._pcb_layers_raw:
            yield z1, z2, material

    # ---- geometry layers -------------------------------------------------

    def iter_geo_layers(self) -> Iterator[GeoLayer]:
        for gl in self._geo_layers:
            yield GeoLayer(name=gl["name"], type=gl["type"], z1=gl["z1"], z2=gl["z2"],
                            matrix_layer=None, _designer=self)

    def _layer_polygons(self, geo_layer: GeoLayer) -> Iterator[LayerPolygon]:
        ox, oy = self._offset_x, self._offset_y
        for p in self._polygons_by_layer.get(geo_layer.name, []):
            xs, ys = _decode_ring(p["outer"])
            holes = [_decode_ring(h) for h in p["holes"]]
            if ox or oy:
                xs = [x + ox for x in xs]
                ys = [y + oy for y in ys]
                holes = [([x + ox for x in hxs], [y + oy for y in hys]) for hxs, hys in holes]
            yield LayerPolygon(points=list(zip(xs, ys)), holes=[list(zip(hxs, hys)) for hxs, hys in holes],
                                z=geo_layer.z, polarity=p["polarity"], layer_name=geo_layer.name, source=None)

    def resolve_layer_polygons(
        self,
        layer: "GeoLayer",
        simplify_delta: float = 1e-6,
        dezigzag: bool = True,
        dezigzag_max_kink_length: Optional[float] = None,
        dezigzag_max_angle_deg: float = 20.0,
        dezigzag_min_neighbor_factor: float = 3.0,
        post_simplify: bool = True,
        post_simplify_delta: Optional[float] = None,
        merge_tol: Optional[float] = None,
        regularize: bool = False,
        regularize_tol: Optional[float] = None,
        regularize_dangle_deg: float = 5.0,
    ) -> list:
        return _resolve_layer_polygons(
            layer, simplify_delta, dezigzag, dezigzag_max_kink_length,
            dezigzag_max_angle_deg, dezigzag_min_neighbor_factor,
            post_simplify, post_simplify_delta, merge_tol,
            regularize, regularize_tol, regularize_dangle_deg,
        )

    # ---- drill holes ---------------------------------------------------------

    def iter_drill_holes(self) -> Iterator[DrillHole]:
        ox, oy = self._offset_x, self._offset_y
        for h in self._drill_holes:
            if ox or oy:
                yield DrillHole(x=h.x + ox, y=h.y + oy, diameter=h.diameter, z1=h.z1, z2=h.z2,
                                 plated=h.plated, layer_name=h.layer_name)
            else:
                yield h

    # ---- plotting --------------------------------------------------------

    def plot_layers(
        self,
        board_color: str = "seagreen",
        copper_color: str = "#B87333",
        figsize: Tuple[float, float] = (8.0, 8.0),
        layer_names: Optional[Iterable[str]] = None,
        simplify_delta: float = 1e-6,
    ) -> None:
        plot_layers_for(self, board_color, copper_color, figsize, layer_names, simplify_delta)
