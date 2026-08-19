"""Layer 7: `odbpp.cache` -- msgpack save/load round-trip of a
PCBView's resolved geometry, and the CachedPCBView drop-in's
interface parity with the real PCBView it was saved from.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from emcad.odbpp import PCBView, save_pcb_cache, load_pcb_cache

EXAMPLE_BOARD = Path(__file__).resolve().parent.parent / "example_files" / "RO4350B_10mil-odb"


def _ring_area(xs, ys) -> float:
    n = len(xs)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(a) / 2.0


def _poly_area(p) -> float:
    a = _ring_area(p.xs, p.ys)
    for h in p.holes:
        a -= _poly_area(h)
    return a


@pytest.fixture(scope="module")
def live_pcbd():
    if not EXAMPLE_BOARD.is_dir():
        pytest.skip(f"example board not found: {EXAMPLE_BOARD}")
    return PCBView(str(EXAMPLE_BOARD), thickness_stack=[35e-6, 508e-6, 35e-6])


@pytest.fixture(scope="module")
def cached_pcbd(live_pcbd, tmp_path_factory):
    cache_path = tmp_path_factory.mktemp("pcbcache") / "board.pcbcache"
    save_pcb_cache(live_pcbd, cache_path)
    return load_pcb_cache(cache_path)


def test_board_outline_matches_exactly(live_pcbd, cached_pcbd):
    assert live_pcbd.get_board_polygon() == cached_pcbd.get_board_polygon()
    assert live_pcbd.get_board_holes() == cached_pcbd.get_board_holes()


def test_pcb_layer_z_stack_matches(live_pcbd, cached_pcbd):
    assert list(live_pcbd.iter_pcb_layers()) == list(cached_pcbd.iter_pcb_layers())


def test_geo_layer_names_and_order_match(live_pcbd, cached_pcbd):
    live_names = [l.name for l in live_pcbd.iter_geo_layers()]
    cached_names = [l.name for l in cached_pcbd.iter_geo_layers()]
    assert live_names == cached_names
    assert len(live_names) > 0


def test_resolved_layer_polygons_match_exactly(live_pcbd, cached_pcbd):
    for l1, l2 in zip(live_pcbd.iter_geo_layers(), cached_pcbd.iter_geo_layers()):
        r1 = live_pcbd.resolve_layer_polygons(l1)
        r2 = cached_pcbd.resolve_layer_polygons(l2)
        assert len(r1) == len(r2), f"layer {l1.name}: polygon count mismatch"
        a1 = sum(_poly_area(p) for p in r1)
        a2 = sum(_poly_area(p) for p in r2)
        assert a1 == pytest.approx(a2, abs=1e-15), f"layer {l1.name}: area mismatch"


def test_drill_holes_match(live_pcbd, cached_pcbd):
    live = list(live_pcbd.iter_drill_holes())
    cached = list(cached_pcbd.iter_drill_holes())
    assert len(live) == len(cached)
    assert {(h.x, h.y, h.diameter, h.z1, h.z2, h.plated) for h in live} == \
           {(h.x, h.y, h.diameter, h.z1, h.z2, h.plated) for h in cached}


def test_centralize_after_load_still_works(cached_pcbd):
    cached_pcbd.centralize()
    xs, ys = cached_pcbd.get_board_polygon()
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    assert abs(cx) < 1e-9
    assert abs(cy) < 1e-9
    # idempotent -- calling again is a no-op
    cached_pcbd.centralize()
    xs2, ys2 = cached_pcbd.get_board_polygon()
    assert xs2 == xs and ys2 == ys


def test_stale_format_version_is_rejected(tmp_path, live_pcbd):
    import msgpack
    cache_path = tmp_path / "bad.pcbcache"
    save_pcb_cache(live_pcbd, cache_path)
    payload = msgpack.unpackb(cache_path.read_bytes(), raw=False, strict_map_key=False)
    payload["format_version"] = 999
    cache_path.write_bytes(msgpack.packb(payload, use_bin_type=True))
    with pytest.raises(ValueError):
        load_pcb_cache(cache_path)
