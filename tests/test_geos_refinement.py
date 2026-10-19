"""Polygon-major GEOS refinement answers exactly the per-pair tree query."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

import mlx_geospatial.join as join
from mlx_geospatial import GridIndex, PolygonSet


def _ring(n, scale=1.0, origin=(0.0, 0.0), wobble=0.1, seed=0):
    angle = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    noise = np.random.default_rng(seed).normal(0.0, 0.002, n)
    radius = 1.0 + wobble * np.sin(7 * angle) + noise
    return (np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]) * scale
            + np.asarray(origin))


def _bowtie(n):
    """A self-intersecting figure-eight ring with ``n`` vertices."""
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.sin(t), np.sin(t) * np.cos(t)])


def _band_points(geoms, rng, per_polygon=60):
    """Vertices, edge midpoints, +-tiny offsets and nearby random points."""
    tol = PolygonSet(geoms).boundary_tolerance
    out = []
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        coords = shapely.get_coordinates(geom)
        pick = rng.choice(len(coords) - 1, min(per_polygon, len(coords) - 1), replace=False)
        p1, p2 = coords[pick], coords[pick + 1]
        mid = 0.5 * (p1 + p2)
        out += [p1, mid, mid + rng.normal(0.0, tol, mid.shape),
                mid + [1e-12, 0.0], mid - [0.0, 1e-12]]
    out.append(np.array([[np.nan, 0.0]]))
    return np.concatenate(out)


def _reference(geoms, points, predicate):
    pt, pg = shapely.STRtree(np.asarray(geoms, dtype=object)).query(
        shapely.points(points), predicate=predicate
    )
    order = np.lexsort((pg, pt))
    return pt[order], pg[order]


def _pairs(polyset, points, predicate):
    pt, pg = join._geos_pairs(polyset, shapely.points(points), predicate)
    order = np.lexsort((pg, pt))
    return pt[order], pg[order]


def _spy(monkeypatch):
    """Record each polygon evaluated polygon-major (one STRtree per group)."""
    built = []
    real = shapely.STRtree

    class Spy(real):
        def query(self, geometry, predicate=None, **kwargs):
            if predicate in ("contains", "covers"):
                built.append(len(self))
            return super().query(geometry, predicate=predicate, **kwargs)

    monkeypatch.setattr(join.shapely, "STRtree", Spy)
    return built


@pytest.mark.parametrize("predicate", ["within", "intersects"])
def test_heavy_single_ring_matches_tree_and_leaves_polygon_unprepared(monkeypatch, predicate):
    rng = np.random.default_rng(1)
    heavy = Polygon(_ring(4096))
    light = [box(1.5, -0.2, 1.9, 0.2), Polygon(_ring(48, 0.2, (0.3, 0.1)))]
    geoms = [heavy, *light, None, Polygon()]
    points = _band_points(geoms, rng)
    built = _spy(monkeypatch)
    polyset = PolygonSet(geoms)
    for actual, expected in zip(_pairs(polyset, points, predicate),
                                _reference(geoms, points, predicate)):
        np.testing.assert_array_equal(actual, expected)
    assert len(built) == 1
    assert not shapely.is_prepared(polyset._shapely).any()
    assert not shapely.is_prepared(heavy)


@pytest.mark.parametrize("predicate", ["within", "intersects"])
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_query_with_prepared_route_matches_geos(backend, predicate):
    rng = np.random.default_rng(2)
    geoms = [Polygon(_ring(2048, 1.0, (3.0 * i, 0.0), seed=i)) for i in range(3)]
    points = np.concatenate([_band_points(geoms, rng), rng.uniform(-1.5, 7.5, (2000, 2))])
    index = GridIndex(geoms)
    pt, pg = _reference(geoms, points, predicate)
    got = index.query(points, backend=backend, all_matches=True, predicate=predicate,
                      exact_boundary=True)
    np.testing.assert_array_equal(got[0], pt)
    np.testing.assert_array_equal(got[1], pg)
    first = np.full(len(points), -1, dtype=np.int32)
    first[pt] = pg  # the rings do not overlap: at most one match per point
    np.testing.assert_array_equal(
        index.query(points, backend=backend, predicate=predicate, exact_boundary=True), first
    )


@pytest.mark.parametrize("predicate", ["within", "intersects"])
def test_invalid_single_rings_keep_relate_semantics(monkeypatch, predicate):
    """Self-intersecting, spiked and repeated-vertex rings: parity on one ring."""
    rng = np.random.default_rng(3)
    spiked = _ring(1024)
    spiked[100] = [-3.0, 0.0]  # a spike across the ring: self-intersecting
    repeated = np.repeat(_ring(600, 0.5, (4.0, 0.0)), 2, axis=0)
    geoms = [Polygon(_bowtie(1024) * 2 + [0.0, 4.0]), Polygon(spiked), Polygon(repeated)]
    assert not any(g.is_valid for g in geoms[:2])
    points = np.concatenate([_band_points(geoms, rng), rng.uniform(-3, 5, (3000, 2))])
    monkeypatch.setattr(join, "_PREPARED_MIN_PAIRS", 1)
    built = _spy(monkeypatch)
    for actual, expected in zip(_pairs(PolygonSet(geoms), points, predicate),
                                _reference(geoms, points, predicate)):
        np.testing.assert_array_equal(actual, expected)
    assert len(built) == 3


@pytest.mark.parametrize("predicate", ["within", "intersects"])
def test_multi_ring_geometry_keeps_the_per_pair_route(monkeypatch, predicate):
    """Invalid holes and overlapping parts differ under prepared parity."""
    rng = np.random.default_rng(4)
    shell = _ring(1024)
    outside_hole = Polygon(shell, holes=[_ring(600, 0.3, (3.0, 0.0))])
    overlapping = MultiPolygon([Polygon(_ring(700, 1.0, (6.0, 0.0))),
                                Polygon(_ring(700, 1.0, (6.5, 0.0), seed=1))])
    valid_holed = Polygon(_ring(1024, 1.0, (0.0, 4.0)), holes=[_ring(600, 0.3, (0.0, 4.0))])
    geoms = [outside_hole, overlapping, valid_holed]
    points = np.concatenate([_band_points(geoms, rng), rng.uniform(-2, 8, (3000, 2))])
    monkeypatch.setattr(join, "_PREPARED_MIN_PAIRS", 1)
    built = _spy(monkeypatch)
    for actual, expected in zip(_pairs(PolygonSet(geoms), points, predicate),
                                _reference(geoms, points, predicate)):
        np.testing.assert_array_equal(actual, expected)
    assert built == []


def test_thresholds_gate_the_prepared_route(monkeypatch):
    rng = np.random.default_rng(5)
    small = Polygon(_ring(400))
    large = Polygon(_ring(12_000, 1.0, (3.0, 0.0)))
    geoms = [small, large]
    coords = [shapely.get_coordinates(g)[:-1] for g in geoms]
    built = _spy(monkeypatch)
    polyset = PolygonSet(geoms)
    # Under 512 edges: never prepared, whatever the pair count.
    join._geos_pairs(polyset, shapely.points(coords[0]), "within")
    assert built == []
    # 30 pairs: below the large-ring minimum (48), so per pair.
    join._geos_pairs(polyset, shapely.points(coords[1][rng.choice(12_000, 30)]), "within")
    assert built == []
    join._geos_pairs(polyset, shapely.points(coords[1][rng.choice(12_000, 60)]), "within")
    assert built == [60]


def test_no_eligible_polygon_is_one_tree_query(monkeypatch):
    rng = np.random.default_rng(6)
    geoms = [Polygon(_ring(48, 0.2, (x, 0.0))) for x in np.linspace(0, 3, 10)]
    points = _band_points(geoms, rng)
    expected = _reference(geoms, points, "intersects")
    calls = []
    polyset = PolygonSet(geoms)
    tree = polyset.strtree
    real = type(tree).query

    def query(self, geometry, predicate=None, **kwargs):
        calls.append(predicate)
        return real(self, geometry, predicate=predicate, **kwargs)

    monkeypatch.setattr(type(tree), "query", query)
    for actual, wanted in zip(_pairs(polyset, points, "intersects"), expected):
        np.testing.assert_array_equal(actual, wanted)
    assert calls == ["intersects"]
