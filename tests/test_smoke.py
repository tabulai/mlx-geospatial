"""End-to-end smoke tests: every public entry point on small inputs."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

import mlx_geospatial as ms
from mlx_geospatial._pip import pip_pairs

from conftest import (
    make_convex_polygons,
    make_donut,
    make_grid_polygons,
    near_boundary_mask,
    reference_assign,
)


def test_version():
    assert ms.__version__


def test_haversine_matches_sklearn(rng):
    from sklearn.metrics.pairwise import haversine_distances

    a = np.column_stack([rng.uniform(-74.3, -73.7, 200), rng.uniform(40.5, 40.9, 200)])
    b = np.column_stack([rng.uniform(-74.3, -73.7, 150), rng.uniform(40.5, 40.9, 150)])
    ours = np.array(ms.haversine_cross(a, b))
    # sklearn expects (lat, lon) in radians and returns unit-sphere distances.
    ref = haversine_distances(np.radians(a[:, ::-1]), np.radians(b[:, ::-1])) * ms.R_EARTH_M
    # fp32 differencing of nearby coordinates bounds absolute error ~ meters,
    # so relative error shrinks with distance: ~1e-4 at 10 km and beyond.
    assert np.abs(ours - ref).max() < 2.0
    far = ref > 10_000.0
    assert np.abs((ours[far] - ref[far]) / ref[far]).max() < 3e-4


def test_haversine_elementwise_and_broadcast():
    nyc = [-74.0060, 40.7128]
    london = [-0.1278, 51.5074]
    d = float(ms.haversine(nyc, london))
    assert d == pytest.approx(5_570_000, rel=0.01)  # ~5570 km
    # broadcasting one-vs-many
    many = np.array([nyc, london, [2.3522, 48.8566]])
    d2 = np.array(ms.haversine(np.array(nyc)[None, :], many))
    assert d2[0] == pytest.approx(0.0, abs=1.0)
    assert d2[1] == pytest.approx(d, rel=1e-6)


def test_haversine_knn_matches_bruteforce(rng):
    q = np.column_stack([rng.uniform(-74.3, -73.7, 50), rng.uniform(40.5, 40.9, 50)])
    r = np.column_stack([rng.uniform(-74.3, -73.7, 300), rng.uniform(40.5, 40.9, 300)])
    dist, idx = ms.haversine_knn(q, r, k=5)
    dist, idx = np.array(dist), np.array(idx)
    full = np.array(ms.haversine_cross(q, r))
    ref_idx = np.argsort(full, axis=1)[:, :5]
    # distances must match the brute-force top-5 (indices may swap on exact ties)
    np.testing.assert_allclose(
        dist, np.take_along_axis(full, ref_idx, axis=1), rtol=1e-6, atol=1e-3
    )
    assert np.all(np.diff(dist, axis=1) >= -1e-3)


def test_pip_known_shapes():
    square_with_hole = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
    )
    triangle_multi = MultiPolygon(
        [
            Polygon([(20, 0), (25, 0), (25, 5)]),
            Polygon([(30, 30), (35, 30), (35, 35)]),
        ]
    )
    polys = np.array([square_with_hole, triangle_multi], dtype=object)
    pts = np.array(
        [
            [2.0, 2.0],  # in square
            [5.0, 5.0],  # in the hole -> outside
            [24.0, 1.0],  # in first part of multi
            [34.0, 31.0],  # in second part of multi
            [50.0, 50.0],  # nowhere
        ]
    )
    got = ms.point_in_polygon(pts, polys, backend="gpu")
    np.testing.assert_array_equal(got, [0, -1, 1, 1, -1])
    got_cpu = ms.point_in_polygon(pts, polys, backend="cpu")
    np.testing.assert_array_equal(got_cpu, [0, -1, 1, 1, -1])


def test_pip_random_golden_vs_shapely(rng):
    polys = make_convex_polygons(80, rng, extent=(-74.3, 40.5, -73.7, 40.9))
    pts = np.column_stack(
        [rng.uniform(-74.35, -73.65, 5000), rng.uniform(40.45, 40.95, 5000)]
    )
    polyset = ms.PolygonSet(polys)
    got = ms.point_in_polygon(pts, polyset, backend="gpu")
    ref = reference_assign(pts, polys)
    ok = ~near_boundary_mask(pts, polys, polyset.boundary_tolerance)
    mismatches = np.flatnonzero((got != ref) & ok)
    assert mismatches.size == 0, f"{mismatches.size} mismatches outside tolerance band"


def test_grid_polygons_shared_edges(rng):
    """Adjacent squares: every strictly-interior point must land in its square."""
    polys = make_grid_polygons(8, 8, cell=1.0)
    ij = rng.integers(0, 8, size=(2000, 2))
    offs = rng.uniform(0.05, 0.95, size=(2000, 2))
    pts = ij + offs
    got = ms.point_in_polygon(pts, polys, backend="gpu")
    ref = reference_assign(pts, polys)
    np.testing.assert_array_equal(got, ref)


def test_donut_hole(rng):
    donut = make_donut(0.0, 0.0, 1.0, 0.5, k=128)
    polys = np.array([donut], dtype=object)
    r = rng.uniform(0, 1.3, 3000)
    theta = rng.uniform(0, 2 * np.pi, 3000)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    got = ms.point_in_polygon(pts, polys, backend="gpu")
    ref = reference_assign(pts, polys)
    polyset = ms.PolygonSet(polys)
    ok = ~near_boundary_mask(pts, polys, polyset.boundary_tolerance)
    assert np.array_equal(got[ok], ref[ok])


def test_dense_matrix_agrees_with_pair_kernel(rng):
    polys = make_convex_polygons(40, rng)
    pts = np.column_stack([rng.uniform(-1.1, 1.1, 2000), rng.uniform(-1.1, 1.1, 2000)])
    polyset = ms.PolygonSet(polys)
    dense = np.array(ms.contains_matrix(pts, polyset))
    # Exhaustive pairs through the Metal kernel.
    n, m = len(pts), len(polys)
    pair_pt = np.repeat(np.arange(n, dtype=np.int32), m)
    pair_pg = np.tile(np.arange(m, dtype=np.int32), n)
    _, pts_local = polyset.recenter_points(pts)
    flags = np.array(
        pip_pairs(pts_local, polyset, mx.array(pair_pt), mx.array(pair_pg))
    ).reshape(n, m)
    np.testing.assert_array_equal(dense.astype(np.uint8), flags)


def test_all_matches_pairs(rng):
    # Two deliberately overlapping polygons.
    polys = np.array(
        [shapely.box(0, 0, 2, 2), shapely.box(1, 1, 3, 3), shapely.box(10, 10, 11, 11)],
        dtype=object,
    )
    pts = np.array([[1.5, 1.5], [0.5, 0.5], [2.5, 2.5], [10.5, 10.5], [-5.0, -5.0]])
    pt_idx, poly_idx = ms.point_in_polygon(pts, polys, all_matches=True, backend="gpu")
    got = set(zip(pt_idx.tolist(), poly_idx.tolist()))
    assert got == {(0, 0), (0, 1), (1, 0), (2, 1), (3, 2)}
    # first-match reduction picks the lowest polygon index
    first = ms.point_in_polygon(pts, polys, backend="gpu")
    np.testing.assert_array_equal(first, [0, 0, 1, 2, -1])


def test_sjoin_pandas_roundtrip(rng):
    import pandas as pd

    polys = make_convex_polygons(30, rng)
    pts = np.column_stack([rng.uniform(-1, 1, 800), rng.uniform(-1, 1, 800)])
    left = pd.DataFrame(
        {"geometry": list(shapely.points(pts)), "trip_id": np.arange(800)}
    )
    right = pd.DataFrame(
        {"geometry": list(polys), "zone_name": [f"zone_{i}" for i in range(30)]},
        index=pd.Index(np.arange(100, 130), name="zone_id"),
    )
    out = ms.sjoin(left, right, how="inner")
    ref = reference_assign(pts, polys)
    ok = ~near_boundary_mask(pts, polys, ms.PolygonSet(polys).boundary_tolerance)
    expected = {(i, 100 + ref[i]) for i in np.flatnonzero((ref >= 0) & ok)}
    got = set(zip(out["trip_id"].tolist(), out["index_right"].tolist()))
    assert expected <= got
    assert "zone_name" in out.columns
    matched = out[out["trip_id"].isin([i for i, _ in expected])]
    assert (matched["zone_name"] == matched["index_right"].map(
        lambda z: f"zone_{z - 100}"
    )).all()
    # left join keeps every row
    out_left = ms.sjoin(left, right, how="left")
    assert len(out_left) == 800


def test_geopandas_sjoin_golden(rng):
    gpd = pytest.importorskip("geopandas")

    polys = make_convex_polygons(50, rng)
    pts = np.column_stack([rng.uniform(-1, 1, 3000), rng.uniform(-1, 1, 3000)])
    left = gpd.GeoDataFrame({"pid": np.arange(3000)}, geometry=list(shapely.points(pts)))
    right = gpd.GeoDataFrame({"zid": np.arange(50)}, geometry=list(polys))
    ours = ms.sjoin(left, right, how="inner")
    theirs = gpd.sjoin(left, right, how="inner", predicate="within")
    ok = ~near_boundary_mask(pts, polys, ms.PolygonSet(polys).boundary_tolerance)
    ours_ok = ours[ours["pid"].isin(np.flatnonzero(ok))]
    theirs_ok = theirs[theirs["pid"].isin(np.flatnonzero(ok))]
    # geopandas returns all matches; ours returns first match. Convex random
    # polygons overlap, so compare the min-index reduction of theirs.
    theirs_first = theirs_ok.groupby("pid")["index_right"].min()
    ours_first = ours_ok.set_index("pid")["index_right"]
    assert theirs_first.to_dict() == ours_first.to_dict()


def test_polygonset_reuse_and_repr(rng):
    polys = make_convex_polygons(10, rng)
    polyset = ms.PolygonSet(polys)
    pts = np.column_stack([rng.uniform(-1, 1, 100), rng.uniform(-1, 1, 100)])
    a = ms.point_in_polygon(pts, polyset)
    b = ms.point_in_polygon(pts, polyset)
    np.testing.assert_array_equal(a, b)
    assert "PolygonSet" in repr(polyset)
    assert len(polyset) == 10


def test_input_validation():
    with pytest.raises(ValueError):
        ms.PolygonSet([shapely.Point(0.0, 0.0)])  # a Point is not a polygon
    with pytest.raises(ValueError):
        ms.as_point_array(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        ms.haversine_knn(np.zeros((2, 2)), np.zeros((3, 2)), k=4)
    with pytest.raises(ValueError):
        ms.point_in_polygon(np.zeros((2, 2)), [shapely.box(0, 0, 1, 1)], backend="tpu")
