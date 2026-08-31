"""Regression pins for the second external-review fixes.

Each test corresponds to a confirmed review finding: sjoin index_right
protection, origin immutability/validation, point-in-polygon overflow at
large coordinates, predicate handling, missing/empty geometry joins, kNN
NaN-path parity, k normalization, XYZM ingestion, radius validation, and
the from_shapely/sjoin keyword forwarding gaps.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pandas as pd
import pytest
import shapely
from shapely.geometry import Point, Polygon

import mlx_spatial as ms
from mlx_spatial._dispatch import metal_available
from mlx_spatial.haversine import _knn_composed, _knn_fused, _prep


# ---------------------------------------------------------------------------
# sjoin: index_right column protection
# ---------------------------------------------------------------------------


def test_sjoin_left_index_right_column_raises():
    """A pre-existing left 'index_right' column must never be overwritten."""
    left = pd.DataFrame(
        {"geometry": [Point(0.5, 0.5)], "index_right": ["user-data"]}
    )
    right = pd.DataFrame({"geometry": [shapely.box(0, 0, 1, 1)]})
    for how in ("inner", "left"):
        with pytest.raises(ValueError, match="index_right"):
            ms.sjoin(left, right, how=how)
    # untouched input, and the same frame joins fine once renamed
    assert left.loc[0, "index_right"] == "user-data"
    out = ms.sjoin(left.rename(columns={"index_right": "keep"}), right)
    assert out.loc[0, "keep"] == "user-data"
    assert out.loc[0, "index_right"] == 0


# ---------------------------------------------------------------------------
# PolygonSet.origin: copied, validated, immutable
# ---------------------------------------------------------------------------


def test_origin_copied_from_caller_array():
    """Mutating the caller's origin array must not desynchronize backends."""
    origin = np.array([0.0, 0.0])
    ps = ms.PolygonSet([shapely.box(0, 0, 1, 1)], origin=origin)
    origin[:] = [100.0, 100.0]
    for backend in ("cpu", "gpu"):
        got = ms.point_in_polygon([[0.5, 0.5]], ps, backend=backend)
        np.testing.assert_array_equal(got, [0])


def test_origin_attribute_read_only():
    ps = ms.PolygonSet([shapely.box(0, 0, 1, 1)])
    with pytest.raises(ValueError, match="read-only"):
        ps.origin[0] = 50.0


def test_origin_validation():
    box = [shapely.box(0, 0, 1, 1)]
    for bad in ([np.nan, 0.0], [0.0, np.inf], [-np.inf, 0.0]):
        with pytest.raises(ValueError, match="finite"):
            ms.PolygonSet(box, origin=bad)
    # an origin so far away that re-centering overflows float32 is rejected
    with pytest.raises(ValueError, match="overflow"):
        ms.PolygonSet(box, origin=[1e300, 1e300])
    # ...but any origin the data can absorb still works
    assert ms.PolygonSet(box, origin=[1e30, -1e30]).n_polygons == 1


# ---------------------------------------------------------------------------
# point-in-polygon: no intermediate overflow at huge coordinates
# ---------------------------------------------------------------------------


def test_pip_no_overflow_at_large_coordinates():
    """The edge-intersection ratio is computed before multiplying, so valid
    geometry at ~1e20 classifies interior points correctly (the old
    multiply-first form overflowed to inf and returned 'outside')."""
    tri = Polygon([(0.0, 0.0), (2e20, 0.0), (0.0, 2e20)])
    ps = ms.PolygonSet([tri])
    pts = np.array(
        [
            [1e19, 1e19],  # deep interior (~1e19 from every edge)
            [5e19, 5e19],  # interior
            [1.9e20, 1.9e20],  # far outside the hypotenuse
            [-1e19, 1e19],  # outside (negative x)
        ]
    )
    expect_inside = np.array([True, True, False, False])
    # every point is thousands of tolerance widths from the boundary
    assert ps.boundary_tolerance < 1e15

    cm = np.array(ms.contains_matrix(pts, ps))[:, 0]
    np.testing.assert_array_equal(cm, expect_inside)
    backends = ("cpu", "gpu") if metal_available() else ("cpu",)
    for backend in backends:
        got = np.asarray(ms.point_in_polygon(pts, ps, backend=backend))
        np.testing.assert_array_equal(got, np.where(expect_inside, 0, -1))


def test_pip_rejects_endpoint_difference_overflow():
    """Finite endpoints can still have an unrepresentable fp32 difference.

    Before the endpoint-difference guard, the diagonal's +/-3e38 subtraction
    became inf in both PIP paths and silently rejected deep-interior points
    hundreds of thousands of boundary tolerances from the edge.
    """
    tri = Polygon(
        [(-3e38, -3e38), (3e38, 3e38), (-3e38, 3e38)]
    )
    assert tri.is_valid
    with pytest.raises(ValueError, match="edge endpoint difference.*float32"):
        ms.PolygonSet([tri])


def test_pip_accepts_far_apart_polygons_with_safe_edges():
    """Dataset-wide span may exceed float32 while every real edge is safe."""
    half_size = 1e35
    polys = [
        shapely.box(-3e38 - half_size, -half_size, -3e38 + half_size, half_size),
        shapely.box(3e38 - half_size, -half_size, 3e38 + half_size, half_size),
    ]
    assert all(poly.is_valid for poly in polys)
    ps = ms.PolygonSet(polys)
    got = np.array(ms.contains_matrix([[-3e38, 0.0], [3e38, 0.0]], ps))
    np.testing.assert_array_equal(got, [[True, False], [False, True]])


# ---------------------------------------------------------------------------
# predicate handling
# ---------------------------------------------------------------------------


def test_sjoin_predicate_honored_on_boundary():
    """Small workloads run on GEOS, where within/intersects differ exactly on
    boundary points — matching geopandas."""
    left = pd.DataFrame({"geometry": [Point(1.0, 0.5), Point(0.5, 0.5)]})
    right = pd.DataFrame({"geometry": [shapely.box(0, 0, 1, 1)]})
    out_within = ms.sjoin(left, right, predicate="within")
    out_intersects = ms.sjoin(left, right, predicate="intersects")
    assert list(out_within.index) == [1]  # boundary point excluded
    assert list(out_intersects.index) == [0, 1]  # boundary point included


def test_point_in_polygon_predicate_param():
    boundary = [[1.0, 0.5]]
    polys = [shapely.box(0, 0, 1, 1)]
    np.testing.assert_array_equal(
        ms.point_in_polygon(boundary, polys, predicate="within", backend="cpu"), [-1]
    )
    np.testing.assert_array_equal(
        ms.point_in_polygon(boundary, polys, predicate="intersects", backend="cpu"), [0]
    )
    with pytest.raises(ValueError, match="predicate"):
        ms.point_in_polygon(boundary, polys, predicate="touches")


def test_predicate_through_prebuilt_index():
    """The index= branch must forward predicate= just like the polygons path."""
    index = ms.GridIndex(ms.PolygonSet([shapely.box(0, 0, 1, 1)]))
    boundary = [[1.0, 0.5]]
    np.testing.assert_array_equal(
        ms.point_in_polygon(boundary, index=index, predicate="within", backend="cpu"),
        [-1],
    )
    np.testing.assert_array_equal(
        ms.point_in_polygon(boundary, index=index, predicate="intersects", backend="cpu"),
        [0],
    )
    with pytest.raises(ValueError, match="predicate"):
        index.query(boundary, predicate="touches")


# ---------------------------------------------------------------------------
# missing and empty geometries through joins
# ---------------------------------------------------------------------------


def test_missing_left_points_are_unmatched():
    pts = [Point(0.5, 0.5), None, Point(5.0, 5.0)]
    out = ms.as_point_array(pts)
    assert out.shape == (3, 2)
    assert np.isnan(out[1]).all()

    left = pd.DataFrame({"geometry": pts, "pid": [0, 1, 2]})
    right = pd.DataFrame({"geometry": [shapely.box(0, 0, 1, 1)], "zid": [9]})
    inner = ms.sjoin(left, right, how="inner")
    assert list(inner["pid"]) == [0]
    left_join = ms.sjoin(left, right, how="left")
    assert len(left_join) == 3
    assert left_join["index_right"].isna().tolist() == [False, True, True]


def test_missing_right_polygons_never_match():
    polys = [None, shapely.box(0, 0, 1, 1), None]
    pts = np.array([[0.5, 0.5], [2.5, 2.5]])
    for backend in ("cpu", "gpu") if metal_available() else ("cpu",):
        got = np.asarray(ms.point_in_polygon(pts, polys, backend=backend))
        np.testing.assert_array_equal(got, [1, -1])
    right = pd.DataFrame({"geometry": polys, "zid": ["a", "b", "c"]})
    left = pd.DataFrame({"geometry": list(shapely.points(pts))})
    out = ms.sjoin(left, right, how="inner")
    assert list(out["index_right"]) == [1]
    assert list(out["zid"]) == ["b"]


def test_all_empty_right_polygons():
    """All-empty polygon sets produce unmatched rows, not an error."""
    empties = [Polygon(), Polygon()]
    got = ms.point_in_polygon([[0.5, 0.5], [1.5, 1.5]], empties)
    np.testing.assert_array_equal(got, [-1, -1])
    pt_idx, pg_idx = ms.point_in_polygon([[0.5, 0.5]], ms.PolygonSet(empties), all_matches=True)
    assert len(pt_idx) == len(pg_idx) == 0

    left = pd.DataFrame({"geometry": [Point(0.5, 0.5)], "pid": [7]})
    right = pd.DataFrame({"geometry": empties, "zid": ["a", "b"]})
    inner = ms.sjoin(left, right, how="inner")
    assert len(inner) == 0
    assert {"pid", "geometry", "index_right", "zid"} == set(inner.columns)
    left_join = ms.sjoin(left, right, how="left")
    assert len(left_join) == 1
    assert left_join["index_right"].isna().all()
    assert left_join["zid"].isna().all()


def test_zero_length_right_frame():
    gpd = pytest.importorskip("geopandas")
    left = gpd.GeoDataFrame({"pid": [0, 1]}, geometry=[Point(0.5, 0.5), Point(2.0, 2.0)])
    right = gpd.GeoDataFrame({"zid": []}, geometry=[])
    inner = ms.sjoin(left, right, how="inner")
    assert len(inner) == 0
    assert {"pid", "geometry", "index_right", "zid"} == set(inner.columns)
    left_join = ms.sjoin(left, right, how="left")
    assert len(left_join) == 2
    assert left_join["index_right"].isna().all()
    assert left_join["zid"].isna().all()
    # argument validation must not depend on the right frame being non-empty
    with pytest.raises(ValueError, match="backend"):
        ms.sjoin(left, right, backend="banana")


def test_zero_length_polygon_sequence_point_in_polygon():
    """The array API treats an empty polygon collection as no matches."""
    points = [[0.0, 0.0], [1.0, 1.0]]
    backends = ("cpu", "gpu") if metal_available() else ("cpu",)
    for backend in backends:
        got = ms.point_in_polygon(points, [], backend=backend)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, [-1, -1])

        point_idx, poly_idx = ms.point_in_polygon(
            points, [], backend=backend, all_matches=True
        )
        assert point_idx.dtype == poly_idx.dtype == np.int64
        assert len(point_idx) == len(poly_idx) == 0

    with pytest.raises(ValueError, match="backend"):
        ms.point_in_polygon(points, [], backend="banana")
    with pytest.raises(ValueError, match="predicate"):
        ms.point_in_polygon(points, [], predicate="touches")


def test_as_point_array_scalar_none_raises():
    """None (a 0-d object array) must fail with a clear ValueError, not an
    opaque TypeError from len()."""
    with pytest.raises(ValueError, match="1-D sequence of points"):
        ms.as_point_array(None)


# ---------------------------------------------------------------------------
# kNN: NaN handling identical on both execution paths
# ---------------------------------------------------------------------------


def _nan_knn_case():
    q = np.array([[np.nan, np.nan], [0.0, 0.0]])
    r = np.array([[0.1, 0.1], [np.nan, np.nan], [0.2, 0.2], [np.nan, np.nan]])
    return _prep(q, True), _prep(r, True)


@pytest.mark.skipif(not metal_available(), reason="fused kNN kernel requires Metal")
@pytest.mark.parametrize("chunk_bytes", [512 * 1024 * 1024, 1])
def test_knn_nan_parity_between_paths(chunk_bytes):
    """chunk_bytes=1 forces one query row per chunk, so the per-chunk NaN
    masking and the multi-chunk concatenation are exercised too."""
    q, r = _nan_knn_case()
    for k in (1, 2, 3, 4):
        df, i_f = _knn_fused(q, r, k, ms.R_EARTH_M)
        dc, i_c = _knn_composed(q, r, k, ms.R_EARTH_M, chunk_bytes)
        mx.eval(df, i_f, dc, i_c)
        np.testing.assert_array_equal(np.array(df), np.array(dc))
        np.testing.assert_array_equal(np.array(i_f), np.array(i_c))


def test_knn_nan_semantics_public_api():
    """NaN query rows and unfillable slots report -1/inf on any path."""
    q = np.array([[np.nan, np.nan], [0.0, 0.0]])
    r = np.array([[0.1, 0.1], [np.nan, np.nan], [0.2, 0.2]])
    d, i = ms.haversine_knn(q, r, k=3)
    d, i = np.array(d), np.array(i)
    assert (i[0] == -1).all() and np.isinf(d[0]).all()  # NaN query row
    np.testing.assert_array_equal(i[1], [0, 2, -1])  # NaN ref fills nothing
    assert np.isfinite(d[1, :2]).all() and np.isinf(d[1, 2])


def test_knn_composed_path_nan_semantics():
    """The composed path alone (as when Metal is unavailable) pads with -1/inf."""
    q, r = _nan_knn_case()
    d, i = _knn_composed(q, r, 3, ms.R_EARTH_M, 512 * 1024 * 1024)
    d, i = np.array(d), np.array(i)
    assert (i[0] == -1).all() and np.isinf(d[0]).all()
    np.testing.assert_array_equal(i[1], [0, 2, -1])
    assert np.isinf(d[1, 2])


# ---------------------------------------------------------------------------
# k normalization and radius validation
# ---------------------------------------------------------------------------


def test_knn_numpy_integer_k():
    q = np.array([[0.0, 0.0]])
    r = np.array([[0.1, 0.1], [0.2, 0.2]])
    for k in (np.int64(2), np.int32(2), np.uint8(2)):
        d, i = ms.haversine_knn(q, r, k=k)
        assert np.array(d).shape == (1, 2)
        np.testing.assert_array_equal(np.array(i), [[0, 1]])


def test_knn_non_integer_k_raises():
    q = np.array([[0.0, 0.0]])
    r = np.array([[0.1, 0.1], [0.2, 0.2]])
    for bad in (1.5, 2.0, "2", np.float64(2)):
        with pytest.raises(TypeError, match="k must be an integer"):
            ms.haversine_knn(q, r, k=bad)


def test_radius_validation():
    a, b = [0.0, 0.0], [1.0, 1.0]
    q = np.array([a])
    r = np.array([b])
    # 1e38 is finite but pi * radius overflows float32 distances to inf,
    # which would collide with the kNN paths' "no neighbor" padding.
    # Positive values below float32's normal range can underflow or be flushed
    # to zero on Metal, destroying the composed path's neighbor ordering.
    smallest_float32_subnormal = float(
        np.nextafter(np.float32(0), np.float32(1))
    )
    subnormal_radius = float(np.finfo(np.float32).tiny) * 0.5
    for bad in (
        0.0,
        -5.0,
        np.nan,
        np.inf,
        -np.inf,
        smallest_float32_subnormal,
        subnormal_radius,
        1e-50,
        1e38,
    ):
        with pytest.raises(ValueError, match="radius"):
            ms.haversine(a, b, radius=bad)
        with pytest.raises(ValueError, match="radius"):
            ms.haversine_cross(q, r, radius=bad)
        with pytest.raises(ValueError, match="radius"):
            ms.haversine_knn(q, r, k=1, radius=bad)

    # The internal engines enforce the same contract, so direct/fallback use
    # cannot reintroduce the fused/composed discrepancy.
    q_mx, r_mx = _prep(q, True), _prep(r, True)
    with pytest.raises(ValueError, match="radius"):
        _knn_fused(q_mx, r_mx, 1, 1e-50)
    with pytest.raises(ValueError, match="radius"):
        _knn_composed(q_mx, r_mx, 1, 1e-50, 512 * 1024 * 1024)

    # The smallest normal float32 radius is the inclusive lower boundary.
    radius_min = float(np.finfo(np.float32).tiny)
    assert np.isfinite(float(ms.haversine(a, b, radius=radius_min)))
    d, i = ms.haversine_knn(q, r, k=1, radius=radius_min)
    assert np.array(d).shape == (1, 1)
    np.testing.assert_array_equal(np.array(i), [[0]])


# ---------------------------------------------------------------------------
# XYZM ingestion
# ---------------------------------------------------------------------------


def test_xyz_and_xyzm_polygons_ingest():
    """Extra ordinates beyond XY are dropped; containment matches the XY twin."""
    xy = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    xyz = shapely.from_wkt("POLYGON Z ((0 0 1, 4 0 1, 4 4 1, 0 4 1, 0 0 1))")
    xyzm = shapely.from_wkt(
        "POLYGON ZM ((0 0 1 2, 4 0 1 2, 4 4 1 2, 0 4 1 2, 0 0 1 2))"
    )
    ref = ms.PolygonSet([xy])
    for poly in (xyz, xyzm):
        ps = ms.PolygonSet([poly])
        assert ps.n_vertices == ref.n_vertices
        assert np.array(ps.coords).shape == (5, 2)
        np.testing.assert_array_equal(np.array(ps.coords), np.array(ref.coords))
        got = ms.point_in_polygon([[2.0, 2.0], [5.0, 5.0]], ps)
        np.testing.assert_array_equal(got, [0, -1])


def test_xyzm_multipolygon_and_mixed_dimension_set():
    """The multipolygon ragged branch also slices Z/M, and a mixed XY+XYZM set
    (shapely pads the missing ordinates with NaN) must still ingest: the slice
    runs before the finite-coordinate check."""
    mp = shapely.from_wkt(
        "MULTIPOLYGON ZM (((0 0 1 2, 4 0 1 2, 4 4 1 2, 0 4 1 2, 0 0 1 2)),"
        " ((10 10 1 2, 12 10 1 2, 12 12 1 2, 10 10 1 2)))"
    )
    ps = ms.PolygonSet([mp])
    got = ms.point_in_polygon([[2.0, 2.0], [11.0, 10.5], [6.0, 6.0]], ps)
    np.testing.assert_array_equal(got, [0, 0, -1])

    mixed = [
        shapely.box(0, 0, 1, 1),
        shapely.from_wkt("POLYGON ZM ((2 2 1 2, 3 2 1 2, 3 3 1 2, 2 2 1 2))"),
    ]
    got = ms.point_in_polygon([[0.5, 0.5], [2.7, 2.4]], mixed)
    np.testing.assert_array_equal(got, [0, 1])


# ---------------------------------------------------------------------------
# keyword forwarding gaps
# ---------------------------------------------------------------------------


def test_from_shapely_forwards_validate():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    with pytest.raises(ValueError, match="even-odd"):
        ms.from_shapely([bowtie], validate=True)
    assert ms.from_shapely([bowtie]).n_polygons == 1


def test_sjoin_backend_override():
    left = pd.DataFrame({"geometry": [Point(0.5, 0.5), Point(9.0, 9.0)]})
    right = pd.DataFrame({"geometry": [shapely.box(0, 0, 1, 1)], "zid": [3]})
    expected = ms.sjoin(left, right, how="left")
    got_cpu = ms.sjoin(left, right, how="left", backend="cpu")
    pd.testing.assert_frame_equal(got_cpu, expected)
    if metal_available():
        got_gpu = ms.sjoin(left, right, how="left", backend="gpu")
        assert got_gpu["index_right"].tolist()[0] == 0
        assert np.isnan(got_gpu["index_right"].tolist()[1])
    with pytest.raises(ValueError, match="backend"):
        ms.sjoin(left, right, backend="banana")
