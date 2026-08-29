"""Deep tests for mlx_spatial.haversine: golden accuracy vs float64/sklearn,
edge cases (poles, dateline, antipodes), input normalization, chunking, and knn.

Tolerances are scale-appropriate per the documented fp32 policy: absolute
error ~1-2 m regardless of distance (asserted with margin), relative error
only asserted for distances where it is meaningful, and a looser band very
near the antipode where arcsin amplification is unavoidable in fp32.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest
import shapely

import mlx_spatial as ms

R = ms.R_EARTH_M
PI_R = np.pi * R  # half-circumference: max possible great-circle distance
SEED = 20260828


def hav64(a, b, radius=R, degrees=True):
    """Float64 numpy reference haversine, (..., 2) (lon, lat) arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lon1, lat1 = a[..., 0], a[..., 1]
    lon2, lat2 = b[..., 0], b[..., 1]
    if degrees:
        lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    sdlat = np.sin((lat2 - lat1) * 0.5)
    sdlon = np.sin((lon2 - lon1) * 0.5)
    h = np.clip(sdlat * sdlat + np.cos(lat1) * np.cos(lat2) * sdlon * sdlon, 0.0, 1.0)
    return 2.0 * radius * np.arcsin(np.sqrt(h))


def city_points(rng, n):
    """NYC-extent points: coordinates spanning ~0.6 deg (distances < ~80 km)."""
    return np.column_stack([rng.uniform(-74.3, -73.7, n), rng.uniform(40.5, 40.9, n)])


def continental_points(rng, n):
    """CONUS-extent points (distances up to ~7000 km)."""
    return np.column_stack([rng.uniform(-130.0, -60.0, n), rng.uniform(25.0, 50.0, n)])


def global_points(rng, n):
    return np.column_stack([rng.uniform(-180.0, 180.0, n), rng.uniform(-90.0, 90.0, n)])


# ---------------------------------------------------------------------------
# Golden accuracy vs float64 numpy and sklearn
# ---------------------------------------------------------------------------


class TestGolden:
    def test_city_scale_vs_float64(self, rng):
        a = city_points(rng, 500)
        b = city_points(rng, 500)
        ours = np.array(ms.haversine(a, b))
        ref = hav64(a, b)
        assert np.abs(ours - ref).max() < 2.0  # documented ~1-2 m absolute

    def test_continental_vs_float64(self, rng):
        a = continental_points(rng, 500)
        b = continental_points(rng, 500)
        ours = np.array(ms.haversine(a, b))
        ref = hav64(a, b)
        err = np.abs(ours - ref)
        assert err.max() < 5.0  # measured ~1.6 m; margin for fp32 at 130 deg
        far = ref > 100_000.0
        assert far.any()
        assert (err[far] / ref[far]).max() < 3e-5  # measured ~1.1e-6

    def test_global_random_vs_float64(self, rng):
        a = global_points(rng, 3000)
        b = global_points(rng, 3000)
        ours = np.array(ms.haversine(a, b))
        ref = hav64(a, b)
        err = np.abs(ours - ref)
        near = ref <= 1_000.0
        assert err[near].max() < 2.0 if near.any() else True
        # measured rel max ~2.9e-6 below 19,500 km; 1e-5 leaves >3x margin
        far = (ref > 1_000.0) & (ref <= 19_500_000.0)
        assert (err[far] / ref[far]).max() < 1e-5
        # within 2.5% of the antipode arcsin amplification kicks in (rel error
        # grows as 1/sqrt(1-h)); measured ~1.6e-5 at 99.9% of pi*R
        anti = ref > 19_500_000.0
        if anti.any():
            assert (err[anti] / ref[anti]).max() < 2e-4

    def test_near_antipodal_vs_float64(self, rng):
        # Deliberately constructed near-antipodes (19,700-20,015 km):
        # arcsin amplification makes fp32 error grow to ~100s of meters.
        a = global_points(rng, 400)
        anti_lon = ((a[:, 0] + 180.0 + rng.uniform(-2, 2, 400) + 180.0) % 360.0) - 180.0
        anti_lat = np.clip(-a[:, 1] + rng.uniform(-2, 2, 400), -90.0, 90.0)
        b = np.column_stack([anti_lon, anti_lat])
        ours = np.array(ms.haversine(a, b))
        ref = hav64(a, b)
        assert ref.min() > 19_000_000.0  # sanity: these really are near-antipodal
        assert (np.abs(ours - ref) / ref).max() < 2e-4  # measured ~3.3e-5

    def test_cross_continental_vs_sklearn(self, rng):
        from sklearn.metrics.pairwise import haversine_distances

        a = continental_points(rng, 120)
        b = continental_points(rng, 90)
        ours = np.array(ms.haversine_cross(a, b))
        # sklearn wants (lat, lon) radians, returns unit-sphere distances
        ref = haversine_distances(np.radians(a[:, ::-1]), np.radians(b[:, ::-1])) * R
        err = np.abs(ours - ref)
        assert err.max() < 5.0
        far = ref > 100_000.0
        assert (err[far] / ref[far]).max() < 3e-5

    def test_unit_sphere_vs_sklearn(self, rng):
        from sklearn.metrics.pairwise import haversine_distances

        a = city_points(rng, 80)
        b = city_points(rng, 60)
        ours = np.array(ms.haversine_cross(a, b, radius=1.0))
        ref = haversine_distances(np.radians(a[:, ::-1]), np.radians(b[:, ::-1]))
        # unit sphere: fp32 resolution of the angle itself (~1e-7)
        assert np.abs(ours - ref).max() < 5e-7


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_identical_points_exact_zero(self):
        p = np.array([[-74.006, 40.7128], [151.2093, -33.8688], [0.0, 0.0], [179.9, -89.9]])
        d = np.array(ms.haversine(p, p))
        assert (d == 0.0).all()
        # mx.array input path (fp32 deg2rad) is still exactly zero: identical
        # fp32 operands difference to exact 0
        pm = mx.array(p.astype(np.float32))
        dm = np.array(ms.haversine(pm, pm))
        assert (dm == 0.0).all()

    def test_same_pole_any_longitude(self):
        # Every (lon, 90) is the same physical point; fp32 cos(pi/2) != 0
        # exactly, but the residual is sub-meter.
        d1 = float(ms.haversine([0.0, 90.0], [120.0, 90.0]))
        d2 = float(ms.haversine([-45.0, -90.0], [170.0, -90.0]))
        assert abs(d1) < 2.0
        assert abs(d2) < 2.0

    def test_pole_to_pole(self):
        d = float(ms.haversine([0.0, 90.0], [0.0, -90.0]))
        assert d == pytest.approx(PI_R, abs=5.0)
        # different longitudes on each pole: still exactly half circumference
        d2 = float(ms.haversine([77.0, 90.0], [-133.0, -90.0]))
        assert d2 == pytest.approx(PI_R, abs=5.0)

    def test_pole_to_equator(self):
        d = float(ms.haversine([37.0, 90.0], [12.0, 0.0]))
        assert d == pytest.approx(PI_R / 2.0, abs=5.0)

    def test_dateline_pair(self):
        # -179.9 vs 179.9: 0.2 deg of longitude through the dateline. The
        # sin(dlon/2) formulation handles the 359.8 deg raw difference.
        lat = 10.0
        d = float(ms.haversine([-179.9, lat], [179.9, lat]))
        ref = float(hav64([-179.9, lat], [179.9, lat]))
        equiv = float(hav64([0.0, lat], [0.2, lat]))  # same physical separation
        assert ref == pytest.approx(equiv, abs=1e-6)  # reference sanity
        assert d == pytest.approx(ref, abs=2.0)
        assert d < 25_000  # ~22 km, NOT half the globe

    def test_dateline_random_band(self, rng):
        # Random pairs straddling the dateline in a narrow band.
        west = np.column_stack([rng.uniform(179.0, 180.0, 300), rng.uniform(-60, 60, 300)])
        east = np.column_stack([rng.uniform(-180.0, -179.0, 300), rng.uniform(-60, 60, 300)])
        ours = np.array(ms.haversine(west, east))
        ref = hav64(west, east)
        assert np.abs(ours - ref).max() < 5.0
        # all separations are small (< ~2 deg lon + 120 deg lat worst case),
        # none should wrap to ~half circumference
        assert ours.max() < 0.75 * PI_R

    def test_exact_antipode_clamp(self, rng):
        # h reaches/exceeds 1.0 in fp32; the clamp must keep results finite
        # and pinned near pi*R (fp32 arcsin amplification allows km-scale dev).
        lats = np.linspace(-89.0, 89.0, 400)
        a = np.column_stack([np.full(400, 30.0), lats])
        b = np.column_stack([np.full(400, -150.0), -lats])
        d = np.array(ms.haversine(a, b))
        assert np.isfinite(d).all()
        assert d.max() <= PI_R + 2.0  # clamp: can never exceed half circumference
        assert np.abs(d - PI_R).max() < 15_000.0  # measured ~6.2 km worst case
        # equatorial antipode too
        deq = float(ms.haversine([0.0, 0.0], [180.0, 0.0]))
        assert deq == pytest.approx(PI_R, abs=15_000.0)

    def test_radius_parameter(self, rng):
        # unit sphere: pole-to-pole is exactly pi
        d_unit = float(ms.haversine([0.0, 90.0], [0.0, -90.0], radius=1.0))
        assert d_unit == pytest.approx(np.pi, abs=1e-6)
        # distances scale linearly (exactly, radius enters as a pure factor)
        a = continental_points(rng, 50)
        b = continental_points(rng, 50)
        d1 = np.array(ms.haversine(a, b, radius=1.0))
        d2 = np.array(ms.haversine(a, b, radius=2.0))
        np.testing.assert_array_equal(2.0 * d1, d2)
        dR = np.array(ms.haversine(a, b))
        np.testing.assert_allclose(dR, d1 * R, rtol=1e-6)

    def test_degrees_false_radians_input(self, rng):
        a = continental_points(rng, 100)
        b = continental_points(rng, 100)
        d_deg = np.array(ms.haversine(a, b))
        # numpy path: degrees=True converts in float64 then casts fp32, which
        # is bitwise-identical to passing float64 radians directly
        d_rad = np.array(ms.haversine(np.radians(a), np.radians(b), degrees=False))
        np.testing.assert_array_equal(d_deg, d_rad)
        # mx radians input: same fp32 values in, same result out
        d_rad_mx = np.array(
            ms.haversine(
                mx.array(np.radians(a).astype(np.float32)),
                mx.array(np.radians(b).astype(np.float32)),
                degrees=False,
            )
        )
        np.testing.assert_array_equal(d_deg, d_rad_mx)
        # cross and knn accept degrees=False too
        c_deg = np.array(ms.haversine_cross(a[:20], b[:15]))
        c_rad = np.array(ms.haversine_cross(np.radians(a[:20]), np.radians(b[:15]), degrees=False))
        np.testing.assert_array_equal(c_deg, c_rad)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


class TestInputHandling:
    def test_mx_array_input(self, rng):
        # mx input converts deg->rad in fp32 (documented); still meter-scale
        a = city_points(rng, 200)
        b = city_points(rng, 200)
        d_mx = np.array(ms.haversine(mx.array(a), mx.array(b)))
        ref = hav64(a, b)
        assert np.abs(d_mx - ref).max() < 3.0  # measured ~0.9 m
        # continental mx input
        a2 = continental_points(rng, 200)
        b2 = continental_points(rng, 200)
        d_mx2 = np.array(ms.haversine(mx.array(a2), mx.array(b2)))
        assert np.abs(d_mx2 - hav64(a2, b2)).max() < 5.0

    def test_list_input(self):
        nyc = [-74.0060, 40.7128]
        london = [-0.1278, 51.5074]
        d = ms.haversine(nyc, london)
        assert isinstance(d, mx.array)
        assert float(d) == pytest.approx(float(hav64(nyc, london)), rel=1e-5)
        # list of pairs
        d2 = np.array(ms.haversine([nyc, london], [london, nyc]))
        assert d2.shape == (2,)
        assert d2[0] == pytest.approx(d2[1], abs=1e-3)  # symmetric

    def test_shapely_point_inputs(self, rng):
        gpd = pytest.importorskip("geopandas")
        import pandas as pd

        a = city_points(rng, 30)
        b = city_points(rng, 30)
        ref = np.array(ms.haversine(a, b))
        # GeoSeries of Points
        gs = gpd.GeoSeries(shapely.points(a))
        np.testing.assert_array_equal(np.array(ms.haversine(gs, b)), ref)
        # plain pandas Series of Points
        ser = pd.Series(list(shapely.points(a)))
        np.testing.assert_array_equal(np.array(ms.haversine(ser, b)), ref)
        # object ndarray of Points
        obj = np.array(list(shapely.points(a)), dtype=object)
        np.testing.assert_array_equal(np.array(ms.haversine(obj, b)), ref)
        # bare single Point geometry
        d1 = np.array(ms.haversine(shapely.Point(a[0]), b[0]))
        np.testing.assert_array_equal(d1.reshape(()), ref[0])

    def test_single_pair_scalar_shape(self):
        d = ms.haversine([-74.0, 40.7], [-73.9, 40.8])
        assert d.shape == ()  # (2,) x (2,) -> scalar
        assert 0 < float(d) < 20_000

    def test_broadcasting_matrix(self, rng):
        a = city_points(rng, 12)
        b = city_points(rng, 9)
        d = np.array(ms.haversine(a[:, None, :], b[None, :, :]))
        assert d.shape == (12, 9)
        ref = np.array(ms.haversine_cross(a, b))
        np.testing.assert_array_equal(d, ref)

    def test_broadcasting_one_vs_many(self, rng):
        b = city_points(rng, 15)
        one = np.array([-74.0, 40.7])
        d = np.array(ms.haversine(one, b))  # (2,) x (15, 2) -> (15,)
        assert d.shape == (15,)
        each = np.array([float(ms.haversine(one, b[i])) for i in range(15)])
        np.testing.assert_array_equal(d, each)

    def test_wrong_last_dim_errors(self):
        good = np.zeros((3, 2))
        with pytest.raises(ValueError):
            ms.haversine(np.zeros((3, 3)), good)
        with pytest.raises(ValueError):
            ms.haversine(good, np.zeros((3, 3)))
        with pytest.raises(ValueError):
            ms.haversine(np.float64(1.0), good)  # 0-d
        with pytest.raises(ValueError):
            ms.haversine(mx.zeros((4, 5)), mx.zeros((4, 5)))  # mx path
        with pytest.raises(ValueError):
            ms.haversine(np.zeros(3), good)  # (3,) is not (..., 2)

    def test_output_type_and_dtype(self, rng):
        a = city_points(rng, 5)
        b = city_points(rng, 5)
        d = ms.haversine(a, b)
        assert isinstance(d, mx.array)
        assert d.dtype == mx.float32
        c = ms.haversine_cross(a, b)
        assert isinstance(c, mx.array)
        assert c.dtype == mx.float32


# ---------------------------------------------------------------------------
# haversine_cross
# ---------------------------------------------------------------------------


class TestCross:
    def test_shape_and_symmetry(self, rng):
        a = city_points(rng, 37)
        b = city_points(rng, 21)
        c = np.array(ms.haversine_cross(a, b))
        assert c.shape == (37, 21)
        ct = np.array(ms.haversine_cross(b, a))
        np.testing.assert_allclose(c, ct.T, atol=1e-3)

    def test_tiny_chunk_bytes_forces_multichunk(self, rng):
        # chunk_bytes=1 -> rows_per_chunk=1 -> N separate chunks; must be
        # bitwise identical to the single-chunk result (same elementwise ops).
        a = city_points(rng, 57)
        b = city_points(rng, 23)
        single = np.array(ms.haversine_cross(a, b))
        chunked = np.array(ms.haversine_cross(a, b, chunk_bytes=1))
        np.testing.assert_array_equal(single, chunked)

    def test_midsize_chunk_ragged_tail(self, rng):
        # rows_per_chunk=10 with N=57: five full chunks + ragged tail of 7
        a = city_points(rng, 57)
        b = city_points(rng, 23)
        cb = 10 * 23 * 4 * 4  # rows_per_chunk = chunk_bytes // (M*16) = 10
        single = np.array(ms.haversine_cross(a, b))
        chunked = np.array(ms.haversine_cross(a, b, chunk_bytes=cb))
        np.testing.assert_array_equal(single, chunked)

    def test_n_or_m_equals_one(self, rng):
        a = city_points(rng, 20)
        b = city_points(rng, 1)
        c = np.array(ms.haversine_cross(a, b))
        assert c.shape == (20, 1)
        np.testing.assert_array_equal(c[:, 0], np.array(ms.haversine(a, b[0])))
        c2 = np.array(ms.haversine_cross(b, a))
        assert c2.shape == (1, 20)
        c3 = np.array(ms.haversine_cross(b, b))
        assert c3.shape == (1, 1)
        assert c3[0, 0] == 0.0

    def test_empty_inputs(self, rng):
        b = city_points(rng, 5)
        assert np.array(ms.haversine_cross(np.zeros((0, 2)), b)).shape == (0, 5)
        assert np.array(ms.haversine_cross(b, np.zeros((0, 2)))).shape == (5, 0)

    def test_dimension_errors(self, rng):
        b = city_points(rng, 5)
        with pytest.raises(ValueError):
            ms.haversine_cross(np.array([0.0, 0.0]), b)  # 1-D
        with pytest.raises(ValueError):
            ms.haversine_cross(b, np.array([0.0, 0.0]))
        with pytest.raises(ValueError):
            ms.haversine_cross(np.zeros((2, 3, 2)), b)  # 3-D


# ---------------------------------------------------------------------------
# haversine_knn
# ---------------------------------------------------------------------------


class TestKnn:
    def test_matches_bruteforce(self, rng):
        q = city_points(rng, 80)
        r = city_points(rng, 200)
        dist, idx = ms.haversine_knn(q, r, k=7)
        dist, idx = np.array(dist), np.array(idx)
        assert dist.shape == (80, 7) and idx.shape == (80, 7)
        full = np.array(ms.haversine_cross(q, r))
        # distances must equal the brute-force sorted top-7 exactly (same
        # fp32 pipeline, selection cannot change values)
        np.testing.assert_array_equal(dist, np.sort(full, axis=1)[:, :7])
        # indices must be consistent with the distances they claim
        np.testing.assert_array_equal(dist, np.take_along_axis(full, idx, axis=1))
        # no duplicate index within a row
        for row in idx:
            assert len(set(row.tolist())) == 7

    def test_distances_sorted_ascending(self, rng):
        q = city_points(rng, 40)
        r = city_points(rng, 100)
        dist, _ = ms.haversine_knn(q, r, k=10)
        assert (np.diff(np.array(dist), axis=1) >= 0).all()

    def test_k1(self, rng):
        q = city_points(rng, 30)
        r = city_points(rng, 50)
        dist, idx = ms.haversine_knn(q, r, k=1)
        dist, idx = np.array(dist), np.array(idx)
        assert dist.shape == (30, 1) and idx.shape == (30, 1)
        full = np.array(ms.haversine_cross(q, r))
        np.testing.assert_array_equal(dist[:, 0], full.min(axis=1))
        np.testing.assert_array_equal(idx[:, 0], np.argmin(full, axis=1))

    def test_k_equals_m_full_sort(self, rng):
        q = city_points(rng, 25)
        r = city_points(rng, 18)
        dist, idx = ms.haversine_knn(q, r, k=18)
        dist, idx = np.array(dist), np.array(idx)
        assert dist.shape == (25, 18)
        full = np.array(ms.haversine_cross(q, r))
        np.testing.assert_array_equal(dist, np.sort(full, axis=1))
        # indices are a permutation of arange(M) per row
        np.testing.assert_array_equal(
            np.sort(idx, axis=1), np.tile(np.arange(18), (25, 1))
        )
        np.testing.assert_array_equal(dist, np.take_along_axis(full, idx, axis=1))

    def test_k_validation_errors(self, rng):
        q = city_points(rng, 5)
        r = city_points(rng, 8)
        with pytest.raises(ValueError):
            ms.haversine_knn(q, r, k=0)
        with pytest.raises(ValueError):
            ms.haversine_knn(q, r, k=9)  # k > M
        with pytest.raises(ValueError):
            ms.haversine_knn(q, r, k=-1)

    def test_chunked_equals_unchunked(self, rng):
        q = city_points(rng, 60)
        r = city_points(rng, 45)
        for k in (5, 45):  # argpartition path and full-sort path
            d0, i0 = ms.haversine_knn(q, r, k=k)
            d1, i1 = ms.haversine_knn(q, r, k=k, chunk_bytes=1)
            np.testing.assert_array_equal(np.array(d0), np.array(d1))
            np.testing.assert_array_equal(np.array(i0), np.array(i1))
            assert i1.dtype == mx.int32

    def test_dtypes(self, rng):
        q = city_points(rng, 10)
        r = city_points(rng, 20)
        dist, idx = ms.haversine_knn(q, r, k=3)
        assert idx.dtype == mx.int32
        assert dist.dtype == mx.float32

    def test_n1_and_m1(self, rng):
        r = city_points(rng, 20)
        # single query
        dist, idx = ms.haversine_knn(r[:1], r, k=4)
        assert np.array(dist).shape == (1, 4)
        assert float(np.array(dist)[0, 0]) == 0.0  # itself
        assert int(np.array(idx)[0, 0]) == 0
        # single reference: everyone's only neighbor is index 0
        dist2, idx2 = ms.haversine_knn(r, r[:1], k=1)
        assert (np.array(idx2) == 0).all()
        np.testing.assert_array_equal(
            np.array(dist2)[:, 0], np.array(ms.haversine(r, r[0]))
        )

    def test_duplicate_references_ties(self, rng):
        # duplicated reference points: tied distances may pick either copy,
        # but the reported distances must still be the true sorted top-k
        r = city_points(rng, 6)
        rdup = np.vstack([r, r])
        q = city_points(rng, 15)
        dist, idx = ms.haversine_knn(q, rdup, k=4)
        dist, idx = np.array(dist), np.array(idx)
        full = np.array(ms.haversine_cross(q, rdup))
        np.testing.assert_array_equal(dist, np.sort(full, axis=1)[:, :4])
        np.testing.assert_array_equal(dist, np.take_along_axis(full, idx, axis=1))

    def test_dimension_error(self, rng):
        r = city_points(rng, 5)
        with pytest.raises(ValueError):
            ms.haversine_knn(np.zeros((2, 3, 2)), r, k=1)
        with pytest.raises(ValueError):
            ms.haversine_knn(np.array([0.0, 0.0]), r, k=1)  # 1-D

    def test_empty_query(self, rng):
        r = city_points(rng, 5)
        dist, idx = ms.haversine_knn(np.zeros((0, 2)), r, k=3)
        assert np.array(dist).shape == (0, 3)
        assert np.array(idx).shape == (0, 3)
