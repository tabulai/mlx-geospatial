"""Fused kNN selection kernel vs the composed argpartition path.

The two paths must agree exactly on selected indices (outside exact-tie
ambiguity) and bit-for-bit on distances, since both report distances through
the same fused ``_hav`` arithmetic. The chord-filtered selectors must match
the plain selectors bit for bit: same indices, distances, order and -1/inf
padding.
"""

from __future__ import annotations

import importlib

import numpy as np
import mlx.core as mx
import pytest

import mlx_geospatial as ms
from mlx_geospatial._dispatch import metal_available
from mlx_geospatial.haversine import (
    _KNN_KERNEL_MAX_K,
    _knn_chord,
    _knn_chord_route,
    _knn_chord_slack,
    _knn_chord_strategy,
    _knn_composed,
    _knn_fused,
    _knn_metal_strategy,
    _knn_simdgroup,
    _prep,
)

pytestmark = pytest.mark.skipif(
    not metal_available(), reason="fused kNN kernel requires Metal"
)


def city_cloud(rng, n):
    return np.column_stack(
        [rng.uniform(-74.3, -73.7, n), rng.uniform(40.5, 40.9, n)]
    )


@pytest.mark.parametrize("k", [1, 5, 32])
@pytest.mark.parametrize("m", [40, 1000, 20000])
def test_fused_matches_composed(rng, k, m):
    if k > m:
        pytest.skip("k > m")
    q = _prep(city_cloud(rng, 500), True)
    r = _prep(city_cloud(rng, m), True)
    df, i_f = _knn_fused(q, r, k, ms.R_EARTH_M)
    dc, i_c = _knn_composed(q, r, k, ms.R_EARTH_M, 512 * 1024 * 1024)
    mx.eval(df, i_f, dc, i_c)
    np.testing.assert_array_equal(np.array(df), np.array(dc))
    np.testing.assert_array_equal(np.array(i_f), np.array(i_c))


def test_public_api_uses_fused_and_matches_bruteforce(rng):
    q = city_cloud(rng, 300)
    r = city_cloud(rng, 5000)
    dist, idx = ms.haversine_knn(q, r, k=7)
    full = np.array(ms.haversine_cross(q, r))
    ref_idx = np.argsort(full, axis=1, kind="stable")[:, :7]
    np.testing.assert_array_equal(np.array(idx), ref_idx)
    np.testing.assert_array_equal(
        np.array(dist), np.take_along_axis(full, ref_idx, axis=1)
    )


# Both references are within about 100 m of the query's antipode.  On the M4
# Max, multiply-add contraction in the selectors collapses their h values to
# the same fp32 value while the public _hav graph keeps them distinct.  Which
# reference the public graph ranks first can vary with the GPU's transcendental
# approximation, so the contract is agreement with that graph, not a fixed
# hardware-specific index.
_ANTIPODAL_QUERY = np.array([[91.06015715, -46.14764379]])
_ANTIPODAL_REFERENCES = np.array(
    [
        [-88.93957964, 46.14706647],
        [-88.94039897, 46.14803763],
    ]
)


def test_public_k1_near_antipode_matches_bruteforce():
    full = np.array(ms.haversine_cross(_ANTIPODAL_QUERY, _ANTIPODAL_REFERENCES))
    expected_idx = np.argsort(full, axis=1, kind="stable")[:, :1]
    expected_dist = np.take_along_axis(full, expected_idx, axis=1)

    dist, idx = ms.haversine_knn(
        _ANTIPODAL_QUERY, _ANTIPODAL_REFERENCES, k=1
    )
    np.testing.assert_array_equal(np.array(idx), expected_idx)
    np.testing.assert_array_equal(np.array(dist), expected_dist)


@pytest.mark.parametrize(
    "selector", [_knn_fused, _knn_simdgroup], ids=["scalar", "simdgroup"]
)
def test_fused_selectors_near_antipode_match_bruteforce(selector):
    q = _prep(_ANTIPODAL_QUERY, True)
    r = _prep(_ANTIPODAL_REFERENCES, True)
    dist, idx = selector(q, r, 1, ms.R_EARTH_M)
    full = np.array(ms.haversine_cross(_ANTIPODAL_QUERY, _ANTIPODAL_REFERENCES))
    expected_idx = np.argsort(full, axis=1, kind="stable")[:, :1]
    expected_dist = np.take_along_axis(full, expected_idx, axis=1)

    np.testing.assert_array_equal(np.array(idx), expected_idx)
    np.testing.assert_array_equal(np.array(dist), expected_dist)


def test_k_above_kernel_cap_falls_back(rng, monkeypatch):
    """k > 32 must still return correct results via the composed path."""
    haversine_module = importlib.import_module("mlx_geospatial.haversine")

    def metal_must_not_run(*args, **kwargs):
        raise AssertionError("k above the cap attempted to run a Metal selector")

    monkeypatch.setattr(haversine_module, "_knn_simdgroup", metal_must_not_run)
    monkeypatch.setattr(haversine_module, "_knn_fused", metal_must_not_run)
    monkeypatch.setattr(haversine_module, "_knn_chord", metal_must_not_run)
    q = city_cloud(rng, 50)
    r = city_cloud(rng, 200)
    k = _KNN_KERNEL_MAX_K + 1
    dist, idx = ms.haversine_knn(q, r, k=k)
    assert np.array(dist).shape == (50, k)
    full = np.array(ms.haversine_cross(q, r))
    ref = np.sort(full, axis=1)[:, :k]
    np.testing.assert_allclose(np.array(dist), ref, rtol=1e-6, atol=1e-3)


def test_duplicated_references_tie_break(rng):
    """Duplicate reference points: earliest index wins, matching stable sort."""
    base = city_cloud(rng, 50)
    r = np.vstack([base, base])  # every point duplicated at index +50
    q = base + 1e-9
    dist, idx = ms.haversine_knn(q, r, k=2)
    idx = np.array(idx)
    # nearest two are the duplicate pair; the lower index must come first
    np.testing.assert_array_equal(idx[:, 0], np.arange(50))
    np.testing.assert_array_equal(idx[:, 1], np.arange(50) + 50)


def test_nan_reference_rows_never_selected(rng):
    r = city_cloud(rng, 100)
    r[10] = np.nan
    r[70] = np.nan
    q = city_cloud(rng, 200)
    _, idx = ms.haversine_knn(q, r, k=5)
    assert not np.isin(np.array(idx), [10, 70]).any()


def test_fewer_finite_refs_than_k():
    """Degenerate: only 2 finite refs but k=4 — kernel pads with -1/inf."""
    q = _prep(np.array([[0.0, 0.0]]), True)
    r = _prep(
        np.array([[0.1, 0.1], [np.nan, np.nan], [0.2, 0.2], [np.nan, np.nan]]), True
    )
    d, i = _knn_fused(q, r, 4, ms.R_EARTH_M)
    d, i = np.array(d), np.array(i)
    np.testing.assert_array_equal(i[0, :2], [0, 2])
    assert (i[0, 2:] == -1).all()
    assert np.isinf(d[0, 2:]).all()
    assert np.isfinite(d[0, :2]).all()


def test_fused_radius_parameter(rng):
    q = city_cloud(rng, 20)
    r = city_cloud(rng, 300)
    d_m, i_m = ms.haversine_knn(q, r, k=3)
    d_u, i_u = ms.haversine_knn(q, r, k=3, radius=1.0)
    np.testing.assert_array_equal(np.array(i_m), np.array(i_u))
    np.testing.assert_allclose(
        np.array(d_m), np.array(d_u) * ms.R_EARTH_M, rtol=1e-6
    )


@pytest.mark.parametrize(
    ("n", "m", "k"),
    [
        (1, 1, 1),
        (3, 31, 1),
        (4, 32, 5),
        (5, 33, 32),
        (7, 63, 5),
        (8, 64, 32),
        (9, 65, 7),
    ],
)
def test_simdgroup_matches_scalar_at_lane_and_launch_boundaries(rng, n, m, k):
    q = city_cloud(rng, n)
    r = city_cloud(rng, m)
    if m > 1:
        r[1] = r[0]  # exact tie in different lane strides
    if m > k:
        r[-1] = np.nan
    q_mx, r_mx = _prep(q, True), _prep(r, True)
    ds, i_s = _knn_simdgroup(q_mx, r_mx, k, ms.R_EARTH_M)
    df, i_f = _knn_fused(q_mx, r_mx, k, ms.R_EARTH_M)
    mx.eval(ds, i_s, df, i_f)
    np.testing.assert_array_equal(np.array(i_s), np.array(i_f))
    np.testing.assert_array_equal(np.array(ds), np.array(df))


def test_simdgroup_global_top_k_can_all_come_from_one_lane():
    """The 32-way merge must retain more than one candidate from each lane."""
    near = np.arange(0, 1024, 32)
    r = np.tile([120.0, 60.0], (1025, 1))
    r[near] = np.column_stack([(np.arange(32) + 1) * 1e-4, np.zeros(32)])
    q = np.array([[0.0, 0.0]])

    d, idx = _knn_simdgroup(_prep(q, True), _prep(r, True), 32, ms.R_EARTH_M)
    full = np.array(ms.haversine_cross(q, r))
    d, idx = np.array(d), np.array(idx)
    np.testing.assert_array_equal(idx[0], near)
    np.testing.assert_array_equal(d[0], full[0, near])


def test_simdgroup_ties_and_nan_padding_match_scalar():
    q = np.array([[0.0, 0.0], [np.nan, 0.0]])
    r = np.full((65, 2), np.nan)
    finite = np.array(
        [0, 1, 2, 5, 10, 15, 20, 25, 30, 31, 32, 33, 34, 40, 47, 50, 55, 62, 63, 64]
    )
    r[finite] = [1.0, 1.0]
    q_mx, r_mx = _prep(q, True), _prep(r, True)
    ds, i_s = _knn_simdgroup(q_mx, r_mx, 32, ms.R_EARTH_M)
    df, i_f = _knn_fused(q_mx, r_mx, 32, ms.R_EARTH_M)
    mx.eval(ds, i_s, df, i_f)
    ds, i_s = np.array(ds), np.array(i_s)
    np.testing.assert_array_equal(i_s, np.array(i_f))
    np.testing.assert_array_equal(ds, np.array(df))
    np.testing.assert_array_equal(i_s[0, :20], finite)
    assert (i_s[0, 20:] == -1).all() and np.isinf(ds[0, 20:]).all()
    assert (i_s[1] == -1).all() and np.isinf(ds[1]).all()


@pytest.mark.parametrize(
    ("n", "m", "k", "expected"),
    [
        (1024, 32, 32, "simdgroup"),
        (1025, 32, 1, "scalar"),
        (32_768, 128, 1, "simdgroup"),
        (32_769, 128, 1, "scalar"),
        (16_384, 128, 5, "simdgroup"),
        (8_193, 128, 10, "scalar"),
        (4_096, 128, 16, "simdgroup"),
        (2_049, 128, 32, "scalar"),
        (10_000, 100_000, 10, "simdgroup"),
        (16_385, 65_536, 10, "scalar"),
        (8_192, 65_536, 16, "simdgroup"),
        (2_049, 100_000, 32, "scalar"),
    ],
)
def test_metal_strategy_crossover(n, m, k, expected):
    assert _knn_metal_strategy(n, m, k) == expected


@pytest.mark.parametrize("threadgroup", [0, 31, 33, 1056])
def test_simdgroup_rejects_invalid_threadgroup(threadgroup):
    q = _prep(np.array([[0.0, 0.0]]), True)
    r = _prep(np.array([[1.0, 0.0]]), True)
    with pytest.raises(ValueError, match="multiple.*32"):
        _knn_simdgroup(q, r, 1, ms.R_EARTH_M, threadgroup=threadgroup)


def test_public_api_respects_cpu_default_device(monkeypatch, rng):
    """An active CPU device must use the composed path, not compile Metal."""
    q, r = city_cloud(rng, 4), city_cloud(rng, 40)

    def metal_must_not_run(*args, **kwargs):
        raise AssertionError("CPU-default kNN attempted to run a Metal kernel")

    haversine_module = importlib.import_module("mlx_geospatial.haversine")
    monkeypatch.setattr(haversine_module, "_knn_simdgroup", metal_must_not_run)
    monkeypatch.setattr(haversine_module, "_knn_fused", metal_must_not_run)
    monkeypatch.setattr(haversine_module, "_knn_chord", metal_must_not_run)

    old_device = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        q_mx, r_mx = _prep(q, True), _prep(r, True)
        expected = _knn_composed(
            q_mx, r_mx, 32, ms.R_EARTH_M, 512 * 1024 * 1024
        )
        got = ms.haversine_knn(q, r, k=32)
        mx.eval(*expected, *got)
    finally:
        mx.set_default_device(old_device)
    np.testing.assert_array_equal(np.array(got[0]), np.array(expected[0]))
    np.testing.assert_array_equal(np.array(got[1]), np.array(expected[1]))


# ---------------------------------------------------------------------------
# chord-filtered selectors
# ---------------------------------------------------------------------------


def _chord_cases():
    """(name, query, reference) in degrees: geometry that stresses the bound,
    ties, padding and the pruning switch."""
    rng = np.random.default_rng(2026)

    def globe(n):
        return np.column_stack(
            [rng.uniform(-180, 180, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))]
        )

    def cluster(n, w, c=(10.0, 20.0)):
        return np.column_stack(
            [c[0] + rng.uniform(-w, w, n), c[1] + rng.uniform(-w, w, n)]
        )

    yield "city", city_cloud(rng, 150), city_cloud(rng, 3000)
    yield "globe", globe(150), globe(3000)
    yield "1m cluster", cluster(100, 1e-5), cluster(2000, 1e-5)
    yield "1e-9 deg cluster", cluster(100, 1e-9), cluster(2000, 1e-9)
    yield "gridded duplicates", np.round(city_cloud(rng, 100), 3), np.round(city_cloud(rng, 2000), 3)
    yield "all duplicate refs", cluster(40, 1e-3), np.tile([[10.0, 20.0]], (500, 1))
    dq = np.column_stack(
        [rng.choice([-1, 1], 150) * rng.uniform(179.99, 180, 150), rng.uniform(-10, 10, 150)]
    )
    dr = np.column_stack(
        [rng.choice([-1, 1], 2000) * rng.uniform(179.99, 180, 2000), rng.uniform(-10, 10, 2000)]
    )
    yield "dateline", dq, dr
    wrapped = dr + np.column_stack([rng.choice([-720, -540, -360, 360, 540, 720], 2000), np.zeros(2000)])
    yield "wrapped longitudes", dq + [540.0, 0.0], wrapped
    poles_q = np.column_stack([rng.uniform(-180, 180, 150), rng.choice([-90.0, 90.0], 150)])
    poles_r = np.column_stack(
        [rng.uniform(-180, 180, 2000), rng.choice([-1, 1], 2000) * rng.uniform(89.999, 90, 2000)]
    )
    poles_r[:300, 1] = rng.choice([-90.0, 90.0], 300)
    yield "exact and near poles", poles_q, poles_r
    aq = globe(150)
    ar = np.column_stack(
        [aq[:, 0] + 180 + rng.normal(0, 1e-3, 150), -aq[:, 1] + rng.normal(0, 1e-3, 150)]
    )
    yield "near antipodes", aq, np.vstack([ar, globe(1500), np.column_stack([aq[:, 0] + 180, -aq[:, 1]])])
    nq = city_cloud(rng, 150)
    nq[::7] = np.nan
    nq[3, 0], nq[5, 1] = np.inf, -np.inf
    nr = city_cloud(rng, 2000)
    nr[::3] = np.nan
    nr[10, 0], nr[11, 1] = np.inf, -np.inf
    yield "nan and inf rows", nq, nr
    yield "latitude 95 (pruning off)", city_cloud(rng, 150), np.vstack([city_cloud(rng, 2000), [[-74.0, 95.0]]])
    yield "latitude -100 query (pruning off)", np.vstack([city_cloud(rng, 150), [[-74.0, -100.0]]]), city_cloud(rng, 2000)
    huge = globe(2000)
    huge[:, 0] = rng.uniform(-5e4, 5e4, 2000)
    yield "longitudes to 5e4 deg", globe(150), huge
    huge[0, 0] = 7e4
    yield "longitude 7e4 deg (pruning off)", globe(150), huge
    few = np.full((70, 2), np.nan)
    few[[3, 40, 66]] = city_cloud(rng, 3)
    yield "fewer finite refs than k", city_cloud(rng, 100), few
    yield "m = 1", city_cloud(rng, 50), city_cloud(rng, 1)
    yield "m = 33", city_cloud(rng, 100), city_cloud(rng, 33)


@pytest.mark.parametrize("radius", [ms.R_EARTH_M, 1.0], ids=["earth", "unit"])
@pytest.mark.parametrize(("name", "q", "r"), list(_chord_cases()), ids=lambda v: v if isinstance(v, str) else "")
def test_chord_selectors_match_plain_selectors_bit_for_bit(name, q, r, radius):
    q_mx, r_mx = _prep(q, True), _prep(r, True)
    for k in (1, 2, 5, 10, 16, 24, 31, 32):
        if k > len(r):
            continue
        expected = _knn_fused(q_mx, r_mx, k, radius)
        mx.eval(*expected)
        expected = tuple(np.array(a) for a in expected)
        for strategy in ("scalar", "simdgroup"):
            got = _knn_chord(q_mx, r_mx, k, radius, strategy)
            mx.eval(*got)
            np.testing.assert_array_equal(np.array(got[1]), expected[1], err_msg=f"{name} k={k} {strategy}")
            np.testing.assert_array_equal(np.array(got[0]), expected[0], err_msg=f"{name} k={k} {strategy}")


@pytest.mark.parametrize("m", [1, 3, 4, 5, 31, 32, 33, 127, 128, 129, 131, 160, 161])
def test_chord_selectors_at_unroll_and_lane_boundaries(rng, m):
    """Four references per iteration (strided by 32 lanes in the SIMDgroup
    kernel) plus a remainder loop; ties and NaN rows straddle both."""
    q = city_cloud(rng, 37)
    r = city_cloud(rng, m)
    if m > 4:
        r[4] = r[0]  # exact tie across unrolled groups
    if m > 3:
        r[2] = np.nan
    q_mx, r_mx = _prep(q, True), _prep(r, True)
    for k in (1, 3, 16, 32):
        if k > m:
            continue
        expected = tuple(np.array(a) for a in _knn_simdgroup(q_mx, r_mx, k, ms.R_EARTH_M))
        for strategy in ("scalar", "simdgroup"):
            d, idx = _knn_chord(q_mx, r_mx, k, ms.R_EARTH_M, strategy)
            np.testing.assert_array_equal(np.array(idx), expected[1], err_msg=f"k={k} {strategy}")
            np.testing.assert_array_equal(np.array(d), expected[0], err_msg=f"k={k} {strategy}")


@pytest.mark.parametrize("k", [16, 32])
def test_chord_simdgroup_shared_threshold_keeps_one_lane_top_k(k):
    """Every true neighbour sits in lane 0's stride while the other lanes see
    only far references: the shared threshold must not prune lane 0."""
    near = np.arange(0, 32 * 64, 32)
    r = np.tile([120.0, 60.0], (32 * 64 + 5, 1))
    r[near] = np.column_stack([(np.arange(64) + 1) * 1e-4, np.zeros(64)])
    q = np.array([[0.0, 0.0]])
    d, idx = _knn_chord(_prep(q, True), _prep(r, True), k, ms.R_EARTH_M, "simdgroup")
    full = np.array(ms.haversine_cross(q, r))
    d, idx = np.array(d), np.array(idx)
    np.testing.assert_array_equal(idx[0], near[:k])
    np.testing.assert_array_equal(d[0], full[0, near[:k]])


def test_chord_slack_switches_pruning_off_only_for_unsafe_inputs():
    city = city_cloud(np.random.default_rng(3), 100)

    def slack(q, r):
        return float(np.array(_knn_chord_slack(_prep(q, True), _prep(r, True)))[0])

    assert 0 < slack(city, city) < 1e-5
    assert slack(np.array([[0.0, 90.0], [0.0, -90.0]]), city) > 0  # fp32 pi/2 is fine
    assert slack(city + [540.0, 0.0], city) > slack(city, city)  # grows with |lon|
    assert slack(np.vstack([city, [[np.nan, np.inf], [np.inf, np.nan]]]), city) > 0
    assert slack(np.array([[0.0, 90.0001]]), city) < 0
    assert slack(city, np.vstack([city, [[-74.0, -95.0]]])) < 0
    assert slack(np.array([[5e4, 0.0]]), city) > 0
    assert slack(np.array([[7e4, 0.0]]), city) < 0


@pytest.mark.parametrize(
    ("n", "m", "k", "strategy", "chord"),
    [
        (1_000, 1_000_000, 10, "simdgroup", True),
        (1_000_000, 10_000, 10, "scalar", True),
        (200_000, 200_000, 10, "scalar", True),
        (1_000_000, 200_000, 10, "scalar", True),
        (16_384, 32, 10, "scalar", True),  # the fused refine pays alone
        (16_383, 32, 10, "scalar", False),
        (512, 32_768, 10, "simdgroup", True),
        (256, 32_768, 10, "simdgroup", False),  # n * m < 2^24
        (2_048, 8_192, 5, "simdgroup", True),
        (2_048, 8_192, 10, "simdgroup", False),
        (100, 1_000, 1, "simdgroup", False),
        (1_000_000, 200_000, 17, "scalar", False),  # k > 16: plain selectors
        (16_384, 32_768, 1, "simdgroup", True),
        (16_385, 32_768, 1, "scalar", True),
        (4_096, 131_072, 16, "simdgroup", True),
        (4_097, 131_072, 16, "scalar", True),
        (2_049, 8_192, 16, "scalar", True),
        (4_096, 2_048, 16, "simdgroup", False),
    ],
)
def test_chord_route_and_strategy(n, m, k, strategy, chord):
    if k <= 16:
        assert _knn_chord_strategy(n, m, k) == strategy
    assert _knn_chord_route(n, m, k, strategy) is chord


@pytest.mark.parametrize(("n", "m", "strategy"), [(16_384, 64, "scalar"), (512, 40_000, "simdgroup")])
def test_public_api_chord_route_matches_bruteforce(monkeypatch, n, m, strategy):
    rng = np.random.default_rng(n + m)
    q, r = city_cloud(rng, n), city_cloud(rng, m)
    r[5] = r[3]  # an exact tie
    r[7] = np.nan
    haversine_module = importlib.import_module("mlx_geospatial.haversine")
    taken = []

    def spy(*args, **kwargs):
        taken.append(args[4])
        return _knn_chord(*args, **kwargs)

    monkeypatch.setattr(haversine_module, "_knn_chord", spy)
    k = 5
    dist, idx = ms.haversine_knn(q, r, k=k)
    assert taken == [strategy]
    full = np.array(ms.haversine_cross(q, r))
    full = np.where(np.isnan(full), np.inf, full)
    ref_idx = np.argsort(full, axis=1, kind="stable")[:, :k]
    np.testing.assert_array_equal(np.array(idx), ref_idx)
    np.testing.assert_array_equal(np.array(dist), np.take_along_axis(full, ref_idx, axis=1))


def test_chord_rejects_unknown_strategy_and_bad_threadgroup():
    q = _prep(np.array([[0.0, 0.0]]), True)
    r = _prep(np.array([[1.0, 0.0]]), True)
    with pytest.raises(ValueError, match="strategy"):
        _knn_chord(q, r, 1, ms.R_EARTH_M, "threadgroup")
    with pytest.raises(ValueError, match="multiple.*32"):
        _knn_chord(q, r, 1, ms.R_EARTH_M, "scalar", threadgroup=48)
