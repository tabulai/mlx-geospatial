"""Fused kNN selection kernel vs the composed argpartition path.

The two paths must agree exactly on selected indices (outside exact-tie
ambiguity) and bit-for-bit on distances, since both report distances through
the same fused ``_hav`` arithmetic.
"""

from __future__ import annotations

import importlib

import numpy as np
import mlx.core as mx
import pytest

import mlx_spatial as ms
from mlx_spatial._dispatch import metal_available
from mlx_spatial.haversine import (
    _KNN_KERNEL_MAX_K,
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


# Both references are within about 100 m of the query's antipode.  With
# multiply-add contraction enabled in the selectors their h values collapse
# to the same fp32 value, while the public _hav graph keeps them distinct.
_ANTIPODAL_QUERY = np.array([[91.06015715, -46.14764379]])
_ANTIPODAL_REFERENCES = np.array(
    [
        [-88.93957964, 46.14706647],
        [-88.94039897, 46.14803763],
    ]
)


def test_public_k1_near_antipode_matches_bruteforce():
    full = np.array(ms.haversine_cross(_ANTIPODAL_QUERY, _ANTIPODAL_REFERENCES))
    assert full[0, 1] < full[0, 0]  # Regression case is sensitive to ordering.

    dist, idx = ms.haversine_knn(
        _ANTIPODAL_QUERY, _ANTIPODAL_REFERENCES, k=1
    )
    np.testing.assert_array_equal(np.array(idx), [[1]])
    np.testing.assert_array_equal(np.array(dist), full[:, 1:2])


@pytest.mark.parametrize(
    "selector", [_knn_fused, _knn_simdgroup], ids=["scalar", "simdgroup"]
)
def test_fused_selectors_near_antipode_match_bruteforce(selector):
    q = _prep(_ANTIPODAL_QUERY, True)
    r = _prep(_ANTIPODAL_REFERENCES, True)
    dist, idx = selector(q, r, 1, ms.R_EARTH_M)
    full = np.array(ms.haversine_cross(_ANTIPODAL_QUERY, _ANTIPODAL_REFERENCES))

    np.testing.assert_array_equal(np.array(idx), [[1]])
    np.testing.assert_array_equal(np.array(dist), full[:, 1:2])


def test_k_above_kernel_cap_falls_back(rng, monkeypatch):
    """k > 32 must still return correct results via the composed path."""
    haversine_module = importlib.import_module("mlx_spatial.haversine")

    def metal_must_not_run(*args, **kwargs):
        raise AssertionError("k above the cap attempted to run a Metal selector")

    monkeypatch.setattr(haversine_module, "_knn_simdgroup", metal_must_not_run)
    monkeypatch.setattr(haversine_module, "_knn_fused", metal_must_not_run)
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

    haversine_module = importlib.import_module("mlx_spatial.haversine")
    monkeypatch.setattr(haversine_module, "_knn_simdgroup", metal_must_not_run)
    monkeypatch.setattr(haversine_module, "_knn_fused", metal_must_not_run)

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
