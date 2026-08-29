"""Fused kNN selection kernel vs the composed argpartition path.

The two paths must agree exactly on selected indices (outside exact-tie
ambiguity) and bit-for-bit on distances, since both report distances through
the same fused ``_hav`` arithmetic.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

import mlx_spatial as ms
from mlx_spatial._dispatch import metal_available
from mlx_spatial.haversine import _KNN_KERNEL_MAX_K, _knn_composed, _knn_fused, _prep

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


def test_k_above_kernel_cap_falls_back(rng):
    """k > 32 must still return correct results via the composed path."""
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
