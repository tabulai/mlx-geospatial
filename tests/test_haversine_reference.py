"""Tests for HaversineReference reusable kNN / cross-distance reference sets."""

from __future__ import annotations

import importlib

import mlx.core as mx
import numpy as np
import pytest

import mlx_geospatial as ms
from mlx_geospatial._dispatch import metal_available

hav = importlib.import_module("mlx_geospatial.haversine")


def _city_cloud(rng, n):
    return np.column_stack(
        [rng.uniform(-74.3, -73.7, n), rng.uniform(40.5, 40.9, n)]
    )


def _globe(rng, n):
    return np.column_stack(
        [rng.uniform(-180.0, 180.0, n), np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))]
    )


@pytest.mark.parametrize("k", [1, 5, 10, 16, 17, 32, 40])
@pytest.mark.parametrize("n", [1, 10, 128])
def test_haversine_reference_matches_raw_knn_bit_for_bit(rng, k, n):
    m = 500 if k > 16 else 9000
    q = _city_cloud(rng, n)
    r = _city_cloud(rng, m)
    ref = ms.HaversineReference(r)
    assert len(ref) == m
    assert ref.n_points == m
    assert ref.shape == (m, 2)
    assert "HaversineReference(" in repr(ref)

    d_raw, i_raw = ms.haversine_knn(q, r, k)
    d_prep, i_prep = ms.haversine_knn(q, ref, k)
    d_meth, i_meth = ref.knn(q, k)

    np.testing.assert_array_equal(np.asarray(i_prep), np.asarray(i_raw))
    np.testing.assert_array_equal(np.asarray(d_prep), np.asarray(d_raw))
    np.testing.assert_array_equal(np.asarray(i_meth), np.asarray(i_raw))
    np.testing.assert_array_equal(np.asarray(d_meth), np.asarray(d_raw))


def test_haversine_reference_adversarial_distributions_match_raw(rng):
    cases = []
    cases.append(("globe", _globe(rng, 25), _globe(rng, 9000)))
    # Dateline & wrapped longitudes
    dq = np.column_stack(
        [rng.choice([-1, 1], 20) * rng.uniform(179.99, 180.0, 20), rng.uniform(-10, 10, 20)]
    )
    dr = np.column_stack(
        [rng.choice([-1, 1], 9000) * rng.uniform(179.99, 180.0, 9000), rng.uniform(-10, 10, 9000)]
    )
    dr_wrap = dr + np.column_stack(
        [rng.choice([-720, -360, 0, 360, 720], 9000), np.zeros(9000)]
    )
    cases.append(("wrapped_dateline", dq + [540.0, 0.0], dr_wrap))
    # Poles & antipodes
    aq = _globe(rng, 20)
    ar = np.vstack([
        np.column_stack([aq[:, 0] + 180.0, -aq[:, 1]]),
        np.column_stack([rng.uniform(-180, 180, 200), rng.choice([-90.0, 90.0], 200)]),
        _globe(rng, 8800),
    ])
    cases.append(("poles_antipodes", aq, ar))
    # NaN / inf rows and fewer finite refs than k
    nq = _city_cloud(rng, 12)
    nq[2] = np.nan
    nq[5, 0] = np.inf
    nr = _city_cloud(rng, 9000)
    nr[::4] = np.nan
    nr[7, 1] = -np.inf
    cases.append(("nan_inf", nq, nr))
    # Latitude > 90 disables chord pruning via cached _max_abs
    bad_lat_r = np.vstack([_city_cloud(rng, 9000), [[-74.0, 95.0]]])
    cases.append(("lat_gt_90", _city_cloud(rng, 10), bad_lat_r))
    # Coordinates > 1e3 rad disable chord pruning via cached _max_abs
    huge_r = _globe(rng, 9000)
    huge_r[0, 0] = 60_000.0
    cases.append(("huge_lon", _city_cloud(rng, 10), huge_r))
    # Fewer finite refs than k
    few_r = np.array([[0.1, 0.1], [np.nan, np.nan], [0.2, 0.2], [np.nan, np.nan]])
    cases.append(("few_finite", np.array([[0.0, 0.0], [np.nan, 0.0]]), few_r))

    for name, q, r in cases:
        k = min(10, len(r))
        ref = ms.HaversineReference(r)
        d_raw, i_raw = ms.haversine_knn(q, r, k)
        d_prep, i_prep = ms.haversine_knn(q, ref, k)
        np.testing.assert_array_equal(np.asarray(i_prep), np.asarray(i_raw), err_msg=name)
        np.testing.assert_array_equal(np.asarray(d_prep), np.asarray(d_raw), err_msg=name)


def test_haversine_reference_degrees_applies_only_to_query(rng):
    q_deg = _city_cloud(rng, 16)
    r_deg = _city_cloud(rng, 500)
    q_rad = np.radians(q_deg)
    r_rad = np.radians(r_deg)

    ref_from_deg = ms.HaversineReference(r_deg, degrees=True)
    ref_from_rad = ms.HaversineReference(r_rad, degrees=False)
    d_expected, i_expected = ms.haversine_knn(q_deg, r_deg, k=5, radius=1.0)

    for ref in (ref_from_deg, ref_from_rad):
        d1, i1 = ms.haversine_knn(q_deg, ref, k=5, degrees=True, radius=1.0)
        d2, i2 = ms.haversine_knn(q_rad, ref, k=5, degrees=False, radius=1.0)
        np.testing.assert_array_equal(np.asarray(i1), np.asarray(i_expected))
        np.testing.assert_array_equal(np.asarray(d1), np.asarray(d_expected))
        np.testing.assert_array_equal(np.asarray(i2), np.asarray(i_expected))
        np.testing.assert_array_equal(np.asarray(d2), np.asarray(d_expected))


def test_haversine_reference_copies_device_input_in_radians():
    r = mx.array([[0.1, 0.1], [0.2, 0.2]])
    q = mx.array([[0.1, 0.1]])
    ref = ms.HaversineReference(r, degrees=False)
    expected_d, expected_i = ref.knn(q, k=1, degrees=False)

    np.asarray(r)[0] = [0.8, 0.8]
    actual_d, actual_i = ref.knn(q, k=1, degrees=False)
    np.testing.assert_array_equal(np.asarray(actual_d), np.asarray(expected_d))
    np.testing.assert_array_equal(np.asarray(actual_i), np.asarray(expected_i))


@pytest.mark.skipif(not metal_available(), reason="Metal GPU required")
def test_haversine_reference_reuses_unit_vectors_and_enables_small_batch_chord(
    rng, monkeypatch
):
    r = _city_cloud(rng, 10_000)
    q = _city_cloud(rng, 4)
    ref = ms.HaversineReference(r)

    uv_shapes = []
    orig_uv = hav._knn_unit_vectors

    def spy_uv(p):
        uv_shapes.append(tuple(p.shape))
        return orig_uv(p)

    chord_calls = []
    orig_chord = hav._knn_chord

    def spy_chord(*args, **kwargs):
        chord_calls.append(kwargs.get("rv") is ref.unit_vectors)
        return orig_chord(*args, **kwargs)

    monkeypatch.setattr(hav, "_knn_unit_vectors", spy_uv)
    monkeypatch.setattr(hav, "_knn_chord", spy_chord)

    d, idx = ms.haversine_knn(q, ref, k=10)
    assert chord_calls == [True]
    # Unit vectors must be computed ONLY for the 4 queries, never for the 10k refs.
    assert uv_shapes == [(4, 2)]

    # Plain path (k=20) reuses cached cos_lat without calling mx.cos on r.
    cos_shapes = []
    orig_cos = hav.mx.cos

    def spy_cos(x, *args, **kwargs):
        cos_shapes.append(tuple(x.shape))
        return orig_cos(x, *args, **kwargs)

    monkeypatch.setattr(hav.mx, "cos", spy_cos)
    d20, i20 = ms.haversine_knn(q, ref, k=20)
    assert (10_000,) not in cos_shapes
    assert cos_shapes[0] == (4,)
    d20_raw, i20_raw = ms.haversine_knn(q, r, k=20)
    np.testing.assert_array_equal(np.asarray(i20), np.asarray(i20_raw))
    np.testing.assert_array_equal(np.asarray(d20), np.asarray(d20_raw))


def test_haversine_reference_snapshot_cross_empty_cpu_and_errors(rng):
    r = _city_cloud(rng, 200)
    q = _city_cloud(rng, 12)
    ref = ms.HaversineReference(r)
    d_before, i_before = ref.knn(q, k=5)

    # Mutate original NumPy array: prepared reference must be isolated.
    r[:] = 0.0
    d_after, i_after = ref.knn(q, k=5)
    np.testing.assert_array_equal(np.asarray(i_before), np.asarray(i_after))
    np.testing.assert_array_equal(np.asarray(d_before), np.asarray(d_after))

    # haversine_cross accepts HaversineReference on either side.
    r2 = _city_cloud(rng, 50)
    ref2 = ms.HaversineReference(r2)
    np.testing.assert_array_equal(
        np.asarray(ms.haversine_cross(q, ref2)), np.asarray(ms.haversine_cross(q, r2))
    )
    np.testing.assert_array_equal(
        np.asarray(ms.haversine_cross(ref2, q)), np.asarray(ms.haversine_cross(r2, q))
    )

    # Empty query against non-empty reference.
    de, ie = ms.haversine_knn(np.empty((0, 2)), ref2, k=3)
    assert de.shape == (0, 3) and ie.shape == (0, 3)

    # Empty reference.
    empty_ref = ms.HaversineReference(np.empty((0, 2)))
    assert len(empty_ref) == 0 and empty_ref.shape == (0, 2)
    with pytest.raises(ValueError, match="k must be in"):
        ms.haversine_knn(q, empty_ref, k=1)

    # CPU default device uses composed path cleanly with HaversineReference.
    old_device = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        ref_cpu = ms.HaversineReference(r2)
        dc, ic = ms.haversine_knn(q, ref_cpu, k=5)
        dr, ir = ms.haversine_knn(q, r2, k=5)
        np.testing.assert_array_equal(np.asarray(ic), np.asarray(ir))
        np.testing.assert_array_equal(np.asarray(dc), np.asarray(dr))
    finally:
        mx.set_default_device(old_device)

    # Error handling.
    with pytest.raises(TypeError, match="already a HaversineReference"):
        ms.HaversineReference(ref2)
    with pytest.raises(TypeError, match="HaversineReference"):
        ms.haversine(q[0], ref2)
    with pytest.raises(TypeError, match="HaversineReference"):
        ms.haversine_knn(ref2, r2, k=1)
    with pytest.raises(ValueError, match="2-D"):
        ms.HaversineReference(np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="2-D"):
        ms.HaversineReference(np.zeros((2, 3, 2)))
