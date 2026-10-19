"""Indexed long-ring predicates retain the NumPy fp32 crossing contract."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

import mlx_geospatial as ms
import mlx_geospatial._edge_index as edge_index
import mlx_geospatial._pip as pip
from mlx_geospatial._dispatch import metal_available
from test_fp_contract import _chord_ring, _first_match, _fma_probes, _numpy_matrix


pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal GPU required")


def _ring(center=(0.0, 0.0), radius=1.0, vertices=9000):
    angle = np.linspace(0.0, 2.0 * np.pi, vertices, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle),
    ])


def _indexed_spy(monkeypatch):
    calls = []
    original = pip.indexed_pip

    def wrapped(kind, *args, **kwargs):
        result = original(kind, *args, **kwargs)
        if result is not None:
            calls.append(kind)
        return result

    monkeypatch.setattr(pip, "indexed_pip", wrapped)
    return calls


def _one_cell(n, m):
    return (
        mx.zeros((n,), dtype=mx.int32),
        mx.array([0, m], dtype=mx.int64),
        mx.arange(m, dtype=mx.int32),
    )


def _check_three_routes(polyset, points, expected, monkeypatch):
    """Require actual indexed results, not merely a successful fallback."""
    assert edge_index.get_edge_index(polyset) is not None
    calls = _indexed_spy(monkeypatch)
    n, m = expected.shape
    local = mx.array(points)
    pair_point = mx.array(np.repeat(np.arange(n, dtype=np.int32), m))
    pair_poly = mx.array(np.tile(np.arange(m, dtype=np.int32), n))
    actual = np.asarray(pip.pip_pairs(local, polyset, pair_point, pair_poly))
    np.testing.assert_array_equal(actual.reshape(n, m), expected)

    cell, offsets, polygons = _one_cell(n, m)
    # Out-of-grid points have to be initialized even when inside a polygon.
    first_cell = np.zeros(n, dtype=np.int32)
    first_cell[::11] = -1
    first = _first_match(expected)
    first[::11] = -1
    np.testing.assert_array_equal(
        pip.pip_first_match(local, polyset, mx.array(first_cell), offsets, polygons), first,
    )
    point, polygon = pip.pip_all_matches(
        local, polyset, mx.arange(n, dtype=mx.int32), cell,
        mx.array(np.arange(n + 1, dtype=np.int64) * m), offsets, polygons,
        pair_begin=0, pair_end=n * m, max_point_pairs=m,
    )
    expected_point, expected_polygon = np.nonzero(expected)
    np.testing.assert_array_equal(point, expected_point)
    np.testing.assert_array_equal(polygon, expected_polygon)
    assert calls == ["pairs", "first", "all"]


@pytest.fixture(scope="module")
def mixed_layer():
    shell = _ring()
    # Traversing the same closed ring twice is invalid geometry but has a
    # well-defined even-odd predicate: every crossing cancels exactly.
    duplicate = _ring((0.0, 2.5), vertices=4500)
    doubled = Polygon(np.concatenate([duplicate, duplicate]))
    assert not doubled.is_valid
    polygons = [
        Polygon(shell, holes=[_ring(radius=0.3, vertices=37)]),
        MultiPolygon([Polygon(_ring((2.5, 0.0))), box(4.0, -0.2, 4.4, 0.2)]),
        doubled,
        Polygon(),
    ]
    polyset = ms.PolygonSet(polygons, origin=(0.0, 0.0))
    index = edge_index.get_edge_index(polyset)
    assert index is not None
    cuts = np.asarray(index.cuts)
    selected = cuts[::max(1, len(cuts) // 24)]
    heights = np.concatenate([
        np.nextafter(selected, np.float32(-np.inf)), selected,
        np.nextafter(selected, np.float32(np.inf)),
    ])
    rng = np.random.default_rng(802)
    points = np.concatenate([
        rng.uniform([-1.3, -1.3], [4.6, 3.8], (80, 2)),
        np.column_stack([np.full(len(heights), 0.6), heights]),
        np.column_stack([np.full(len(heights), 2.6), heights]),
        shell[::1000],
        [[0, 0], [0.5, 0], [2.5, 0], [4.2, 0], [0, 2.5],
         [np.nan, 0], [0, np.nan], [np.inf, 0], [-np.inf, 0], [0, np.inf]],
    ]).astype(np.float32)
    expected = _numpy_matrix(polyset, points)
    assert expected[:, :2].any(axis=0).all()
    assert not expected[:, 2:].any()
    assert not expected[-5:].any()
    return polyset, points, expected


@pytest.mark.parametrize("cpu_default", [False, True])
def test_indexed_routes_match_fp32_oracle_at_cuts_and_invalid_inputs(
    mixed_layer, monkeypatch, cpu_default,
):
    polyset, points, expected = mixed_layer
    old_device = mx.default_device()
    try:
        if cpu_default:
            mx.set_default_device(mx.cpu)
        _check_three_routes(polyset, points, expected, monkeypatch)
    finally:
        mx.set_default_device(old_device)


def test_indexed_routes_preserve_unfused_intersections(monkeypatch):
    polyset = ms.PolygonSet([_chord_ring((0.0, 0.0), 1.0, n_dense=9000)], origin=(0, 0))
    edges = polyset._edges_np()
    mask = np.abs(edges[:, 2] - edges[:, 0]) > 0.2
    fma, _, exact, _ = _fma_probes(polyset, mask, np.random.default_rng(713), samples=100)
    assert len(fma) >= 25 and len(exact) >= 100
    points = np.concatenate([fma, exact[:120]])
    expected = _numpy_matrix(polyset, points)
    assert expected.any() and not expected.all()
    _check_three_routes(polyset, points, expected, monkeypatch)


def test_indexed_all_match_caps_split_points_and_preserve_global_ids(mixed_layer, monkeypatch):
    polyset, points, expected = mixed_layer
    selected = np.arange(0, min(len(points), 150), 3, dtype=np.int32)
    m = polyset.n_polygons
    local = mx.array(points)
    cell, offsets, polygons = _one_cell(len(selected), m)
    cumulative = mx.array(np.arange(len(selected) + 1, dtype=np.int64) * m)
    calls = _indexed_spy(monkeypatch)
    begin, end, cap = 3, len(selected) * m - 1, 17
    actual_point, actual_polygon = [], []
    for start in range(begin, end, cap):
        point, polygon = pip.pip_all_matches(
            local, polyset, mx.array(selected), cell, cumulative, offsets, polygons,
            pair_begin=start, pair_end=min(start + cap, end),
            max_point_pairs=min(m, end - start),
        )
        actual_point.append(np.asarray(point))
        actual_polygon.append(np.asarray(polygon))
    conceptual = np.arange(begin, end)
    expected_point = selected[conceptual // m]
    expected_polygon = conceptual % m
    keep = expected[expected_point, expected_polygon]
    assert keep.any()
    np.testing.assert_array_equal(np.concatenate(actual_point), expected_point[keep])
    np.testing.assert_array_equal(np.concatenate(actual_polygon), expected_polygon[keep])
    assert calls == ["all"] * len(actual_point)


def test_cold_tiny_query_defers_index_then_reuses_warmed_index(monkeypatch):
    polyset = ms.PolygonSet([Polygon(_ring())], origin=(0, 0))
    points = mx.array([[0.5, 0.0], [2.0, 0.0]], dtype=mx.float32)
    pair_point, pair_poly = mx.array([0, 1]), mx.array([0, 0])
    calls = _indexed_spy(monkeypatch)
    np.testing.assert_array_equal(pip.pip_pairs(points, polyset, pair_point, pair_poly), [1, 0])
    assert not calls and polyset._y_edge_index is False
    index = edge_index.get_edge_index(polyset)
    assert index is not None
    assert edge_index.get_edge_index(polyset) is index
    np.testing.assert_array_equal(pip.pip_pairs(points, polyset, pair_point, pair_poly), [1, 0])
    assert calls == ["pairs"]


@pytest.mark.parametrize("reason", ["zigzag", "reference_cap", "many_polygons"])
def test_rejected_edge_index_is_cached_before_reference_expansion(monkeypatch, reason):
    if reason == "zigzag":
        ring = np.column_stack([
            np.linspace(-1.0, 1.0, 9000), np.tile([-1.0, 1.0], 4500),
        ])
    else:
        ring = _ring()
    polygons = [Polygon(ring)]
    if reason == "many_polygons":
        polygons.extend([box(2, 2, 3, 3)] * 1024)
    polyset = ms.PolygonSet(polygons, origin=(0, 0))
    if reason == "reference_cap":
        monkeypatch.setattr(edge_index, "_MAX_REFERENCES", 100)
    builds = []
    original = edge_index._build_edge_index

    def build(layer):
        builds.append(layer)
        return original(layer)

    def no_expansion(*args, **kwargs):
        raise AssertionError("reference arrays expanded before checking the budget")

    monkeypatch.setattr(edge_index, "_build_edge_index", build)
    with monkeypatch.context() as context:
        context.setattr(edge_index.np, "repeat", no_expansion)
        assert edge_index.get_edge_index(polyset) is None
        assert edge_index.get_edge_index(polyset) is None
    assert builds == [polyset] and polyset._y_edge_index is None
    # A rejected cached index must continue through the ordinary predicate,
    # including invalid zigzags whose answer is fp32 parity, not GEOS validity.
    points = np.array([[0.25, 0.125], [-0.5, -0.25], [4.0, 4.0]], dtype=np.float32)
    expected = _numpy_matrix(polyset, points)[:, 0]
    calls = _indexed_spy(monkeypatch)
    actual = pip.pip_pairs(
        mx.array(points), polyset, mx.arange(len(points), dtype=mx.int32),
        mx.zeros((len(points),), dtype=mx.int32),
    )
    np.testing.assert_array_equal(actual, expected)
    assert not calls and builds == [polyset]


def test_short_rings_never_build_edge_index(monkeypatch):
    polyset = ms.PolygonSet([box(0, 0, 1, 1)])

    def unexpected(*args):
        raise AssertionError("short-ring query attempted to build a long-ring index")

    monkeypatch.setattr(edge_index, "_build_edge_index", unexpected)
    assert edge_index.get_edge_index(polyset) is None
    assert polyset._y_edge_index is False
