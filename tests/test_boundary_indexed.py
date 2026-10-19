"""Sparse indexed verification agrees exactly with the original host oracle."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

import mlx_geospatial._boundary_indexed as indexed
from mlx_geospatial._edge_index import get_edge_index
from mlx_geospatial.join import GridIndex


def _ring(n=8192, scale=1.0, origin=(0.0, 0.0)):
    angle = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    radius = 1.0 + 0.1 * np.sin(7 * angle)
    return (np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]) * scale
            + np.asarray(origin))


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_sparse_index_matches_original_oracle_without_edge_tree(scale):
    coords = _ring(scale=scale)
    polygons = [Polygon(coords), shapely.Polygon(), None]
    index = GridIndex(polygons)
    assert get_edge_index(index.polygons) is not None
    tol = index.polygons.boundary_tolerance
    anchors = coords[::53]
    points = np.concatenate([
        anchors, anchors + [0.95 * tol, 0], anchors + [1.05 * tol, 0],
        np.random.default_rng(91).uniform(-1.2, 1.2, (300, 2)) * scale,
        [[np.nan, 0.0], [np.inf, 0.0]],
    ])
    actual = indexed.indexed_near_boundary(index, points, tol)
    assert actual is not None
    assert index.polygons._edge_box_strtree is None
    expected = index._near_boundary(points, tol)
    assert len(expected) > 200
    np.testing.assert_array_equal(actual, expected)


def test_all_ring_bounds_keep_invalid_hole_outside_shell_and_empty_parts():
    shell = _ring()
    hole = _ring(2048, scale=0.2, origin=(5.0, 0.0))
    invalid = Polygon(shell, holes=[hole])
    assert not invalid.is_valid
    polygon = MultiPolygon([invalid, Polygon(_ring(2048, 0.1, (-4.0, 0.0)))])
    index = GridIndex([shapely.Polygon(), polygon, None])
    assert get_edge_index(index.polygons) is not None
    tol = index.polygons.boundary_tolerance
    points = np.concatenate([
        hole[::31], hole[::31] + [0.95 * tol, 0],
        shell[::137], [[5.5, 0.1], [4.9, 0.0], [-4.0, 0.1]],
    ])
    actual = indexed.indexed_near_boundary(index, points, tol)
    assert actual is not None
    assert index.polygons._edge_box_strtree is None
    np.testing.assert_array_equal(actual, index._near_boundary(points, tol))


def test_2048_edge_layer_reuses_cached_index_and_preserves_zero_length_edges():
    coords = _ring(2048)
    # Repeated vertices remain edges in the original ragged oracle.
    coords = np.insert(coords, [10, 35, 120], coords[[10, 35, 120]], axis=0)
    polygons = [Polygon(coords + [3.0 * i, 0.0]) for i in range(4)]
    index = GridIndex(polygons)
    assert get_edge_index(index.polygons) is not None
    tol = index.polygons.boundary_tolerance
    points = np.concatenate([coords[::59] + [3.0 * i, 0] for i in range(4)])
    points = np.concatenate([points, points + [0.95 * tol, 0], points + [1.05 * tol, 0]])
    actual = indexed.indexed_near_boundary(index, points, tol)
    assert actual is not None
    assert index.polygons._edge_box_strtree is None
    np.testing.assert_array_equal(actual, index._near_boundary(points, tol))


def test_pair_budget_falls_back_before_float64_edge_allocation(monkeypatch):
    index = GridIndex([Polygon(_ring())])
    assert get_edge_index(index.polygons) is not None
    monkeypatch.setattr(indexed, "_MAX_EDGE_PAIRS", 1)
    assert indexed.indexed_near_boundary(index, np.array([[0.3, 0.4]]),
                                         index.polygons.boundary_tolerance) is None
    assert index.polygons._edges_f64_cache is None
    assert index.polygons._edge_box_strtree is None


def test_uncached_index_and_nonstandard_tolerance_keep_original_route():
    index = GridIndex([Polygon(_ring())])
    points = np.array([[0.3, 0.4]])
    tol = index.polygons.boundary_tolerance
    assert indexed.indexed_near_boundary(index, points, tol) is None
    assert index.polygons._y_edge_index is False
    assert get_edge_index(index.polygons) is not None
    assert indexed.indexed_near_boundary(index, points, 2 * tol) is None
    empty = indexed.indexed_near_boundary(index, np.array([[np.nan, 0.0]]), tol)
    np.testing.assert_array_equal(empty, np.empty(0, dtype=np.int64))


def test_large_original_coordinate_rounding_keeps_host_oracle():
    offset = 1e20
    coords = _ring(scale=100 * np.spacing(offset), origin=(offset, offset))
    index = GridIndex([Polygon(coords)])
    assert get_edge_index(index.polygons) is not None
    assert np.spacing(offset) > index.polygons.boundary_tolerance
    assert indexed.indexed_near_boundary(
        index, coords[::1000], index.polygons.boundary_tolerance
    ) is None


def test_chunked_projection_matches_same_oracle(monkeypatch):
    coords = _ring()
    index = GridIndex([Polygon(coords)])
    assert get_edge_index(index.polygons) is not None
    tol = index.polygons.boundary_tolerance
    points = np.concatenate([coords[::71], coords[::71] + [0.9 * tol, 0]])
    monkeypatch.setattr(indexed, "_PAIR_CHUNK", 19)
    actual = indexed.indexed_near_boundary(index, points, tol)
    assert actual is not None
    np.testing.assert_array_equal(actual, index._near_boundary(points, tol))
