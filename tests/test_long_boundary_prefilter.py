"""Long-ring edge bands preserve the exact boundary oracle's full band."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

import mlx_geospatial._boundary as boundary
from mlx_geospatial._dispatch import metal_available
from mlx_geospatial._edge_index import get_edge_index
from mlx_geospatial.join import GridIndex


pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal GPU required")


def _ring(vertices=20_000, *, scale=1.0):
    angle = np.linspace(0.0, 2.0 * np.pi, vertices, endpoint=False)
    radius = 1.0 + 0.12 * np.sin(7 * angle) + 0.025 * np.sin(53 * angle)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]) * scale


def _candidate_rows(index, points, **kwargs):
    # These small fixtures exercise a warmed edge index. Cold small queries
    # deliberately retain the original path to avoid construction overhead.
    get_edge_index(index.polygons)
    local = index.polygons._local_from_f64(points)
    cell, _, _, _ = index._device_candidates(local)
    return boundary.boundary_candidate_rows(index, local, cell, **kwargs)


def test_cold_small_boundary_query_defers_edge_index():
    index = GridIndex([Polygon(_ring())])
    local = index.polygons._local_from_f64(np.array([[0.1, 0.2], [3.0, 4.0]]))
    cell, _, _, _ = index._device_candidates(local)
    assert boundary.boundary_candidate_rows(index, local, cell) is None
    assert index.polygons._y_edge_index is False


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_long_ring_candidates_preserve_band_and_reject_distant_rows(scale):
    rng = np.random.default_rng(916)
    coords = _ring(scale=scale)
    index = GridIndex([Polygon(coords)])
    tol = index.polygons.boundary_tolerance
    chosen = np.arange(0, len(coords), 97)
    anchors = np.concatenate([
        coords[chosen], (coords[chosen] + coords[(chosen + 1) % len(coords)]) * 0.5,
    ])
    angles = rng.uniform(0, 2 * np.pi, len(anchors))
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    points = np.concatenate([
        anchors, anchors + directions * (0.95 * tol),
        anchors + directions * (1.05 * tol),
        rng.uniform(-1.5, 1.5, (2000, 2)) * scale,
        np.array([[-3, -3], [3, 3]]) * scale,
        [[np.nan, 0], [np.inf, 0]],
    ])
    rows = _candidate_rows(index, points)
    assert rows is not None
    expected = index._near_boundary(points, tol)
    assert expected.size > 500
    assert np.isin(expected, rows).all()
    assert len(points) - 4 not in rows and len(points) - 3 not in rows
    assert len(points) - 2 in rows and len(points) - 1 in rows
    np.testing.assert_array_equal(rows[index._near_boundary(points[rows], tol)], expected)


def test_long_band_cutoffs_holes_and_multiple_polygons():
    outer = _ring()
    inner = _ring(9000) * 0.25
    polygons = [Polygon(outer, holes=[inner]), shapely.box(-0.1, -0.1, 0.2, 0.2)]
    index = GridIndex(polygons)
    edges = get_edge_index(index.polygons)
    assert edges is not None
    cuts = np.asarray(edges.cuts, dtype=np.float64)
    tol = index.polygons.boundary_tolerance
    # Include the exact shared bin cuts and their adjacent fp32 values; every
    # narrow edge band must retain matching geometry on either side.
    selected = cuts[::max(1, len(cuts) // 100)]
    y_values = np.concatenate([
        selected, np.nextafter(selected.astype(np.float32), np.float32(-np.inf)),
        np.nextafter(selected.astype(np.float32), np.float32(np.inf)),
    ])
    # Intersect horizontal lines with the boundary to obtain guaranteed
    # boundary anchors at the actual band transitions.
    lines = shapely.linestrings(np.stack([
        np.column_stack([np.full(len(y_values), -2.0), y_values]),
        np.column_stack([np.full(len(y_values), 2.0), y_values]),
    ], axis=1))
    anchors = shapely.get_coordinates(shapely.intersection(lines, shapely.boundary(polygons[0])))
    points = np.concatenate([
        anchors, anchors + [0.5 * tol, 0], anchors - [0.5 * tol, 0],
        inner[::127], [[-0.1, 0.07], [0.03, 0.2]],
    ])
    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        rows = _candidate_rows(index, points)
    finally:
        mx.set_default_device(previous)
    assert rows is not None
    expected = index._near_boundary(points, tol)
    assert np.isin(expected, rows).all()
    np.testing.assert_array_equal(rows[index._near_boundary(points[rows], tol)], expected)


def test_long_boundary_batches_preserve_global_row_numbers(monkeypatch):
    coords = _ring()
    index = GridIndex([Polygon(coords)])
    points = np.concatenate([coords[::1000], [[3, 3], [0.123, 0.234]]])
    expected = _candidate_rows(index, points)
    assert expected is not None
    original = boundary._get_long_candidate_kernel()
    sizes = []

    def kernel(**kwargs):
        sizes.append(kwargs["grid"][0])
        return original(**kwargs)

    monkeypatch.setattr(boundary, "_get_long_candidate_kernel", lambda: kernel)
    actual = _candidate_rows(index, points, max_pairs_per_batch=3)
    np.testing.assert_array_equal(actual, expected)
    assert len(sizes) > 1 and max(sizes) <= 3


def test_long_boundary_retains_large_origin_oracle_fallback():
    offset = 1e20
    step = np.spacing(offset)
    triangle = Polygon([
        (offset, offset), (offset + 2 * step, offset + step),
        (offset, offset + 2 * step),
    ])
    # Repeated vertices make a long ring without changing the cancellation
    # fixture's geometric edges or the original-coordinate host projection.
    coords = np.repeat(np.asarray(triangle.exterior.coords)[:-1], 3000, axis=0)
    long_triangle = Polygon(coords)
    index = GridIndex([MultiPolygon([long_triangle, long_triangle])], cell_size=20000)
    points = np.array([[offset + step, offset + step]])
    np.testing.assert_array_equal(index._near_boundary(points, index.polygons.boundary_tolerance), [0])
    assert _candidate_rows(index, points) is None


def test_noisy_150k_coastline_boundary_filter():
    rng = np.random.default_rng(44)
    angle = np.linspace(0, 2 * np.pi, 150_000, endpoint=False)
    radius = 1.0 + 0.12 * np.sin(7 * angle) + 0.06 * np.sin(41 * angle + 1.0)
    radius += 0.03 * np.sin(211 * angle) + 0.01 * rng.normal(size=len(angle))
    coords = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    index = GridIndex([Polygon(coords)])
    tol = index.polygons.boundary_tolerance
    anchors = coords[::997]
    points = np.concatenate([
        anchors, anchors + [0.5 * tol, 0],
        rng.uniform(-2, 2, (1500, 2)), [[-4, -4]],
    ])
    rows = _candidate_rows(index, points)
    assert rows is not None
    expected = index._near_boundary(points, tol)
    assert np.isin(expected, rows).all()
    assert len(rows) < len(points) // 3
    np.testing.assert_array_equal(rows[index._near_boundary(points[rows], tol)], expected)
