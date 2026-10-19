"""The Metal broad phase must never discard an exact boundary-band row."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

from mlx_geospatial._boundary import boundary_candidate_rows
from mlx_geospatial._dispatch import metal_available
from mlx_geospatial.join import GridIndex

from conftest import make_convex_polygons, make_donut


pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal GPU required")


def _candidate_rows(index, points, **kwargs):
    local = index.polygons._local_from_f64(points)
    cell, _, _, _ = index._device_candidates(local)
    return boundary_candidate_rows(index, local, cell, **kwargs)


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_filter_preserves_band_at_edges_vertices_and_grid_borders(scale):
    rng = np.random.default_rng(183)
    polygons = make_convex_polygons(18, rng, n_vertices=(8, 16), radius_frac=0.12)
    polygons = shapely.transform(polygons, lambda p: p * scale)
    index = GridIndex(polygons)
    tol = index.polygons.boundary_tolerance
    edges = index.polygons._edges_f64()
    anchors = np.concatenate([edges[:, :2], (edges[:, :2] + edges[:, 2:]) * 0.5])
    angles = rng.uniform(0.0, 2.0 * np.pi, len(anchors))
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    points = np.concatenate([
        anchors,
        anchors + directions * tol * 0.95,
        anchors + directions * tol * 1.05,
        rng.uniform(-1.2, 1.2, (3000, 2)) * scale,
    ])
    rows = _candidate_rows(index, points)
    assert rows is not None
    expected = index._near_boundary(points, tol)
    assert expected.size > 100
    assert np.isin(expected, rows).all()
    # The unchanged float64 oracle on the retained subset is exactly equal,
    # including points just inside and outside the Euclidean band.
    actual = rows[index._near_boundary(points[rows], tol)]
    np.testing.assert_array_equal(actual, expected)


def test_filter_retains_holes_overlaps_outer_edges_and_nonfinite_rows():
    polygons = [
        shapely.box(-2, -2, 2, 2),
        make_donut(0.0, 0.0, 1.5, 0.4, k=40),
        Polygon([(2, -1), (3, -1), (3, -1), (3, 1), (2, 1)]),
    ]
    index = GridIndex(polygons)
    tol = index.polygons.boundary_tolerance
    points = np.array([
        [0.4, 0.0], [0.4 + 0.95 * tol, 0.0],
        [3 + 0.95 * tol, 0.2], [-2 - 0.95 * tol, 0.2],
        [3.0, -1.0], [0.371, 0.317],
        [np.nan, 0.0], [np.inf, 0.0], [-np.inf, np.nan],
    ])
    rows = _candidate_rows(index, points)
    assert rows is not None
    expected = index._near_boundary(points, tol)
    assert np.isin(expected, rows).all()
    assert np.isin([6, 7, 8], rows).all()
    assert 5 not in rows
    np.testing.assert_array_equal(rows[index._near_boundary(points[rows], tol)], expected)


def test_filter_recentered_large_origin():
    polygons = [shapely.box(1e9, -1e9, 1e9 + 5, -1e9 + 3)]
    index = GridIndex(polygons)
    tol = index.polygons.boundary_tolerance
    points = np.array([
        [1e9 + 0.5 * tol, -1e9 + 0.321],
        [1e9 - 0.5 * tol, -1e9 + 0.321],
        [1e9 + 0.321, -1e9 + 0.321],
    ])
    rows = _candidate_rows(index, points)
    assert rows is not None
    expected = index._near_boundary(points, tol)
    assert np.isin(expected, rows).all()
    assert 2 not in rows


def test_long_rings_and_subnormal_scales_preserve_host_fallback():
    angles = np.linspace(0, 2 * np.pi, 5000, endpoint=False)
    long_index = GridIndex([Polygon(np.column_stack([np.cos(angles), np.sin(angles)]))])
    assert _candidate_rows(long_index, np.array([[0.2, 0.3]])) is None
    tiny_index = GridIndex([shapely.box(0, 0, 1e-40, 1e-40)])
    assert _candidate_rows(tiny_index, np.array([[0.2e-40, 0.3e-40]])) is None


def test_detailed_polygon_and_cpu_default_device():
    angles = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    coords = np.column_stack([np.cos(angles), np.sin(angles)])
    index = GridIndex([Polygon(coords)])
    tol = index.polygons.boundary_tolerance
    anchors = coords[::31]
    points = np.concatenate([
        anchors, anchors * (1.0 + 0.95 * tol), anchors * (1.0 - 0.95 * tol),
        [[0.123, 0.234]],
    ])
    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        rows = _candidate_rows(index, points)
    finally:
        mx.set_default_device(previous)
    assert rows is not None
    assert len(points) - 1 not in rows
    expected = index._near_boundary(points, tol)
    np.testing.assert_array_equal(rows[index._near_boundary(points[rows], tol)], expected)


def test_empty_points_do_not_launch_kernel():
    index = GridIndex([shapely.box(0, 0, 1, 1)])
    rows = boundary_candidate_rows(
        index, mx.zeros((0, 2), dtype=mx.float32), mx.zeros((0,), dtype=mx.int32)
    )
    assert rows.dtype == np.int64
    assert rows.size == 0


def test_pair_cap_fallback_and_batched_rows_match_single_launch(monkeypatch):
    import mlx_geospatial._boundary as boundary

    index = GridIndex([shapely.box(0, 0, 1, 1), shapely.box(0.2, 0.2, 0.8, 0.8)])
    points = np.array([
        [0.0, 0.3], [0.37, 0.37], [0.2, 0.31], [0.37, 0.37],
        [0.8, 0.31], [1.0, 0.31], [0.37, 0.37], [np.nan, 0.0],
        [0.0, 0.3], [0.37, 0.37], [0.2, 0.31],
    ])
    expected = _candidate_rows(index, points)
    assert expected is not None
    assert _candidate_rows(index, points, max_pairs_per_batch=1) is None
    original = boundary._get_candidate_kernel()
    batch_sizes = []

    def kernel(**kwargs):
        batch_sizes.append(kwargs["grid"][0])
        return original(**kwargs)

    monkeypatch.setattr(boundary, "_get_candidate_kernel", lambda: kernel)
    cap = index._max_cell_count * 3
    actual = _candidate_rows(index, points, max_pairs_per_batch=cap)
    np.testing.assert_array_equal(actual, expected)
    assert batch_sizes == [3, 3, 3, 2]
    assert all(size * index._max_cell_count <= cap for size in batch_sizes)


def test_many_short_rings_retain_host_route():
    # Each individual ring is short, but the aggregate work of one candidate
    # list must not serialize an unbounded number of edge tests in one lane.
    index = GridIndex([shapely.box(0, 0, 1, 1)] * 8193)
    assert _candidate_rows(index, np.array([[0.37, 0.31]])) is None


def test_line_strip_discards_diagonal_envelope_false_positives():
    index = GridIndex([Polygon([(0, 0), (1, 1), (2, 0)])])
    tol = index.polygons.boundary_tolerance
    points = np.array([
        [0.37, 0.37], [0.37, 0.37 + 0.95 * tol], [0.37, 0.37 - 0.95 * tol],
        [0.37, 0.41], [0.37, 0.31],
    ])
    rows = _candidate_rows(index, points)
    np.testing.assert_array_equal(rows, [0, 1, 2])


def test_tiny_edges_keep_conservative_envelopes_and_huge_scales_fall_back():
    # Very short diagonal edges cannot establish a reliable fp32 direction.
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (1e-6, 1e-6), (0, 0)])
    index = GridIndex([polygon])
    tol = index.polygons.boundary_tolerance
    points = np.array([[0.5e-6, 0.5e-6], [0.5e-6, 0.5e-6 + 0.95 * tol]])
    rows = _candidate_rows(index, points)
    expected = index._near_boundary(points, tol)
    assert rows is not None
    assert np.isin(expected, rows).all()
    huge = GridIndex([shapely.box(-1e38, -1e38, 1e38, 1e38)])
    assert _candidate_rows(huge, np.array([[0.1e38, 0.2e38]])) is None


def test_large_original_offset_preserves_host_projection_rounding():
    offset = 1e20
    step = np.spacing(offset)
    triangle = Polygon([
        (offset, offset), (offset + 2 * step, offset + step),
        (offset, offset + 2 * step),
    ])
    # Invalid duplicate parts intentionally distinguish the fp32 even-odd
    # answer from the GEOS answer used by exact boundary refinement.
    index = GridIndex([MultiPolygon([triangle, triangle])], cell_size=20000)
    points = np.array([[offset + step, offset + step]])
    tol = index.polygons.boundary_tolerance
    assert tol < np.spacing(offset)
    # Original-coordinate projection addition rounds onto this point, so
    # the existing host oracle includes it even though the true line is far.
    np.testing.assert_array_equal(index._near_boundary(points, tol), [0])
    assert _candidate_rows(index, points) is None
    np.testing.assert_array_equal(
        index.query(points, backend="gpu", exact_boundary=True),
        index.query(points, backend="cpu", exact_boundary=True),
    )
    gpu_pairs = index.query(points, backend="gpu", exact_boundary=True, all_matches=True)
    cpu_pairs = index.query(points, backend="cpu", exact_boundary=True, all_matches=True)
    np.testing.assert_array_equal(gpu_pairs[0], cpu_pairs[0])
    np.testing.assert_array_equal(gpu_pairs[1], cpu_pairs[1])
