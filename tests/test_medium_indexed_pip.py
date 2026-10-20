"""The 2048-vertex workload reuses edge bands without changing predicates."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

import mlx_geospatial as ms
import mlx_geospatial._boundary as boundary
import mlx_geospatial._edge_index as edge_index
import mlx_geospatial._pip as pip
from mlx_geospatial._dispatch import metal_available
from test_fp_contract import _numpy_matrix
from test_indexed_pip import _check_three_routes, _indexed_spy, _one_cell, _ring


pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal GPU required")


@pytest.fixture(scope="module")
def medium_layer():
    centers = [(i - 1.5, j - 1.5) for j in range(4) for i in range(4)]
    polygons = []
    for i, center in enumerate(centers):
        shell = _ring(center, radius=0.8, vertices=2047)
        holes = [_ring(center, radius=0.19, vertices=41)] if i % 5 == 0 else []
        polygons.append(Polygon(shell, holes=holes))
    polygons[3] = MultiPolygon([polygons[3], box(3.0, -0.2, 3.4, 0.2)])
    polygons.append(Polygon())
    assert all(p.is_valid for p in polygons)
    polyset = ms.PolygonSet(polygons, origin=(0.0, 0.0))
    # 2048 stored coordinates include the repeated closing coordinate.
    assert polyset.max_edges_per_ring == 2047
    accepted = edge_index.get_edge_index(polyset)
    assert accepted is not None
    cuts = np.asarray(accepted.cuts)
    selected = cuts[::max(1, len(cuts) // 24)]
    heights = np.concatenate([
        np.nextafter(selected, np.float32(-np.inf)), selected,
        np.nextafter(selected, np.float32(np.inf)),
    ])
    rng = np.random.default_rng(2048)
    points = np.concatenate([
        rng.uniform(-2.4, 2.4, (96, 2)),
        np.column_stack([np.full(len(heights), 0.1), heights]),
        np.asarray(centers),
        [[0.0, 0.0], [3.2, 0.0], [9.0, 9.0], [np.nan, 0.0], [np.inf, 0.0]],
    ]).astype(np.float32)
    expected = _numpy_matrix(polyset, points)
    assert (expected.sum(axis=1) >= 2).any(), "overlap coverage is required"
    assert expected[:, :-1].any(axis=0).all()
    assert not expected[:, -1].any() and not expected[-3:].any()
    return polygons, polyset, points, expected


@pytest.mark.parametrize("edges,eligible", [(2046, False), (2047, True), (2048, True)])
def test_medium_index_threshold_counts_edges_not_closing_vertex(edges, eligible):
    polyset = ms.PolygonSet([Polygon(_ring(vertices=edges))])
    assert polyset.max_edges_per_ring == edges
    assert (edge_index.get_edge_index(polyset) is not None) == eligible


def test_medium_many_polygon_routes_match_numpy_at_band_cuts(medium_layer, monkeypatch):
    _, polyset, points, expected = medium_layer
    _check_three_routes(polyset, points, expected, monkeypatch)


def test_medium_all_matches_split_candidate_ranges(medium_layer, monkeypatch):
    _, polyset, points, expected = medium_layer
    selected = np.arange(0, 45, 2, dtype=np.int32)
    m = polyset.n_polygons
    cell, offsets, polygons = _one_cell(len(selected), m)
    cumulative = mx.array(np.arange(len(selected) + 1, dtype=np.int64) * m)
    calls = _indexed_spy(monkeypatch)
    begin, end, cap = 5, len(selected) * m - 3, 29
    outputs = []
    for start in range(begin, end, cap):
        result = pip.pip_all_matches(
            mx.array(points), polyset, mx.array(selected), cell, cumulative, offsets, polygons,
            pair_begin=start, pair_end=min(start + cap, end),
            max_point_pairs=min(m, end - start),
        )
        outputs.append(tuple(np.asarray(part) for part in result))
    conceptual = np.arange(begin, end)
    expected_point, expected_polygon = selected[conceptual // m], conceptual % m
    keep = expected[expected_point, expected_polygon]
    assert keep.any()
    np.testing.assert_array_equal(np.concatenate([p for p, _ in outputs]), expected_point[keep])
    np.testing.assert_array_equal(np.concatenate([g for _, g in outputs]), expected_polygon[keep])
    assert calls == ["all"] * len(outputs)


def _boundary_rows(index, points, **kwargs):
    local = index.polygons._local_from_f64(points)
    cell, _, _, _ = index._device_candidates(local)
    return boundary.boundary_candidate_rows(index, local, cell, **kwargs)


def _spy_boundary_kernel(monkeypatch, name):
    original = getattr(boundary, name)()
    sizes = []

    def launch(**kwargs):
        sizes.append(kwargs["grid"][0])
        return original(**kwargs)

    monkeypatch.setattr(boundary, name, lambda: launch)
    return sizes


@pytest.mark.parametrize("predicate", ["within", "intersects"])
def test_medium_exact_boundary_reuses_index_and_matches_cpu_and_geos(
    medium_layer, monkeypatch, predicate,
):
    polygons, polyset, _, _ = medium_layer
    index = ms.GridIndex(polyset)
    edges = polyset._edges_f64()[::211]
    anchors = np.concatenate([edges[:, :2], (edges[:, :2] + edges[:, 2:]) * 0.5])
    tol = polyset.boundary_tolerance
    points = np.concatenate([
        anchors, anchors + [0.5 * tol, 0], anchors - [0.5 * tol, 0],
        [[0.0, 0.0], [3.2, 0.0], [9.0, 9.0]],
    ])
    sizes = _spy_boundary_kernel(monkeypatch, "_get_long_candidate_kernel")

    def no_linear():
        raise AssertionError("accepted medium edge index was not reused for the boundary filter")

    monkeypatch.setattr(boundary, "_get_candidate_kernel", no_linear)
    # Verify conservative row selection independently of final GEOS refinement.
    cap = max(1, index._max_cell_count) * 37
    rows = _boundary_rows(index, points, max_pairs_per_batch=cap)
    assert rows is not None and sizes and max(sizes) <= 37
    oracle = index._near_boundary(points, tol)
    assert len(oracle) >= len(anchors)
    np.testing.assert_array_equal(rows[index._near_boundary(points[rows], tol)], oracle)

    reference = shapely.STRtree(polygons).query(shapely.points(points), predicate=predicate)
    order = np.lexsort((reference[1], reference[0]))
    reference = reference[:, order]
    cpu = index.query(points, backend="cpu", exact_boundary=True, all_matches=True,
                      predicate=predicate)
    gpu = index.query(points, backend="gpu", exact_boundary=True, all_matches=True,
                      predicate=predicate, max_pairs_per_batch=cap)
    np.testing.assert_array_equal(cpu[0], reference[0])
    np.testing.assert_array_equal(cpu[1], reference[1])
    np.testing.assert_array_equal(gpu[0], cpu[0])
    np.testing.assert_array_equal(gpu[1], cpu[1])
    np.testing.assert_array_equal(
        index.query(points, backend="gpu", exact_boundary=True, predicate=predicate),
        index.query(points, backend="cpu", exact_boundary=True, predicate=predicate),
    )


def test_medium_small_cold_query_keeps_index_unbuilt(monkeypatch):
    polyset = ms.PolygonSet([Polygon(_ring(vertices=2047))], origin=(0, 0))
    points = np.array([[0.2, 0.1], [2.0, 0.0]], dtype=np.float32)
    calls = _indexed_spy(monkeypatch)
    cell, offsets, polygons = _one_cell(len(points), 1)
    np.testing.assert_array_equal(
        pip.pip_first_match(mx.array(points), polyset, cell, offsets, polygons), [0, -1],
    )
    np.testing.assert_array_equal(
        pip.pip_pairs(mx.array(points), polyset, mx.array([0, 1]), mx.array([0, 0])), [1, 0],
    )
    p, g = pip.pip_all_matches(
        mx.array(points), polyset, mx.array([0, 1]), cell,
        mx.array([0, 1, 2], dtype=mx.int64), offsets, polygons,
        pair_begin=0, pair_end=2, max_point_pairs=1,
    )
    np.testing.assert_array_equal(p, [0])
    np.testing.assert_array_equal(g, [0])
    assert not calls and polyset._y_edge_index is False
    sizes = _spy_boundary_kernel(monkeypatch, "_get_candidate_kernel")
    assert _boundary_rows(ms.GridIndex(polyset), points.astype(np.float64)) is not None
    assert sizes and polyset._y_edge_index is False


def test_medium_rejected_index_keeps_predicate_and_linear_boundary_fallback(monkeypatch):
    coords = np.column_stack([
        np.linspace(-1.0, 1.0, 2047), np.resize([-1.0, 1.0], 2047),
    ])
    polyset = ms.PolygonSet([Polygon(coords)], origin=(0, 0))
    assert polyset.max_edges_per_ring == 2047
    assert edge_index.get_edge_index(polyset) is None
    assert polyset._y_edge_index is None
    index = ms.GridIndex(polyset)
    points = np.concatenate([coords[::511], [[0.123, 0.234], [4.0, 4.0]]])
    calls = _indexed_spy(monkeypatch)
    for all_matches in (False, True):
        cpu = index.query(points, backend="cpu", all_matches=all_matches)
        gpu = index.query(points, backend="gpu", all_matches=all_matches)
        if all_matches:
            np.testing.assert_array_equal(gpu[0], cpu[0])
            np.testing.assert_array_equal(gpu[1], cpu[1])
        else:
            np.testing.assert_array_equal(gpu, cpu)
    assert not calls
    sizes = _spy_boundary_kernel(monkeypatch, "_get_candidate_kernel")
    rows = _boundary_rows(index, points)
    assert rows is not None and sizes
    tol = polyset.boundary_tolerance
    np.testing.assert_array_equal(
        rows[index._near_boundary(points[rows], tol)], index._near_boundary(points, tol),
    )
    assert polyset._y_edge_index is None


def test_48_edge_layers_keep_existing_predicate_and_boundary_routes(monkeypatch):
    coords = _ring(vertices=48)
    polyset = ms.PolygonSet([Polygon(coords)], origin=(0, 0))
    assert polyset.max_edges_per_ring == 48

    def no_build(*args):
        raise AssertionError("48-edge geometry attempted to build a medium-ring index")

    monkeypatch.setattr(edge_index, "_build_edge_index", no_build)
    assert edge_index.get_edge_index(polyset) is None
    calls = _indexed_spy(monkeypatch)
    sizes = _spy_boundary_kernel(monkeypatch, "_get_candidate_kernel")
    index = ms.GridIndex(polyset)
    points = np.concatenate([coords[::5], [[0.0, 0.0], [2.0, 0.0]]])
    for all_matches in (False, True):
        cpu = index.query(points, backend="cpu", exact_boundary=True, all_matches=all_matches)
        gpu = index.query(points, backend="gpu", exact_boundary=True, all_matches=all_matches)
        if all_matches:
            np.testing.assert_array_equal(gpu[0], cpu[0])
            np.testing.assert_array_equal(gpu[1], cpu[1])
        else:
            np.testing.assert_array_equal(gpu, cpu)
    assert not calls and sizes and polyset._y_edge_index is False
