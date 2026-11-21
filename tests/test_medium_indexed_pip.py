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


def test_medium_first_match_extends_cooperative_traversal_past_4096(monkeypatch):
    """4,097..65,535 points stay on cooperative SIMD32 without cold index build."""
    import mlx_geospatial._pip_indexed as pip_indexed

    polyset = ms.PolygonSet([Polygon(_ring(vertices=2047))], origin=(0, 0))
    index = ms.GridIndex(polyset)
    rng = np.random.default_rng(4097)
    pts = rng.uniform(-1.2, 1.2, (4097, 2))
    simd_calls = []
    orig_simd = pip._pip_first_match_simdgroup

    def spy_simd(*args, **kwargs):
        simd_calls.append(int(args[0].shape[0]))
        return orig_simd(*args, **kwargs)

    monkeypatch.setattr(pip, "_pip_first_match_simdgroup", spy_simd)
    calls = _indexed_spy(monkeypatch)
    got = index.query(pts, backend="gpu")
    expected = index.query(pts, backend="cpu")
    np.testing.assert_array_equal(got, expected)
    assert simd_calls == [4097]
    assert not calls and polyset._y_edge_index is False
    # Repeated 4,097-point batches stay below the per-batch indexed crossover
    # (4097 * 2047 < _REPEAT_MIN_EDGE_WORK), so they never trigger an index build.
    assert 4097 * polyset.max_edges_per_ring < pip_indexed._REPEAT_MIN_EDGE_WORK
    for _ in range(3):
        np.testing.assert_array_equal(index.query(pts, backend="gpu"), expected)
    assert polyset._y_edge_index is False and polyset._y_edge_index_work == 0


def test_repeated_medium_batches_amortize_edge_index_build(monkeypatch):
    """Repeated medium batches build and reuse the edge index once amortized."""
    import mlx_geospatial._pip_indexed as pip_indexed

    polyset = ms.PolygonSet([Polygon(_ring(vertices=2047))], origin=(0, 0))
    index = ms.GridIndex(polyset)
    rng = np.random.default_rng(202611)
    # 16,384 * 2047 = 33.5M >= _REPEAT_MIN_EDGE_WORK (32M).
    pts = rng.uniform(-0.95, 0.95, (16_384, 2))
    expected = index.query(pts, backend="cpu")
    # Lower the cumulative threshold so the test exercises amortization in 3 calls.
    per_batch = len(pts) * polyset.max_edges_per_ring
    monkeypatch.setattr(pip_indexed, "_REPEAT_BUILD_MIN_EDGE_WORK", 3 * per_batch)
    monkeypatch.setattr(pip_indexed, "_REPEAT_BUILD_EDGES_MULTIPLIER", 1)
    calls = _indexed_spy(monkeypatch)

    for call_num in range(1, 3):
        got = index.query(pts, backend="gpu")
        np.testing.assert_array_equal(got, expected)
        assert polyset._y_edge_index is False, f"built too early on call {call_num}"
        assert not calls

    # Third batch crosses cumulative threshold -> builds and uses edge index.
    got = index.query(pts, backend="gpu")
    np.testing.assert_array_equal(got, expected)
    assert polyset._y_edge_index is not False and polyset._y_edge_index is not None
    assert calls == ["first"]

    # Fourth call (even a tiny batch) reuses the now-warm edge index.
    tiny = pts[:16]
    np.testing.assert_array_equal(index.query(tiny, backend="gpu"), expected[:16])
    assert calls == ["first", "first"]


@pytest.mark.parametrize("all_matches", [False, True])
def test_cold_build_follows_used_cell_work(all_matches):
    ring = _ring(vertices=2048)
    annulus = Polygon(ring, holes=[ring * 0.8])
    index = ms.GridIndex([annulus] * 32)
    points = np.full((65_536, 2), 5.0)
    points[0] = [0.9, 0.0]

    sparse = index.query(points, backend="gpu", all_matches=all_matches)
    assert index.polygons._y_edge_index is False
    assert index.polygons._y_edge_index_work == 0

    # The same number of rows now walks all the long candidate rings, but
    # points in the hole still produce exactly the same matches.
    points[1:] = 0.0
    dense = index.query(points, backend="gpu", all_matches=all_matches)
    assert index.polygons._y_edge_index
    if all_matches:
        for actual, expected in zip(dense, sparse, strict=True):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(dense[1], np.arange(32))
    else:
        np.testing.assert_array_equal(dense, sparse)
        assert dense[0] == 0


def test_cold_exact_query_reuses_boundary_index_for_containment(monkeypatch):
    index = ms.GridIndex([Polygon(_ring(vertices=2047))])
    points = np.array([[0.9, 0.0], [1.0, 0.0], [1.1, 0.0]])
    monkeypatch.setattr(boundary, "_COLD_LINEAR_BOUNDARY_POINTS", len(points))
    calls = _indexed_spy(monkeypatch)

    actual = index.query(points, backend="gpu", exact_boundary=True, predicate="within")
    np.testing.assert_array_equal(actual, [0, -1, -1])
    assert index.polygons._y_edge_index
    assert calls == ["first"]
