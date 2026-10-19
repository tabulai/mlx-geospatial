"""The host float64 edge grid finds exactly the edge-box STRtree oracle's rows."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

import mlx_geospatial._edge_grid as edge_grid
import mlx_geospatial.join as join
from mlx_geospatial import GridIndex, PolygonSet
from mlx_geospatial._edge_grid import HostEdgeGrid


def _ring(n=8192, scale=1.0, origin=(0.0, 0.0)):
    angle = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    radius = 1.0 + 0.1 * np.sin(7 * angle)
    return (np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]) * scale
            + np.asarray(origin))


def _probe_points(index, rng, n_random=2000):
    """Vertices, on-edge points, +-tol*(1+-1e-9) offsets, envelope corners +-1 ulp."""
    polyset = index.polygons
    tol = polyset.boundary_tolerance
    edges = polyset._edges_f64()
    pick = rng.choice(len(edges), min(len(edges), 400), replace=False)
    p1, p2 = edges[pick, :2], edges[pick, 2:]
    delta = p2 - p1
    length = np.hypot(delta[:, 0], delta[:, 1])
    length[length == 0] = 1.0
    normal = np.column_stack([-delta[:, 1], delta[:, 0]]) / length[:, None]
    on = p1 + rng.uniform(0, 1, (len(pick), 1)) * delta
    points = [p1, on, p1 + 0.5 * delta]
    for factor in (0.5, 1 - 1e-9, 1.0, 1 + 1e-9, 2.0):
        points += [on + factor * tol * normal, on - factor * tol * normal]
    points.append(on + rng.normal(0.0, tol, on.shape))
    lo = np.minimum(p1, p2) - tol
    hi = np.maximum(p1, p2) + tol
    for x in (lo[:, 0], hi[:, 0]):
        for y in (lo[:, 1], hi[:, 1]):
            points.append(np.column_stack([x, y]))
            points.append(np.column_stack([np.nextafter(x, -np.inf), y]))
            points.append(np.column_stack([x, np.nextafter(y, np.inf)]))
    bounds = polyset.bboxes[np.isfinite(polyset.bboxes).all(axis=1)]
    x0, y0 = bounds[:, :2].min(axis=0)
    x1, y1 = bounds[:, 2:].max(axis=0)
    w, h = x1 - x0, y1 - y0
    points.append(np.column_stack([
        rng.uniform(x0 - 0.1 * w, x1 + 0.1 * w, n_random),
        rng.uniform(y0 - 0.1 * h, y1 + 0.1 * h, n_random),
    ]))
    points.append(np.array([[np.nan, 0.0], [np.inf, 0.0], [0.0, -np.inf], [np.nan, np.nan]]))
    points = np.concatenate(points)
    return points[rng.permutation(len(points))]


def _assert_matches_oracle(index, points, rng):
    polyset = index.polygons
    tol = polyset.boundary_tolerance
    grid = polyset._host_edge_grid()
    assert grid is not None
    expected = index._near_boundary(points, tol)
    actual = grid.near_rows(points)
    assert actual.dtype == expected.dtype == np.int64
    np.testing.assert_array_equal(actual, expected)
    rows = np.sort(rng.choice(len(points), len(points) // 5, replace=False))
    local = grid.near_rows(points, rows)
    np.testing.assert_array_equal(local, index._near_boundary(points[rows], tol))
    np.testing.assert_array_equal(index._near_rows(points, rows),
                                  rows[index._near_boundary(points[rows], tol)])
    return expected


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_grid_matches_oracle_across_scales(scale):
    rng = np.random.default_rng(1)
    index = GridIndex([Polygon(_ring(scale=scale)), shapely.Polygon(), None])
    near = _assert_matches_oracle(index, _probe_points(index, rng), rng)
    assert near.size > 1000


@pytest.mark.parametrize("offset", [(5e5, 4e6), (1e7, 1e7), (179.5, 89.5), (-179.5, -89.5)])
def test_grid_matches_oracle_at_offsets(offset):
    rng = np.random.default_rng(2)
    polygons = [Polygon(_ring(512, scale=0.2, origin=(offset[0] + dx, offset[1])))
                for dx in (-0.3, 0.0, 0.3)]
    index = GridIndex(polygons)
    _assert_matches_oracle(index, _probe_points(index, rng), rng)


def test_invalid_hole_outside_shell_multipolygons_and_empty_parts():
    rng = np.random.default_rng(3)
    shell = _ring(2048)
    hole = _ring(512, scale=0.2, origin=(5.0, 0.0))
    invalid = Polygon(shell, holes=[hole])
    assert not invalid.is_valid
    polygon = MultiPolygon([invalid, Polygon(_ring(512, 0.1, (-4.0, 0.0)))])
    index = GridIndex([shapely.Polygon(), polygon, None, box(-0.2, -0.2, 0.2, 0.2)])
    points = np.concatenate([hole, hole + [0.5 * index.polygons.boundary_tolerance, 0.0],
                             _probe_points(index, rng)])
    _assert_matches_oracle(index, points, rng)


def test_shared_edge_lattice_and_stacked_overlaps():
    rng = np.random.default_rng(4)
    # Dyadic lattice: shared edges fall on exact multiples of the cell side.
    tiles = [box(i / 64, j / 64, (i + 1) / 64, (j + 1) / 64)
             for i in range(64) for j in range(64)]
    stacked = [box(0.25, 0.25, 0.75, 0.75)] * 300
    index = GridIndex(tiles + stacked)
    xs = np.arange(65) / 64
    lattice = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    points = np.concatenate([lattice, lattice + [0.5 / 64, 0.0], _probe_points(index, rng)])
    near = _assert_matches_oracle(index, points, rng)
    assert near.size > lattice.shape[0]


def test_repeated_vertices_keep_zero_length_edges():
    rng = np.random.default_rng(5)
    coords = _ring(2048)
    coords = np.insert(coords, [10, 35, 120], coords[[10, 35, 120]], axis=0)
    index = GridIndex([Polygon(coords + [3.0 * i, 0.0]) for i in range(4)])
    _assert_matches_oracle(index, _probe_points(index, rng), rng)


def test_pair_chunks_split_one_points_candidates(monkeypatch):
    rng = np.random.default_rng(6)
    index = GridIndex([Polygon(_ring(4096))])
    points = _probe_points(index, rng)
    expected = index._near_boundary(points, index.polygons.boundary_tolerance)
    monkeypatch.setattr(edge_grid, "_PAIR_CHUNK", 7)
    np.testing.assert_array_equal(index.polygons._host_edge_grid().near_rows(points), expected)


def test_two_pass_radix_order_is_stable():
    rng = np.random.default_rng(7)
    for n_cells in (300, 1 << 16, (1 << 16) + 1, 1 << 20):
        cells = rng.integers(0, n_cells, 50_000)
        np.testing.assert_array_equal(
            edge_grid._stable_cell_order(cells, n_cells), np.argsort(cells, kind="stable")
        )


def test_thin_strip_uses_more_than_65536_cells():
    rng = np.random.default_rng(8)
    tiles = [box(i * 1e-3, 0.0, (i + 1) * 1e-3, 1e-6) for i in range(20_000)]
    index = GridIndex(tiles)
    grid = index.polygons._host_edge_grid()
    assert grid.n_cells > 1 << 16
    _assert_matches_oracle(index, _probe_points(index, rng), rng)


def test_guard_falls_back_to_strtree_with_identical_results(monkeypatch):
    rng = np.random.default_rng(9)
    polygons = [Polygon(_ring(256, 0.1, (0.01 * i, 0.0))) for i in range(40)]
    points = _probe_points(GridIndex(polygons), rng)
    expected = {
        backend: GridIndex(polygons).query(points, backend=backend, all_matches=True,
                                           exact_boundary=True)
        for backend in ("cpu", "gpu")
    }
    monkeypatch.setattr(join, "_NEAR_GRID_PAIRS_PER_ROW", 0)
    monkeypatch.setattr(join, "_NEAR_GRID_PAIRS_PER_EDGE", 0)
    index = GridIndex(polygons)
    assert index.polygons._host_edge_grid().near_rows(points, max_pairs=0) is None
    tol = index.polygons.boundary_tolerance
    np.testing.assert_array_equal(index._near_rows(points, None),
                                  index._near_boundary(points, tol))
    for backend, (pt, pg) in expected.items():
        got = index.query(points, backend=backend, all_matches=True, exact_boundary=True)
        np.testing.assert_array_equal(got[0], pt)
        np.testing.assert_array_equal(got[1], pg)
    assert index.polygons._edge_box_strtree is not None


def test_empty_rows_nonfinite_rows_and_unbuildable_grid():
    index = GridIndex([Polygon(_ring(64))])
    grid = index.polygons._host_edge_grid()
    empty = grid.near_rows(np.empty((0, 2)))
    assert empty.dtype == np.int64 and empty.size == 0
    rows = grid.near_rows(np.array([[np.nan, 0.0], [np.inf, np.inf], [1e9, 0.0]]),
                          np.array([0, 1, 2]))
    assert rows.dtype == np.int64 and rows.size == 0
    assert HostEdgeGrid.build(np.empty((0, 4)), 1e-6) is None
    assert HostEdgeGrid.build(np.zeros((3, 4)), np.nan) is None
    # Every envelope collapses to one point: no positive cell side exists.
    assert HostEdgeGrid.build(np.zeros((3, 4)), 0.0) is None


def test_edges_f64_matches_ragged_table():
    holed = Polygon(_ring(64), holes=[_ring(16, 0.2)])
    layers = [
        [Polygon(), holed, None, MultiPolygon([holed, Polygon(_ring(32, 0.1, (5, 5)))])],
        [shapely.from_wkt("POLYGON Z ((0 0 1, 1 0 2, 1 1 3, 0 0 1))"), Polygon()],
        [Polygon(), None],
    ]
    for geoms in layers:
        polyset = PolygonSet(geoms)
        _, coords, _ = shapely.to_ragged_array(polyset._shapely)
        coords = np.asarray(coords, dtype=np.float64)[:, :2]
        keep = np.ones(len(coords), dtype=bool)
        keep[polyset._ring_offsets[1:] - 1] = False
        start = np.flatnonzero(keep)
        expected = np.concatenate([coords[start], coords[start + 1]], axis=1)
        actual = polyset._edges_f64()
        assert actual.dtype == np.float64 and actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, expected.reshape(-1, 4))


def test_exact_boundary_queries_use_the_grid_and_match_geos():
    rng = np.random.default_rng(10)
    polygons = [box(i, j, i + 1, j + 1) for i in range(6) for j in range(6)]
    index = GridIndex(polygons)
    xs = np.arange(7, dtype=np.float64)
    lattice = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    points = np.concatenate([lattice, lattice + [0.5, 0.0], rng.uniform(-1, 7, (500, 2))])
    for predicate in ("within", "intersects"):
        reference = shapely.STRtree(polygons).query(shapely.points(points), predicate=predicate)
        reference = reference[:, np.lexsort((reference[1], reference[0]))]
        for backend in ("cpu", "gpu"):
            pt, pg = index.query(points, backend=backend, all_matches=True,
                                 predicate=predicate, exact_boundary=True)
            np.testing.assert_array_equal(pt, reference[0])
            np.testing.assert_array_equal(pg, reference[1])
    assert index.polygons._edge_box_strtree is None



def test_budget_counts_the_cold_tree_build(monkeypatch):
    rng = np.random.default_rng(11)
    cluster = [Polygon(_ring(48, 1e-4, (rng.uniform(0, 1e-3), rng.uniform(0, 1e-3))))
               for _ in range(100)]
    spread = [Polygon(_ring(48, 0.5, (rng.uniform(0, 100), rng.uniform(0, 100))))
              for _ in range(20)]
    index = GridIndex(cluster + spread)
    tol = index.polygons.boundary_tolerance
    points = np.concatenate([rng.uniform(0, 1e-3, (50, 2)), _probe_points(index, rng)])
    grid = index.polygons._host_edge_grid()
    expected = grid.near_rows(points)
    np.testing.assert_array_equal(expected, index._near_boundary(points, tol))
    index.polygons._edge_box_strtree = None
    x, y = points[:, 0], points[:, 1]
    inside = (x >= grid.x0) & (x <= grid.x1) & (y >= grid.y0) & (y <= grid.y1)
    cell = (edge_grid._cells(y[inside], grid.y0, grid.inv, grid.ny) * grid.nx
            + edge_grid._cells(x[inside], grid.x0, grid.inv, grid.nx))
    pairs = int((grid.offsets[cell + 1] - grid.offsets[cell]).sum())
    answers = []
    near_rows = HostEdgeGrid.near_rows

    def spy(self, *args, **kwargs):
        answers.append(near_rows(self, *args, **kwargs) is not None)
        return near_rows(self, *args, **kwargs)

    monkeypatch.setattr(HostEdgeGrid, "near_rows", spy)
    monkeypatch.setattr(join, "_NEAR_GRID_PAIRS_PER_ROW", 0)
    monkeypatch.setattr(join, "_NEAR_GRID_PAIRS_PER_EDGE", -(-pairs // len(grid.edges)))
    # Cold: the tree's build is part of the budget, so the grid answers.
    np.testing.assert_array_equal(index._near_rows(points, None), expected)
    assert answers == [True] and index.polygons._edge_box_strtree is None
    # Once the tree exists, only per-row work remains in the budget.
    index.polygons._edge_box_tree()
    np.testing.assert_array_equal(index._near_rows(points, None), expected)
    assert answers == [True, False]


def test_envelope_test_rejects_rounded_distances_outside_the_box():
    # A horizontal edge and a point one ulp above its widened envelope: the
    # float64 projection rounds the distance to at most tol, but the oracle's
    # envelope query excludes the point, so the grid must as well.
    tol = 0.0078125
    y = -0.009291351336486483
    edges = np.array([[4033.305755571224, y, 4587.643202856107, y],
                      [4000.0, 1.0, 4600.0, 1.0]])
    point = np.array([[4176.84045375799, np.nextafter(y + tol, np.inf)]])
    seg = edges[0]
    t = np.clip((point[0, 0] - seg[0]) / (seg[2] - seg[0]), 0.0, 1.0)
    assert (point[0, 0] - (seg[0] + t * (seg[2] - seg[0]))) ** 2 + (point[0, 1] - y) ** 2 <= tol * tol
    boxes = shapely.box(np.minimum(edges[:, 0], edges[:, 2]) - tol,
                        np.minimum(edges[:, 1], edges[:, 3]) - tol,
                        np.maximum(edges[:, 0], edges[:, 2]) + tol,
                        np.maximum(edges[:, 1], edges[:, 3]) + tol)
    assert shapely.STRtree(boxes).query(shapely.points(point)).size == 0
    grid = HostEdgeGrid.build(edges, tol)
    assert grid.near_rows(point).size == 0
    inside = np.array([[point[0, 0], y + tol]])
    np.testing.assert_array_equal(grid.near_rows(inside), [0])
