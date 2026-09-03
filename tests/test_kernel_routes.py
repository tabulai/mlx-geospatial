"""Every Metal kernel is launched through its public route and agrees bit for bit.

``_pip.py`` compiles 15 kernels: scalar, SIMD32 and TG128 variants of the
pair, dense and first-match predicates, the point-centric all-match count and
write kernels, and the pair-parallel all-match count (scalar, SIMD32, TG128)
and write kernels. The selectors (``_pip_long_polygon_strategy`` on the work
count and the longest ring, ``_all_matches_strategy`` on the compaction
shape) pick among them, so a test that only checks answers can silently
exercise a kernel other than the one it was written for. These tests spy the
kernel getters, derive the expected strategy from the same selectors and
bounds the entry point sees, assert the spy matches it, and compare against
the NumPy fp32 fallback (``backend="cpu"``), the repo's bit-identity arbiter.
Every ``index.query`` leg passes ``backend="gpu"`` so auto dispatch cannot
swap in the fallback unnoticed; ``contains_matrix`` has no backend switch and
is pinned by the spy count alone.

The direct SIMD32/TG128 matrix at the end is regression insurance for lane
tails, sub-32-edge rings inside long-ring sets, partial last groups and
threadgroup sizes; no current bug is known there.

Helpers are local to this file.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

import mlx_geospatial as ms
import mlx_geospatial._pip as pip
from mlx_geospatial._dispatch import metal_available
from mlx_geospatial._ingest import LocalPoints
from mlx_geospatial._pip import _pip_long_polygon_strategy

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal not available")

GETTERS = sorted(n for n in dir(pip) if n.startswith("_get_") and n.endswith("_kernel"))
assert len(GETTERS) == 15, GETTERS


def _spy_getters(monkeypatch) -> dict[str, int]:
    """Count launches per kernel getter, keyed without the _get_/_kernel affixes."""
    counts = {}

    def wrap(name, original):
        def wrapped(*args, **kwargs):
            counts[name] = counts.get(name, 0) + 1
            return original(*args, **kwargs)

        return wrapped

    for name in GETTERS:
        monkeypatch.setattr(pip, name, wrap(name[len("_get_") : -len("_kernel")], getattr(pip, name)))
    return counts


def _ring(center, radius: float, k: int, rotation: float = 0.0) -> Polygon:
    ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False) + rotation
    return Polygon(np.column_stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)]))


def _long_layer(rng, n_rings: int = 64, k: int = 1100, extra_k: int = 1039):
    """Overlapping long rings near the origin sharing one grid cell."""
    polys = [_ring(rng.uniform(-0.05, 0.05, 2), rng.uniform(0.9, 1.0), k, rng.uniform(0, 1)) for _ in range(n_rings)]
    polys.append(_ring((0.02, -0.03), 0.95, extra_k, 0.3))
    polys = np.array(polys, dtype=object)
    polyset = ms.PolygonSet(polys, origin=(0.0, 0.0))
    index = ms.GridIndex(polyset, cell_size=100.0)
    assert index.n_cells == 1 and index._max_cell_count == len(polys)
    return polys, polyset, index


def _bbox_points(rng, polys, n: int) -> np.ndarray:
    """Points sampled inside the layer's bboxes (roughly half of them hit)."""
    bb = shapely.bounds(np.asarray(polys, dtype=object))
    bb = bb[np.isfinite(bb).all(axis=1)]
    pick = rng.integers(0, len(bb), n)
    return rng.uniform(bb[pick, :2], bb[pick, 2:])


def _assert_same_join(index, pts, *, all_matches=False, launches=1, **kw):
    cpu = index.query(pts, backend="cpu", all_matches=all_matches, **kw)
    for _ in range(launches):
        gpu = index.query(pts, backend="gpu", all_matches=all_matches, **kw)
        if all_matches:
            np.testing.assert_array_equal(gpu[0], cpu[0])
            np.testing.assert_array_equal(gpu[1], cpu[1])
        else:
            np.testing.assert_array_equal(gpu, cpu)
    if all_matches:
        assert len(cpu[0]) > 0
        return cpu
    hits = np.count_nonzero(cpu >= 0)
    assert 0.2 * len(pts) <= hits <= 0.9 * len(pts), hits
    return cpu


def _with_outside(rng, pts: np.ndarray, n_outside: int = 50) -> np.ndarray:
    """Replace the tail of ``pts`` by out-of-grid points (cell -1).

    The public route hands such points to the first-match kernels, which
    must write -1 for them before returning; a stale output buffer would
    otherwise leak old matches.
    """
    out = pts.copy()
    out[-n_outside:] = rng.uniform(50, 60, (n_outside, 2))
    return out


# ---------------------------------------------------------------------------
# first match: TG128 for n <= longest ring, SIMD32 up to 4096, scalar above
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_points", "expected"),
    [(1000, "threadgroup"), (1100, "threadgroup"), (1101, "simdgroup"), (4096, "simdgroup"),
     (4097, "scalar")],
)
def test_first_match_routes_by_point_count(n_points, expected, rng, monkeypatch):
    """The bound is the longest ring among the cell's candidates (1100 here),
    not their number: eight rings keep the NumPy oracle fast (the 64-candidate
    cooperative loop is covered by tests/test_pip_adversarial.py section 9).
    Fifty points lie outside the grid and must come back -1 on every launch."""
    polys, polyset, index = _long_layer(rng, n_rings=8)
    bound = int(index._cell_max_ring_edges.max())
    assert bound == 1100
    assert _pip_long_polygon_strategy(n_points, bound, max_work=4096) == expected
    counts = _spy_getters(monkeypatch)
    pts = _with_outside(rng, _bbox_points(rng, polys, n_points))
    cpu = _assert_same_join(index, pts, launches=3)
    assert (cpu[-50:] == -1).all()
    name = "first_match" if expected == "scalar" else f"first_match_{expected}"
    assert counts == {name: 3}, counts


def test_first_match_cooperation_needs_every_candidate_long(rng, monkeypatch):
    """One short polygon in the cell demotes first match to the scalar kernel;
    an empty polygon or a long shell with short holes does not."""
    polys, polyset, index = _long_layer(rng, n_rings=8)
    n = 1200
    pts = _with_outside(rng, _bbox_points(rng, polys, n))

    demoted = np.append(polys, [shapely.box(-0.2, -0.2, 0.2, 0.2)])
    index_d = ms.GridIndex(ms.PolygonSet(demoted, origin=(0.0, 0.0)), cell_size=100.0)
    counts = _spy_getters(monkeypatch)
    _assert_same_join(index_d, pts)
    assert counts == {"first_match": 1}, counts

    holed = Polygon(
        np.asarray(_ring((0.0, 0.0), 0.97, 1057).exterior.coords)[:-1],
        holes=[np.asarray(_ring((0.3, 0.0), 0.1, 5).exterior.coords)[:-1][::-1],
               np.asarray(_ring((-0.3, 0.0), 0.1, 40).exterior.coords)[:-1][::-1]],
    )
    assert holed.is_valid
    kept = np.append(polys, [Polygon(), holed])
    index_k = ms.GridIndex(ms.PolygonSet(kept, origin=(0.0, 0.0)), cell_size=100.0)
    counts.clear()
    cpu = _assert_same_join(index_k, pts, launches=3)
    assert (cpu[-50:] == -1).all()
    assert counts == {"first_match_simdgroup": 3}, counts


# ---------------------------------------------------------------------------
# all matches: long rings force the pair-parallel compactor; its count kernel
# routes on the batch's pair count. Short rings with many active points take
# the point-centric count/write kernels.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_in_cell", "expected"), [(16, "threadgroup"), (31, "simdgroup"), (67, "simdgroup"), (68, "scalar")]
)
def test_all_matches_pair_route_by_batch_pairs(n_in_cell, expected, rng, monkeypatch):
    polys, polyset, index = _long_layer(rng)
    m = polyset.n_polygons  # 65 candidates for every in-cell point
    assert _pip_long_polygon_strategy(n_in_cell * m, 1100) == expected
    pts = np.concatenate([_bbox_points(rng, polys, n_in_cell), rng.uniform(50, 60, (3000, 2))])
    counts = _spy_getters(monkeypatch)
    _assert_same_join(index, pts, all_matches=True)
    name = "all_matches_pair_count" if expected == "scalar" else f"all_matches_pair_{expected}_count"
    assert counts == {name: 1, "all_matches_pair_write": 1}, counts


def test_all_matches_pair_route_batches_independently(rng, monkeypatch):
    """A pair cap turns one SIMD32 batch into several TG128 batches."""
    polys, polyset, index = _long_layer(rng)
    pts = _bbox_points(rng, polys, 31)  # 2015 pairs
    counts = _spy_getters(monkeypatch)
    _assert_same_join(index, pts, all_matches=True, max_pairs_per_batch=1000)
    assert counts == {"all_matches_pair_threadgroup_count": 3, "all_matches_pair_write": 3}, counts


def test_all_matches_point_route_on_short_rings(rng, monkeypatch):
    polys = np.array([shapely.box(i, j, i + 1, j + 1) for i in range(4) for j in range(4)], dtype=object)
    index = ms.GridIndex(ms.PolygonSet(polys))
    pts = rng.uniform(0.05, 3.95, (1500, 2))  # >= 1024 active points, <= 4 candidates each
    counts = _spy_getters(monkeypatch)
    _assert_same_join(index, pts, all_matches=True)
    assert counts == {"all_matches_count": 1, "all_matches_write": 1}, counts


# ---------------------------------------------------------------------------
# explicit pairs: the oversized-cell first-match fallback batches pairs by
# max_pairs_per_batch; each batch routes on its own pair count
# ---------------------------------------------------------------------------


def test_pairs_route_simdgroup_in_exact_batches(rng, monkeypatch):
    """2200 x 1039-edge rings in one cell, cap 1100: every batch is 1100 pairs."""
    polys = np.array(
        [_ring(rng.uniform(-0.02, 0.02, 2), 1.0, 1039, rng.uniform(0, 1)) for _ in range(2200)],
        dtype=object,
    )
    index = ms.GridIndex(ms.PolygonSet(polys, origin=(0.0, 0.0)), cell_size=100.0)
    assert index._max_cell_count == 2200
    assert _pip_long_polygon_strategy(1100, 1039) == "simdgroup"
    # three in-cell points (the last is inside every bbox but outside every
    # ring): 3 x 2200 pairs = 6 batches of exactly 1100
    pts = np.array([[0.0, 0.0], [0.5, 0.5], [0.95, 0.95]])
    counts = _spy_getters(monkeypatch)
    got = index.query(pts, backend="gpu", max_pairs_per_batch=1100)
    np.testing.assert_array_equal(got, index.query(pts, backend="cpu"))
    assert got[0] >= 0 and got[2] == -1
    assert counts == {"pair_simdgroup": 6}, counts


def test_pairs_route_threadgroup_and_mixed_tail_batch(rng, monkeypatch):
    polys, polyset, index = _long_layer(rng)  # 65 long candidates
    pts = _bbox_points(rng, polys, 40)
    counts = _spy_getters(monkeypatch)
    _assert_same_join(index, pts, max_pairs_per_batch=64)  # 2600 pairs: 40 full + 1 tail batch
    assert set(counts) == {"pair_threadgroup"} and counts["pair_threadgroup"] == 41, counts

    # cap 1100 on 1101 x 1025-edge rings: four SIMD32 batches and one TG128
    # tail batch answer one public call; the answer is route-independent.
    polys = np.array(
        [_ring(rng.uniform(-0.02, 0.02, 2), 1.0, 1025, rng.uniform(0, 1)) for _ in range(1101)],
        dtype=object,
    )
    index = ms.GridIndex(ms.PolygonSet(polys, origin=(0.0, 0.0)), cell_size=100.0)
    pts = np.array([[0.0, 0.0], [0.3, 0.1], [-0.2, 0.4], [0.1, -0.5]])
    counts.clear()
    np.testing.assert_array_equal(
        index.query(pts, backend="gpu", max_pairs_per_batch=1100), index.query(pts, backend="cpu")
    )
    assert counts == {"pair_simdgroup": 4, "pair_threadgroup": 1}, counts


def test_pairs_route_scalar_on_short_rings(rng, monkeypatch):
    # a sliver fixes the grid extent to [-1, 1]^2; the 70 boxes cover only
    # |y| <= 0.5, so about half of the points miss every candidate
    polys = [shapely.box(-1, -1, -0.98, 1)]
    polys += [shapely.box(-1 + 0.01 * i, -0.5, 1, 0.5) for i in range(70)]
    polys = np.array(polys, dtype=object)
    index = ms.GridIndex(ms.PolygonSet(polys, origin=(0.0, 0.0)), cell_size=100.0)
    pts = rng.uniform(-0.97, 0.99, (40, 2))  # all in the grid: 40 x 71 = 2840 pairs
    counts = _spy_getters(monkeypatch)
    _assert_same_join(index, pts, max_pairs_per_batch=50)
    assert counts == {"pair": 57}, counts


def test_pairs_route_bounds_each_batch_by_its_own_candidates(rng, monkeypatch):
    """A batch of short candidates in a long-ring set runs the scalar kernel.

    _query_gpu_pairs takes the ring bound from the polygons in each batch, not
    from the whole set: two boxes far from the 65 long rings share the grid
    but not a cell, and with a cap of 2 pairs the point among the boxes forms
    its own batch.
    """
    polys, polyset, index = _long_layer(rng)
    mixed = np.concatenate(
        [np.array([shapely.box(9, -1, 11, 1), shapely.box(9.5, -0.5, 10.5, 0.5)], dtype=object), polys]
    )
    index = ms.GridIndex(ms.PolygonSet(mixed, origin=(0.0, 0.0)), cell_size=5.0)
    pts = np.array([[10.0, 0.0], [0.05, 0.02]])  # 2 short candidates; 65 long ones
    counts = _spy_getters(monkeypatch)
    gpu = index.query(pts, backend="gpu", max_pairs_per_batch=2)
    np.testing.assert_array_equal(gpu, index.query(pts, backend="cpu"))
    assert gpu[0] == 0 and gpu[1] >= 2
    assert counts == {"pair": 1, "pair_threadgroup": 33}, counts


def test_all_matches_route_bounds_by_the_used_cells(rng, monkeypatch):
    """A far-away long ring must not push a short-ring query onto the
    pair-parallel compactor (the bound comes from the cells the points use)."""
    polys = [shapely.box(i, j, i + 1, j + 1) for i in range(4) for j in range(4)]
    polys.append(_ring((60.0, 60.0), 1.0, 1100))
    index = ms.GridIndex(ms.PolygonSet(np.array(polys, dtype=object)))
    pts = rng.uniform(0.05, 3.95, (1500, 2))
    counts = _spy_getters(monkeypatch)
    _assert_same_join(index, pts, all_matches=True)
    assert counts == {"all_matches_count": 1, "all_matches_write": 1}, counts


# ---------------------------------------------------------------------------
# dense: contains_matrix routes on N * M and the whole set's longest ring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "m", "expected"), [(20, 50, "threadgroup"), (22, 50, "threadgroup"), (23, 50, "simdgroup"),
                             (50, 60, "simdgroup"), (88, 50, "simdgroup"), (89, 50, "scalar")]
)
def test_dense_routes_by_matrix_size(n, m, expected, rng, monkeypatch):
    polys = [_ring(rng.uniform(-0.05, 0.05, 2), rng.uniform(0.9, 1.0), 1100, rng.uniform(0, 1)) for _ in range(m - 1)]
    polys.append(shapely.box(-0.3, -0.3, 0.3, 0.3))  # one short polygon does not demote
    polys = np.array(polys, dtype=object)
    polyset = ms.PolygonSet(polys, origin=(0.0, 0.0))
    assert _pip_long_polygon_strategy(n * m, polyset.max_edges_per_ring) == expected
    pts = _bbox_points(rng, polys, n)
    counts = _spy_getters(monkeypatch)
    got = ms.contains_matrix(pts, polyset)
    name = "dense" if expected == "scalar" else f"dense_{expected}"
    assert counts == {name: 1}, counts
    index = ms.GridIndex(polyset, cell_size=100.0)
    pt, pg = index.query(pts, backend="cpu", all_matches=True)
    expected_matrix = np.zeros((n, m), dtype=bool)
    expected_matrix[pt, pg] = True
    np.testing.assert_array_equal(got, expected_matrix)
    hits = expected_matrix.sum()  # overlapping rings: most bbox points hit most rings
    assert 0.05 * n * m <= hits <= 0.95 * n * m, hits


# ---------------------------------------------------------------------------
# direct SIMD32 / TG128 matrix: lane tails, short rings among long ones,
# partial groups, threadgroup sizes
# ---------------------------------------------------------------------------


def _hazard_layer(rng, k: int) -> np.ndarray:
    shell = np.asarray(_ring((0.0, 0.0), 1.0, k).exterior.coords)[:-1]
    centers = [(0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4), (0.35, 0.35)]
    holes = [
        np.asarray(_ring(c, 0.12, h).exterior.coords)[:-1][::-1]
        for c, h in zip(centers, (3, 31, 32, 33, 40))
    ]
    holed = Polygon(shell, holes=[holes[0], holes[1]])
    holed2 = Polygon(np.asarray(_ring((0.0, 0.0), 0.98, k).exterior.coords)[:-1], holes=holes[2:])
    dup = np.asarray(_ring((0.1, 0.1), 0.9, k).exterior.coords)[:-1]
    dup = np.insert(dup, [1, 1, 500], dup[[0, 0, 499]], axis=0)  # duplicate vertices
    multi = MultiPolygon([_ring((0.0, 0.0), 0.95, k, 0.2), shapely.box(1.5, 1.5, 1.7, 1.7)])
    polys = [holed, holed2, Polygon(dup), multi, Polygon(), _ring((-0.1, 0.05), 0.93, k, 0.7)]
    assert all(p.is_valid for p in polys)
    return np.array(polys, dtype=object)


def _hazard_points(rng, polys, n: int) -> np.ndarray:
    coords = shapely.get_coordinates(polys)
    vy = coords[rng.integers(0, len(coords), 60), 1]
    pts = np.concatenate(
        [
            _bbox_points(rng, polys, n),
            coords[rng.integers(0, len(coords), 40)],  # exact vertices
            np.column_stack([rng.uniform(-1.2, 1.2, 60), vy]),  # rays through vertices
            [[np.nan, 0.0], [0.5, np.inf], [-np.inf, -np.inf]],
        ]
    )
    return pts


@pytest.mark.parametrize("k", [1025, 1039, 1056, 1150, 2049])
def test_cooperative_kernels_match_scalar_on_hazard_layers(k, rng, monkeypatch):
    """Shell lengths cover SIMD lane tails of 1, 15, 0 and 30 (mod 32) and
    threadgroup tails of 1, 15, 32 and 126 (mod 128); the duplicate-vertex
    ring has k + 3 edges."""
    polys = _hazard_layer(rng, k)
    polyset = ms.PolygonSet(polys, origin=(0.0, 0.0))
    assert polyset.max_edges_per_ring == k + 3
    pts = _hazard_points(rng, polys, 300)
    n, m = len(pts), polyset.n_polygons
    local = mx.array(pts.astype(np.float32))
    pair_pt = mx.array(np.repeat(np.arange(n, dtype=np.int32), m))
    pair_pg = mx.array(np.tile(np.arange(m, dtype=np.int32), n))
    scalar = np.array(pip.pip_pairs(local, polyset, pair_pt, pair_pg, max_ring_edges=0)).reshape(n, m)
    index = ms.GridIndex(polyset, cell_size=100.0)
    pt, pg = index.query(LocalPoints(local, polyset.origin), backend="cpu", all_matches=True)
    fallback = np.zeros((n, m), dtype=np.uint8)
    fallback[np.asarray(pt), np.asarray(pg)] = 1
    np.testing.assert_array_equal(scalar, fallback)
    assert 0.2 * n <= fallback.any(axis=1).sum() <= 0.8 * n

    simd = np.array(pip._pip_pairs_simdgroup(local, polyset, pair_pt, pair_pg)).reshape(n, m)
    tg = np.array(pip._pip_pairs_threadgroup(local, polyset, pair_pt, pair_pg)).reshape(n, m)
    np.testing.assert_array_equal(simd, scalar, err_msg="pairs simdgroup")
    np.testing.assert_array_equal(tg, scalar, err_msg="pairs threadgroup")
    np.testing.assert_array_equal(np.array(pip._pip_dense_simdgroup(local, polyset)), scalar.astype(bool))
    np.testing.assert_array_equal(np.array(pip._pip_dense_threadgroup(local, polyset)), scalar.astype(bool))

    point_cell = mx.array(np.zeros(n, dtype=np.int32))
    cell_offsets = mx.array(np.array([0, m], dtype=np.int64))
    cell_poly = mx.array(np.arange(m, dtype=np.int32))
    first = np.where(scalar.any(axis=1), scalar.argmax(axis=1), -1).astype(np.int32)
    for fn in (pip._pip_first_match_simdgroup, pip._pip_first_match_threadgroup):
        np.testing.assert_array_equal(
            np.array(fn(local, polyset, point_cell, cell_offsets, cell_poly)), first, err_msg=fn.__name__
        )
    # all-match pair compaction: a 7-pair range that splits the second point
    # (7 <= k: TG128), the rest of the pairs (k < n*m - 7 <= 4k: SIMD32), and
    # one full pass with the ring bound forced to 0 (scalar count kernel)
    point_id = mx.array(np.arange(n, dtype=np.int32))
    candidate_offsets = mx.array(np.arange(0, (n + 1) * m, m, dtype=np.int64))
    expected_pt, expected_pg = np.nonzero(scalar)
    assert k < n * m - 7 <= 4 * k
    counts = _spy_getters(monkeypatch)
    plan = [(0, 7, None, "threadgroup"), (7, n * m, None, "simdgroup"), (0, n * m, 0, "scalar")]
    got = {}
    for a, b, bound, expected in plan:
        kw = {} if bound is None else {"max_ring_edges": bound}
        counts.clear()
        p, g = pip.pip_all_matches(
            local, polyset, point_id, point_cell, candidate_offsets, cell_offsets, cell_poly,
            pair_begin=a, pair_end=b, max_point_pairs=min(m, b - a), **kw,
        )
        got[(a, b)] = (np.asarray(p), np.asarray(g))
        name = "all_matches_pair_count" if expected == "scalar" else f"all_matches_pair_{expected}_count"
        assert counts == {name: 1, "all_matches_pair_write": 1}, (expected, counts)
    np.testing.assert_array_equal(np.concatenate([got[(0, 7)][0], got[(7, n * m)][0]]), expected_pt)
    np.testing.assert_array_equal(np.concatenate([got[(0, 7)][1], got[(7, n * m)][1]]), expected_pg)
    np.testing.assert_array_equal(got[(0, n * m)][0], expected_pt)
    np.testing.assert_array_equal(got[(0, n * m)][1], expected_pg)


@pytest.mark.parametrize("threadgroup", [32, 64, 96, 128, 256, 512, 1024])
@pytest.mark.parametrize("n_pairs", [1, 33])
def test_simdgroup_kernels_accept_every_threadgroup_size(threadgroup, n_pairs, rng):
    """n_pairs * 32 lanes rarely divide the group: partial groups and tails.

    An oversized threadgroup would abort the process inside MLX rather than
    fail this test; 1024 is the Metal maximum on every Apple GPU.
    """
    polys = np.array([_ring((0.0, 0.0), 1.0, 1039), _ring((0.1, 0.0), 0.9, 1057, 0.4)], dtype=object)
    polyset = ms.PolygonSet(polys, origin=(0.0, 0.0))
    pts = rng.uniform(-1.1, 1.1, (n_pairs, 2)).astype(np.float32)
    local = mx.array(pts)
    pair_pt = mx.array(np.arange(n_pairs, dtype=np.int32))
    pair_pg = mx.array(rng.integers(0, 2, n_pairs).astype(np.int32))
    scalar = np.array(pip.pip_pairs(local, polyset, pair_pt, pair_pg, max_ring_edges=0))
    got = np.array(pip._pip_pairs_simdgroup(local, polyset, pair_pt, pair_pg, threadgroup=threadgroup))
    np.testing.assert_array_equal(got, scalar)
    dense_scalar = np.array(pip._pip_dense_composed(local, polyset))
    got = np.array(pip._pip_dense_simdgroup(local, polyset, threadgroup=threadgroup))
    np.testing.assert_array_equal(got, dense_scalar)
    point_cell, cell_offsets, cell_poly = (
        mx.array(np.zeros(n_pairs, dtype=np.int32)),
        mx.array(np.array([0, 2], dtype=np.int64)),
        mx.array(np.arange(2, dtype=np.int32)),
    )
    first = np.where(dense_scalar.any(axis=1), dense_scalar.argmax(axis=1), -1).astype(np.int32)
    got = np.array(
        pip._pip_first_match_simdgroup(local, polyset, point_cell, cell_offsets, cell_poly, threadgroup=threadgroup)
    )
    np.testing.assert_array_equal(got, first)
