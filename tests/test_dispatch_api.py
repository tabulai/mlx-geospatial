"""Deep tests for backend dispatch (_dispatch.py) and the public API surface.

Covers:
- ``resolve_backend``: auto threshold, explicit pass-through, env var
  precedence (requested= wins over ``MLX_SPATIAL_BACKEND``), invalid values.
- End-to-end backend routing: env-forced CPU on a workload that would
  otherwise auto-dispatch to the GPU, auto-CPU below the threshold, explicit
  GPU on tiny workloads. Routing is observed by monkeypatching the kernel
  entry points in ``mlx_spatial.join`` at test time (no library code is
  modified on disk).
- ``contains_matrix`` chunking equivalence (tiny chunk_bytes == default,
  bitwise).
- Degenerate sizes: zero points, zero candidate pairs, a single point.
- Public API: ``__all__`` resolves with docstrings, ``__version__`` matches
  pyproject.toml, clean import in a fresh subprocess, dispatch cost-model
  sanity.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest
import shapely

import mlx_spatial as ms
import mlx_spatial._dispatch as _dispatch
from mlx_spatial._dispatch import (
    GPU_LAUNCH_US,
    gpu_min_points,
    metal_available,
    resolve_backend,
)
from mlx_spatial._pip import (
    _all_matches_strategy,
    _pip_dense_composed,
    _pip_long_polygon_strategy,
    pip_all_matches,
    pip_dense,
    pip_first_match,
    pip_pairs,
)

from conftest import make_convex_polygons, make_grid_polygons, reference_assign

requires_metal = pytest.mark.skipif(
    not metal_available(), reason="Metal GPU not available"
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_matches_strategy_accounts_for_max_point_skew():
    """One pathological point must not hide behind a low chunk average."""
    assert _all_matches_strategy(100_000, 199_999, 100_000) == "pair"
    assert _all_matches_strategy(100_000, 800_000, 8) == "point"
    # Long candidate geometries force the candidate-parallel path even when
    # candidate degree alone would favor the point-centric implementation.
    assert _all_matches_strategy(100_000, 6_400_000, 64, 8191) == "point"
    assert _all_matches_strategy(100_000, 6_400_000, 64, 8192) == "pair"
    assert _all_matches_strategy(100_000, 6_400_000, 64, 1, 1024) == "pair"


@pytest.mark.parametrize(
    ("n_work", "max_ring_edges", "expected"),
    [
        (0, 100_000, "scalar"),
        (1, 1023, "scalar"),
        (1024, 1024, "threadgroup"),
        (1025, 1024, "simdgroup"),
        (4096, 1024, "simdgroup"),
        (4097, 1024, "scalar"),
        (4096, 4096, "threadgroup"),
        (4097, 4096, "simdgroup"),
        (16_384, 4096, "simdgroup"),
        (16_385, 4096, "scalar"),
    ],
)
def test_pip_long_polygon_strategy_boundaries(
    n_work, max_ring_edges, expected
):
    assert _pip_long_polygon_strategy(n_work, max_ring_edges) == expected


@pytest.fixture(scope="module")
def grid_workload():
    """8x8 unit-square tiling + 100k strictly-interior points.

    Far more points than gpu_min_points() for this layer, so 'auto' picks
    the GPU.
    Points stay >= 0.05 from every edge -- far outside the fp32 tolerance
    band -- so GPU and CPU answers are both exactly the golden STRtree
    assignment.
    """
    rng = np.random.default_rng(987654321)
    polys = make_grid_polygons(8, 8, cell=1.0)
    polyset = ms.PolygonSet(polys)
    index = ms.GridIndex(polyset)
    n = 100_000
    ij = rng.integers(0, 8, size=(n, 2)).astype(np.float64)
    offs = rng.uniform(0.05, 0.95, size=(n, 2))
    pts = ij + offs
    ref = reference_assign(pts, polys)
    return polys, polyset, index, pts, ref


# ---------------------------------------------------------------------------
# resolve_backend unit behavior
# ---------------------------------------------------------------------------


@requires_metal
def test_auto_threshold_boundary():
    """'auto' switches to GPU exactly at gpu_min_points() (>= rule)."""
    t = gpu_min_points()
    assert resolve_backend(None, t) == "gpu"
    assert resolve_backend(None, t - 1) == "cpu"
    assert resolve_backend(None, 0) == "cpu"
    assert resolve_backend("auto", t + 1) == "gpu"
    assert resolve_backend("auto", 1) == "cpu"
    # Device-resident inputs never wait for the CPU path under auto.
    assert resolve_backend(None, None) == "gpu"
    # Detailed polygons make the host fallback slower per point, so the GPU wins earlier.
    heavy = gpu_min_points(edges_per_point=6000.0)
    assert heavy < t
    assert resolve_backend(None, heavy, edges_per_point=6000.0) == "gpu"
    assert resolve_backend(None, heavy, edges_per_point=0.0) == "cpu"
    # All-match compaction has a larger fixed GPU cost, so it switches later.
    t_all = gpu_min_points(all_matches=True)
    assert t_all > t
    assert resolve_backend(None, t, all_matches=True) == "cpu"
    assert resolve_backend(None, t_all, all_matches=True) == "gpu"
    # NaN edge work is treated as zero rather than raising.
    assert gpu_min_points(float("nan")) == gpu_min_points(0.0)
    with pytest.raises(ValueError):
        resolve_backend(3, 10)


def test_explicit_passthrough_ignores_workload_size():
    assert resolve_backend("cpu", 10**9) == "cpu"
    assert resolve_backend("cpu", 0) == "cpu"
    if metal_available():
        assert resolve_backend("gpu", 0) == "gpu"
        assert resolve_backend("gpu", 1) == "gpu"


def test_resolve_backend_never_returns_auto():
    for n in (0, gpu_min_points() - 1, gpu_min_points(), 10**9, None):
        assert resolve_backend(None, n) in ("gpu", "cpu")
        assert resolve_backend("auto", n) in ("gpu", "cpu")


def test_requested_is_case_insensitive():
    assert resolve_backend("CPU", 10**9) == "cpu"
    assert resolve_backend("Cpu", 10**9) == "cpu"
    if metal_available():
        assert resolve_backend("GPU", 1) == "gpu"
        assert resolve_backend("Auto", 10**9) == "gpu"


def test_invalid_requested_raises_valueerror_with_value():
    with pytest.raises(ValueError) as ei:
        resolve_backend("tpu", 10)
    assert "tpu" in str(ei.value)
    with pytest.raises(ValueError) as ei:
        resolve_backend("metal", 10**9)
    assert "metal" in str(ei.value)


def test_env_var_respected(monkeypatch):
    monkeypatch.setenv("MLX_SPATIAL_BACKEND", "cpu")
    assert resolve_backend(None, 10**9) == "cpu"
    if metal_available():
        monkeypatch.setenv("MLX_SPATIAL_BACKEND", "gpu")
        assert resolve_backend(None, 1) == "gpu"


def test_env_var_case_insensitive(monkeypatch):
    monkeypatch.setenv("MLX_SPATIAL_BACKEND", "CPU")
    assert resolve_backend(None, 10**9) == "cpu"


def test_requested_overrides_env(monkeypatch):
    monkeypatch.setenv("MLX_SPATIAL_BACKEND", "cpu")
    if metal_available():
        assert resolve_backend("gpu", 1) == "gpu"
        # explicit 'auto' also takes precedence over the env var
        assert resolve_backend("auto", 10**9) == "gpu"
    monkeypatch.setenv("MLX_SPATIAL_BACKEND", "gpu")
    assert resolve_backend("cpu", 10**9) == "cpu"


def test_invalid_env_raises_valueerror_with_value(monkeypatch):
    monkeypatch.setenv("MLX_SPATIAL_BACKEND", "quantum")
    with pytest.raises(ValueError) as ei:
        resolve_backend(None, 10)
    assert "quantum" in str(ei.value)
    # a valid requested= never even consults the (broken) env var
    assert resolve_backend("cpu", 10) == "cpu"


def test_metal_unavailable_fallback(monkeypatch):
    """Without Metal: 'auto' degrades to cpu, explicit 'gpu' raises."""
    monkeypatch.setattr(_dispatch, "metal_available", lambda: False)
    assert _dispatch.resolve_backend(None, 10**9) == "cpu"
    assert _dispatch.resolve_backend("auto", 10**9) == "cpu"
    assert _dispatch.resolve_backend("cpu", 1) == "cpu"
    with pytest.raises(RuntimeError):
        _dispatch.resolve_backend("gpu", 10**9)


def test_metal_available_returns_bool():
    assert isinstance(metal_available(), bool)


def test_dispatch_cost_model_sane():
    assert _dispatch.CPU_US_PER_POINT > 0 and _dispatch.CPU_US_PER_EDGE > 0
    assert GPU_LAUNCH_US > 0
    t = gpu_min_points()
    assert isinstance(t, int) and not isinstance(t, bool)
    assert 1 <= t < 10**6
    # More candidate edge work per point can only lower the threshold.
    thresholds = [gpu_min_points(e) for e in (0.0, 150.0, 1000.0, 10_000.0)]
    assert thresholds == sorted(thresholds, reverse=True)
    assert _dispatch.GPU_ALL_MATCHES_LAUNCH_US > GPU_LAUNCH_US
    for name in (
        "CPU_US_PER_POINT",
        "CPU_US_PER_EDGE",
        "GPU_LAUNCH_US",
        "GPU_ALL_MATCHES_LAUNCH_US",
        "gpu_min_points",
    ):
        assert name in _dispatch.__all__


# ---------------------------------------------------------------------------
# End-to-end backend routing
# ---------------------------------------------------------------------------


@requires_metal
def test_env_cpu_forces_cpu_path_end_to_end(grid_workload, monkeypatch):
    """MLX_SPATIAL_BACKEND=cpu must route a large workload off the GPU."""
    polys, polyset, index, pts, ref = grid_workload
    # This workload would auto-dispatch to the GPU...
    assert len(pts) >= gpu_min_points(index._edges_per_point)

    monkeypatch.setenv("MLX_SPATIAL_BACKEND", "cpu")

    def _boom(*args, **kwargs):  # pragma: no cover - only on failure
        raise AssertionError("GPU kernel invoked despite MLX_SPATIAL_BACKEND=cpu")

    monkeypatch.setattr("mlx_spatial.join.pip_first_match", _boom)
    monkeypatch.setattr("mlx_spatial.join.pip_pairs", _boom)
    got = ms.point_in_polygon(pts, polyset, index=index)
    assert got.shape == (len(pts),)
    np.testing.assert_array_equal(got, ref)


@requires_metal
def test_auto_uses_gpu_above_threshold_end_to_end(grid_workload, monkeypatch):
    """Same workload, no env override: the fused first-match kernel runs."""
    polys, polyset, index, pts, ref = grid_workload
    calls: list[int] = []

    def _spy(points_local, polyset_, point_cell, cell_offsets, cell_poly, **kwargs):
        calls.append(int(points_local.shape[0]))
        return pip_first_match(
            points_local,
            polyset_,
            point_cell,
            cell_offsets,
            cell_poly,
            **kwargs,
        )

    monkeypatch.setattr("mlx_spatial.join.pip_first_match", _spy)
    got = ms.point_in_polygon(pts, polyset, index=index)
    assert calls, "auto dispatch above the threshold never launched the kernel"
    np.testing.assert_array_equal(got, ref)


def test_auto_small_workload_takes_cpu_path(rng, monkeypatch):
    """Below the threshold, auto must use the NumPy fallback (no kernel)."""
    polys = make_grid_polygons(4, 4, cell=1.0)
    polyset = ms.PolygonSet(polys)
    index = ms.GridIndex(polyset)
    ij = rng.integers(0, 4, size=(500, 2)).astype(np.float64)
    pts = ij + rng.uniform(0.05, 0.95, size=(500, 2))
    assert len(pts) < gpu_min_points(index._edges_per_point)

    def _boom(*args, **kwargs):  # pragma: no cover - only on failure
        raise AssertionError("GPU kernel invoked below the auto threshold")

    monkeypatch.setattr("mlx_spatial.join.pip_first_match", _boom)
    monkeypatch.setattr("mlx_spatial.join.pip_pairs", _boom)
    got = ms.point_in_polygon(pts, polyset, index=index)
    np.testing.assert_array_equal(got, reference_assign(pts, polys))


@requires_metal
@requires_metal
def test_all_matches_auto_switches_later_than_first_match(rng, monkeypatch):
    """Between the two thresholds, auto keeps all_matches on the host fallback (cheaper)."""
    polys = make_grid_polygons(6, 6, cell=1.0)
    polyset = ms.PolygonSet(polys)
    index = ms.GridIndex(polyset)
    t_first = gpu_min_points(index._edges_per_point)
    t_all = gpu_min_points(index._edges_per_point, all_matches=True)
    assert t_first < t_all
    pts = rng.uniform(0.05, 5.95, size=(t_all, 2))

    cpu_calls, gpu_calls = [], []
    orig_cpu = ms.GridIndex._query_cpu
    orig_all = ms.GridIndex._query_gpu_all_matches

    def cpu_spy(self, *args, **kwargs):
        cpu_calls.append(1)
        return orig_cpu(self, *args, **kwargs)

    def gpu_spy(self, *args, **kwargs):
        gpu_calls.append(1)
        return orig_all(self, *args, **kwargs)

    monkeypatch.setattr(ms.GridIndex, "_query_cpu", cpu_spy)
    monkeypatch.setattr(ms.GridIndex, "_query_gpu_all_matches", gpu_spy)
    monkeypatch.delenv("MLX_SPATIAL_BACKEND", raising=False)

    index.query(pts[:t_first], all_matches=True)  # first-match threshold: still the fallback
    assert (len(cpu_calls), len(gpu_calls)) == (1, 0)
    index.query(pts[:t_first])  # the same point count runs first match on the GPU
    assert (len(cpu_calls), len(gpu_calls)) == (1, 0)
    index.query(pts, all_matches=True)  # at the all-match threshold: GPU compaction
    assert (len(cpu_calls), len(gpu_calls)) == (1, 1)


def test_explicit_gpu_small_workload_uses_kernel(rng, monkeypatch):
    """backend='gpu' forces fused first-match even below the threshold."""
    polys = make_grid_polygons(4, 4, cell=1.0)
    polyset = ms.PolygonSet(polys)
    index = ms.GridIndex(polyset)
    ij = rng.integers(0, 4, size=(200, 2)).astype(np.float64)
    pts = ij + rng.uniform(0.05, 0.95, size=(200, 2))
    calls: list[int] = []

    def _spy(points_local, polyset_, point_cell, cell_offsets, cell_poly, **kwargs):
        calls.append(int(points_local.shape[0]))
        return pip_first_match(
            points_local,
            polyset_,
            point_cell,
            cell_offsets,
            cell_poly,
            **kwargs,
        )

    monkeypatch.setattr("mlx_spatial.join.pip_first_match", _spy)
    got = ms.point_in_polygon(pts, polyset, index=index, backend="gpu")
    assert calls, "backend='gpu' did not launch the kernel"
    np.testing.assert_array_equal(got, reference_assign(pts, polys))


@requires_metal
def test_explicit_gpu_all_matches_uses_gpu_compactor(rng, monkeypatch):
    polys = make_grid_polygons(4, 4, cell=1.0)
    index = ms.GridIndex(polys)
    pts = rng.uniform(0.05, 3.95, size=(200, 2))
    calls: list[int] = []

    def _spy(*args, pair_begin, pair_end, **kwargs):
        calls.append(pair_end - pair_begin)
        return pip_all_matches(
            *args,
            pair_begin=pair_begin,
            pair_end=pair_end,
            **kwargs,
        )

    def other_kernel_must_not_run(*args, **kwargs):
        raise AssertionError("all_matches launched a non-compacting kernel")

    monkeypatch.setattr("mlx_spatial.join.pip_all_matches", _spy)
    monkeypatch.setattr("mlx_spatial.join.pip_first_match", other_kernel_must_not_run)
    monkeypatch.setattr("mlx_spatial.join.pip_pairs", other_kernel_must_not_run)
    index.query(pts, all_matches=True, backend="gpu")
    assert calls, "all_matches never launched the GPU compactor"


@requires_metal
def test_join_long_polygon_routing_is_limited_to_candidate_cells(monkeypatch):
    """An unrelated long ring must not change a short-cell query's route."""
    k = 1031
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    long_far_away = shapely.Polygon(
        np.column_stack(
            [100.0 + np.cos(angles), 100.0 + np.sin(angles)]
        )
    )
    polyset = ms.PolygonSet([shapely.box(0.0, 0.0, 1.0, 1.0), long_far_away])
    index = ms.GridIndex(polyset, cell_size=10.0)
    pts = np.array([[0.25, 0.25], [0.75, 0.75]])
    _, _, _, point_cell = index._candidates(pts)
    assert index._candidate_edge_bounds(point_cell) == (4, 4)

    def cooperative_must_not_run(*args, **kwargs):  # pragma: no cover
        raise AssertionError("an unrelated long polygon forced cooperative PIP")

    for name in (
        "_pip_first_match_threadgroup",
        "_pip_first_match_simdgroup",
        "_get_all_matches_pair_threadgroup_count_kernel",
        "_get_all_matches_pair_simdgroup_count_kernel",
    ):
        monkeypatch.setattr(f"mlx_spatial._pip.{name}", cooperative_must_not_run)

    np.testing.assert_array_equal(index.query(pts, backend="gpu"), [0, 0])
    point_idx, poly_idx = index.query(pts, all_matches=True, backend="gpu")
    np.testing.assert_array_equal(point_idx, [0, 1])
    np.testing.assert_array_equal(poly_idx, [0, 0])


def test_invalid_env_propagates_from_point_in_polygon(monkeypatch):
    monkeypatch.setenv("MLX_SPATIAL_BACKEND", "banana")
    with pytest.raises(ValueError) as ei:
        ms.point_in_polygon([[0.5, 0.5]], [shapely.box(0, 0, 1, 1)])
    assert "banana" in str(ei.value)


def test_invalid_backend_argument_propagates():
    with pytest.raises(ValueError) as ei:
        ms.point_in_polygon(
            [[0.5, 0.5]], [shapely.box(0, 0, 1, 1)], backend="npu"
        )
    assert "npu" in str(ei.value)


# ---------------------------------------------------------------------------
# contains_matrix chunking
# ---------------------------------------------------------------------------


def test_contains_matrix_tiny_chunks_bitwise_equal(rng, monkeypatch):
    """Composed fallback chunking is bitwise equal to direct dense Metal."""
    polys = make_convex_polygons(20, rng, radius_frac=0.25)
    polyset = ms.PolygonSet(polys)
    pts = np.column_stack(
        [rng.uniform(-1.1, 1.1, 137), rng.uniform(-1.1, 1.1, 137)]
    )
    default = np.array(ms.contains_matrix(pts, polyset))
    assert default.shape == (137, 20)
    assert default.any(), "degenerate all-False matrix; chunk test is vacuous"

    # Force the retained non-Metal implementation. On the direct Metal path
    # chunk_bytes is intentionally inert because no N x E intermediate exists.
    monkeypatch.setattr("mlx_spatial._pip.metal_available", lambda: False)
    fallback = np.array(ms.contains_matrix(pts, polyset))
    np.testing.assert_array_equal(fallback, default)

    tiny = np.array(ms.contains_matrix(pts, polyset, chunk_bytes=1))
    np.testing.assert_array_equal(tiny, default)

    # A middling chunk size that does not divide 137 evenly (ragged last chunk).
    n_edges = int(polyset.edges.edges.shape[0])
    mid = np.array(
        ms.contains_matrix(pts, polyset, chunk_bytes=7 * n_edges * 4 * 5)
    )
    np.testing.assert_array_equal(mid, default)


@requires_metal
def test_contains_matrix_uses_composed_fallback_on_cpu_device(monkeypatch):
    """Metal availability does not imply the active MLX device is the GPU."""
    polyset = ms.PolygonSet(
        [shapely.box(0, 0, 1, 1), shapely.box(2, 2, 3, 3)]
    )
    pts = np.array([[0.5, 0.5], [2.5, 2.5], [4.0, 4.0]])
    _, points_local = polyset.recenter_points(pts)
    expected = np.array(_pip_dense_composed(points_local, polyset))

    def metal_must_not_run():
        raise AssertionError("CPU-device dense PIP attempted to compile Metal")

    monkeypatch.setattr("mlx_spatial._pip._get_dense_kernel", metal_must_not_run)
    old_device = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        got = np.array(pip_dense(points_local, polyset, chunk_bytes=1))
    finally:
        mx.set_default_device(old_device)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# Degenerate sizes: zero points, zero pairs, single point
# ---------------------------------------------------------------------------


def test_pip_pairs_zero_pairs():
    polyset = ms.PolygonSet([shapely.box(0, 0, 1, 1)])
    _, pts_local = polyset.recenter_points(np.array([[0.5, 0.5]]))
    out = pip_pairs(
        pts_local,
        polyset,
        mx.array(np.empty(0, dtype=np.int32)),
        mx.array(np.empty(0, dtype=np.int32)),
    )
    assert out.shape == (0,)
    assert out.dtype == mx.uint8


def test_pip_first_match_zero_points():
    polyset = ms.PolygonSet([shapely.box(0, 0, 1, 1)])
    _, pts_local = polyset.recenter_points(np.empty((0, 2)))
    out = pip_first_match(
        pts_local,
        polyset,
        mx.array(np.empty(0, dtype=np.int32)),
        mx.array(np.array([0, 1], dtype=np.int64)),
        mx.array(np.array([0], dtype=np.int32)),
    )
    assert out.shape == (0,)
    assert out.dtype == mx.int32


@requires_metal
def test_pip_all_matches_uses_int64_global_candidate_offsets():
    polyset = ms.PolygonSet([shapely.box(0, 0, 1, 1)])
    _, points_local = polyset.recenter_points(
        np.array([[0.5, 0.5], [2.0, 2.0]])
    )
    base = 2**31 + 17
    candidate_offsets = mx.array([base, base + 1], dtype=mx.int64)
    cell_offsets = mx.array([0, 1], dtype=mx.int64)
    cell_poly = mx.array([0], dtype=mx.int32)

    hit_pt, hit_pg = pip_all_matches(
        points_local,
        polyset,
        mx.array([0], dtype=mx.int32),
        mx.array([0], dtype=mx.int32),
        candidate_offsets,
        cell_offsets,
        cell_poly,
        pair_begin=base,
        pair_end=base + 1,
    )
    np.testing.assert_array_equal(np.array(hit_pt), [0])
    np.testing.assert_array_equal(np.array(hit_pg), [0])
    assert hit_pt.dtype == mx.int64 and hit_pg.dtype == mx.int64

    miss_pt, miss_pg = pip_all_matches(
        points_local,
        polyset,
        mx.array([1], dtype=mx.int32),
        mx.array([0], dtype=mx.int32),
        candidate_offsets,
        cell_offsets,
        cell_poly,
        pair_begin=base,
        pair_end=base + 1,
    )
    assert miss_pt.shape == miss_pg.shape == (0,)
    assert miss_pt.dtype == mx.int64 and miss_pg.dtype == mx.int64


def test_pip_dense_zero_points():
    polyset = ms.PolygonSet([shapely.box(0, 0, 1, 1), shapely.box(2, 2, 3, 3)])
    _, pts_local = polyset.recenter_points(np.empty((0, 2)))
    out = pip_dense(pts_local, polyset)
    assert out.shape == (0, 2)
    assert out.dtype == mx.bool_


def test_contains_matrix_zero_points():
    polyset = ms.PolygonSet([shapely.box(0, 0, 1, 1), shapely.box(2, 2, 3, 3)])
    out = ms.contains_matrix(np.empty((0, 2)), polyset)
    assert isinstance(out, np.ndarray)  # host input -> NumPy container
    assert out.shape == (0, 2)
    assert out.dtype == np.bool_


def test_contains_matrix_all_empty_polygons():
    polyset = ms.PolygonSet([shapely.Polygon(), shapely.Polygon()])
    out = ms.contains_matrix([[0.0, 0.0], [1.0, 1.0]], polyset, chunk_bytes=1)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    assert out.dtype == np.bool_
    assert not out.any()


def test_point_in_polygon_zero_points():
    polys = [shapely.box(0, 0, 1, 1)]
    backends = ["cpu"] + (["gpu"] if metal_available() else [])
    for backend in backends:
        res = ms.point_in_polygon(np.empty((0, 2)), polys, backend=backend)
        assert res.shape == (0,)
        pt_idx, poly_idx = ms.point_in_polygon(
            np.empty((0, 2)), polys, all_matches=True, backend=backend
        )
        assert len(pt_idx) == 0
        assert len(poly_idx) == 0


@requires_metal
def test_gpu_zero_candidate_pairs():
    """Points entirely outside the grid extent: zero pairs, all -1."""
    polys = [shapely.box(0, 0, 1, 1)]
    pts = np.array([[100.0, 100.0], [-50.0, 3.0], [0.5, 99.0]])
    res = ms.point_in_polygon(pts, polys, backend="gpu")
    np.testing.assert_array_equal(res, [-1, -1, -1])
    pt_idx, poly_idx = ms.point_in_polygon(
        pts, polys, all_matches=True, backend="gpu"
    )
    assert len(pt_idx) == 0
    assert len(poly_idx) == 0


def test_point_in_polygon_single_point():
    polyset = ms.PolygonSet([shapely.box(0, 0, 1, 1), shapely.box(2, 2, 3, 3)])
    backends = ["cpu"] + (["gpu"] if metal_available() else [])
    for backend in backends:
        # 1-D (x, y) list is promoted to a single-row point array
        got = ms.point_in_polygon([0.5, 0.5], polyset, backend=backend)
        assert got.shape == (1,)
        assert got[0] == 0
        # a bare shapely Point works too
        got = ms.point_in_polygon(shapely.Point(2.5, 2.5), polyset, backend=backend)
        np.testing.assert_array_equal(got, [1])
        # single miss
        got = ms.point_in_polygon(np.array([[5.0, 5.0]]), polyset, backend=backend)
        np.testing.assert_array_equal(got, [-1])


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_all_names_resolve_with_docstrings():
    assert ms.__doc__ and ms.__doc__.strip()
    for name in ms.__all__:
        obj = getattr(ms, name)  # AttributeError -> broken __all__
        if callable(obj):  # functions and classes
            assert obj.__doc__ and obj.__doc__.strip(), (
                f"{name} is public but has no docstring"
            )


def test_all_covers_expected_surface():
    expected = {
        "__version__",
        "PolygonSet",
        "LocalPoints",
        "PrecisionWarning",
        "as_point_array",
        "from_shapely",
        "contains_matrix",
        "R_EARTH_M",
        "haversine",
        "haversine_cross",
        "haversine_knn",
        "GridIndex",
        "point_in_polygon",
        "sjoin",
    }
    assert set(ms.__all__) == expected
    assert len(ms.__all__) == len(set(ms.__all__)), "duplicates in __all__"


def test_constants_have_sane_values():
    assert isinstance(ms.__version__, str) and ms.__version__
    assert isinstance(ms.R_EARTH_M, float)
    assert 6.3e6 < ms.R_EARTH_M < 6.4e6  # meters, mean Earth radius


def test_version_matches_pyproject():
    tomllib = pytest.importorskip("tomllib")  # stdlib from 3.11; package supports 3.10

    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == ms.__version__


def test_fresh_subprocess_import():
    """The package must import cleanly outside this process."""
    res = subprocess.run(
        [sys.executable, "-c", "import mlx_spatial; print(mlx_spatial.__version__)"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(Path(__file__).resolve().parent),
    )
    assert res.returncode == 0, f"import failed:\n{res.stderr}"
    assert res.stdout.strip() == ms.__version__
    assert "Traceback" not in res.stderr
