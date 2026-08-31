"""Deep tests for mlx_spatial.join: GridIndex construction, candidate
completeness, batching, backend dispatch, and edge-case geometry.

Complements test_smoke.py (which covers the public wrappers end-to-end) by
attacking the index machinery directly.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

import mlx_spatial as ms
import mlx_spatial._dispatch as dispatch
from mlx_spatial._dispatch import metal_available
from mlx_spatial._pip import pip_all_matches, pip_first_match
from mlx_spatial.join import GridIndex

from conftest import (
    make_convex_polygons,
    make_donut,
    make_grid_polygons,
    near_boundary_mask,
    reference_assign,
)

needs_metal = pytest.mark.skipif(not metal_available(), reason="Metal GPU required")


# ---------------------------------------------------------------------------
# helpers (local to this file per suite rules)
# ---------------------------------------------------------------------------


def true_pairs(points_xy: np.ndarray, geoms) -> set[tuple[int, int]]:
    """All (point, polygon) containment pairs via float64 shapely."""
    tree = shapely.STRtree(np.asarray(geoms, dtype=object))
    pt, pg = tree.query(
        shapely.points(np.asarray(points_xy, dtype=np.float64)), predicate="intersects"
    )
    return set(zip(pt.tolist(), pg.tolist()))


def restrict(pairs, keep_point: np.ndarray) -> set[tuple[int, int]]:
    """Drop pairs whose point index is masked out (tolerance band)."""
    return {(p, g) for p, g in pairs if keep_point[p]}


def assert_lexsorted_unique(pt_idx: np.ndarray, poly_idx: np.ndarray, n_polys: int):
    """Pairs strictly increasing under (point, polygon) lexicographic order."""
    key = pt_idx.astype(np.int64) * (n_polys + 1) + poly_idx.astype(np.int64)
    assert np.all(np.diff(key) > 0), "all_matches pairs not strictly lex-sorted"


def assert_matches_reference(index: GridIndex, geoms, pts: np.ndarray, backend: str):
    """First-match result equals the shapely golden outside the tolerance band."""
    got = np.asarray(index.query(pts, backend=backend))
    ref = reference_assign(pts, geoms)
    ok = ~near_boundary_mask(pts, geoms, index.polygons.boundary_tolerance)
    bad = np.flatnonzero((got != ref) & ok)
    assert bad.size == 0, f"{bad.size} mismatches outside band, e.g. point {bad[:5]}"


def skewed_polygons(rng) -> np.ndarray:
    """Varied sizes plus one giant polygon covering everything (skewed grid)."""
    small = make_convex_polygons(120, rng, radius_frac=0.02)
    big = make_convex_polygons(20, rng, radius_frac=0.15)
    giant = np.array([shapely.box(-1.2, -1.2, 1.2, 1.2)], dtype=object)
    return np.concatenate([small, big, giant])


# ---------------------------------------------------------------------------
# candidate completeness invariant
# ---------------------------------------------------------------------------


@needs_metal
class TestCandidateCompleteness:
    def test_random_varied_sizes(self, rng):
        polys = np.concatenate(
            [
                make_convex_polygons(60, rng, radius_frac=0.03),
                make_convex_polygons(60, rng, radius_frac=0.10),
            ]
        )
        pts = rng.uniform(-1.3, 1.3, size=(4000, 2))
        index = GridIndex(polys)
        gpt, gpg = index.query(pts, all_matches=True, backend="gpu")
        keep = ~near_boundary_mask(pts, polys, index.polygons.boundary_tolerance)
        ref = restrict(true_pairs(pts, polys), keep)
        got = restrict(set(zip(gpt.tolist(), gpg.tolist())), keep)
        missing = ref - got
        assert not missing, f"{len(missing)} true containment pairs missing: {sorted(missing)[:5]}"
        extra = got - ref
        assert not extra, f"{len(extra)} spurious pairs outside band: {sorted(extra)[:5]}"

    def test_skewed_giant_polygon(self, rng):
        """One polygon covering every cell + many small ones: worst-case skew."""
        polys = skewed_polygons(rng)
        giant_idx = len(polys) - 1
        pts = rng.uniform(-1.4, 1.4, size=(4000, 2))
        index = GridIndex(polys)
        gpt, gpg = index.query(pts, all_matches=True, backend="gpu")
        keep = ~near_boundary_mask(pts, polys, index.polygons.boundary_tolerance)
        ref = restrict(true_pairs(pts, polys), keep)
        got = restrict(set(zip(gpt.tolist(), gpg.tolist())), keep)
        assert ref - got == set(), f"missing pairs under skew: {sorted(ref - got)[:5]}"
        assert got - ref == set()
        # sanity: the giant really is matched by every kept in-extent point
        inside_giant = keep & (np.abs(pts) < 1.19).all(axis=1)
        matched_giant = np.zeros(len(pts), dtype=bool)
        matched_giant[gpt[gpg == giant_idx]] = True
        assert matched_giant[inside_giant].all()

    def test_overlapping_lattice_exact(self, rng):
        """Overlapping 1.5x1.5 boxes on a unit lattice; points keep >=0.2
        clearance from every edge, so pair sets must match exactly."""
        polys = np.array(
            [shapely.box(i, j, i + 1.5, j + 1.5) for j in range(8) for i in range(8)],
            dtype=object,
        )
        ij = rng.integers(0, 9, size=(3000, 2)).astype(np.float64)
        offs = rng.choice([0.2, 0.7], size=(3000, 2))
        pts = ij + offs  # never within 0.2 of any integer or half-integer edge
        index = GridIndex(polys)
        gpt, gpg = index.query(pts, all_matches=True, backend="gpu")
        got = set(zip(gpt.tolist(), gpg.tolist()))
        ref = true_pairs(pts, polys)
        assert got == ref
        # overlap really occurred (multiple matches for some points)
        assert len(gpt) > len(np.unique(gpt))

    def test_completeness_survives_forced_coarsening(self, rng):
        """A grid coarsened down to a tight entry budget must not lose hits."""
        polys = skewed_polygons(rng)
        pts = rng.uniform(-1.3, 1.3, size=(2000, 2))
        index = GridIndex(polys, cell_size=1e-4, max_entries=2000)
        assert len(index._cell_poly) <= 2000
        gpt, gpg = index.query(pts, all_matches=True, backend="gpu")
        keep = ~near_boundary_mask(pts, polys, index.polygons.boundary_tolerance)
        ref = restrict(true_pairs(pts, polys), keep)
        got = restrict(set(zip(gpt.tolist(), gpg.tolist())), keep)
        assert ref - got == set()
        assert got - ref == set()


# ---------------------------------------------------------------------------
# GridIndex parameters
# ---------------------------------------------------------------------------


class TestGridIndexParameters:
    def test_tiny_cell_size_coarsens_to_budgets(self, rng):
        polys = make_convex_polygons(50, rng)
        index = GridIndex(polys, cell_size=1e-4, max_entries=1000)
        assert index.n_cells <= 4_194_304
        assert len(index._cell_poly) <= 1000
        assert index.cell_size > 1e-4  # actually coarsened
        assert index.nx >= 1 and index.ny >= 1
        pts = rng.uniform(-1.2, 1.2, size=(800, 2))
        assert_matches_reference(index, polys, pts, backend="cpu")
        if metal_available():
            assert_matches_reference(index, polys, pts, backend="gpu")

    def test_huge_cell_size_single_cell(self, rng):
        polys = make_convex_polygons(40, rng)
        index = GridIndex(polys, cell_size=1e6)
        assert (index.nx, index.ny) == (1, 1)
        assert index.n_cells == 1
        # single cell holds every polygon, ascending
        np.testing.assert_array_equal(index._cell_poly, np.arange(40, dtype=np.int32))
        pts = rng.uniform(-1.2, 1.2, size=(500, 2))
        assert_matches_reference(index, polys, pts, backend="cpu")
        if metal_available():
            assert_matches_reference(index, polys, pts, backend="gpu")

    def test_max_cells_budget_respected(self, rng):
        polys = make_convex_polygons(50, rng)
        for budget in (1, 4, 16, 100):
            index = GridIndex(polys, max_cells=budget)
            assert index.n_cells <= budget
        index = GridIndex(polys, max_cells=16)
        pts = rng.uniform(-1.2, 1.2, size=(500, 2))
        assert_matches_reference(index, polys, pts, backend="cpu")

    def test_repr(self, rng):
        polys = make_convex_polygons(7, rng)
        r = repr(GridIndex(polys, cell_size=0.25))
        assert "GridIndex" in r
        assert "nx=" in r and "ny=" in r and "polygons=7" in r

    def test_invalid_parameters_raise(self, rng):
        polys = make_convex_polygons(10, rng)
        with pytest.raises(ValueError):
            GridIndex(polys, cell_size=0.0)
        with pytest.raises(ValueError):
            GridIndex(polys, cell_size=-1.0)
        # entry budget below one entry per polygon can never be met
        with pytest.raises(ValueError):
            GridIndex(polys, max_entries=1)


# ---------------------------------------------------------------------------
# query batching
# ---------------------------------------------------------------------------


@needs_metal
class TestBatching:
    def test_batched_equals_unbatched(self, rng):
        polys = make_grid_polygons(10, 10)
        pts = rng.uniform(-0.5, 10.5, size=(6000, 2))
        index = GridIndex(polys)
        total_pairs = index._candidates(np.asarray(pts, dtype=np.float64))[2]
        assert total_pairs > 10_000  # workload really is >> 1000 pairs

        base = np.asarray(index.query(pts, backend="gpu"))
        base_pt, base_pg = index.query(pts, all_matches=True, backend="gpu")
        for batch in (1000, 997):  # aligned-ish and deliberately misaligned
            got = np.asarray(index.query(pts, backend="gpu", max_pairs_per_batch=batch))
            np.testing.assert_array_equal(got, base)
            gpt, gpg = index.query(
                pts, all_matches=True, backend="gpu", max_pairs_per_batch=batch
            )
            np.testing.assert_array_equal(gpt, base_pt)
            np.testing.assert_array_equal(gpg, base_pg)

    def test_oversized_point_lists_respect_hard_cap(self, monkeypatch):
        """Dense points split at the cap despite intervening zero-count rows."""
        polys = np.array([shapely.box(0, 0, 1, 1) for _ in range(17)], dtype=object)
        index = GridIndex(polys, cell_size=2.0)
        points = [[-1.0, -1.0], [0.5, 0.5], [9.0, 9.0], [0.25, 0.25], [2.0, 2.0]]
        calls = []

        def all_hit(_points, _polyset, pair_point, _pair_poly):
            calls.append(int(pair_point.shape[0]))
            return mx.ones(pair_point.shape, dtype=mx.uint8)

        monkeypatch.setattr("mlx_spatial.join.pip_pairs", all_hit)
        got = np.asarray(
            index.query(points, backend="gpu", max_pairs_per_batch=4)
        )
        np.testing.assert_array_equal(got, [-1, 0, -1, 0, -1])
        assert calls == [4] * 8 + [2]

        calls.clear()
        compact_calls = []

        def compact_spy(*args, pair_begin, pair_end, **kwargs):
            compact_calls.append(pair_end - pair_begin)
            return pip_all_matches(
                *args,
                pair_begin=pair_begin,
                pair_end=pair_end,
                **kwargs,
            )

        monkeypatch.setattr("mlx_spatial.join.pip_all_matches", compact_spy)
        point_idx, poly_idx = index.query(
            points,
            backend="gpu",
            all_matches=True,
            max_pairs_per_batch=4,
        )
        assert compact_calls == [4] * 8 + [2]
        assert calls == [], "all_matches fell back to explicit candidate pairs"
        np.testing.assert_array_equal(
            point_idx,
            np.repeat(np.array([1, 3], dtype=np.int64), 17),
        )
        np.testing.assert_array_equal(
            poly_idx,
            np.tile(np.arange(17, dtype=np.int64), 2),
        )

    def test_fused_whole_point_batches_avoid_pair_expansion(self, rng, monkeypatch):
        polys = make_grid_polygons(10, 10)
        pts = rng.uniform(-0.5, 10.5, size=(6000, 2))
        index = GridIndex(polys)
        expected = np.asarray(index.query(pts, backend="gpu"))
        total_pairs = index._candidates(pts)[2]
        calls: list[int] = []
        candidate_work: list[int] = []

        def fused_spy(points_local, polyset, point_cell, offsets, cell_poly, **kwargs):
            calls.append(int(points_local.shape[0]))
            cells = np.array(point_cell)
            csr_offsets = np.array(offsets)
            valid = cells >= 0
            candidate_work.append(
                int(
                    np.sum(
                        csr_offsets[cells[valid] + 1] - csr_offsets[cells[valid]]
                    )
                )
            )
            return pip_first_match(
                points_local,
                polyset,
                point_cell,
                offsets,
                cell_poly,
                **kwargs,
            )

        def pairs_must_not_run(*args, **kwargs):
            raise AssertionError("ordinary first-match query expanded candidate pairs")

        monkeypatch.setattr("mlx_spatial.join.pip_first_match", fused_spy)
        monkeypatch.setattr("mlx_spatial.join.pip_pairs", pairs_must_not_run)
        got = np.asarray(
            index.query(pts, backend="gpu", max_pairs_per_batch=1000)
        )
        assert len(calls) > 1, "candidate-work cap did not split fused launches"
        assert max(candidate_work) <= 1000
        assert sum(candidate_work) == total_pairs
        np.testing.assert_array_equal(got, expected)

    def test_all_matches_sparse_split_batches_skip_long_zero_runs(self, monkeypatch):
        """Split points and zero-hit chunks retain global lexicographic order."""
        triangle = Polygon([(0, 0), (2, 0), (0, 2)])
        polys = np.array([triangle for _ in range(17)], dtype=object)
        index = GridIndex(polys, cell_size=3.0)
        outside = np.full((100_005, 2), 9.0)
        points = np.vstack([[0.25, 0.25], outside, [1.75, 1.75], [0.5, 0.5]])
        calls: list[int] = []

        def compact_spy(*args, pair_begin, pair_end, **kwargs):
            calls.append(pair_end - pair_begin)
            return pip_all_matches(
                *args,
                pair_begin=pair_begin,
                pair_end=pair_end,
                **kwargs,
            )

        def pairs_must_not_run(*args, **kwargs):
            raise AssertionError("all_matches expanded explicit candidate pairs")

        monkeypatch.setattr("mlx_spatial.join.pip_all_matches", compact_spy)
        monkeypatch.setattr("mlx_spatial.join.pip_pairs", pairs_must_not_run)
        point_idx, poly_idx = index.query(
            points,
            backend="gpu",
            all_matches=True,
            max_pairs_per_batch=4,
        )

        assert calls == [4] * 12 + [3]
        expected_points = np.repeat(
            np.array([0, len(points) - 1], dtype=np.int64), 17
        )
        np.testing.assert_array_equal(point_idx, expected_points)
        np.testing.assert_array_equal(
            poly_idx, np.tile(np.arange(17, dtype=np.int64), 2)
        )

    def test_fused_matches_pair_reduction_on_mixed_geometry(self, monkeypatch):
        donut = Polygon(
            [(0, 0), (6, 0), (6, 6), (0, 6)],
            holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]],
        )
        multi = MultiPolygon(
            [shapely.box(10, 0, 12, 2), shapely.box(14, 0, 16, 2)]
        )
        polys = np.array(
            [
                donut,
                Polygon(),
                shapely.box(2.5, 2.5, 3.5, 3.5),
                shapely.box(1, -1, 5, 7),
                multi,
            ],
            dtype=object,
        )
        pts = np.array(
            [
                [3, 3],
                [1.5, 1.5],
                [3.8, 3],
                [11, 1],
                [13, 1],
                [15, 1],
                [100, 100],
                [np.nan, np.nan],
                [np.nan, 1],
                [2, 3],
            ]
        )
        index = GridIndex(polys, cell_size=2.0)
        counts, _, total_pairs, _ = index._candidates(pts)
        assert total_pairs > 3 and counts.max() <= 3

        pt_idx, poly_idx = index.query(pts, all_matches=True, backend="gpu")
        reduced = np.full(len(pts), -1, dtype=np.int32)
        if len(pt_idx):
            first_pt, first_at = np.unique(pt_idx, return_index=True)
            reduced[first_pt] = poly_idx[first_at].astype(np.int32)

        def pairs_must_not_run(*args, **kwargs):
            raise AssertionError("fused first-match expanded candidate pairs")

        monkeypatch.setattr("mlx_spatial.join.pip_pairs", pairs_must_not_run)
        fused = np.asarray(
            index.query(pts, backend="gpu", max_pairs_per_batch=3)
        )
        np.testing.assert_array_equal(fused, reduced)
        np.testing.assert_array_equal(fused, [2, 0, 3, 4, -1, 4, -1, -1, -1, 3])


# ---------------------------------------------------------------------------
# backend equivalence and dispatch
# ---------------------------------------------------------------------------


@needs_metal
class TestBackends:
    def test_cpu_gpu_agree_outside_band(self, rng):
        polys = skewed_polygons(rng)
        pts = rng.uniform(-1.4, 1.4, size=(3000, 2))
        index = GridIndex(polys)
        cpu = np.asarray(index.query(pts, backend="cpu"))
        gpu = np.asarray(index.query(pts, backend="gpu"))
        assert cpu.dtype == np.int32 and gpu.dtype == np.int32
        keep = ~near_boundary_mask(pts, polys, index.polygons.boundary_tolerance)
        np.testing.assert_array_equal(cpu[keep], gpu[keep])
        cpt, cpg = index.query(pts, all_matches=True, backend="cpu")
        gpt, gpg = index.query(pts, all_matches=True, backend="gpu")
        assert restrict(set(zip(cpt.tolist(), cpg.tolist())), keep) == restrict(
            set(zip(gpt.tolist(), gpg.tolist())), keep
        )

    def test_auto_dispatch_threshold(self, rng, monkeypatch):
        polys = make_convex_polygons(30, rng)
        index = GridIndex(polys)
        pts = rng.uniform(-1, 1, size=(400, 2))

        calls = []
        orig = GridIndex._query_cpu

        def spy(self, pts64, *, all_matches, predicate="intersects"):
            calls.append(1)
            return orig(self, pts64, all_matches=all_matches, predicate=predicate)

        monkeypatch.setattr(GridIndex, "_query_cpu", spy)
        monkeypatch.delenv("MLX_SPATIAL_BACKEND", raising=False)

        # tiny workload, real default threshold: auto must pick the CPU path
        assert index._candidates(np.asarray(pts, dtype=np.float64))[2] < (
            dispatch.GPU_MIN_CANDIDATE_PAIRS
        )
        r_default = np.asarray(index.query(pts))
        assert len(calls) == 1

        # threshold forced high: still CPU
        monkeypatch.setattr(dispatch, "GPU_MIN_CANDIDATE_PAIRS", 10**18)
        index.query(pts)
        assert len(calls) == 2

        # threshold forced to zero: auto must take the GPU path (no CPU call)
        monkeypatch.setattr(dispatch, "GPU_MIN_CANDIDATE_PAIRS", 0)
        r_gpu = np.asarray(index.query(pts))
        assert len(calls) == 2
        keep = ~near_boundary_mask(pts, polys, index.polygons.boundary_tolerance)
        np.testing.assert_array_equal(r_default[keep], r_gpu[keep])

    def test_env_var_override(self, rng, monkeypatch):
        polys = make_convex_polygons(20, rng)
        index = GridIndex(polys)
        pts = rng.uniform(-1, 1, size=(200, 2))

        calls = []
        orig = GridIndex._query_cpu

        def spy(self, pts64, *, all_matches, predicate="intersects"):
            calls.append(1)
            return orig(self, pts64, all_matches=all_matches, predicate=predicate)

        monkeypatch.setattr(GridIndex, "_query_cpu", spy)
        monkeypatch.setattr(dispatch, "GPU_MIN_CANDIDATE_PAIRS", 0)
        monkeypatch.setenv("MLX_SPATIAL_BACKEND", "cpu")
        index.query(pts)
        assert len(calls) == 1  # env var forced CPU despite zero threshold

        # explicit backend= argument beats the env var
        monkeypatch.setenv("MLX_SPATIAL_BACKEND", "gpu")
        index.query(pts, backend="cpu")
        assert len(calls) == 2

        monkeypatch.setenv("MLX_SPATIAL_BACKEND", "quantum")
        with pytest.raises(ValueError):
            index.query(pts)

    def test_explicit_gpu_works_with_cpu_default_device(self):
        """backend='gpu' selects a GPU stream even when MLX defaults to CPU."""
        old_device = mx.default_device()
        try:
            mx.set_default_device(mx.cpu)
            polys = np.array(
                [shapely.box(0, 0, 2, 2), shapely.box(1, 1, 3, 3)],
                dtype=object,
            )
            index = GridIndex(polys)
            pts = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [9.0, 9.0]])
            np.testing.assert_array_equal(
                index.query(pts, backend="gpu"), [0, 0, 1, -1]
            )
            pt_idx, poly_idx = index.query(pts, backend="gpu", all_matches=True)
            np.testing.assert_array_equal(pt_idx, [0, 1, 1, 2])
            np.testing.assert_array_equal(poly_idx, [0, 0, 1, 1])
        finally:
            mx.set_default_device(old_device)

    def test_device_csr_is_lazy_and_reused(self):
        index = GridIndex(
            np.array([shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)])
        )
        pts = np.array([[0.5, 0.5], [1.5, 0.5]])
        assert index._cell_offsets_mx is None and index._cell_poly_mx is None
        index.query(pts, backend="cpu")
        assert index._cell_offsets_mx is None and index._cell_poly_mx is None

        index.query(pts, backend="gpu")
        offsets_mx = index._cell_offsets_mx
        poly_mx = index._cell_poly_mx
        assert offsets_mx is not None and poly_mx is not None
        index.query(pts, backend="gpu")
        assert index._cell_offsets_mx is offsets_mx
        assert index._cell_poly_mx is poly_mx


# ---------------------------------------------------------------------------
# edge-case geometry
# ---------------------------------------------------------------------------


class TestEdgeGeometry:
    def test_all_points_outside_grid(self, rng):
        polys = make_convex_polygons(20, rng, extent=(0.0, 0.0, 1.0, 1.0))
        index = GridIndex(polys)
        pts = rng.uniform(10.0, 20.0, size=(300, 2))
        for backend in (["cpu", "gpu"] if metal_available() else ["cpu"]):
            got = np.asarray(index.query(pts, backend=backend))
            np.testing.assert_array_equal(got, np.full(300, -1, dtype=np.int32))
            pt_idx, poly_idx = index.query(pts, all_matches=True, backend=backend)
            assert len(pt_idx) == 0 and len(poly_idx) == 0
            assert pt_idx.dtype == np.int64 and poly_idx.dtype == np.int64

    def test_zero_width_bbox_polygons(self):
        """Vertical segment-like degenerate polygons: no crash, sane grid."""
        slivers = np.array(
            [
                Polygon([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]),
                Polygon([(0.5, 0.0), (0.5, 1.0), (0.5, 2.0)]),
            ],
            dtype=object,
        )
        index = GridIndex(slivers)
        assert index.nx >= 1 and index.ny >= 1
        assert np.isfinite(index.cell_size) and index.cell_size > 0
        pts = np.array([[0.0, 1.0], [0.25, 1.0], [9.0, 9.0]])
        cpu = np.asarray(index.query(pts, backend="cpu"))
        assert cpu[1] == -1 and cpu[2] == -1  # off-sliver points never match
        if metal_available():
            gpu = np.asarray(index.query(pts, backend="gpu"))
            assert gpu[1] == -1 and gpu[2] == -1
            assert gpu[0] in (-1, 0)  # on the degenerate boundary: band, either way

    def test_zero_width_grid_extent(self):
        """Every bbox at the same x: total grid width is exactly zero."""
        slivers = np.array(
            [
                Polygon([(1.0, 0.0), (1.0, 1.0), (1.0, 2.0)]),
                Polygon([(1.0, 3.0), (1.0, 4.0), (1.0, 5.0)]),
            ],
            dtype=object,
        )
        index = GridIndex(slivers)
        assert index.nx == 1 and index.ny >= 1
        assert index.n_cells <= 4_194_304
        pts = np.array([[0.9, 1.0], [1.1, 4.0], [1.0, 10.0]])
        for backend in (["cpu", "gpu"] if metal_available() else ["cpu"]):
            got = np.asarray(index.query(pts, backend=backend))
            np.testing.assert_array_equal(got, [-1, -1, -1])

    def test_single_polygon(self, rng):
        donut = make_donut(0.0, 0.0, 1.0, 0.5, k=64)
        index = GridIndex(np.array([donut], dtype=object))
        pts = np.array([[0.75, 0.0], [0.0, 0.0], [2.0, 2.0], [0.0, -0.75]])
        expected = np.array([0, -1, -1, 0], dtype=np.int32)  # hole center excluded
        np.testing.assert_array_equal(np.asarray(index.query(pts, backend="cpu")), expected)
        if metal_available():
            np.testing.assert_array_equal(
                np.asarray(index.query(pts, backend="gpu")), expected
            )

    def test_empty_polygon_mixed_in(self):
        polys = np.array(
            [shapely.box(0, 0, 1, 1), Polygon(), shapely.box(2, 2, 3, 3)], dtype=object
        )
        index = GridIndex(polys)
        pts = np.array([[0.5, 0.5], [2.5, 2.5], [5.0, 5.0]])
        for backend in (["cpu", "gpu"] if metal_available() else ["cpu"]):
            got = np.asarray(index.query(pts, backend=backend))
            np.testing.assert_array_equal(got, [0, 2, -1])
            pt_idx, poly_idx = index.query(pts, all_matches=True, backend=backend)
            assert 1 not in poly_idx  # the empty polygon never matches
            assert set(zip(pt_idx.tolist(), poly_idx.tolist())) == {(0, 0), (1, 2)}

    def test_all_empty_polygons_degenerate_grid(self):
        """All-empty sets build a degenerate grid: every query is unmatched."""
        index = GridIndex(np.array([Polygon(), Polygon()], dtype=object))
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [np.nan, np.nan]])
        for backend in (["cpu", "gpu"] if metal_available() else ["cpu"]):
            np.testing.assert_array_equal(
                np.asarray(index.query(pts, backend=backend)), [-1, -1, -1]
            )
            pt_idx, poly_idx = index.query(pts, all_matches=True, backend=backend)
            assert len(pt_idx) == 0 and len(poly_idx) == 0


# ---------------------------------------------------------------------------
# all_matches output contract
# ---------------------------------------------------------------------------


class TestAllMatchesOutput:
    @pytest.fixture
    def overlap_workload(self, rng):
        polys = np.concatenate(
            [
                make_convex_polygons(40, rng, radius_frac=0.15),
                np.array([shapely.box(-1.1, -1.1, 1.1, 1.1)], dtype=object),
            ]
        )
        pts = rng.uniform(-1.2, 1.2, size=(2500, 2))
        return polys, pts, GridIndex(polys)

    @pytest.mark.parametrize("backend", ["cpu", "gpu"])
    def test_dtype_sorted_first_match_consistent(self, overlap_workload, backend):
        if backend == "gpu" and not metal_available():
            pytest.skip("Metal GPU required")
        polys, pts, index = overlap_workload
        pt_idx, poly_idx = index.query(pts, all_matches=True, backend=backend)

        assert pt_idx.dtype == np.int64
        assert poly_idx.dtype == np.int64
        assert len(pt_idx) == len(poly_idx)
        assert len(pt_idx) > len(np.unique(pt_idx))  # overlaps present -> meaningful
        assert_lexsorted_unique(pt_idx, poly_idx, len(polys))
        assert poly_idx.min() >= 0 and poly_idx.max() < len(polys)
        assert pt_idx.min() >= 0 and pt_idx.max() < len(pts)

        # min-per-point reduction of all_matches == first-match result
        first = np.asarray(index.query(pts, backend=backend))
        reduced = np.full(len(pts), -1, dtype=np.int64)
        # pairs are lex-sorted, so the first pair per point is its minimum
        seen_first = np.unique(pt_idx, return_index=True)
        reduced[seen_first[0]] = poly_idx[seen_first[1]]
        np.testing.assert_array_equal(reduced, first.astype(np.int64))


# ---------------------------------------------------------------------------
# inclusive in-grid test at the max corner
# ---------------------------------------------------------------------------


@needs_metal
class TestGridMaxCorner:
    def test_point_on_max_corner_is_candidate(self):
        """A point exactly at (x1, y1) maps into the last cell (inclusive
        bounds) and the corner-touching polygon must appear as a candidate.
        Containment itself is boundary-band-undefined on GPU, but the grid
        must not drop the point."""
        polys = np.array(
            [shapely.box(0, 0, 4, 4), shapely.box(4, 4, 10, 10)], dtype=object
        )
        index = GridIndex(polys)
        assert (index.x1, index.y1) == (10.0, 10.0)
        corner = np.array([[10.0, 10.0], [10.0, 7.0]])  # max corner + right-edge point
        counts, starts, total, _ = index._candidates(corner)
        assert counts[0] >= 1, "point at exact grid max corner was dropped"
        assert counts[1] >= 1, "point at x == x1 was dropped"
        for i in range(2):
            cands = index._cell_poly[starts[i] : starts[i] + counts[i]]
            assert 1 in cands, "corner-touching polygon missing from candidates"

        # full GPU query must run and only ever report the touching polygon
        got = np.asarray(index.query(corner, backend="gpu"))
        assert got[0] in (-1, 1) and got[1] in (-1, 1)
        # CPU semantics (shapely intersects) deterministically match boundaries
        np.testing.assert_array_equal(np.asarray(index.query(corner, backend="cpu")), [1, 1])

    def test_point_just_inside_corner_found_on_gpu(self):
        polys = np.array(
            [shapely.box(0, 0, 4, 4), shapely.box(4, 4, 10, 10)], dtype=object
        )
        index = GridIndex(polys)
        eps = 1e-3  # >> boundary_tolerance (~8 ulps of ~5), < cell size
        assert eps > 10 * index.polygons.boundary_tolerance
        inside = np.array([[10.0 - eps, 10.0 - eps]])
        np.testing.assert_array_equal(np.asarray(index.query(inside, backend="gpu")), [1])
        # just past the corner: outside the grid entirely -> no match
        outside = np.array([[10.0 + eps, 10.0], [10.0, 10.0 + eps]])
        np.testing.assert_array_equal(
            np.asarray(index.query(outside, backend="gpu")), [-1, -1]
        )


@needs_metal
def test_fused_join_uses_float64_planned_cell_at_rounding_boundary():
    """The fused kernel must consume the CPU-planned cell, not recompute it.

    The point lies just left of x=1 in float64, but rounds to exactly x=1 in
    the kernel's local float32 coordinates. Recomputing its cell on Metal
    would incorrectly expose the right-hand polygon as a candidate.
    """
    polys = np.array(
        [shapely.box(0.0, 0.0, 0.5, 1.0), shapely.box(1.0, 0.0, 2.0, 1.0)],
        dtype=object,
    )
    polyset = ms.PolygonSet(polys, origin=(0.0, 0.0))
    index = GridIndex(polyset, cell_size=1.0)
    x = 1.0 - float(np.finfo(np.float32).eps) / 4.0
    pts = np.array([[x, 0.5]])
    assert x < 1.0 and np.float32(x) == np.float32(1.0)
    counts, starts, total, point_cell = index._candidates(pts)
    assert total == counts[0] == 1 and point_cell[0] == 0
    np.testing.assert_array_equal(
        index._cell_poly[starts[0] : starts[0] + counts[0]], [0]
    )

    first = np.asarray(index.query(pts, backend="gpu"))
    pt_idx, poly_idx = index.query(pts, backend="gpu", all_matches=True)
    assert len(pt_idx) == len(poly_idx) == 0
    np.testing.assert_array_equal(first, [-1])
