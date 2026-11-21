#!/usr/bin/env python3
"""Public-API benchmarks for mlx-geospatial.

Unlike ``bench/e2e/run_e2e.py`` (which stays frozen with its separate
query-plus-assembly adapter for cross-revision comparability), this script
times the public ``ms.sjoin``, ``ms.PreparedJoin``, ``ms.HaversineReference``,
and detailed-polygon first-match paths directly.

Usage:
    python bench/bench_public_api.py
    python bench/bench_public_api.py --suite sjoin
    python bench/bench_public_api.py --suite pip
    python bench/bench_public_api.py --suite knn
    python bench/bench_public_api.py --suite batching
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import shapely
import mlx.core as mx

import mlx_geospatial as ms
from mlx_geospatial._edge_index import get_edge_index
import bench


def _median_ms(fn, *, warmup: int = 1, repeats: int = 5) -> float:
    for _ in range(warmup):
        out = fn()
        del out
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
        del out
    return statistics.median(samples)


def bench_pip_detailed(seed: int = 0) -> None:
    """First-match latency curve on 260 x 2,048-vertex zones (4,096 -> 4,097+)."""
    print("\n### 1. First-match detailed polygons (260 zones x 2,048 vertices)")
    rng = np.random.default_rng(seed)
    zones = bench.densify_zones(bench.make_zones(260, rng), 2048)
    # Keep the original 100k-point prefix identical to earlier runs.
    pts_all = np.concatenate([bench.make_points(100_000, rng), bench.make_points(900_000, rng)])

    ps_cold = ms.PolygonSet(zones)
    idx_cold = ms.GridIndex(ps_cold)
    ps_warm = ms.PolygonSet(zones)
    idx_warm = ms.GridIndex(ps_warm)
    # Make the warm measurements independent of the cold dispatch policy.
    assert get_edge_index(ps_warm) is not None
    idx_warm.query(pts_all[:65_536], backend="gpu")
    mx.synchronize()

    # Ramp GPU clocks before measuring.
    idx_cold.query(pts_all[:1024], backend="gpu")
    mx.synchronize()

    print(f"  {'points':>9}  {'cold edge cache':>18}  {'warm edge cache':>16}  cold route")
    for n in (
        1_024, 2_048, 4_096, 4_097, 8_192, 16_384, 32_768, 50_000, 65_536,
        100_000, 262_144, 524_288, 1_000_000,
    ):
        pts = pts_all[:n]

        def run_cold():
            ps_cold._y_edge_index = False
            ps_cold._y_edge_index_work = 0
            return idx_cold.query(pts, backend="gpu")

        t_cold = _median_ms(run_cold, warmup=2, repeats=9)
        t_warm = _median_ms(lambda: idx_warm.query(pts, backend="gpu"), warmup=2, repeats=9)
        np.testing.assert_array_equal(run_cold(), idx_warm.query(pts, backend="gpu"))
        route = "build index" if ps_cold._y_edge_index else "traverse"
        print(f"  {n:>9,}  {t_cold:>15.2f} ms  {t_warm:>13.2f} ms  {route}")

    # A new index gets the same medium batch repeatedly. Include the build in
    # the timings and check that crossing the build threshold keeps the labels.
    repeated_polygons = ms.PolygonSet(zones)
    repeated_index = ms.GridIndex(repeated_polygons)
    medium = pts_all[:32_768]
    expected = idx_warm.query(medium, backend="gpu")
    first_build = None
    timings = []
    for batch in range(1, 97):
        start = time.perf_counter()
        labels = repeated_index.query(medium, backend="gpu")
        timings.append((time.perf_counter() - start) * 1e3)
        if first_build is None and repeated_polygons._y_edge_index:
            first_build = batch
            np.testing.assert_array_equal(labels, expected)
    np.testing.assert_array_equal(labels, expected)
    build = "unbuilt"
    if first_build is not None:
        build = f"built on batch {first_build} ({timings[first_build - 1]:.2f} ms)"
    print(
        f"  96 × 32,768 points: {sum(timings):.2f} ms total, index {build}; "
        f"last 10 median {statistics.median(timings[-10:]):.2f} ms; equality=PASS"
    )


def bench_all_match_batches(seed: int = 0) -> None:
    """Measure the planner with single and split GPU batches on identical inputs."""
    rng = np.random.default_rng(seed)
    zones = bench.make_zones(260, rng)
    points = bench.make_points(100_000, rng)
    index = ms.GridIndex(zones)
    expected = index.query(points, all_matches=True, backend="cpu")

    print("\n### 4. All-match batching (100k points x 260 zones, 48 vertices)")
    for cap in (None, 65_536, 8_192, 2_048):
        def run():
            return index.query(points, all_matches=True, backend="gpu", max_pairs_per_batch=cap)

        elapsed = _median_ms(run)
        for actual, target in zip(run(), expected, strict=True):
            np.testing.assert_array_equal(actual, target)
        label = "default" if cap is None else f"{cap:,}"
        print(f"  cap {label:>7}: {elapsed:6.2f} ms, {len(expected[0]):,} pairs, equality=PASS")


def _make_geoframes(
    n: int,
    m: int,
    rng: np.random.Generator,
    *,
    vertices: int = 48,
    extra_left: int = 0,
    extra_right: int = 0,
):
    import geopandas as gpd
    from geopandas.array import GeometryArray
    import pandas as pd

    pts = bench.make_points(n, rng)
    zones = bench.make_zones(m, rng, vertices=48)
    if vertices != 48:
        zones = bench.densify_zones(zones, vertices)
    pid = np.arange(n, dtype=np.int64)
    zid = np.arange(len(zones), dtype=np.int64)
    left_data: dict[str, object] = {
        "point_id": pid,
        "value": (pid % 1009).astype(np.float64) / 1009.0,
    }
    for i in range(extra_left):
        if i % 3 == 0:
            left_data[f"lf_{i}"] = (pid % (100 + i)).astype(np.float64)
        elif i % 3 == 1:
            left_data[f"li_{i}"] = pd.array(pid % (50 + i), dtype="Int64")
        else:
            left_data[f"ls_{i}"] = pd.Categorical.from_codes(
                pid % 5, categories=["a", "b", "c", "d", "e"]
            )
    left = gpd.GeoDataFrame(
        left_data,
        geometry=GeometryArray(
            np.asarray(shapely.points(pts), dtype=object), crs="EPSG:4326"
        ),
        index=pd.Index(pid * 2 + 101),
        crs="EPSG:4326",
    )
    right_data: dict[str, object] = {
        "zone_id": zid,
        "zone_name": pd.array([f"zone_{i:05d}" for i in zid], dtype="string"),
        "zone_weight": zid.astype(np.float64) * 0.125 + 0.5,
    }
    for i in range(extra_right):
        if i % 4 == 0:
            right_data[f"rf_{i}"] = zid.astype(np.float64) + i
        elif i % 4 == 1:
            right_data[f"ri_{i}"] = pd.array(zid + i, dtype="Int64")
        elif i % 4 == 2:
            right_data[f"rc_{i}"] = pd.Categorical.from_codes(
                zid % 4, categories=["w", "x", "y", "z"]
            )
        else:
            right_data[f"rt_{i}"] = pd.date_range(
                "2026-01-01", periods=len(zones), freq="h", tz="UTC"
            )
    right = gpd.GeoDataFrame(
        right_data,
        geometry=GeometryArray(np.asarray(zones, dtype=object), crs="EPSG:4326"),
        index=pd.Index(zid * 3 + 1001),
        crs="EPSG:4326",
    )
    return left, right


def bench_sjoin_public(seed: int = 0, *, raw_only: bool = False) -> None:
    """Public ms.sjoin (cold and PreparedJoin) on narrow and wide GeoDataFrames."""
    import geopandas as gpd
    from geopandas.testing import assert_geodataframe_equal

    print("\n### 2. Public ms.sjoin & PreparedJoin (exact_boundary=True, predicate='within')")
    rng = np.random.default_rng(seed)
    cases = [
        ("1M x 260 (48v, narrow, inner, all)", 1_000_000, 260, 48, 0, 0, "inner", True),
        ("1M x 260 (48v, narrow, left, all)", 1_000_000, 260, 48, 0, 0, "left", True),
        ("1M x 260 (48v, wide, inner, all)", 1_000_000, 260, 48, 10, 12, "inner", True),
        ("1M x 260 (2048v, narrow, inner, all)", 1_000_000, 260, 2048, 0, 0, "inner", True),
        ("100k x 260 (2048v, narrow, inner, all)", 100_000, 260, 2048, 0, 0, "inner", True),
        ("1M x 10k (48v, narrow, inner, all)", 1_000_000, 10_000, 48, 0, 0, "inner", True),
        ("1M x 260 (2048v, narrow, inner, first)", 1_000_000, 260, 2048, 0, 0, "inner", False),
    ]
    print(
        f"  {'case':<40}  {'cold':>8}  {'prep':>7}  {'warm1':>8}  {'warm':>8}  {'ratio':>6}  {'gpd':>9}"
    )
    for label, n, m, v, el, er, how, all_m in cases:
        gc.collect()
        left, right = _make_geoframes(n, m, rng, vertices=v, extra_left=el, extra_right=er)

        def run_cold():
            return ms.sjoin(
                left, right, how=how, predicate="within",
                all_matches=all_m, exact_boundary=True, backend="gpu",
            )

        def run_prepared():
            return ms.sjoin(
                left, prepared, how=how, predicate="within",
                all_matches=all_m, exact_boundary=True, backend="gpu",
            )

        def run_gpd():
            return gpd.sjoin(left, right, how=how, predicate="within")

        t_cold = _median_ms(run_cold)
        prepared_cells = f"  {'n/a':>7}  {'n/a':>8}  {'n/a':>8}  {'n/a':>6}"
        if not raw_only:
            t_prep = _median_ms(lambda: ms.PreparedJoin(right))
            prepared = ms.PreparedJoin(right)
            t_warm1 = _median_ms(run_prepared, warmup=0, repeats=1)
            t_warm = _median_ms(run_prepared)
            prepared_cells = (
                f"  {t_prep:4.1f} ms  {t_warm1:5.1f} ms  {t_warm:5.1f} ms  {t_cold / t_warm:5.2f}x"
            )
        gpd_cell = "      n/a"
        if all_m and el == 0 and er == 0 and how == "inner":
            # One native join before the validation step builds a spatial index.
            gpd_cell = f"{_median_ms(run_gpd, warmup=0, repeats=1):6.1f} ms"

        actual = run_cold()
        expected = run_gpd().sort_values(["point_id", "index_right"])
        if not all_m:
            expected = expected.drop_duplicates("point_id")
        assert_geodataframe_equal(actual, expected, check_geom_type=True)
        if not raw_only:
            assert_geodataframe_equal(run_prepared(), actual, check_geom_type=True)
        print(
            f"  {label:<40}  {t_cold:5.1f} ms{prepared_cells}  {gpd_cell}  "
            f"{len(actual):,} rows, equality=PASS"
        )


def bench_knn_prepared(seed: int = 0, *, raw_only: bool = False) -> None:
    """haversine_knn: raw vs HaversineReference vs prebuilt BallTree (m=1M, k=10)."""
    print("\n### 3. haversine_knn with HaversineReference (m=1,000,000 refs, k=10)")
    rng = np.random.default_rng(seed)
    m = 1_000_000
    r = bench.make_points(m, rng)

    ref = None
    if not raw_only:
        t_prep = _median_ms(lambda: ms.HaversineReference(r))
        ref = ms.HaversineReference(r)
        print(f"  prepare HaversineReference (1M): {t_prep:.2f} ms")

    bt = None
    try:
        from sklearn.neighbors import BallTree

        r_rad = np.radians(r[:, ::-1])
        t0 = time.perf_counter()
        bt = BallTree(r_rad, metric="haversine")
        print(f"  BallTree build (1M): {(time.perf_counter() - t0) * 1e3:.1f} ms")
    except ImportError:
        pass

    def run_knn(query, reference):
        distances, indices = ms.haversine_knn(query, reference, 10)
        mx.eval(distances, indices)
        return distances, indices

    print(
        f"  {'queries':>8}  {'raw knn':>10}  {'prepared':>10}  {'speedup':>8}  {'BallTree':>10}"
    )
    for n in (1, 10, 100, 500, 1_000, 4_096):
        q = bench.make_points(n, rng)
        t_raw = _median_ms(lambda: run_knn(q, r), warmup=2, repeats=7)
        ref_cells = f"  {'n/a':>10}  {'n/a':>8}"
        if ref is not None:
            t_ref = _median_ms(lambda: run_knn(q, ref), warmup=2, repeats=7)
            for actual, target in zip(run_knn(q, ref), run_knn(q, r), strict=True):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(target))
            ref_cells = f"  {t_ref:7.2f} ms  {t_raw / t_ref:7.2f}x"
        bt_s = "n/a"
        if bt is not None:
            q_rad = np.radians(q[:, ::-1])
            t_bt = _median_ms(lambda: bt.query(q_rad, k=10), warmup=1, repeats=5)
            bt_s = f"{t_bt:7.2f} ms"
        print(
            f"  {n:>8,}  {t_raw:7.2f} ms{ref_cells}  {bt_s:>10}"
            + ("  equality=PASS" if ref is not None else "")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["all", "pip", "sjoin", "knn", "batching"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--raw-only", action="store_true", help="compare with a release without prepared APIs"
    )
    args = parser.parse_args()
    print(f"machine: {bench.machine()}")
    print(f"package: {ms.__file__}; seed: {args.seed}")
    if args.suite in ("all", "pip"):
        bench_pip_detailed(args.seed)
    if args.suite in ("all", "sjoin"):
        bench_sjoin_public(args.seed, raw_only=args.raw_only)
    if args.suite in ("all", "knn"):
        bench_knn_prepared(args.seed, raw_only=args.raw_only)
    if args.suite in ("all", "batching"):
        bench_all_match_batches(args.seed)


if __name__ == "__main__":
    main()
