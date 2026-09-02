"""Benchmark harness for mlx-spatial.

Honesty rules:
- All timings are end-to-end from host-resident numpy/shapely inputs,
  including ingestion, host->device input transfer, device result evaluation,
  and synchronization. Host-input join rows return NumPy labels as a
  zero-copy view of the MLX result (no copy-back is timed); other MLX results
  stay MLX arrays.
- The explicitly labeled dense-PIP microbenchmark is the exception: it uses
  a pre-ingested PolygonSet and GPU-resident points to isolate the replacement
  kernel from its retained composed oracle.
- ``cold`` numbers include building the PolygonSet + grid index; ``warm``
  numbers reuse a prebuilt index (the realistic repeated-query pattern).
- Baselines that are unavailable are reported as ``n/a``, never silently
  dropped.

Usage:
    python bench/bench.py --op haversine --n 20000 --m 20000
    python bench/bench.py --op knn --n 200000 --m 200000 --k 10
    python bench/bench.py --op dense-pip --n 100000 --m 32 --vertices 48
    python bench/bench.py --op join --n 1000000 --m 260
    python bench/bench.py --op crossover --m 260   # CPU/GPU dispatch sweep + constant fit
    python bench/bench.py --all            # quick representative sweep
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import time

import numpy as np
import shapely

import mlx.core as mx
import mlx_spatial as ms
from mlx_spatial._dispatch import metal_available
from mlx_spatial._pip import _pip_dense_composed, pip_dense

EXTENT = (-74.3, 40.5, -73.7, 40.9)  # roughly NYC


# ---------------------------------------------------------------- utilities

def sync():
    mx.synchronize()


def timeit(fn, *, warmup: int = 1, repeats: int = 3) -> float:
    """Best-of-repeats wall time in seconds."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def machine() -> str:
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        chip = platform.processor()
    return f"{chip}, macOS {platform.mac_ver()[0]}, mlx {mx.__version__}"


def make_points(n: int, rng: np.random.Generator) -> np.ndarray:
    x0, y0, x1, y1 = EXTENT
    return np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])


def make_zones(m: int, rng: np.random.Generator, vertices: int = 48) -> np.ndarray:
    """Convex 'zones' tiling the extent statistically (like taxi zones)."""
    x0, y0, x1, y1 = EXTENT
    scale = np.sqrt((x1 - x0) * (y1 - y0) / m)
    centers = np.column_stack([rng.uniform(x0, x1, m), rng.uniform(y0, y1, m)])
    ang = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    polys = []
    for cx, cy in centers:
        radii = scale * rng.uniform(0.4, 0.8, vertices)
        polys.append(
            shapely.Polygon(
                np.column_stack([cx + radii * np.cos(ang), cy + radii * np.sin(ang)])
            )
        )
    return np.array(polys, dtype=object)


# ---------------------------------------------------------------- dense PIP

def bench_dense_pip(n: int, m: int, vertices: int, rng) -> list[tuple[str, float | None]]:
    """Warm, GPU-resident comparison of direct and composed dense PIP."""
    if not metal_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("the dense-PIP Metal benchmark requires an active GPU device")
    pts = make_points(n, rng)
    polyset = ms.PolygonSet(make_zones(m, rng, vertices=vertices))
    _, points_local = polyset.recenter_points(pts)

    def direct():
        out = pip_dense(points_local, polyset)
        mx.eval(out)
        sync()

    def composed():
        out = _pip_dense_composed(points_local, polyset)
        mx.eval(out)
        sync()

    # Correctness is checked outside the timed region. Keeping the composed
    # implementation here makes performance regressions reproducible while it
    # remains the non-Metal fallback and independent arithmetic oracle.
    direct_out = np.array(pip_dense(points_local, polyset))
    composed_out = np.array(_pip_dense_composed(points_local, polyset))
    np.testing.assert_array_equal(direct_out, composed_out)

    return [
        ("mlx-spatial dense PIP (direct 2-D Metal)", timeit(direct)),
        ("composed MLX dense PIP oracle", timeit(composed)),
    ]


# ---------------------------------------------------------------- haversine

def bench_haversine(n: int, m: int, rng) -> list[tuple[str, float | None]]:
    a, b = make_points(n, rng), make_points(m, rng)
    rows = []

    def ours():
        d = ms.haversine_cross(a, b)
        mx.eval(d)
        sync()

    rows.append(("mlx-spatial haversine_cross", timeit(ours)))

    if n * m <= 400_000_000:
        def np64():
            lon1, lat1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
            lon2, lat2 = np.radians(b[:, 0])[None, :], np.radians(b[:, 1])[None, :]
            h = (
                np.sin((lat2 - lat1) / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
            )
            return 2 * ms.R_EARTH_M * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

        rows.append(("numpy float64", timeit(np64)))
    else:
        rows.append(("numpy float64", None))

    try:
        from sklearn.metrics.pairwise import haversine_distances

        if n * m <= 400_000_000:
            def sk():
                haversine_distances(
                    np.radians(a[:, ::-1]), np.radians(b[:, ::-1])
                )

            rows.append(("sklearn haversine_distances", timeit(sk)))
        else:
            rows.append(("sklearn haversine_distances", None))
    except ImportError:
        rows.append(("sklearn haversine_distances", None))
    return rows


def bench_knn(n: int, m: int, k: int, rng) -> list[tuple[str, float | None]]:
    q, r = make_points(n, rng), make_points(m, rng)
    rows = []

    def ours():
        d, i = ms.haversine_knn(q, r, k)
        mx.eval(d, i)
        sync()

    rows.append((f"mlx-spatial haversine_knn (k={k})", timeit(ours)))

    try:
        from sklearn.neighbors import BallTree

        def sk():
            tree = BallTree(np.radians(r[:, ::-1]), metric="haversine")
            tree.query(np.radians(q[:, ::-1]), k=k)

        rows.append(("sklearn BallTree(haversine)", timeit(sk)))
    except ImportError:
        rows.append(("sklearn BallTree(haversine)", None))
    return rows


# ---------------------------------------------------------------- join

def densify_zones(zones: np.ndarray, vertices: int) -> np.ndarray:
    """Resample each zone's exterior ring to ``vertices`` points (same shape)."""
    out = []
    for poly in zones:
        ring = np.asarray(poly.exterior.coords)
        t_src = np.arange(len(ring))
        t_dst = np.linspace(0.0, len(ring) - 1, vertices)
        out.append(
            shapely.Polygon(
                np.column_stack([np.interp(t_dst, t_src, ring[:, 0]), np.interp(t_dst, t_src, ring[:, 1])])
            )
        )
    return np.array(out, dtype=object)


def bench_crossover(m: int, rng) -> None:
    """Where does the GPU start winning for host-resident points?

    Sweeps N for forced CPU (shapely STRtree fallback), forced GPU and the
    default auto dispatch, for first-match and all-match queries, on ``m``
    zones with 48 and 2048 vertices, then fits the constants of the dispatch
    cost model used by ``mlx_spatial._dispatch`` (GEOS cost per point and per
    candidate edge versus the fixed GPU cost of each route) so they can be
    re-derived on another machine.
    """
    from mlx_spatial import _dispatch

    pts_all = make_points(200_000, rng)
    ns = [100, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000]
    layers = [
        ("48-vertex zones", make_zones(m, rng)),
        ("2048-vertex zones", densify_zones(make_zones(m, rng), 2048)),
    ]
    slopes = []
    gpu_fixed, gpu_all_fixed = [], []
    for label, zones in layers:
        polyset = ms.PolygonSet(zones)
        index = ms.GridIndex(polyset)
        edges = index._edges_per_point
        print(f"\n### crossover: {m:,} {label} (~{edges:.0f} candidate edges per point)")
        print(
            f"  {'points':>8}  {'first match: cpu':>16}  {'gpu':>8}  {'auto':>8}  pick"
            f"  | {'all matches: cpu':>16}  {'gpu':>8}  {'auto':>8}  pick"
        )
        xs, ys = [], []
        for n in ns:
            pts = pts_all[:n]
            row = []
            for all_matches in (False, True):
                kw = {"all_matches": True} if all_matches else {}
                cpu = timeit(lambda: index.query(pts, backend="cpu", **kw), warmup=2, repeats=7)
                gpu = timeit(lambda: index.query(pts, backend="gpu", **kw), warmup=2, repeats=7)
                auto = timeit(lambda: index.query(pts, **kw), warmup=2, repeats=7)
                picks = _dispatch.resolve_backend(None, n, edges_per_point=edges, all_matches=all_matches)
                flag = "" if (picks == "gpu") == (gpu <= cpu) else "!"
                row.append(f"{fmt(cpu):>16}  {fmt(gpu):>8}  {fmt(auto):>8}  {picks}{flag:<1}")
                if not all_matches:
                    xs.append(n)
                    ys.append(cpu * 1e6)
                    if n <= 5_000:
                        gpu_fixed.append(gpu * 1e6)
                elif n <= 2_000:
                    gpu_all_fixed.append(gpu * 1e6)
            print(f"  {n:>8,}  {row[0]}  | {row[1]}")
        slope = float(np.polyfit(xs, ys, 1)[0])  # us per point on the CPU path
        slopes.append((edges, slope))
    print("  ('!' marks a row where auto picked the slower side)")
    (e0, s0), (e1, s1) = slopes
    per_edge = (s1 - s0) / (e1 - e0) if e1 != e0 else 0.0
    per_point = s0 - per_edge * e0
    launch = float(np.median(gpu_fixed))
    launch_all = float(np.median(gpu_all_fixed))
    print("\n  dispatch constants          current -> measured here")
    print(f"  CPU_US_PER_POINT            {_dispatch.CPU_US_PER_POINT:.3f} -> {per_point:.3f}")
    print(f"  CPU_US_PER_EDGE             {_dispatch.CPU_US_PER_EDGE:.5f} -> {per_edge:.5f}")
    print(f"  GPU_LAUNCH_US               {_dispatch.GPU_LAUNCH_US:.0f} -> {launch:.0f}")
    print(f"  GPU_ALL_MATCHES_LAUNCH_US   {_dispatch.GPU_ALL_MATCHES_LAUNCH_US:.0f} -> {launch_all:.0f}")
    e48, e2048 = slopes[0][0], slopes[1][0]
    print(
        f"  gpu_min_points()            first match {_dispatch.gpu_min_points(e48)} / "
        f"{_dispatch.gpu_min_points(e2048)} points (48- / 2048-vertex layer); all matches "
        f"{_dispatch.gpu_min_points(e48, all_matches=True)} / {_dispatch.gpu_min_points(e2048, all_matches=True)}"
    )


def bench_join(n: int, m: int, rng) -> list[tuple[str, float | None]]:
    pts = make_points(n, rng)
    zones = make_zones(m, rng)
    rows = []

    def ours_cold():
        polyset = ms.PolygonSet(zones)
        ms.point_in_polygon(pts, polyset, backend="gpu")

    rows.append(("mlx-spatial point_in_polygon (cold)", timeit(ours_cold)))

    polyset = ms.PolygonSet(zones)
    index = ms.GridIndex(polyset)

    def ours_warm():
        # numpy in -> numpy out: host re-centering plus a zero-copy view of
        # the MLX result; the label array is fully materialized on return.
        index.query(pts, backend="gpu")

    rows.append(("mlx-spatial point_in_polygon (warm index)", timeit(ours_warm)))

    local = polyset.to_local(pts)  # re-center once; the array-native pattern

    def ours_device():
        mx.eval(index.query(local))

    rows.append(
        ("mlx-spatial point_in_polygon (warm index, device-resident points)", timeit(ours_device))
    )

    def strtree():
        tree = shapely.STRtree(zones)
        tree.query(shapely.points(pts), predicate="intersects")

    rows.append(("shapely STRtree bulk query", timeit(strtree)))

    try:
        import geopandas as gpd

        left = gpd.GeoDataFrame(geometry=list(shapely.points(pts)))
        right = gpd.GeoDataFrame(geometry=list(zones))

        def gpd_join():
            gpd.sjoin(left, right, how="inner", predicate="within")

        rows.append(("geopandas sjoin", timeit(gpd_join, warmup=0, repeats=1 if n >= 5_000_000 else 3)))
    except ImportError:
        rows.append(("geopandas sjoin", None))

    rows.append(("duckdb spatial join", _bench_duckdb_join(pts, zones)))
    return rows


def _bench_duckdb_join(pts: np.ndarray, zones: np.ndarray) -> float | None:
    try:
        import duckdb
        import pandas as pd

        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        pts_df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1]})
        zones_df = pd.DataFrame({"wkb": [shapely.to_wkb(z) for z in zones]})
        con.register("pts_raw", pts_df)
        con.register("zones_raw", zones_df)
        con.execute(
            "CREATE TABLE zones AS SELECT ST_GeomFromWKB(wkb) AS geom FROM zones_raw"
        )

        def q():
            con.execute(
                "SELECT count(*) FROM pts_raw JOIN zones "
                "ON ST_Contains(zones.geom, ST_Point(pts_raw.x, pts_raw.y))"
            ).fetchall()

        return timeit(q, warmup=1, repeats=2)
    except Exception:
        return None


# ---------------------------------------------------------------- driver

def report(title: str, rows: list[tuple[str, float | None]]):
    print(f"\n### {title}")
    # Join reports cold first but external baselines are comparable to its
    # reused-index row. Other operations keep their primary result first.
    ours = next(
        (
            secs
            for name, secs in rows
            if name.startswith("mlx-spatial") and "warm index" in name
        ),
        rows[0][1],
    )
    width = max(len(name) for name, _ in rows)
    for name, secs in rows:
        speedup = ""
        if (
            secs is not None
            and ours is not None
            and secs > 0
            and not name.startswith("mlx-spatial")
        ):
            speedup = f"  ({secs / ours:.1f}x vs ours)"
        print(f"  {name:<{width}}  {fmt(secs):>10}{speedup}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=["haversine", "knn", "dense-pip", "join", "crossover"])
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--m", type=int, default=260)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--vertices", type=int, default=48)
    parser.add_argument("--all", action="store_true", help="run a representative sweep")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"machine: {machine()}")

    if args.all:
        report("haversine_cross 20k x 20k", bench_haversine(20_000, 20_000, rng))
        report(
            "haversine_knn 1k queries, 1M refs, k=10",
            bench_knn(1_000, 1_000_000, 10, rng),
        )
        report(
            "haversine_knn 1M queries, 10k refs, k=10",
            bench_knn(1_000_000, 10_000, 10, rng),
        )
        report(
            "haversine_knn 200k queries, 200k refs, k=10",
            bench_knn(200_000, 200_000, 10, rng),
        )
        report(
            "dense PIP 100,000 points x 32 polygons x 48 vertices",
            bench_dense_pip(100_000, 32, 48, rng),
        )
        for n in (100_000, 1_000_000, 10_000_000):
            report(f"join {n:,} points x 260 zones", bench_join(n, 260, rng))
        report("join 1,000,000 points x 10,000 zones", bench_join(1_000_000, 10_000, rng))
        bench_crossover(260, rng)
        return

    if args.op == "haversine":
        report(f"haversine_cross {args.n:,} x {args.m:,}", bench_haversine(args.n, args.m, rng))
    elif args.op == "knn":
        report(
            f"haversine_knn {args.n:,} queries, {args.m:,} refs, k={args.k}",
            bench_knn(args.n, args.m, args.k, rng),
        )
    elif args.op == "dense-pip":
        report(
            f"dense PIP {args.n:,} points x {args.m:,} polygons "
            f"x {args.vertices} vertices",
            bench_dense_pip(args.n, args.m, args.vertices, rng),
        )
    elif args.op == "join":
        report(f"join {args.n:,} points x {args.m:,} zones", bench_join(args.n, args.m, rng))
    elif args.op == "crossover":
        bench_crossover(args.m, rng)
    else:
        parser.error("pass --op or --all")


if __name__ == "__main__":
    main()
