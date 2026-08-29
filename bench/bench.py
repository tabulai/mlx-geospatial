"""Benchmark harness for mlx-spatial.

Honesty rules:
- All timings are end-to-end from host-resident numpy/shapely inputs,
  including ingestion, host<->device transfer, and result materialization.
- ``cold`` numbers include building the PolygonSet + grid index; ``warm``
  numbers reuse a prebuilt index (the realistic repeated-query pattern).
- Baselines that are unavailable are reported as ``n/a``, never silently
  dropped.

Usage:
    python bench/bench.py --op haversine --n 20000 --m 20000
    python bench/bench.py --op knn --n 200000 --m 200000 --k 10
    python bench/bench.py --op join --n 1000000 --m 260
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
        index.query(pts, backend="gpu")

    rows.append(("mlx-spatial point_in_polygon (warm index)", timeit(ours_warm)))

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
    ours = rows[0][1]
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
    parser.add_argument("--op", choices=["haversine", "knn", "join"])
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--m", type=int, default=260)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--all", action="store_true", help="run a representative sweep")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"machine: {machine()}")

    if args.all:
        report("haversine_cross 20k x 20k", bench_haversine(20_000, 20_000, rng))
        report(
            "haversine_knn 1M queries, 10k refs, k=10",
            bench_knn(1_000_000, 10_000, 10, rng),
        )
        report(
            "haversine_knn 200k queries, 200k refs, k=10",
            bench_knn(200_000, 200_000, 10, rng),
        )
        for n in (100_000, 1_000_000, 10_000_000):
            report(f"join {n:,} points x 260 zones", bench_join(n, 260, rng))
        report("join 1,000,000 points x 10,000 zones", bench_join(1_000_000, 10_000, rng))
        return

    if args.op == "haversine":
        report(f"haversine_cross {args.n:,} x {args.m:,}", bench_haversine(args.n, args.m, rng))
    elif args.op == "knn":
        report(
            f"haversine_knn {args.n:,} queries, {args.m:,} refs, k={args.k}",
            bench_knn(args.n, args.m, args.k, rng),
        )
    elif args.op == "join":
        report(f"join {args.n:,} points x {args.m:,} zones", bench_join(args.n, args.m, rng))
    else:
        parser.error("pass --op or --all")


if __name__ == "__main__":
    main()
