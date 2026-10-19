"""Benchmark harness for mlx-geospatial.

Honesty rules:
- All timings are end-to-end from host-resident numpy/shapely inputs,
  including ingestion, host->device input transfer, device result evaluation,
  and synchronization. Host-input join rows return NumPy labels as a
  zero-copy view of the MLX result (no copy-back is timed); other MLX results
  stay MLX arrays.
- Like against like: the join baselines evaluate every (point, polygon)
  pair (shapely STRtree and GeoPandas sjoin return them; DuckDB is timed as
  ``count(*)`` over the join and never materializes them), so their speedups
  are computed against our ``all_matches=True`` row, not the cheaper
  first-match row, which is reported alongside. GEOS is given its best
  configuration: the polygons are the query side of the STRtree (and the left
  frame of ``sjoin``), because shapely prepares the geometries it is queried
  with, and prepared polygons carry an indexed point-in-area locator that is
  8-60x faster on detailed polygons than the unprepared orientation. The kNN
  baseline (sklearn BallTree) is reported both with the tree built inside the
  timed region and query-only on a prebuilt tree; our brute-force kNN has no
  index to prebuild.
- Thread counts are stated: shapely/GEOS, GeoPandas and sklearn run on one
  core; DuckDB uses its default thread pool (all cores); the GPU rows use the
  whole GPU. The machine line lists the core count.
- The synthetic "zones" are random star-shaped polygons that overlap and leave
  gaps (they do not tile the extent and are not convex); the join report
  prints the measured coverage. Detailed geometry (2,048-vertex zones, a
  150k-vertex coastline-like polygon) is benchmarked as well, because that is
  where baseline configuration matters most (prepared vs unprepared polygons
  differ 8-60x) and where our per-candidate ring walk meets GEOS's indexed
  locator.
- The explicitly labeled dense-PIP microbenchmark is the exception to the
  end-to-end rule: it uses a pre-ingested PolygonSet and GPU-resident points to
  isolate the replacement kernel from its retained composed oracle.
- ``cold`` numbers include building the PolygonSet + grid index; ``warm``
  numbers reuse a prebuilt index (the realistic repeated-query pattern) and
  are steady-state burst numbers: GPU clocks are warmed before each timing, as
  in the crossover sweep, because the fixed launch cost is 3-4x higher right
  after the GPU has idled.
- Baselines that are unavailable are reported as ``n/a``, never silently
  dropped.

Usage:
    python bench/bench.py --op haversine --n 20000 --m 20000
    python bench/bench.py --op knn --n 200000 --m 200000 --k 10
    python bench/bench.py --op dense-pip --n 100000 --m 32 --vertices 48
    python bench/bench.py --op join --n 1000000 --m 260 [--vertices 2048]
    python bench/bench.py --op join --n 1000000 --coastline 150000   # one detailed polygon
    python bench/bench.py --op crossover --m 260   # CPU/GPU dispatch sweep + constant fit
    python bench/bench.py --all            # the README sweep
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import time

import numpy as np
import shapely

import mlx.core as mx
import mlx_geospatial as ms
from mlx_geospatial._dispatch import metal_available
from mlx_geospatial._pip import _pip_dense_composed, pip_dense

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
    cores = f"{os.cpu_count()} cores"
    try:
        perf = subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True).strip()
        eff = subprocess.check_output(["sysctl", "-n", "hw.perflevel1.physicalcpu"], text=True).strip()
        cores = f"{os.cpu_count()} cores ({perf} performance + {eff} efficiency)"
    except Exception:
        pass
    return (
        f"{chip}, {cores}, macOS {platform.mac_ver()[0]}, mlx {mx.__version__}, "
        f"Python {platform.python_version()}"
    )


def duckdb_threads() -> int | None:
    try:
        import duckdb

        return int(duckdb.connect().execute("select current_setting('threads')").fetchone()[0])
    except Exception:
        return None


def make_points(n: int, rng: np.random.Generator) -> np.ndarray:
    x0, y0, x1, y1 = EXTENT
    return np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])


def make_zones(m: int, rng: np.random.Generator, vertices: int = 48) -> np.ndarray:
    """``m`` random star-shaped 'zones' with ``vertices`` vertices each.

    Centers are uniform over the extent and every vertex gets its own radius
    (0.4-0.8 of the mean cell side), so the zones are neither convex nor a
    tiling: they overlap and leave gaps. About two thirds of the extent is
    covered and a covered point lies in 1.6 zones on average (the join report
    prints the measured coverage). Vertex counts model taxi-zone style
    polygons (48) or detailed administrative boundaries (2,048).
    """
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
        ("mlx-geospatial dense PIP (direct 2-D Metal)", timeit(direct)),
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

    rows.append(("mlx-geospatial haversine_cross", timeit(ours)))

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
    """Exact brute-force kNN versus sklearn's BallTree (one core).

    The BallTree is reported twice: with the tree built inside the timed
    region (a one-off query, comparable to our end-to-end number, which has no
    index to build) and query-only on a prebuilt tree (repeated queries
    against a fixed reference set, where the tree amortizes).
    """
    q, r = make_points(n, rng), make_points(m, rng)
    rows = []

    def ours():
        d, i = ms.haversine_knn(q, r, k)
        mx.eval(d, i)
        sync()

    rows.append((f"mlx-geospatial haversine_knn (k={k}, brute force)", timeit(ours)))

    try:
        from sklearn.neighbors import BallTree

        r_rad = np.radians(r[:, ::-1])
        q_rad = np.radians(q[:, ::-1])

        def sk_build_and_query():
            tree = BallTree(r_rad, metric="haversine")
            tree.query(q_rad, k=k)

        rows.append(("sklearn BallTree(haversine), build + query, 1 core", timeit(sk_build_and_query)))
        tree = BallTree(r_rad, metric="haversine")

        def sk_query():
            tree.query(q_rad, k=k)

        rows.append(("sklearn BallTree(haversine), query only (prebuilt), 1 core", timeit(sk_query)))
    except ImportError:
        rows.append(("sklearn BallTree(haversine), build + query, 1 core", None))
        rows.append(("sklearn BallTree(haversine), query only (prebuilt), 1 core", None))
    return rows


# ---------------------------------------------------------------- join

def make_tiling(m: int) -> np.ndarray:
    """Axis-aligned square tiling of the extent with about ``m`` tiles.

    Four edges per tile and about nine candidate edges per point: the
    low-edge layer that pins the dispatch model's per-point cost.
    """
    x0, y0, x1, y1 = EXTENT
    side = max(1, int(round(np.sqrt(m))))
    xs = np.linspace(x0, x1, side + 1)
    ys = np.linspace(y0, y1, side + 1)
    return np.array(
        [shapely.box(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(side) for i in range(side)],
        dtype=object,
    )


def _gpu_warm(fn, seconds: float = 0.05) -> None:
    """Run ``fn`` for a short while so GPU clocks ramp before timing it."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        fn()


def _launch_probe(index, pts, *, idle_s: float = 0.2, repeats: int = 7) -> dict:
    """Fixed GPU cost per route with warm clocks and after the GPU has idled.

    Returns microseconds: ``{"first": (warm, idle), "all": (warm, idle)}``.
    The idle number is the median single query after ``idle_s`` of sleep (a
    host-only program issuing one small query); the warm number is the best of
    a back-to-back burst (a loop of small GPU queries).
    """
    out = {}
    for key, kw in (("first", {}), ("all", {"all_matches": True})):
        fn = lambda: index.query(pts, backend="gpu", **kw)  # noqa: E731
        _gpu_warm(fn)
        warm = timeit(fn, warmup=2, repeats=repeats)
        idle = []
        for _ in range(repeats):
            time.sleep(idle_s)
            t = time.perf_counter()
            fn()
            idle.append(time.perf_counter() - t)
        out[key] = (warm * 1e6, float(np.median(idle)) * 1e6)
    return out


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

    Sweeps N for forced CPU (the NumPy even-odd fallback), forced GPU and the
    default auto dispatch, for first-match and all-match queries, on ``m``
    zones with 48 and 2048 vertices and on an axis-aligned square tiling of
    about ``m`` tiles, then fits the constants of the dispatch cost model used
    by ``mlx_geospatial._dispatch`` (the host fallback's cost per point and per
    candidate edge versus the fixed GPU cost of each route) so they can be
    re-derived on another machine. The tiling, with only ~9 candidate edges
    per point, pins the per-point term; from the two zone layers alone it
    would be the difference of two nearly equal numbers.
    """
    from mlx_geospatial import _dispatch

    pts_all = make_points(200_000, rng)
    ns = [100, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000]
    layers = [
        ("48-vertex zones", make_zones(m, rng)),
        ("square tiling", make_tiling(m)),
        ("2048-vertex zones", densify_zones(make_zones(m, rng), 2048)),
    ]
    fits = []  # (candidate edges per point, us per point, us fixed at n=1) per layer
    gpu_fixed, gpu_all_fixed = [], []
    probe = None
    for label, zones in layers:
        polyset = ms.PolygonSet(zones)
        index = ms.GridIndex(polyset)
        edges = index._edges_per_point
        fixed = timeit(lambda: index.query(pts_all[:1], backend="cpu"), warmup=5, repeats=50) * 1e6
        if probe is None:
            probe = _launch_probe(index, pts_all[:500])
        print(f"\n### crossover: {len(zones):,} {label} (~{edges:.0f} candidate edges per point)")
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
                # Let the GPU clocks ramp after the CPU-only timings so the
                # forced-GPU column (and the fitted launch cost) is not biased.
                _gpu_warm(lambda: index.query(pts, backend="gpu", **kw))
                gpu = timeit(lambda: index.query(pts, backend="gpu", **kw), warmup=2, repeats=7)
                auto = timeit(lambda: index.query(pts, **kw), warmup=2, repeats=7)
                picks = _dispatch.resolve_backend(None, n, edges_per_point=edges, all_matches=all_matches)
                # flag a wrong pick only beyond a 5% tie band
                wrong = (picks == "gpu") != (gpu <= cpu) and abs(gpu - cpu) > 0.05 * min(gpu, cpu)
                flag = "!" if wrong else ""
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
        fits.append((edges, slope, fixed))
    print("  ('!' marks a row where auto picked the slower side by more than 5%)")
    # slope(layer) = per_point + per_edge * edges(layer): least squares over the
    # three layers; the per-point term cannot be negative.
    e = np.array([f[0] for f in fits])
    slopes = np.array([f[1] for f in fits])
    per_point, per_edge = np.linalg.lstsq(np.column_stack([np.ones_like(e), e]), slopes, rcond=None)[0]
    per_point = max(float(per_point), 0.0)
    per_edge = float(per_edge)
    fixed = float(np.median([f[2] for f in fits]))
    launch = float(np.median(gpu_fixed))
    launch_all = float(np.median(gpu_all_fixed))
    print("\n  dispatch constants          current -> measured here")
    print(f"  CPU_US_PER_POINT            {_dispatch.CPU_US_PER_POINT:.3f} -> {per_point:.3f}")
    print(f"  CPU_US_PER_EDGE             {_dispatch.CPU_US_PER_EDGE:.5f} -> {per_edge:.5f}")
    print(
        f"  GPU_LAUNCH_US               {_dispatch.GPU_LAUNCH_US:.0f} -> {launch:.0f} warm"
        f" ({probe['first'][0]:.0f} warm / {probe['first'][1]:.0f} after the GPU idled 200 ms, 500 points)"
    )
    print(
        f"  GPU_ALL_MATCHES_LAUNCH_US   {_dispatch.GPU_ALL_MATCHES_LAUNCH_US:.0f} -> {launch_all:.0f} warm"
        f" ({probe['all'][0]:.0f} warm / {probe['all'][1]:.0f} idle)"
    )
    def _where(constant, warm_idle):
        warm, idle = warm_idle
        if constant < warm:
            return f"sits below the warm cost measured here ({warm:.0f} us)"
        if constant > idle:
            return f"sits above the idle cost measured here ({idle:.0f} us)"
        return "sits between the warm and the idle cost measured here"

    print(
        f"  (GPU columns are steady-state, clocks warmed before each timing. GPU_LAUNCH_US"
        f" {_where(_dispatch.GPU_LAUNCH_US, probe['first'])}; GPU_ALL_MATCHES_LAUNCH_US"
        f" {_where(_dispatch.GPU_ALL_MATCHES_LAUNCH_US, probe['all'])}. The host fallback's"
        f" ~{fixed:.0f} us fixed cost per call is not a model term.)"
    )
    names = [label for label, _ in layers]
    print(
        "  gpu_min_points()            first match "
        + " / ".join(f"{_dispatch.gpu_min_points(f[0])}" for f in fits)
        + " points; all matches "
        + " / ".join(f"{_dispatch.gpu_min_points(f[0], all_matches=True)}" for f in fits)
        + f" ({', '.join(names)})"
    )


def make_coastline(vertices: int, rng: np.random.Generator) -> np.ndarray:
    """One detailed, coastline-like polygon with ``vertices`` vertices.

    A star-shaped ring around the extent's center whose radius carries
    low- and high-frequency undulation plus noise, so it is valid (simple),
    covers about 22% of the extent (its bounding box about 39%) and has no
    short cuts for a crossing-number test: every containment decision walks
    the whole ring on the scalar paths, which is the case detailed
    administrative or natural boundaries present.
    """
    x0, y0, x1, y1 = EXTENT
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    base = 0.32 * min(x1 - x0, y1 - y0)
    ang = np.linspace(0.0, 2.0 * np.pi, vertices, endpoint=False)
    r = base * (
        1.0 + 0.12 * np.sin(7 * ang) + 0.06 * np.sin(41 * ang + 1.0)
        + 0.03 * np.sin(211 * ang) + 0.01 * rng.normal(size=vertices)
    )
    poly = shapely.Polygon(np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)]))
    assert poly.is_valid
    return np.array([poly], dtype=object)


def bench_join(
    n: int, m: int, rng, *, vertices: int = 48, coastline: int | None = None
) -> list[tuple[str, float | None]]:
    """Point-in-polygon join against all-pairs CPU baselines.

    ``coastline`` replaces the zone layer by one polygon with that many
    vertices. Baselines evaluate every (point, polygon) pair (DuckDB as a
    count), so the row they are compared with is our ``all_matches=True``
    query; first match (the cheaper contract most joins want) is reported as
    well. Warm rows are timed after warming the GPU clocks.
    """
    pts = make_points(n, rng)
    if coastline:
        zones = make_coastline(coastline, rng)
    else:
        zones = make_zones(m, rng)
        if vertices != 48:
            zones = densify_zones(zones, vertices)
    rows = []

    def ours_cold():
        polyset = ms.PolygonSet(zones)
        ms.point_in_polygon(pts, polyset, backend="gpu")

    rows.append(("mlx-geospatial point_in_polygon, first match (cold: ingest + index)", timeit(ours_cold)))

    polyset = ms.PolygonSet(zones)
    index = ms.GridIndex(polyset)

    def ours_warm():
        # numpy in -> numpy out: host re-centering plus a zero-copy view of
        # the MLX result; the label array is fully materialized on return.
        index.query(pts, backend="gpu")

    _gpu_warm(ours_warm)
    rows.append(("mlx-geospatial point_in_polygon, first match (warm index)", timeit(ours_warm)))

    def ours_all():
        # numpy in -> two numpy int64 pair arrays out, materialized on return
        index.query(pts, backend="gpu", all_matches=True)

    _gpu_warm(ours_all)
    rows.append(("mlx-geospatial point_in_polygon, all pairs (warm index)", timeit(ours_all)))

    local = polyset.to_local(pts)  # re-center once; the array-native pattern

    def ours_device():
        mx.eval(index.query(local))

    _gpu_warm(ours_device)
    rows.append(
        ("mlx-geospatial point_in_polygon, first match (warm index, device-resident points)", timeit(ours_device))
    )

    # Coverage, for reading the numbers: how many pairs the all-pairs rows
    # produce per point, and how many points match at all.
    pt, _ = index.query(pts, backend="gpu", all_matches=True)
    n_pairs = int(pt.shape[0])
    covered = int(np.unique(np.asarray(pt)).size)
    rows.append(
        (
            f"  [coverage: {covered / n:.0%} of points lie in a polygon, "
            f"{n_pairs / max(covered, 1):.2f} polygons per such point, {n_pairs:,} pairs]",
            None,
        )
    )

    def strtree():
        # Points in the tree, polygons as the query: shapely prepares the
        # query geometries, so this gives GEOS its indexed point-in-area
        # locator (8-60x faster on detailed polygons than querying a polygon
        # tree with points). Point geometries are built inside the timing,
        # as our rows ingest raw NumPy inside theirs.
        tree = shapely.STRtree(shapely.points(pts))
        tree.query(zones, predicate="intersects")

    rows.append(("shapely STRtree bulk query (prepared polygons), all pairs, 1 core", timeit(strtree)))

    try:
        import geopandas as gpd

        polygons_df = gpd.GeoDataFrame(geometry=list(zones))  # prebuilt, like DuckDB's zone table

        def gpd_join():
            # polygons on the left: sjoin queries the right frame's tree with
            # the left geometries, so the polygons are the prepared side
            points_df = gpd.GeoDataFrame(geometry=gpd.GeoSeries(shapely.points(pts)))
            gpd.sjoin(polygons_df, points_df, how="inner", predicate="intersects")

        rows.append(
            (
                "geopandas sjoin (polygons left, prepared), all pairs, 1 core",
                timeit(gpd_join, warmup=0, repeats=1 if n >= 5_000_000 else 3),
            )
        )
    except ImportError:
        rows.append(("geopandas sjoin (polygons left, prepared), all pairs, 1 core", None))

    threads = duckdb_threads()
    label = (
        f"duckdb spatial join, count(*) over all pairs, {threads} threads"
        if threads
        else "duckdb spatial join, count(*) over all pairs"
    )
    secs, note = _bench_duckdb_join(pts, zones)
    rows.append((label + note, secs))
    return rows


def _bench_duckdb_join(
    pts: np.ndarray, zones: np.ndarray, deadline_s: float = 300.0
) -> tuple[float | None, str]:
    """DuckDB spatial join timing, or ``(None, note)`` when unavailable or
    interrupted after ``deadline_s`` (detailed polygons can take DuckDB far
    longer than any other baseline; a bounded wait keeps the sweep finite)."""
    try:
        import threading

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
            timer = threading.Timer(deadline_s, con.interrupt)
            timer.start()
            try:
                con.execute(
                    "SELECT count(*) FROM pts_raw JOIN zones "
                    "ON ST_Intersects(zones.geom, ST_Point(pts_raw.x, pts_raw.y))"
                ).fetchall()
            finally:
                timer.cancel()

        try:
            return timeit(q, warmup=1, repeats=2), ""
        except duckdb.InterruptException:
            return None, f" (interrupted after {deadline_s:.0f} s)"
    except Exception:
        return None, ""


# ---------------------------------------------------------------- driver

def report(title: str, rows: list[tuple[str, float | None]]):
    print(f"\n### {title}")
    # Speedups are computed against the row that answers the same question
    # as the baselines: the all-pairs warm-index join, or otherwise the first
    # mlx-geospatial row.
    ours_name, ours = next(
        (
            (name, secs)
            for name, secs in rows
            if name.startswith("mlx-geospatial") and "all pairs (warm index)" in name
        ),
        rows[0],
    )
    width = max(len(name) for name, _ in rows)
    for name, secs in rows:
        if secs is None and name.startswith("  ["):
            print(f"  {name}")
            continue
        speedup = ""
        if (
            secs is not None
            and ours is not None
            and secs > 0
            and not name.startswith("mlx-geospatial")
        ):
            if "all pairs (warm index)" in ours_name:
                what = "all pairs (warm index)"
            elif "brute force" in ours_name:
                what = "brute force"
            else:
                what = "ours"
            speedup = f"  ({secs / ours:.1f}x vs {what})"
        print(f"  {name:<{width}}  {fmt(secs):>10}{speedup}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=["haversine", "knn", "dense-pip", "join", "crossover"])
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--m", type=int, default=260)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--vertices", type=int, default=48, help="vertices per zone (join, dense-pip)")
    parser.add_argument(
        "--coastline", type=int, default=None,
        help="join against one coastline-like polygon with this many vertices instead of zones",
    )
    parser.add_argument("--all", action="store_true", help="run the README sweep")
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
            "haversine_knn 1M queries, 200k refs, k=10",
            bench_knn(1_000_000, 200_000, 10, rng),
        )
        report(
            "dense PIP 100,000 points x 32 polygons x 48 vertices",
            bench_dense_pip(100_000, 32, 48, rng),
        )
        for n in (100_000, 1_000_000, 10_000_000):
            report(f"join {n:,} points x 260 zones (48 vertices)", bench_join(n, 260, rng))
        report("join 1,000,000 points x 10,000 zones (48 vertices)", bench_join(1_000_000, 10_000, rng))
        report(
            "join 1,000,000 points x 260 zones (2,048 vertices)",
            bench_join(1_000_000, 260, rng, vertices=2048),
        )
        report(
            "join 1,000,000 points x one 150,000-vertex coastline polygon",
            bench_join(1_000_000, 1, rng, coastline=150_000),
        )
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
        if args.coastline and (args.vertices != 48 or args.m != 260):
            parser.error("--coastline replaces the zone layer; --m and --vertices do not apply")
        if args.coastline:
            title = f"join {args.n:,} points x one {args.coastline:,}-vertex coastline polygon"
        else:
            title = f"join {args.n:,} points x {args.m:,} zones ({args.vertices:,} vertices)"
        report(title, bench_join(args.n, args.m, rng, vertices=args.vertices, coastline=args.coastline))
    elif args.op == "crossover":
        bench_crossover(args.m, rng)
    else:
        parser.error("pass --op or --all")


if __name__ == "__main__":
    main()
