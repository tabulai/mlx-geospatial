# Historical benchmark measurements

For the current six-workload spatial-join comparison, see the
[benchmark guide](README.md). Fresh component measurements, including kNN,
distances, warm queries and CPU/GPU crossover tables, are in the
[current component report](results/2026-10-19-opus-speedups/COMPONENTS.md).
The material below preserves earlier measurements
and their original timing boundaries; it is not the performance summary for
the current workspace build.

The first sections are pre-release M4 Max array-query, distance, and dispatch
measurements. They use cached polygon indexes for warm queries, omit exact
boundary refinement, and produce different outputs from the CPU baselines.
They must not be quoted as equivalent, fresh-index GeoDataFrame join speedups.
The distance and kNN measurements also retain their stated float32/float64 and
index-reuse distinctions. Their original values are retained here; the current
component report contains a separate rerun on the latest measured source.

The dated optimization studies in this archive describe
intermediate workspace builds. Their measurements remain useful for tracing
those changes, but all six spatial-join rows are superseded by the current
complete run. Do not combine rows from different revisions into a current table.

## Original pre-release report

These measurements show where mlx-geospatial is fast and where a CPU library
can be a better choice. The largest gains come from large point-in-polygon
queries and distance matrices. Small queries pay for GPU launch overhead,
and a prebuilt BallTree can beat brute-force GPU nearest-neighbor search.

## Test machine and runtime

Apple M4 Max, 16 CPU cores (12 performance + 4 efficiency), macOS 26.2,
Python 3.13, MLX 0.31.2.

These are pre-release reference measurements. The v0.1.0 release requires
MLX ≥ 0.32.0 for an upstream custom-kernel cache correctness fix. Reproduce
the measurements on your supported runtime before making capacity decisions.
The tables describe these workloads on one machine, not every dataset or
geospatial operation. The superseded end-to-end updates at the end of this file
use a newer runtime and a different, explicitly stated timing protocol.

## What the timings include

- **Host inputs:** unless stated otherwise, MLX timings start from NumPy
  points and include conversion, re-centering where needed, input transfer,
  execution, and result evaluation/synchronization. Host-input joins return
  fully materialized NumPy results, using a zero-copy view of the MLX result.
  Distance functions return MLX arrays. “NumPy” in a join label describes
  the input/output container; the timed query runs on the GPU.
- **Warm and cold joins:** warm queries reuse a `PolygonSet` and `GridIndex`;
  cold queries build them inside the timer. Warm rows also warm the GPU clocks
  before measurement, so they represent repeated queries in a burst. An
  isolated query after the GPU has idled has a larger launch cost.
- **Prepared points:** the `LocalPoints` rows exclude the one-time
  re-centering/upload with `PolygonSet.to_local`. They return MLX arrays and
  include `mx.eval` inside the timer.
- **CPU baselines:** Shapely/GEOS, GeoPandas and scikit-learn use one core.
  DuckDB uses its default 16 threads. MLX uses the whole GPU.
- **Repeated measurements:** the default is best of 3 after one warmup.
  DuckDB uses best of 2 after one warmup. GeoPandas has no warmup and uses
  3 runs, except at 10M points, where it uses one run. The crossover tables
  use best of 7 after warmup. These are minimum wall times, not latency
  distributions or error bars.
- **Unavailable baselines:** the harness reports `n/a` rather than silently
  omitting them. DuckDB queries are interrupted after 300 seconds.

The benchmark does not time `exact_boundary=True`. That option adds a
conservative GPU filter, float64 host boundary checks and GEOS refinement
(or the full host scan on fallback routes), so these GPU timings should not
be used to estimate its cost.

## Point-in-polygon joins

### All matching pairs

Every system evaluates all matching `(point, polygon)` pairs. MLX, Shapely
and GeoPandas materialize the matches. DuckDB returns only `count(*)`, which
gives it a lighter output contract. All ratios below use MLX
`all_matches=True` as the denominator.

Each baseline cell shows elapsed time and how many times longer it took than
MLX. For example, `771 / 7.7 ≈ 100`, so 771 ms is **100× slower** than 7.7 ms.

| Workload | mlx-geospatial all pairs (NumPy) | Shapely STRtree, prepared, 1 core | GeoPandas sjoin, prepared, 1 core | DuckDB spatial `count(*)`, 16 threads |
| :--- | ---: | ---: | ---: | ---: |
| 100k points × 260 zones, 48 vertices | **1.9 ms** | 49 ms · **26× slower** | 56 ms · **30× slower** | 127 ms · **67× slower** |
| 1M points × 260 zones, 48 vertices | **7.7 ms** | 771 ms · **100× slower** | 805 ms · **105× slower** | 557 ms · **72× slower** |
| 10M points × 260 zones, 48 vertices | **69 ms** | 8.5 s · **124× slower** | 9.0 s · **131× slower** | 6.4 s · **93× slower** |
| 1M points × 10k zones, 48 vertices | **8.1 ms** | 884 ms · **109× slower** | 930 ms · **115× slower** | 566 ms · **70× slower** |
| 1M points × 260 zones, 2,048 vertices | **48 ms** | 1.07 s · **22× slower** | 1.11 s · **23× slower** | 2.9 s · **60× slower** |
| 1M points × one 150k-vertex polygon | **233 ms** | 1.79 s · **7.7× slower** | 1.79 s · **7.7× slower** | 25 s · **107× slower** |

MLX is fastest in all 18 displayed comparisons: **7.7–131× faster**, with a
**71× median of the displayed ratios**. The ratios are retained from the
reference measurements; rounding the displayed times may give slightly
different ratios.

### First match per point

First-match mode returns the lowest-index containing polygon for each point,
or `-1` if none contains it. Its output differs from finding every match, so
these times are not the denominator for the CPU comparisons above. This mode
usually runs faster, with the huge-polygon exception explained below.

| Workload | NumPy input → NumPy output | Prepared `LocalPoints` → MLX output |
| :--- | ---: | ---: |
| 100k points × 260 zones, 48 vertices | 0.48 ms | 0.33 ms |
| 1M points × 260 zones, 48 vertices | 2.5 ms | 1.5 ms |
| 10M points × 260 zones, 48 vertices | 26 ms | 14 ms |
| 1M points × 10k zones, 48 vertices | 2.6 ms | 1.5 ms |
| 1M points × 260 zones, 2,048 vertices | 39 ms | 38 ms |
| 1M points × one 150k-vertex polygon | 548 ms | 547 ms |

### Cold first-match queries

These timings include polygon ingestion and index construction, as well as
the NumPy-input first-match query. The preceding two tables reuse the index.

| Workload | Cold first match |
| :--- | ---: |
| 100k points × 260 zones, 48 vertices | 1.8 ms |
| 1M points × 260 zones, 48 vertices | 4.0 ms |
| 1M points × 10k zones, 48 vertices | 22 ms |
| 1M points × 260 zones, 2,048 vertices | 45 ms |
| 1M points × one 150k-vertex polygon | 0.55 s |

### Data and baseline setup

The zones are synthetic, random star-shaped polygons with varying radii.
They overlap and leave gaps; they are neither convex nor a tiling. About two
thirds of points fall in a zone, with 1.6 zones per matched point on average,
producing roughly 1.1 output pairs per input point. The coastline case uses
one 150,000-vertex polygon covering 22% of the extent. The harness prints
coverage for each generated dataset.

Shapely uses `STRtree(points).query(polygons, predicate="intersects")`.
GeoPandas places polygons on the left of `sjoin`. Both orientations let GEOS
prepare the polygons and use its indexed point-in-area locator. On detailed
polygons this is 8–60× faster than querying a polygon tree with points, with
the same matching pairs. Earlier versions of this harness used the slower
orientation and reported detailed-geometry speedups in the hundreds; those
figures are not the comparison reported here.

Shapely point geometries and the point STRtree are built inside the timer.
GeoPandas likewise builds the point GeoDataFrame and its index inside the
timer; its polygon GeoDataFrame is prebuilt. DuckDB prebuilds its polygon
table, including WKB conversion, and registers the point DataFrame before
timing. Its timed query scans the registered points, constructs point
geometries and counts the matches. These setup boundaries matter when
comparing a one-off query with repeated queries.

### Why the speedup changes

On 48-vertex zones, MLX is 26–131× faster than the single-core GEOS baselines.
The advantage falls to 22–23× on 2,048-vertex zones and 7.7× on the
150,000-vertex polygon. These measurements predate the y-band edge index:
the GPU kernels then walked every edge of each candidate ring, while GEOS
could use its indexed locator. More detailed geometry therefore reduced
MLX's advantage. DuckDB, which uses 16 threads but has no prepared
geometries in this comparison, takes 60–107× longer.

For simple zones, much of MLX's time goes to cell assignment and
count/scan/compaction, rather than edge tests. The all-pairs route has about
1.1 ms of fixed launch cost: over half the 100k-point time and about 14% of
the 1M-point time. The CPU baselines are within about 2.5× of one another
(1.4–1.6× from 1M points upward).

First match is slower than all pairs on the huge polygon. About 61% of
points lie outside its bounding box and exit immediately; their GPU lanes
then sit idle while neighboring lanes walk the 150,000 edges. The all-pairs
route first compacts candidate pairs. Reordering the same points so the 39%
with candidates are contiguous makes first match as fast as all pairs.
These observations also predate the y-band edge index. Current sufficiently
large queries against long rings use bounded edge bands for both containment
and exact-boundary filtering, with construction cost included in the
separate end-to-end benchmark.

## When CPU queries beat GPU queries

These crossover measurements use a warm index and host-resident NumPy
points. “CPU” is mlx-geospatial's NumPy implementation of the same fp32
even-odd predicate, not Shapely. `auto` shows the backend selected by the
dispatch cost model. GPU times are measured in a burst after warming its
clocks; isolated queries cost more, as explained below.

### First match: 260 zones

| Points | 48-vertex zones: CPU / GPU | auto | 2,048-vertex zones: CPU / GPU | auto |
| ---: | --- | --- | --- | --- |
| 100 | **188 µs** / 419 µs | cpu | 4.9 ms / **343 µs** | gpu |
| 200 | 311 µs / **277 µs** | cpu | 10.3 ms / **334 µs** | gpu |
| 500 | 705 µs / **196 µs** | gpu | 26.0 ms / **344 µs** | gpu |
| 1,000 | 1.4 ms / **190 µs** | gpu | 52.6 ms / **329 µs** | gpu |
| 2,000 | 2.7 ms / **200 µs** | gpu | 108 ms / **358 µs** | gpu |
| 10,000 | 13.5 ms / **216 µs** | gpu | 540 ms / **1.8 ms** | gpu |
| 50,000 | 68.9 ms / **289 µs** | gpu | 2.74 s / **3.2 ms** | gpu |

### Square tiling: 256 tiles

This axis-aligned tiling has only about 9 candidate edges per point, so the
CPU fallback stays competitive for longer.

| Points | First match: CPU / GPU | auto | All matches: CPU / GPU | auto |
| ---: | --- | --- | --- | --- |
| 500 | **99 µs** / 145 µs | cpu | **94 µs** / 1.1 ms | cpu |
| 1,000 | **154 µs** / 156 µs | cpu | **148 µs** / 1.0 ms | cpu |
| 2,000 | 276 µs / **148 µs** | cpu | **265 µs** / 1.0 ms | cpu |
| 5,000 | 607 µs / **148 µs** | gpu | **588 µs** / 1.0 ms | cpu |
| 10,000 | 1.2 ms / **145 µs** | gpu | 1.1 ms / **1.1 ms** | gpu |
| 50,000 | 6.5 ms / **193 µs** | gpu | 6.3 ms / **1.3 ms** | gpu |

### All matches: 260 zones

Finding all matches adds count, scan and compaction passes on the GPU, so
the crossover occurs at larger point counts.

| Points | 48-vertex zones: CPU / GPU | auto | 2,048-vertex zones: CPU / GPU | auto |
| ---: | --- | --- | --- | --- |
| 100 | **198 µs** / 1.5 ms | cpu | 4.8 ms / **1.3 ms** | gpu |
| 500 | **701 µs** / 1.2 ms | cpu | 26.1 ms / **1.3 ms** | gpu |
| 1,000 | 1.3 ms / **1.1 ms** | cpu | 52.5 ms / **1.4 ms** | gpu |
| 2,000 | 2.6 ms / **1.2 ms** | gpu | 108 ms / **1.3 ms** | gpu |
| 50,000 | 68.0 ms / **1.5 ms** | gpu | 2.77 s / **4.1 ms** | gpu |

### Dispatch and GPU idle time

`auto` chose the faster backend in every displayed row except three near a
threshold: first match at 200 simple points and 2,000 tiling points, and all
matches at 1,000 simple points. Each difference was within about 0.2 ms.

Inside a burst, fixed GPU costs are about 0.2 ms for first match and 1.1 ms
for all matches. After tens of milliseconds of host-only work let the GPU
idle, these rise to roughly 0.8–1.1 ms and 2.5 ms. The harness reports both
regimes. Dispatch uses intermediate constants, `GPU_LAUNCH_US = 350` and
`GPU_ALL_MATCHES_LAUNCH_US = 1500`.

Near a threshold, choosing CPU while the GPU is warm can cost about 0.2 ms
extra (0.45 ms for all matches); choosing GPU after it has idled can cost
about 0.6 ms extra (1 ms for all matches). A loop of small queries interleaved
with heavy host work may benefit from `backend="cpu"` for a few hundred
points beyond the normal threshold.

The model switches first-match queries at about 250 points for 48-vertex
zones, 2,100 for the square tiling, and under ten for 2,048-vertex zones.
All-match thresholds are roughly four times later. GPU time remains near its
launch cost until workloads reach tens of thousands of points for simple
polygons, or a few thousand for detailed polygons. CPU work grows with both
point count and candidate edge count. Device-resident inputs bypass this
comparison and use the GPU whenever Metal is available.

The fallback prioritizes consistency with the GPU predicate, not beating
GEOS: it is about 1.5–2× slower per point on 48-vertex zones and up to 7×
slower on 2,048-vertex zones, though faster on sparse tilings. It also has
about 50 µs of fixed setup, giving roughly 0.05–0.2 ms per tiny query.

## Haversine distance matrices

MLX uses compiled, fused distance operations. All rows start from host points;
the MLX result is an evaluated MLX array. NumPy computes in float64.

| Workload | mlx-geospatial | NumPy float64, 1 core | scikit-learn, 1 core |
| :--- | ---: | ---: | ---: |
| All pairs, 20k × 20k points | **10.2 ms** | 2.69 s · **263× slower** | 3.98 s · **389× slower** |

## Haversine nearest neighbors

MLX performs exact brute-force selection under its computed distance metric;
it has no search index to build. The BallTree columns show two different
uses: building a tree for a one-off query, and querying an existing tree.
BallTree input conversion to radians occurs before both timers. The MLX
timer includes input conversion, execution and evaluation of both returned
arrays. Every row uses `k=10`.

| Queries × references | mlx-geospatial | BallTree build + query | BallTree query only |
| :--- | ---: | ---: | ---: |
| 1k × 1M | 26 ms | 305 ms · 11.7× slower | **9.1 ms · 0.35× MLX time** |
| 1M × 10k | **147 ms** | 4.53 s · 31× slower | 4.52 s · 31× slower |
| 200k × 200k | **548 ms** | 1.19 s · 2.2× slower | 1.14 s · 2.1× slower |
| 1M × 200k | **2.62 s** | 5.77 s · 2.2× slower | 5.70 s · 2.2× slower |

The prebuilt BallTree wins the first row: 9.1 ms versus 26 ms. A larger query
batch does not necessarily reverse that result. For 1M references, the tree
costs about 10 µs per query here, versus 20–25 µs for brute force; at 100k
queries, even build + query beats brute force. With 200k references, the
measured costs are about 6 µs per query for the tree and 3 µs for MLX, giving
MLX roughly a 2× advantage. With 10k references its advantage is about 30×.

For `k <= 32`, fused Metal selectors avoid storing a full query × reference
matrix or sorting the entire reference set. A 32-lane group cooperates on a
query when query parallelism is scarce; larger batches use one thread per
query. For `k > 32`, or without an active Metal device, kNN uses chunked
distance matrices and `argpartition`. That path can be limited by selection
cost, and BallTree can also win on large reference sets.

## Reproduce the results

Run from the repository root with the benchmark dependencies installed:

```bash
python -m pip install -e ".[bench]"
python bench/bench.py --all
python bench/bench.py --op join --n 10000000 --m 260
python bench/bench.py --op join --n 1000000 --m 260 --vertices 2048
python bench/bench.py --op join --n 1000000 --coastline 150000
python bench/bench.py --op haversine --n 20000 --m 20000
python bench/bench.py --op knn --n 1000 --m 1000000 --k 10
python bench/bench.py --op knn --n 200000 --m 200000 --k 10
python bench/bench.py --op dense-pip --n 100000 --m 32 --vertices 48
python bench/bench.py --op crossover --m 260
```

`--op crossover` also fits the dispatch cost model for the current machine.
`--op dense-pip` is a separate kernel microbenchmark: it starts with a
pre-ingested `PolygonSet` and GPU-resident points, comparing the direct 2-D
Metal kernel with its retained composed implementation. It checks matching
outputs outside the timer and does not measure end-to-end ingestion.

## Superseded long-polygon study — 2026-10-19

The shared y-band edge index improves the 1M-point × one 150k-vertex case
with exact boundaries and complete joined GeoDataFrame outputs:

| Machine | Prior optimized MLX | Edge-index MLX | Improvement | GeoPandas CPU | MLX speedup over GeoPandas |
|---|---:|---:|---:|---:|---:|
| M4 Max, 40 GPU cores | 588 ms | **219 ms** | **2.68×** | 1,660 ms | **7.57×** |
| M1 Ultra, 64 GPU cores | 794 ms | **298 ms** | **2.66×** | 2,748 ms | **9.22×** |

These are medians of three alternating paired trials after one full warmup
per method. Every timed call constructs fresh indexes, starting from the
same ready GeoDataFrames. MLX includes polygon ingestion, the new edge index,
point extraction, all-pair GPU containment, exact float64/GEOS refinement,
attribute assembly and sorting. GeoPandas uses native `sjoin(how="inner",
predicate="within")`, with the same output and sorting requirements. Input
frame creation, runtime initialization, I/O and validation are excluded.
The prior MLX version already includes the short-ring boundary filter and
join-output optimizations; the table isolates the additional long-ring work.

All 16 paired output comparisons across the two implementations and both
machines passed, with identical attributes, indexes, dtypes, geometry and
CRS for all **217,315 rows**. The original seed-0 benchmark's complete input
generation sequence was replayed and its fingerprints verified. Both hosts
used Python 3.14.2, MLX 0.32.4, NumPy 2.5.4, Shapely 2.2.0, pandas 3.0.6 and
GeoPandas 1.2.0; macOS was 26.6.2 on M4 and 26.5.1 on M1. Metal validation
was off for timings and on for the full suite: **591 tests passed per host**.
These measurements describe this synthetic workload, including index build
cost; folded polygons with excessive edge replication retain prior traversal.

## Superseded full end-to-end run — 2026-10-19

Fresh measurements of all six workloads with all spatial-join optimizations
enabled, using the same final package on both machines. This is a separate
full-suite run from the long-polygon-only measurements above. All values are
medians of three alternating paired trials after one full warmup per method.

| Points × polygons | Vertices | M1 Ultra MLX | M1 GeoPandas | Speedup | M4 Max MLX | M4 GeoPandas | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100k × 260 | 48 | 34.2 ms | 85.4 ms | **2.50×** | 25.7 ms | 49.2 ms | **1.91×** |
| 1M × 260 | 48 | 199 ms | 1.70 s | **8.56×** | 148 ms | 779 ms | **5.27×** |
| 10M × 260 | 48 | 2.70 s | 22.57 s | **8.35×** | 1.76 s | 11.32 s | **6.44×** |
| 1M × 10k | 48 | 495 ms | 1.63 s | **3.29×** | 364 ms | 932 ms | **2.56×** |
| 1M × 260 | 2,048 | 750 ms | 1.52 s | **2.03×** | 451 ms | 1.07 s | **2.38×** |
| 1M × 1 | 150,000 | 296 ms | 2.82 s | **9.53×** | 206 ms | 1.64 s | **7.97×** |

Every timer includes fresh spatial index construction, the MLX edge index
where applicable, point extraction, exact float64/GEOS boundary refinement,
all matching pairs, complete joined GeoDataFrames with attributes, and
canonical sorting. Both methods start with ready input GeoDataFrames; input
construction/copies, imports, cold kernel compilation, I/O and validation are
excluded. MLX uses the unchanged query-plus-attribute-assembly adapter; native
`ms.sjoin` is validated separately rather than timed in this comparison.

**All 48 paired full-output comparisons passed** across the two hosts. Input
fingerprints, output pair hashes, dependency versions and package source
hashes matched. MLX was faster in all six cases on each host: **2.03–9.53×**
on M1 Ultra (median ratio **5.82×**) and **1.91–7.97×** on M4 Max (median
ratio **3.91×**). The M4 completed the MLX joins **1.33–1.66×** faster than
the M1. These are descriptive results for six synthetic workloads; individual
trial samples and ranges are retained with the results. Runtime versions are
the same as the preceding end-to-end update. The unchanged wheel had already
passed 591 tests per host; this rerun checked every benchmark output afresh.

## Superseded 2,048-vertex optimization study — 2026-10-19

A subsequent change extends the shared edge bands to 2,048-vertex rings and
reuses them for sparse float64 boundary checks. This avoids a second full
edge STRtree while preserving the original exact-boundary arithmetic and
bounded fallbacks. Fresh before/after measurements of **1M points × 260
polygons, 2,048 vertices** include all index builds and complete outputs:

| Machine | Prior optimized MLX | MLX at that revision | Improvement | GeoPandas CPU | MLX speedup over GeoPandas |
|---|---:|---:|---:|---:|---:|
| M1 Ultra, 64 GPU cores | 749 ms | **303 ms** | **2.47×** | 1,519 ms | **5.02×** |
| M4 Max, 40 GPU cores | 450 ms | **222 ms** | **2.03×** | 1,155 ms | **5.21×** |

The unchanged end-to-end adapter, ready input frames, exact `within`/inner
semantics, full attribute assembly and canonical sorting follow the protocol
above. Each median uses three alternating paired trials after one warmup;
GeoPandas times in this table are from the current-build paired run. The
baseline is the already optimized package from the preceding full rerun.
All **16 paired comparisons** passed, with identical **1,076,107 output rows**,
attributes, indexes, dtypes, geometry and CRS. Input fingerprints and matching
pair hashes agree across versions and hosts. **610 tests passed per host**
with Metal validation enabled. These are measurements of one synthetic
workload; they supersede its row in the earlier full-run table, not the other
workloads' timings.

## Superseded complete E2E run — 2026-10-19, before the latest speedups

The preceding complete six-case comparison measured the build incorporated in
`fc0bb6e`, before threaded point extraction, the host float64 edge grid,
polygon-major GEOS refinement, and chord-distance kNN rejection. Its
[raw results, tables, charts and manifest](results/2026-10-19-full-e2e/) remain
unchanged. The [current benchmark guide](README.md) points to a new complete
run of commit `8805228`; do not combine timings from the two source revisions.
