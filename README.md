# mlx-geospatial

GPU-accelerated spatial operations for Apple silicon, built on
[MLX](https://github.com/ml-explore/mlx): haversine distances,
point-in-polygon, and grid-indexed spatial joins.

The pitch beyond speed: mlx-geospatial is **array-native in unified memory**.
Re-center a point table once with `PolygonSet.to_local`, and zone
assignment, distance-to-POI features and nearest-neighbor lookups run
entirely on the GPU and come back as MLX arrays: no NumPy round trip, no
copies, no serialization, straight into an MLX training loop or embedding
model (see `examples/mlx_feature_pipeline.py`). GeoPandas users keep NumPy:
the predicates return the container you pass in.

## Install

```bash
pip install mlx-geospatial
```

Requires macOS on Apple silicon (Metal), Python ≥ 3.10, MLX ≥ 0.32.0, `numpy`,
`shapely ≥ 2.0`. `geopandas` is optional (`pip install "mlx-geospatial[geo]"`).

## Quickstart

The first block runs as written (synthetic data stands in for a real zone
layer); the second continues from it and elides the model.

**GeoPandas / NumPy users**: pass NumPy, shapely or GeoSeries, get NumPy back.

```python
import numpy as np, shapely
import mlx_geospatial as ms

rng = np.random.default_rng(0)
pickups = np.column_stack([rng.uniform(-74.03, -73.9, 200_000), rng.uniform(40.7, 40.82, 200_000)])
stations = np.column_stack([rng.uniform(-74.03, -73.9, 500), rng.uniform(40.7, 40.82, 500)])
gx, gy = np.linspace(-74.03, -73.9, 21), np.linspace(40.7, 40.82, 21)
zones = [shapely.box(gx[i], gy[j], gx[i + 1], gy[j + 1]) for j in range(20) for i in range(20)]
# real data: zones = gpd.read_file("taxi_zones.geojson").geometry

# --- haversine distances (coords are (lon, lat) degrees; meters out) ------
d = ms.haversine(pickups[:-1], pickups[1:])              # elementwise, mx.array
dist, idx = ms.haversine_knn(pickups, stations, k=5)     # (N, 5) each, mx.array

# --- point-in-polygon / spatial join --------------------------------------
zone_idx = ms.point_in_polygon(pickups, zones)          # (N,) numpy int32, -1 = none
polyset = ms.PolygonSet(zones)                           # ingest + index once, reuse
labels = ms.point_in_polygon(pickups, polyset)

# GeoPandas-flavored join (pandas DataFrames with a geometry column work too)
# joined = ms.sjoin(trips_gdf, zones_gdf, how="inner")   # adds index_right + zone cols
```

**MLX users**: re-center once, slice per batch, everything stays on the GPU.

```python
import mlx.core as mx, mlx.nn as nn

polyset = ms.PolygonSet(zones)
index = ms.GridIndex(polyset)
local = polyset.to_local(pickups)         # float64 -> fp32 local frame, once; LocalPoints
pickups_mx = mx.array(np.radians(pickups).astype(np.float32))   # raw lon/lat for distances
pois_mx = mx.array(np.radians(stations[:32]).astype(np.float32))
n_zones = len(polyset)
embed = nn.Embedding(n_zones + 1, 16)     # extra row for "no zone"

for start in range(0, len(local), 65_536):
    batch = local[start : start + 65_536]                  # slicing keeps the frame
    zone = index.query(batch)                              # lazy mx.array int32, GPU only
    zone = mx.where(zone < 0, n_zones, zone)               # -1 would read the last row
    dist = ms.haversine_cross(pickups_mx[start : start + 65_536], pois_mx, degrees=False)
    feats = mx.concatenate([embed(zone), dist / 1e4], axis=1)
    mx.eval(feats)                                         # your mlx.nn step goes here
```

`LocalPoints` are re-centered coordinates, so the distance functions reject
them by design; keep the raw lon/lat around for `haversine*`.

## API

| Function | Description |
| --- | --- |
| `haversine(a, b)` | elementwise/broadcast great-circle distance |
| `haversine_cross(a, b)` | all-pairs `(N, M)` distance matrix |
| `haversine_knn(q, r, k)` | k nearest references per query, `(dist, idx)` |
| `point_in_polygon(pts, polys)` | first containing polygon per point (`-1` none); `all_matches=True` for pairs; `exact_boundary=True` for exact GEOS `predicate` semantics on boundary points |
| `contains_matrix(pts, polys)` | dense `(N, M)` bool containment (small M) |
| `sjoin(left, right)` | GeoPandas-flavored point-in-polygon join (`predicate="intersects"` default, `exact_boundary=`) |
| `PolygonSet(geoms)` | ingested, re-centered, GPU-resident polygons |
| `PolygonSet.to_local(points)` | re-center points once into the set's fp32 frame → `LocalPoints` |
| `LocalPoints` | device-resident query points; sliceable; joins on them return MLX arrays |
| `GridIndex(polyset)` | reusable uniform-grid spatial index |
| `PrecisionWarning` | warning emitted for float32/16-bit MLX query points, see below |
| `from_shapely(geoms)` | alias for `PolygonSet(geoms)` |
| `as_point_array(points)` | normalize any host point input to float64 `(N, 2)` |
| `R_EARTH_M` | mean Earth radius (meters) used by the haversine functions |

Output containers: the predicates (`point_in_polygon`, `GridIndex.query`,
`contains_matrix`) return the container you pass in — NumPy for NumPy, lists,
shapely and GeoSeries; MLX arrays for `mx.array` and `LocalPoints`, computed
without leaving unified memory (float32 inputs and `LocalPoints` entirely on
the GPU; float64 re-centering on the MLX CPU stream). The distance functions
always return MLX arrays; `sjoin` returns pandas.

`GridIndex` cells default to a quarter of the median polygon bbox's longer
side (about three candidates per point on typical zone layers), backing off
to a half or the median itself when a few large polygons among many small
ones would inflate the index, and coarsening further to fit the cell/entry
budgets. Construction is adaptive: low-coverage bbox expansions use stable
integer sorting, more expanded grids use lower-memory packed-key sorting, and
material high-occupancy grids use a sort-free direct CSR fill. Every route
retains ascending polygon IDs within each cell. Pass a prebuilt `PolygonSet`
to `point_in_polygon`/`sjoin`, or a `GridIndex` via
`point_in_polygon(points, index=grid)`, to reuse ingestion and the index
across calls.

Conventions: coordinates are `(x, y)` = `(lon, lat)`, degrees; planar math
for containment (project first if you need metric correctness near poles);
spherical math for haversine. No reprojection — that stays pyproj's job. When
both sides carry a CRS (GeoSeries/GeoDataFrame), `sjoin`, `point_in_polygon`,
`contains_matrix` and `PolygonSet.to_local` raise on a mismatch; a CRS-less
side is never checked.

## Precision, honestly

Metal GPUs are fp32-only, and raw lon/lat in fp32 carries ~1 m rounding —
enough to flip containment at polygon edges. mlx-geospatial handles this
structurally rather than hopefully:

- All GPU predicates run in **locally re-centered coordinates**: a float64
  origin (dataset bbox center) is subtracted before casting to fp32,
  restoring ~millimeter resolution at city/state extents.
- **Boundary tolerance policy**: points within `PolygonSet.boundary_tolerance`
  (a few fp32 ulps of the local extent — centimeters for a metro area) of a
  polygon edge may classify either way. Outside that band, results match
  shapely/GEOS; the test suite enforces exactly this contract. `predicate`
  (`"intersects"`, the default, or `"within"`) only matters for points exactly
  on a boundary, inside that band, so by default both give the same answer;
  pass `exact_boundary=True` to re-check the near-boundary points with GEOS
  and get the predicate honored exactly. Finding them is a host pass over all
  points (an STRtree over tolerance-widened edge boxes plus a float64 distance
  test: about 0.3 s per million points on 48-vertex zones, 0.5 s on 10k
  zones), cheaper than a shapely join but far from GPU speed, so reserve it
  for boundary-sensitive work; only the band is re-checked, so interior points
  of invalid geometry keep the even-odd answer. The band scales with the
  *dataset* extent: one stray geometry far from the rest (a placeholder at
  (0, 0), a wrong-hemisphere typo) widens it for everyone, so clean outliers
  before ingesting.
- Cell assignment runs on the GPU in fp32. Polygon cell ranges are planned in
  float64 and padded by 16 ulps of the local extent — more than the kernel's
  bbox widening plus the fp32 assignment error — so whichever cell a point
  near a cell line lands in lists every polygon whose kernel bbox accepts it:
  rounding can only add candidate pairs, never lose a true hit, and host and
  device inputs give bit-identical answers. Tilings whose edges fall on cell
  lines pay for this with each tile listed on both sides of those lines.
- **Device inputs**: host float64 coordinates and `mx.array` float64 inputs
  (build them with `mx.array(x, dtype=mx.float64)`; a plain `mx.array` of a
  float64 NumPy array is silently float32) are re-centered in float64 and keep
  this contract. A float32 (or 16-bit) MLX array already carries ~0.5 ulp of
  raw lon/lat rounding (about 0.4 m at longitude 74), wider than the band at
  city or state extents, so the join re-centers it on the device and emits `mlx_geospatial.PrecisionWarning`
  once per call site; silence it with `warnings.filterwarnings("ignore",
  category=ms.PrecisionWarning)` or make it strict with
  `warnings.simplefilter("error", ms.PrecisionWarning)`.
- **One containment contract**: containment is the even-odd rule over all
  rings on every backend. The CPU fallback assigns grid cells with the device
  graph's fp32 arithmetic and evaluates the same fp32 predicate in NumPy with
  the kernels' operation order, so `backend="cpu"` and `backend="gpu"` are
  bit-identical for every input and auto dispatch can never change an answer
  (the one escape is fp32 subnormals, below ~1e-38, which Metal flushes to
  zero and NumPy keeps: a coordinate, an edge delta or an intermediate of the
  crossing test that small can differ, which needs local coordinates
  themselves near 1e-38 and never occurs for real-world units). Even-odd
  equals GEOS
  fill semantics for valid polygons; on invalid input (self-intersections,
  overlapping multipolygon parts) it differs from GEOS, which you meet only
  through `exact_boundary=True`, and only for the near-boundary points it
  re-checks. Pass `PolygonSet(geoms, validate=True)` to reject invalid
  geometry up front.
- **Long polygons are adaptive**: Metal containment chooses scalar traversal,
  one 32-lane SIMDgroup, or one 128-thread group per predicate. Lanes evaluate
  disjoint edges and XOR integer crossing parity, so the cooperative paths
  preserve the scalar fp32 even-odd result exactly. They are enabled only for
  rings with at least 1,024 edges and bounded work sizes; indexed first-match
  stays scalar unless every candidate in every used cell has a long ring.
- Haversine: absolute error ~1-2 m from fp32 differencing (relative ~1e-4 at
  10 km), far below the ~0.5% sphere-vs-ellipsoid modeling error any
  haversine implementation carries. One exception: within ~2° of the exact
  antipode, arcsin amplification grows absolute error to 10²-10⁴ m (relative
  error stays < 2e-4).

Small host-resident workloads automatically fall back to the NumPy evaluation
of the same predicate: auto dispatch compares its cost (per point plus per
candidate polygon edge, estimated from the index) with the fixed GPU cost of
the route, which puts the crossover at a few hundred points for typical zone
layers, a couple of thousand for sparse tilings and under ten for detailed
polygons, later for all-match queries whose compaction passes cost more
(measured in the crossover table below; `python bench/bench.py --op crossover`
re-derives the constants for another machine). The fallback exists for
consistency, not speed: it is about 1.5-2x slower than GEOS per point on
48-vertex zones (up to ~7x on 2048-vertex zones, faster on sparse tilings)
and carries ~50 µs of fixed setup, so a loop of tiny queries pays roughly
0.05-0.2 ms per call. Device-resident inputs (`mx.array`, `LocalPoints`) run on
the GPU whenever Metal is available, because the fallback would need a host
copy. Force a path with `backend="gpu" | "cpu"` or `MLX_GEOSPATIAL_BACKEND`;
`backend="cpu"` on device inputs copies the fp32 local coordinates to the host
and runs the same predicate there (a parity/debug path).

## Benchmarks

Apple M4 Max (16 cores: 12 performance + 4 efficiency), macOS 26.2, mlx
0.31.2, Python 3.13. These are pre-release reference measurements; v0.1.0
requires MLX ≥ 0.32.0 for an upstream custom-kernel cache correctness fix,
so reproduce them on your supported runtime before making capacity decisions.
End-to-end wall time from host-resident numpy/shapely inputs — ingestion,
host→device transfer, and device result
evaluation/synchronization included; host-input join rows return NumPy labels
as a zero-copy view of the MLX result; best of 3 (exceptions in
`bench/README.md`). CPU baselines: shapely/GEOS, GeoPandas and scikit-learn
run on one core, DuckDB on its default 16 threads; the GPU rows use the whole
GPU, with clocks warmed before each warm-index timing (steady-state burst
numbers, like the crossover tables). Reproduce with `python bench/bench.py
--all` (methodology in `bench/README.md`).

**Point-in-polygon joins: all-pairs comparison.** The comparison denominator
is always `mlx-geospatial` with `all_matches=True`, not the cheaper first-match
mode. Every system evaluates all matching `(point, polygon)` pairs.
`mlx-geospatial`, shapely and GeoPandas materialize the pair indices; DuckDB
only computes `count(*)` and does not materialize them, a lighter output
contract that favors DuckDB. Each baseline cell shows its elapsed time and how
many times longer it took than `mlx-geospatial`. Lower time is better; a larger
“× slower” value means a larger `mlx-geospatial` advantage.

| workload | **mlx-geospatial all pairs (NumPy)** | shapely STRtree, prepared, 1 core | GeoPandas sjoin, prepared, 1 core | DuckDB spatial `count(*)`, 16 threads |
| :--- | ---: | ---: | ---: | ---: |
| 100k pts × 260 zones, 48 vertices | **1.9 ms** | 49 ms · **26× slower** | 56 ms · **30× slower** | 127 ms · **67× slower** |
| 1M pts × 260 zones, 48 vertices | **7.7 ms** | 771 ms · **100× slower** | 805 ms · **105× slower** | 557 ms · **72× slower** |
| 10M pts × 260 zones, 48 vertices | **69 ms** | 8.5 s · **124× slower** | 9.0 s · **131× slower** | 6.4 s · **93× slower** |
| 1M pts × 10k zones, 48 vertices | **8.1 ms** | 884 ms · **109× slower** | 930 ms · **115× slower** | 566 ms · **70× slower** |
| 1M pts × 260 zones, 2,048 vertices | **48 ms** | 1.07 s · **22× slower** | 1.11 s · **23× slower** | 2.9 s · **60× slower** |
| 1M pts × one 150k-vertex polygon | **233 ms** | 1.79 s · **7.7× slower** | 1.79 s · **7.7× slower** | 25 s · **107× slower** |

For example, on 1M points × 260 zones, shapely takes 771 ms versus 7.7 ms
for `mlx-geospatial`: `771 / 7.7 ≈ 100`, hence **100× slower**. Across the 18
displayed baseline comparisons, `mlx-geospatial` is fastest in every one:
**7.7–131× faster, with a median speedup of 71×**. These are results for the
large, warm-index point-in-polygon workloads shown here on the tested M4 Max,
not a claim that `mlx-geospatial` is faster for every geospatial operation,
dataset or machine.

GEOS gets its best configuration: the polygons are the prepared side
(`STRtree(points).query(polygons)`, polygons on the left of `sjoin`), which
is 8-60x faster on detailed polygons than querying a polygon tree with points
and returns the same pairs. The synthetic zones are random star-shaped
polygons that overlap and leave gaps (not a tiling, not convex): about two
thirds of the points fall in a zone, in 1.6 zones on average, so the all-pairs
outputs hold ~1.1 pairs per point. The coastline row is one 150,000-vertex
polygon covering 22% of the extent.

**First-match mode (a different, cheaper operation).** These queries return
only the lowest-index containing polygon for each point, rather than every
matching pair, so they must not be compared directly with the all-pairs
baselines above. The NumPy path measures NumPy input to NumPy output. The
array-native path assumes the points were re-centered once as `LocalPoints`,
returns an MLX array, and includes `mx.eval` in the timing.

| workload | first match (NumPy) | first match (prepared `LocalPoints`) |
| :--- | ---: | ---: |
| 100k pts × 260 zones, 48 vertices | 0.48 ms | 0.33 ms |
| 1M pts × 260 zones, 48 vertices | 2.5 ms | 1.5 ms |
| 10M pts × 260 zones, 48 vertices | 26 ms | 14 ms |
| 1M pts × 10k zones, 48 vertices | 2.6 ms | 1.5 ms |
| 1M pts × 260 zones, 2,048 vertices | 39 ms | 38 ms |
| 1M pts × one 150k-vertex polygon | 548 ms | 547 ms |

Both tables use a warm polygon index. Cold runs, including polygon ingestion
and index construction, take 1.8 ms at 100k × 260, 4.0 ms at 1M × 260,
22 ms at 1M × 10k, 45 ms on the 2,048-vertex zones, and 0.55 s on the
coastline.

Three things these results show that a headline would hide. Against single-core
GEOS in its best configuration the advantage is 26-131x on 48-vertex zones
and *shrinks* with geometric detail: 22x on 2,048-vertex zones, 7.7x on the
150,000-vertex polygon. Prepared GEOS polygons carry an indexed point-in-area
locator that costs roughly log(edges) per test, while our kernels walk every
edge of each candidate ring; GPU parallelism absorbs that up to a few thousand
edges per ring and no further (indexing long rings is on the backlog). The
unprepared orientation, `STRtree(polygons).query(points)`, is what this
harness used before and what produced detailed-geometry speedups in the
hundreds; those numbers measured GEOS's worst configuration, not ours.
DuckDB, which has no prepared geometries but uses 16 threads, stays 60-107x
behind throughout. On the 48-vertex zones per-point overhead dominates our
time (cell assignment and the count/scan/compaction passes, plus ~1.1 ms of
fixed launch cost for the all-pairs route, the floor the crossover table
shows: over half of the 100k row, about 14% at 1M), not edge tests, and the
baselines are within about 2.5x of each other (1.4-1.6x from 1M points up). And on the single
huge polygon first match is slower than all pairs: the first-match kernel
gives every point a thread, and the 61% of points outside the polygon's bbox
(the grid covers only that bbox, so they have no candidates) exit at once and
leave their SIMD lanes idle while their neighbours in the same 32-lane group
walk 150,000 edges, whereas the all-pairs route compacts candidate pairs
first; reordering the same points so the candidate-bearing 39% are contiguous
makes first match as fast as all pairs. Compacting first-match candidates is
a known follow-up.

**Crossover** (`python bench/bench.py --op crossover`; 260 zones, warm index,
host-resident NumPy points, best of 7). The CPU column is the host fallback the
auto dispatch uses: the same fp32 even-odd predicate evaluated in NumPy, so it
returns exactly what the GPU returns. GPU cells are steady-state numbers,
measured in a back-to-back burst with the GPU clocks warmed first; the note
below the tables says what an isolated query costs. First-match queries:

| points | 48-vertex zones: NumPy / GPU | auto | 2048-vertex zones: NumPy / GPU | auto |
| --- | --- | --- | --- | --- |
| 100 | **188 µs** / 419 µs | cpu | 4.9 ms / **343 µs** | gpu |
| 200 | 311 µs / **277 µs** | cpu | 10.3 ms / **334 µs** | gpu |
| 500 | 705 µs / **196 µs** | gpu | 26.0 ms / **344 µs** | gpu |
| 1,000 | 1.4 ms / **190 µs** | gpu | 52.6 ms / **329 µs** | gpu |
| 2,000 | 2.7 ms / **200 µs** | gpu | 108 ms / **358 µs** | gpu |
| 10,000 | 13.5 ms / **216 µs** | gpu | 540 ms / **1.8 ms** | gpu |
| 50,000 | 68.9 ms / **289 µs** | gpu | 2.74 s / **3.2 ms** | gpu |

An axis-aligned square tiling of 256 tiles has only ~9 candidate edges per
point, so the fallback stays cheap for longer (first match / all matches):

| points | square tiling: NumPy / GPU | auto | all matches: NumPy / GPU | auto |
| --- | --- | --- | --- | --- |
| 500 | **99 µs** / 145 µs | cpu | **94 µs** / 1.1 ms | cpu |
| 1,000 | **154 µs** / 156 µs | cpu | **148 µs** / 1.0 ms | cpu |
| 2,000 | 276 µs / **148 µs** | cpu | **265 µs** / 1.0 ms | cpu |
| 5,000 | 607 µs / **148 µs** | gpu | **588 µs** / 1.0 ms | cpu |
| 10,000 | 1.2 ms / **145 µs** | gpu | 1.1 ms / **1.1 ms** | gpu |
| 50,000 | 6.5 ms / **193 µs** | gpu | 6.3 ms / **1.3 ms** | gpu |

All-match queries (`all_matches=True`) carry a larger fixed GPU cost for the
count, scan and compaction passes, so they switch later:

| points | 48-vertex zones: NumPy / GPU | auto | 2048-vertex zones: NumPy / GPU | auto |
| --- | --- | --- | --- | --- |
| 100 | **198 µs** / 1.5 ms | cpu | 4.8 ms / **1.3 ms** | gpu |
| 500 | **701 µs** / 1.2 ms | cpu | 26.1 ms / **1.3 ms** | gpu |
| 1,000 | 1.3 ms / **1.1 ms** | cpu | 52.5 ms / **1.4 ms** | gpu |
| 2,000 | 2.6 ms / **1.2 ms** | gpu | 108 ms / **1.3 ms** | gpu |
| 50,000 | 68.0 ms / **1.5 ms** | gpu | 2.77 s / **4.1 ms** | gpu |

`auto` picked the faster side in every row except three near the thresholds
(first match at 200 simple points and at 2,000 tiles, all matches at 1,000
simple points), each within about 0.2 ms. The fixed GPU cost depends on what the
GPU was doing just before: about 0.2 ms per first-match query and 1.1 ms per
all-match query inside a burst, but roughly 0.8-1.1 ms and 2.5 ms for a query
issued after tens of milliseconds of host-only work have let the GPU idle
(`bench.py --op crossover` prints both). The dispatch constants
(`GPU_LAUNCH_US = 350`, `GPU_ALL_MATCHES_LAUNCH_US = 1500`) sit between the
two regimes: picking the CPU while the GPU was warm loses at most ~0.2 ms
(~0.45 ms for all matches), picking the GPU after it has idled loses up to
~0.6 ms (~1 ms for all matches) just above the threshold, so a loop of small
host queries interleaved with heavy host work may prefer `backend="cpu"` for a
few hundred points more. The GPU path is a near-constant
launch cost until kernel work shows (tens of thousands of points for simple
polygons, a few thousand for detailed ones) while the fallback grows per point
and per candidate edge, so the model switches first-match queries at about 250
points for 48-vertex zones, about 2,100 for the square tiling and under ten
for 2048-vertex zones, and all-match queries roughly four times later.
Device-resident inputs skip the comparison and run on the GPU whenever Metal is
available.

**Haversine** (`mx.compile`-fused distances; kNN via adaptive scalar/SIMDgroup
selection kernels). Our kNN is exact brute force with nothing to prebuild, so
the BallTree baseline is shown both with the tree built inside the timing (a
one-off query) and query-only on a prebuilt tree (repeated queries against a
fixed reference set, where the tree amortizes):

| workload | mlx-geospatial | numpy float64, 1 core | scikit-learn, 1 core |
| --- | --- | --- | --- |
| cross 20k × 20k | **10.2 ms** | 2.69 s (263x) | 3.98 s (389x) |

| kNN workload (k=10) | mlx-geospatial (brute force) | BallTree build + query | BallTree query only (prebuilt) |
| --- | --- | --- | --- |
| 1k queries × 1M refs | 26 ms | 305 ms (11.7x) | **9.1 ms (0.35x)** |
| 1M queries × 10k refs | **147 ms** | 4.53 s (31x) | 4.52 s (31x) |
| 200k queries × 200k refs | **548 ms** | 1.19 s (2.2x) | 1.14 s (2.1x) |
| 1M queries × 200k refs | **2.62 s** | 5.77 s (2.2x) | 5.70 s (2.2x) |

Honesty notes: `haversine_knn` is exact brute force. For `k <= 32` it runs a
shape-adaptive fused Metal selector: 32 lanes cooperate on each query when
query parallelism is scarce, while large query batches retain the faster
one-thread-per-query kernel. Neither materializes an `(N, M)` matrix or sorts
the full reference set. Brute force costs queries × references, a BallTree
queries × log(references), so the reference-set size decides the winner, not
the query count: against 200k references brute force is cheaper per query
(about 3 µs vs 6 µs here) and the GPU wins 2x, against 10k references 30x;
against 1M references a prebuilt tree is cheaper per query (about 10 µs vs
20-25 µs) at any query count, and only the tree-building cost puts us ahead
(1k × 1M above; at 100k queries even build + query beats brute force). With
`k > 32` (or without an active Metal device) kNN falls
back to a chunked distance-matrix + `argpartition` path that is sort-bound; at
large reference sets a BallTree can win there too. Below a few hundred host
points (fewer for detailed polygons) the join auto-dispatches to the host
fallback, because the fixed GPU launch overhead dominates there; the
crossover table shows both sides.

## Limitations (v0.1)

- Points-in-polygons only: no polygon-polygon predicates, overlays, or
  distance-to-polygon (backlog).
- Uniform grid index; heavily skewed polygon sizes may want the quadtree
  planned for v0.2.
- `all_matches=True` uses bounded adaptive GPU count/scan/ordered compaction
  without NumPy candidate-pair expansion. A long-ring candidate or at least
  8,192 candidate edges in one used grid cell selects the pair-parallel route;
  its returned arrays still require memory proportional to the number of hits.
- Containment walks every edge of a candidate ring, so on rings with tens of
  thousands of vertices single-core GEOS with prepared polygons (an indexed
  point-in-area locator) comes within about 8x of the GPU. Indexing long
  rings is on the backlog.
- First match gives every point a GPU thread, so on a layer dominated by one
  huge polygon (the 150k-vertex coastline row) points outside its bbox idle
  their SIMD lanes while neighbours walk the ring, and `all_matches=True`,
  which compacts candidate pairs first, is faster. Compacting first-match
  candidates the same way is on the backlog.
- `sjoin` is a first-match (lowest polygon index) point-in-polygon join with
  the GeoPandas default predicate `intersects`; `predicate="within"` only
  differs on boundary points and needs `exact_boundary=True` to be honored.
  Use `all_matches=True` on `point_in_polygon` for multi-match joins.

## License

MIT
