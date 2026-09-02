# mlx-spatial

GPU-accelerated spatial operations for Apple silicon, built on
[MLX](https://github.com/ml-explore/mlx): haversine distances,
point-in-polygon, and grid-indexed spatial joins.

The pitch beyond speed: mlx-spatial is **array-native in unified memory**.
Re-center a point table once with `PolygonSet.to_local`, and zone
assignment, distance-to-POI features and nearest-neighbor lookups run
entirely on the GPU and come back as MLX arrays: no NumPy round trip, no
copies, no serialization, straight into an MLX training loop or embedding
model (see `examples/mlx_feature_pipeline.py`). GeoPandas users keep NumPy:
the predicates return the container you pass in.

## Install

```bash
pip install mlx-spatial          # once published; for now:
pip install -e .
```

Requires macOS on Apple silicon (Metal), Python ≥ 3.10, `mlx`, `numpy`,
`shapely ≥ 2.0`. `geopandas` is optional (`pip install "mlx-spatial[geo]"`).

## Quickstart

The first block runs as written (synthetic data stands in for a real zone
layer); the second continues from it and elides the model.

**GeoPandas / NumPy users**: pass NumPy, shapely or GeoSeries, get NumPy back.

```python
import numpy as np, shapely
import mlx_spatial as ms

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
| `point_in_polygon(pts, polys)` | first containing polygon per point (`-1` none); `all_matches=True` for pairs |
| `contains_matrix(pts, polys)` | dense `(N, M)` bool containment (small M) |
| `sjoin(left, right)` | GeoPandas-flavored point-in-polygon join |
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
enough to flip containment at polygon edges. mlx-spatial handles this
structurally rather than hopefully:

- All GPU predicates run in **locally re-centered coordinates**: a float64
  origin (dataset bbox center) is subtracted before casting to fp32,
  restoring ~millimeter resolution at city/state extents.
- **Boundary tolerance policy**: points within `PolygonSet.boundary_tolerance`
  (a few fp32 ulps of the local extent — centimeters for a metro area) of a
  polygon edge may classify either way. Outside that band, results match
  shapely/GEOS; the test suite enforces exactly this contract. The band
  scales with the *dataset* extent: one stray geometry far from the rest
  (a placeholder at (0, 0), a wrong-hemisphere typo) widens it for everyone,
  so clean outliers before ingesting.
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
  city or state extents, so the join re-centers it on the device and emits `mlx_spatial.PrecisionWarning`
  once per call site; silence it with `warnings.filterwarnings("ignore",
  category=ms.PrecisionWarning)` or make it strict with
  `warnings.simplefilter("error", ms.PrecisionWarning)`.
- **Valid geometry assumed**: containment is the even-odd rule over all rings,
  which equals GEOS fill semantics for valid polygons. On invalid input
  (self-intersections, overlapping multipolygon parts) even-odd differs from
  GEOS, and the CPU fallback will disagree with the GPU. Pass
  `PolygonSet(geoms, validate=True)` to reject invalid geometry up front.
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

Small host-resident workloads (< ~100k candidate pairs) automatically fall
back to shapely — single-threaded GEOS beats a kernel launch there.
Device-resident inputs (`mx.array`, `LocalPoints`) run on the GPU whenever
Metal is available, because the fallback would need a host copy. Force a path
with `backend="gpu" | "cpu"` or `MLX_SPATIAL_BACKEND`; `backend="cpu"` on
device inputs runs GEOS on the fp32 local coordinates (a parity/debug path).

## Benchmarks

Apple M4 Max, macOS 26.2, mlx 0.31.2, Python 3.13. End-to-end wall time from
host-resident numpy/shapely inputs — ingestion, host→device transfer, and
device result evaluation/synchronization included; host-input join rows
return NumPy labels as a zero-copy view of the MLX result; best of 3. Reproduce with `python bench/bench.py --all` (methodology in
`bench/README.md`).

**Point-in-polygon join** (uniform random points × synthetic 48-vertex zones,
NYC-like extent; mlx-spatial numbers use a warm index. The first mlx-spatial
column is NumPy in → NumPy out; the second is the array-native pattern:
`LocalPoints` re-centered once, MLX array out, `mx.eval` inside the timing.
Speedups are against the NumPy column. The harness also reports cold runs with
polygon ingestion and index construction included: 3.8 ms at 1M × 260 zones,
19 ms at 1M × 10k zones):

| workload | mlx-spatial (NumPy) | mlx-spatial (LocalPoints) | shapely STRtree | GeoPandas sjoin | DuckDB spatial |
| --- | --- | --- | --- | --- | --- |
| 100k pts × 260 zones | **1.0 ms** | 0.95 ms | 61 ms (59x) | 28 ms (28x) | 118 ms (116x) |
| 1M pts × 260 zones | **2.5 ms** | 1.5 ms | 608 ms (247x) | 581 ms (236x) | 550 ms (224x) |
| 10M pts × 260 zones | **24 ms** | 13 ms | 6.4 s (262x) | 10.3 s (426x) | 6.6 s (273x) |
| 1M pts × 10k zones | **3.1 ms** | 1.6 ms | 892 ms (290x) | 692 ms (225x) | 576 ms (188x) |

**Haversine** (`mx.compile`-fused distances; kNN via adaptive scalar/SIMDgroup
selection kernels):

| workload | mlx-spatial | numpy float64 | scikit-learn |
| --- | --- | --- | --- |
| cross 20k × 20k | **9.6 ms** | 2.67 s (278x) | 3.91 s (406x) |
| knn 1k queries × 1M refs, k=10 | **26.6 ms** | — | BallTree 328 ms (12.3x) |
| knn 1M queries × 10k refs, k=10 | **154 ms** | — | BallTree 4.54 s (30x) |
| knn 200k queries × 200k refs, k=10 | **562 ms** | — | BallTree 1.22 s (2.2x) |
| knn 1M queries × 200k refs, k=10 | **2.63 s** | — | BallTree 6.02 s (2.3x) |

Honesty notes: `haversine_knn` is exact brute force. For `k <= 32` it runs a
shape-adaptive fused Metal selector: 32 lanes cooperate on each query when
query parallelism is scarce, while large query batches retain the faster
one-thread-per-query kernel. Neither materializes an `(N, M)` matrix or sorts
the full reference set. In the 1k × 1M row, the GPU-resident selector core is
24.8 ms versus 355.4 ms for the retained scalar kernel (14.3x); the table's
26.6 ms includes host ingestion and transfer. With `k > 32` (or without an
active Metal device) kNN falls back to a chunked distance-matrix +
`argpartition` path that is sort-bound; at large reference sets a BallTree can
win there. Below ~1e5 candidate pairs the join auto-dispatches to shapely,
because kernel-launch overhead dominates at small sizes.

## Limitations (v0.1)

- Points-in-polygons only: no polygon-polygon predicates, overlays, or
  distance-to-polygon (backlog).
- Uniform grid index; heavily skewed polygon sizes may want the quadtree
  planned for v0.2.
- `all_matches=True` uses bounded adaptive GPU count/scan/ordered compaction
  without NumPy candidate-pair expansion. A long-ring candidate or at least
  8,192 candidate edges in one used grid cell selects the pair-parallel route;
  its returned arrays still require memory proportional to the number of hits.
- `sjoin` implements point-`within`-polygon with first-match (lowest polygon
  index) semantics; use `all_matches=True` on `point_in_polygon` for
  multi-match joins.

## License

MIT
