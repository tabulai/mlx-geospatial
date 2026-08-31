# mlx-spatial

GPU-accelerated spatial operations for Apple silicon, built on
[MLX](https://github.com/ml-explore/mlx): haversine distances,
point-in-polygon, and grid-indexed spatial joins.

The pitch beyond speed: mlx-spatial is **array-native in unified memory**.
Spatial feature engineering — zone assignment, distance-to-POI features,
nearest-neighbor lookups — flows straight into an MLX training loop or
embedding model with zero copies and zero serialization, something neither
GeoPandas nor DuckDB can offer.

## Install

```bash
pip install mlx-spatial          # once published; for now:
pip install -e .
```

Requires macOS on Apple silicon (Metal), Python ≥ 3.10, `mlx`, `numpy`,
`shapely ≥ 2.0`. `geopandas` is optional (`pip install "mlx-spatial[geo]"`).

## Quickstart

```python
import numpy as np
import mlx_spatial as ms

# --- haversine distances (coords are (lon, lat) degrees; meters out) ------
pickups  = np.column_stack([lons, lats])          # (N, 2)
stations = np.column_stack([st_lons, st_lats])    # (M, 2)

d      = ms.haversine(pickups[:-1], pickups[1:])          # elementwise
dmat   = ms.haversine_cross(pickups, stations)            # (N, M)
dist, idx = ms.haversine_knn(pickups, stations, k=5)      # (N, 5) each

# --- point-in-polygon / spatial join --------------------------------------
import geopandas as gpd
zones = gpd.read_file("taxi_zones.geojson")

zone_idx = ms.point_in_polygon(pickups, zones.geometry)   # (N,) int32, -1 = none

# GeoPandas-flavored join (pandas DataFrames with a geometry column work too)
trips = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
joined = ms.sjoin(trips, zones, how="inner")              # adds index_right + zone cols

# reuse ingestion + index across many queries
polyset = ms.PolygonSet(zones.geometry)
for batch in point_batches:
    labels = ms.point_in_polygon(batch, polyset)
```

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
| `GridIndex(polyset)` | reusable uniform-grid spatial index |

Conventions: coordinates are `(x, y)` = `(lon, lat)`, degrees; planar math
for containment (project first if you need metric correctness near poles);
spherical math for haversine. No CRS handling — that stays pyproj's job;
`sjoin` only checks that left/right CRS match.

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
- Grid planning runs in float64 and kernel bboxes are widened so rounding can
  only add candidate pairs, never lose a true hit.
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

Small workloads (< ~100k candidate pairs) automatically fall back to shapely —
single-threaded GEOS beats a kernel launch there. Force a path with
`backend="gpu" | "cpu"` or `MLX_SPATIAL_BACKEND`.

## Benchmarks

Apple M4 Max, macOS 26.2, mlx 0.31.2, Python 3.13. End-to-end wall time from
host-resident numpy/shapely inputs — ingestion, host→device transfer, and
device result evaluation/synchronization included (no NumPy copy-back); best
of 3. Reproduce with `python bench/bench.py --all` (methodology in
`bench/README.md`).

**Point-in-polygon join** (uniform random points × convex zones, NYC-like
extent; mlx-spatial numbers use a warm index; the benchmark harness also
reports cold runs with polygon ingestion and index construction included):

| workload | mlx-spatial | shapely STRtree | GeoPandas sjoin | DuckDB spatial |
| --- | --- | --- | --- | --- |
| 100k pts × 260 zones | **2.2 ms** | 63 ms (29x) | 41 ms (19x) | 126 ms (57x) |
| 1M pts × 260 zones | **13 ms** | 648 ms (50x) | 690 ms (53x) | 547 ms (42x) |
| 10M pts × 260 zones | **136 ms** | 6.8 s (50x) | 11.1 s (82x) | 6.4 s (47x) |
| 1M pts × 10k zones | **12 ms** | 1.06 s (88x) | 786 ms (65x) | 554 ms (46x) |

**Haversine** (`mx.compile`-fused distances; kNN via adaptive scalar/SIMDgroup
selection kernels):

| workload | mlx-spatial | numpy float64 | scikit-learn |
| --- | --- | --- | --- |
| cross 20k × 20k | **19 ms** | 2.64 s (139x) | 3.91 s (206x) |
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
