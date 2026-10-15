# mlx-geospatial

Fast point-in-polygon queries, spatial joins, and great-circle distances on
Apple silicon, powered by [MLX](https://github.com/ml-explore/mlx).

Use it to assign GPS points to zones, find nearby places, or build spatial
features for an MLX model. It accepts NumPy and Shapely data and can keep
query results as MLX arrays for further GPU processing.

## Install

```bash
pip install mlx-geospatial
```

Requires an Apple silicon Mac, Python 3.10+, and MLX 0.32.0+.
NumPy and Shapely are installed automatically. For GeoPandas support:

```bash
pip install "mlx-geospatial[geo]"
```

## Quickstart

Find which polygon contains each point:

```python
import numpy as np
from shapely.geometry import box
import mlx_geospatial as ms

# Coordinates are (longitude, latitude). Keep host coordinates in float64.
points = np.array([
    [-74.00, 40.72],
    [-73.98, 40.75],
    [-73.90, 40.80],
])
zones = [
    box(-74.02, 40.70, -73.99, 40.74),
    box(-74.00, 40.73, -73.96, 40.78),
]

labels = ms.point_in_polygon(points, zones)
print(labels)  # [ 0  1 -1]
```

Each label is a polygon's position in `zones`; `-1` means no match.
If polygons overlap, the lowest polygon index wins. To get every match:

```python
point_ids, polygon_ids = ms.point_in_polygon(points, zones, all_matches=True)
```

Containment returns NumPy arrays for host inputs and MLX arrays for MLX inputs.
Points very close to polygon edges need extra care; see [accuracy](#accuracy-and-limits).

### Reuse polygons and points

Build the polygon set and index once when querying the same zones repeatedly:

```python
polygons = ms.PolygonSet(zones)
index = ms.GridIndex(polygons)
labels = index.query(points)  # NumPy result
```

For an MLX pipeline, prepare the points once too. `to_local` shifts them to
the polygons' local coordinate system before converting to float32, preserving
more precision than casting raw longitude and latitude directly.

```python
import mlx.core as mx

local = polygons.to_local(points)
labels_mx = index.query(local)  # MLX result; local[:batch_size] works too
mx.eval(labels_mx)
```

These results can feed MLX operations directly. The
[feature pipeline example](https://github.com/tabulai/mlx-geospatial/blob/main/examples/mlx_feature_pipeline.py)
combines zone embeddings and distance features in a small neural network.

### Distances and nearest neighbors

Haversine functions take `(longitude, latitude)` in degrees and return meters
as MLX arrays. Use `degrees=False` for radians.

```python
landmarks = np.array([
    [-73.9857, 40.7484],
    [-74.0445, 40.6892],
])

distance = ms.haversine(points[0], points[1])            # one distance
matrix = ms.haversine_cross(points, landmarks)           # shape (3, 2)
distances, indices = ms.haversine_knn(points, landmarks, k=1)  # shape (3, 1) each
```

Keep the original coordinates for these functions: `LocalPoints` are shifted
coordinates and cannot be used for distances.

## Performance

On the tested **M4 Max**, the large point-in-polygon workloads below ran
**7.7–131× faster** than the listed alternatives. These are repeated queries
with the polygon index already built and the GPU warmed up.

Every comparison uses MLX's **all-matches** mode. Each competitor cell shows
its time and how many times longer it took than MLX. For example, 771 ms
versus 7.7 ms means Shapely took about **100× as long**.

| Workload | mlx-geospatial | Shapely | GeoPandas | DuckDB Spatial |
| :--- | ---: | ---: | ---: | ---: |
| 100k points × 260 zones, 48 vertices | **1.9 ms** | 49 ms · **26× slower** | 56 ms · **30× slower** | 127 ms · **67× slower** |
| 1M points × 260 zones, 48 vertices | **7.7 ms** | 771 ms · **100× slower** | 805 ms · **105× slower** | 557 ms · **72× slower** |
| 10M points × 260 zones, 48 vertices | **69 ms** | 8.5 s · **124× slower** | 9.0 s · **131× slower** | 6.4 s · **93× slower** |
| 1M points × 10k zones, 48 vertices | **8.1 ms** | 884 ms · **109× slower** | 930 ms · **115× slower** | 566 ms · **70× slower** |
| 1M points × 260 zones, 2,048 vertices | **48 ms** | 1.07 s · **22× slower** | 1.11 s · **23× slower** | 2.9 s · **60× slower** |
| 1M points × one 150k-vertex polygon | **233 ms** | 1.79 s · **7.7× slower** | 1.79 s · **7.7× slower** | 25 s · **107× slower** |

Shapely and GeoPandas use prepared polygons on one CPU core; DuckDB uses
16 threads and returns only a count, which gives it less output work.
MLX uses the GPU and returns all matching pairs as NumPy arrays. Its timings
include point preparation and result evaluation, but exclude polygon and
index setup. The polygons are synthetic; exact boundary refinement is off.

First-match queries return one polygon index per point: the 1M-point, 260-zone
case takes **2.5 ms** from NumPy inputs, or **1.5 ms** with points already
prepared as `LocalPoints`.
Those timings are separate from the all-matches comparison above.

Distance benchmarks also show substantial gains:

- **20k × 20k distance matrix:** 10.2 ms, versus 2.69 s for NumPy (**263×**)
  and 3.98 s for scikit-learn (**389×**). MLX computes in float32; these CPU
  baselines use float64.
- **10 nearest neighbors for 1M queries against 10k references:** 147 ms,
  versus 4.52 s for a prebuilt scikit-learn BallTree (**31×**).
- **A case where BallTree wins:** 1k queries against 1M references take
  26 ms in MLX and 9.1 ms with a prebuilt BallTree—about **2.9× faster**
  for BallTree. Including tree construction raises its time to 305 ms.

Measurements used macOS 26.2, Python 3.13, and MLX 0.31.2. The released
package requires MLX 0.32.0+ for a correctness fix, so these are reference
results from before release. Most timings are best of three.
See [full results and methodology](https://github.com/tabulai/mlx-geospatial/blob/main/bench/README.md)
for timing details, CPU/GPU crossover measurements, and reproduction commands.

## API at a glance

| API | Result |
| :--- | :--- |
| `point_in_polygon(points, polygons)` | One polygon index per point; `all_matches=True` returns every pair |
| `contains_matrix(points, polygons)` | Dense boolean matrix of shape `(points, polygons)` |
| `sjoin(points_df, polygons_df)` | DataFrame with polygon attributes; at most one match per point |
| `haversine(a, b)` | Elementwise or broadcast great-circle distances |
| `haversine_cross(a, b)` | All-pairs distance matrix |
| `haversine_knn(queries, references, k)` | Nearest distances and reference indices |
| `PolygonSet(geometries)` | Polygons prepared for reuse; `to_local(points)` prepares query points |
| `GridIndex(polygons)` | Reusable spatial index; call `query(points)` |

`sjoin` supports `how="inner"` and `how="left"` and returns a pandas
DataFrame. For overlapping polygons, use `point_in_polygon(..., all_matches=True)`
when you need multiple matches per point.

## Accuracy and limits

- **Polygon edges:** containment uses float32 in local coordinates. Points
  inside a narrow tolerance band around an edge may fall on either side.
  Use `exact_boundary=True` with host inputs in `point_in_polygon`,
  `GridIndex.query`, or `sjoin` when exact Shapely boundary behavior matters.
  This adds a CPU scan and can substantially increase query time.
- **Input precision:** prefer float64 NumPy coordinates or prepare them with
  `PolygonSet.to_local`. Raw float32 MLX coordinates may already have lost
  precision; queries emit `PrecisionWarning`.
- **Coordinates:** containment is planar and accepts coordinates in a shared
  coordinate system. Haversine uses a spherical Earth model. The library
  does not reproject data; it checks for a CRS mismatch when both inputs
  carry CRS metadata.
- **Scope:** supports points against polygons, including holes and
  multipolygons. It does not provide polygon overlays or polygon-to-polygon
  predicates. Use `PolygonSet(zones, validate=True)` to reject invalid geometry.
- **Workload matters:** small host queries can use the CPU automatically.
  Very detailed polygons reduce the GPU advantage, and a prebuilt BallTree
  can be faster for nearest-neighbor queries against large reference sets.
  Dense distance matrices and all-match outputs still need memory for their
  results.

The [usage guide](https://github.com/tabulai/mlx-geospatial/blob/main/docs/USAGE.md)
covers boundary predicates, precision, output types, backend selection, and
the remaining API helpers.
[Design notes](https://github.com/tabulai/mlx-geospatial/blob/main/docs/DESIGN.md)
describe the implementation.

## License

[MIT](https://github.com/tabulai/mlx-geospatial/blob/main/LICENSE)
