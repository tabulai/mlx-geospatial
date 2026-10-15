# Usage and API guide

For installation, a quick example, and benchmark highlights, see the
[README](../README.md). This guide covers output formats, repeated queries,
and the precision choices that affect results.

## Choose an operation

All examples use `import mlx_geospatial as ms`.

| API | Result |
| --- | --- |
| `point_in_polygon(points, polygons)` | One polygon position per point: an `(N,)` int32 array; `-1` means no match. |
| `point_in_polygon(points, polygons, all_matches=True)` | Two int64 arrays, `(point_indices, polygon_indices)`, sorted by point and then polygon. |
| `contains_matrix(points, polygons)` | An `(N, M)` boolean matrix. Best for small polygon collections. |
| `sjoin(left, right)` | Left table rows with `index_right` and the right table's non-geometry columns. At most one match per point. |
| `haversine(a, b)` | Elementwise or broadcast distances. |
| `haversine_cross(a, b)` | An `(N, M)` distance matrix. |
| `haversine_knn(query, reference, k)` | `(distances, indices)`, each `(N, k)`, sorted by increasing distance per row. |

Polygon indices are zero-based positions in the input collection, not
GeoDataFrame index labels. If polygons overlap, first-match queries choose
the lowest position. Use `all_matches=True` when you need every overlap.

Containment functions return NumPy arrays for host inputs: NumPy arrays,
Python sequences, Shapely points, GeoSeries, or GeoDataFrames. They return
MLX arrays for `mx.array` or `LocalPoints` inputs. Distance functions always
return MLX arrays; call `np.asarray(result)` when you need NumPy.

MLX can defer computation. Call `mx.eval(result)` before reading a completed
GPU timing. First-match device queries can return a lazy result; all-match
queries may synchronize to determine output sizes.

## Coordinates and geometry

Point arrays have shape `(N, 2)`, ordered `(x, y)`. For geographic data,
that means **longitude, latitude**. Distance functions expect degrees by
default; use `degrees=False` for radians.

Containment is planar. Points and polygons must use the same coordinate
system; projected coordinates work too. The library does not reproject,
wrap polygons across the antimeridian, or perform spherical containment.
Prepare those geometries in an appropriate coordinate system first.
Haversine functions take geographic coordinates, not projected coordinates.

`PolygonSet`, `point_in_polygon`, and `contains_matrix` accept Shapely
Polygon and MultiPolygon collections, including holes, or a GeoSeries or
GeoDataFrame containing them. Predicates use only two dimensions. Polygon
vertices must be finite.

When both inputs carry a CRS, containment queries, `sjoin`, and
`PolygonSet.to_local` reject a mismatch. A side without CRS metadata cannot
be checked; the caller must ensure the coordinates agree. Haversine
functions do not interpret CRS metadata.

Empty or missing point geometries retain their row positions and never
match a polygon. Empty or missing polygons also retain their positions and
never match. `PolygonSet` requires at least one entry, although
`point_in_polygon(points, [])` is allowed and returns no matches.

## Reuse polygons and query points

Build a `PolygonSet` once when the same polygons serve several queries.
`point_in_polygon` creates and caches its grid index on the first query.

```python
import numpy as np
import shapely
import mlx_geospatial as ms

zones = [shapely.box(0, 0, 2, 2), shapely.box(1, 0, 3, 2)]
points = np.array([[0.5, 1], [1.5, 1], [4, 1]], dtype=np.float64)
polygons = ms.PolygonSet(zones, validate=True)

labels = ms.point_in_polygon(points, polygons)  # [0, 0, -1]
point_ids, polygon_ids = ms.point_in_polygon(
    points, polygons, all_matches=True
)  # point_ids: [0, 1, 1]; polygon_ids: [0, 0, 1]
```

For MLX pipelines, convert points once with `to_local` and reuse the result:

```python
import mlx.core as mx

index = ms.GridIndex(polygons)
local = polygons.to_local(points)
labels_mx = index.query(local)             # MLX int32 array
batch_labels = index.query(local[:2])      # slicing keeps the coordinate frame
mx.eval(labels_mx, batch_labels)
```

`LocalPoints` holds float32 coordinates relative to the polygon set's
origin. It supports slices and one-dimensional integer index arrays, not
single-row integer indexing or boolean masks. Its `.coords` attribute
exposes the MLX array. A query against a different origin raises an error.
To share local points between layers, construct each layer with
`PolygonSet(other_zones, origin=polygons.origin)` and a compatible CRS.

Keep the original longitude/latitude coordinates for distance calculations:
`haversine*` rejects `LocalPoints` because its translated coordinates no
longer represent longitude and latitude.

You can also query an explicit index with
`ms.point_in_polygon(points, index=index)`. `GridIndex` chooses its cell size
automatically. Advanced controls are `cell_size`, `max_cells`, and
`max_entries`; the planner may enlarge cells to stay within its budgets.
`index.query(..., max_pairs_per_batch=...)` limits candidate work per GPU
launch. This must be a positive integer below `2**31`; it does not limit the
size of the final result.

## Table joins

`sjoin(left, right, how="inner")` accepts GeoPandas objects or pandas
DataFrames with a `geometry` column. The left side contains points and the
right side polygons. Install the `geo` extra for GeoPandas support.

The result preserves the matched left rows and adds the selected right
index label as `index_right`. Right-side attributes are attached by polygon
position, so duplicate right index labels are safe. Column collisions add
`_right` suffixes. An existing left column named `index_right` must be
renamed before joining.

`how="left"` retains unmatched points with missing right-side values.
Only `"inner"` and `"left"` are supported. Each point gets at most one
polygon, even when polygons overlap; this differs from GeoPandas' usual
all-pairs join. Use `point_in_polygon(..., all_matches=True)` to obtain every
matching pair.

## Boundaries and precision

GPU containment uses float32 arithmetic. To preserve useful precision,
the library subtracts the polygon dataset's center in float64 before
converting coordinates to float32. Start with float64 host coordinates, or
explicitly construct MLX inputs with `mx.array(points, dtype=mx.float64)`.
A plain `mx.array` created from a float64 NumPy array defaults to float32
and has already discarded precision before the query.

`PolygonSet.boundary_tolerance` describes a narrow band around polygon
edges, in the input coordinate units. The default result may classify
points inside this band either way. Even a point exactly on a shared edge
may match either polygon, both, or neither. For valid geometries and
float64 input, the documented agreement with Shapely/GEOS applies outside
the band.

The band grows with the whole dataset's extent. Local translation typically
gives millimeter coordinate resolution and a centimeter-scale tolerance
for city-sized geographic layers. A distant erroneous geometry can widen
the tolerance for the entire layer; clean coordinate outliers before
ingestion.

Raw float32 or 16-bit MLX coordinates cannot recover precision already lost
in longitude/latitude. They are translated on the GPU and emit
`ms.PrecisionWarning`. For example, longitude near 74° can already have
roughly 0.4 m of rounding error, larger than a city's local tolerance band.
The same limitation applies to host data that was previously rounded to
float32, even if it is later converted back to float64.

To turn the device-input warning into an error:

```python
import warnings
warnings.simplefilter("error", ms.PrecisionWarning)
```

Use `warnings.filterwarnings("ignore", category=ms.PrecisionWarning)` only
when that loss of precision is acceptable for your data.

### Exact boundary handling

`point_in_polygon`, `GridIndex.query`, and `sjoin` accept
`predicate="intersects"` (the default) or `predicate="within"`. In the
default tolerance mode, both use the same containment rule. To distinguish
boundary points exactly, pass `exact_boundary=True` with original host
coordinates:

```python
labels = ms.point_in_polygon(
    points, polygons, predicate="within", exact_boundary=True
)
```

GEOS then re-checks points near edges. `"intersects"` includes points on
the boundary; `"within"` excludes them. This option is unavailable for
`mx.array` and `LocalPoints` inputs. `contains_matrix` does not expose it.

Finding near-edge points requires a host pass over every point, even if
few need correction. Reference measurements took about 0.3–0.5 seconds
per million points, depending on the polygon layer. The fast benchmark
tables do not include this extra work.

Containment otherwise follows the even-odd rule across all rings. It
agrees with GEOS fill semantics for valid polygons, but can differ for
self-intersections or overlapping MultiPolygon parts. Use
`PolygonSet(geometries, validate=True)` to reject invalid geometry.
`exact_boundary=True` does not repair invalid geometry or replace all
interior results with GEOS results; it re-checks only the boundary band.

## CPU and GPU execution

Indexed queries use `backend="auto"` by default. Small host inputs may use
the NumPy fallback to avoid GPU launch overhead; larger inputs use Metal.
The choice depends on point count and polygon complexity. Device inputs
use the GPU whenever Metal is available.

Pass `backend="cpu"` or `backend="gpu"` to `point_in_polygon`,
`GridIndex.query`, or `sjoin` to force a route. Set
`MLX_GEOSPATIAL_BACKEND=cpu`, `gpu`, or `auto` for a process-wide default;
an explicit argument takes precedence. Requesting GPU execution without
Metal raises an error. Forcing CPU execution on device inputs transfers
their local coordinates to the host; it is useful for checking parity.

Both indexed backends evaluate the same float32 containment rule. CPU
fallback is not a double-precision or GEOS mode. CPU and GPU results are
bit-identical except for float32 subnormal arithmetic near `1e-38`, which
Metal flushes to zero. The fallback favors consistent answers over peak
CPU speed; Shapely can be faster for small or CPU-only workloads.

These `backend` controls apply to indexed joins, not the distance
functions or `contains_matrix`.

## Distance accuracy and memory

`haversine*` uses a sphere with mean radius `ms.R_EARTH_M = 6_371_008.8`
meters. Pass `radius=` to change it; the output uses the radius's units.
The radius must be finite and in `[np.finfo(np.float32).tiny, 1e37)`.

Distances are float32. Typical rounding error away from antipodal points
is roughly 1–2 m; it can grow to hundreds or thousands of meters within
about 2° of the antipode. The spherical model also differs from an
ellipsoidal geodesic by up to roughly 0.5%. Use an ellipsoidal geodesic
implementation when your accuracy requirements demand it.

`haversine_cross` materializes all `N × M` distances. Its `chunk_bytes=`
option limits work per launch but still builds the complete matrix;
concatenating chunks can use about twice the result's memory. Similarly,
`contains_matrix` allocates an `N × M` boolean result. Prefer indexed
containment or k-nearest neighbors when you do not need a dense matrix.

`haversine_knn` is a brute-force search, not a spatial tree. On Metal,
`k <= 32` avoids materializing the full distance matrix. Larger `k` uses
a chunked matrix-based fallback, where a CPU BallTree can be faster for
large reference sets. `k` must be an integer from 1 through the reference
count. Near-ties at float32 distance resolution can select either neighbor.
NaN reference rows are excluded; unfillable slots, including NaN query
rows, return index `-1` and distance `inf`.

## Other helpers

- `from_shapely(geometries, origin=None, validate=False)` constructs a
  `PolygonSet`.
- `as_point_array(points)` normalizes point inputs to a host float64
  `(N, 2)` NumPy array. It accepts MLX inputs too, which requires host
  access, but rejects `LocalPoints`.
- `PolygonSet.recenter_points(points)` returns original-coordinate
  float64 host points and the local MLX array. For device inputs, the
  host values are reconstructed from the rounded local coordinates;
  prefer `to_local` for reusable GPU pipelines.

See [design notes](DESIGN.md) for kernel and index internals, and the
[benchmark guide](../bench/README.md) for measurement methodology.
