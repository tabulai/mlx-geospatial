# mlx-spatial v0.1 — design notes

One coherent vertical: **distance → predicate → indexed join**, sharing
ingestion, precision handling, and test infrastructure.

## Scope

In: haversine (pairwise / all-pairs / k-nearest), vectorized point-in-polygon,
grid-indexed point-to-polygon spatial join with an `sjoin()` wrapper.

Out (backlog): quadtree index for skewed data, polygon-polygon predicates,
point-polygon distance, trajectory ops, geometry construction, CRS handling
(stays pyproj's job), exact int32 fixed-point path, H3 interop.

## Precision (the defining problem)

Metal is fp32-only; raw lon/lat in fp32 has ~1 m worst-case rounding — enough
to flip containment near edges. Structural mitigations:

1. **Local re-centering** (`_precision.py`): subtract the float64 dataset bbox
   center on the CPU before casting to fp32. A pure translation, so all
   distances/topology are preserved; at city/state extents coordinates carry
   millimeter resolution. `PolygonSet` owns the origin; query points are
   re-centered with the same origin.
2. **Boundary tolerance policy**: points within `PolygonSet.boundary_tolerance`
   (8 ulps of the max local coordinate) of an edge may classify either way.
   Documented, tested against explicitly — golden tests exclude the band.
   GEOS itself has definitional edge behavior, so this is defensible when
   stated.
3. **Conservative planning**: grid planning runs in float64 on original
   coordinates; kernel bboxes are widened by 4 ulps so fp32 rounding can only
   add candidates, never drop a true hit. A parity test is the final arbiter.
4. Haversine: degree→radian conversion in float64 on the CPU when possible;
   fp32 differencing bounds absolute error ~1-2 m (relative ~1e-4 at 10 km),
   except within ~2° of the exact antipode where arcsin amplification grows
   absolute error to 10²-10⁴ m (relative stays < 2e-4).

## Ingestion (never build a geometry engine)

`shapely.to_ragged_array()` → flat coords + ring/part offsets (GeoArrow
layout) → `mx.array`. Multipolygon part level is collapsed
(`geom_ring_offsets = part_ring_offsets[geom_part_offsets]`): the even-odd
crossing-number rule over *all rings of a geometry* handles holes and
multiparts with zero special cases. Planar coordinates for PiP, spherical for
haversine (same split cuSpatial uses).

## Point-in-polygon

Two paths, identical semantics, cross-validated in tests:

- `pip_pairs`: custom Metal kernel via `mx.fast.metal_kernel`, one thread per
  candidate (point, polygon) pair; bbox pre-test then serial crossing loop
  over that polygon's rings. Used by the join.
- `pip_dense`: custom 2-D Metal kernel, one thread per `(point, polygon)`
  output. Points occupy the x-axis and polygons the y-axis, so a SIMDgroup
  reads contiguous points while sharing polygon metadata and edge addresses.
  It writes the final boolean matrix directly, avoiding the former `N x E`
  crossing matrix and cumulative sum. The composed MLX implementation remains
  the non-Metal fallback and an independent arithmetic oracle in tests.

Boundary semantics: the half-open crossing rule avoids double-counting ray
crossings at vertices. Within the tolerance band — including every exact
boundary point — results are undefined, so a shared-edge point may match either
adjacent polygon, both, or neither. Valid geometry is assumed: even-odd equals
GEOS fill semantics only for valid polygons; on self-intersections or
overlapping multipolygon parts the GPU (even-odd) and the CPU fallback (GEOS)
diverge — `PolygonSet(geoms, validate=True)` rejects such input at ingestion.

## k-nearest neighbors

`haversine_knn` with `k <= 32` on Metal runs a fused selection kernel: one
thread per query streams the reference set, computes the haversine `h` term
inline (~10 flops, no asin/sqrt in the loop — `h` is monotone in distance),
and keeps its k best in registers with an insertion that costs one compare
per candidate. No `(N, M)` matrix, no sort. Selected survivors then get
their distances recomputed through the exact `_hav` graph, so reported
distances are bit-identical to `haversine_cross`, and a final `(distance,
index)` sort fixes tie order. Techniques adapted from mlxlearn's neighbor
index (`mlxlearn/_kernels/distance.py`): rank by a cheap monotone surrogate,
refine survivors exactly, keep selection on-device end to end. The composed
fallback (chunked matrix + `mx.argpartition`, which in MLX costs a full
sort) remains for `k > 32` and non-Metal.

## Grid-indexed join

Uniform grid over polygon bounds (cell ≈ median polygon bbox side, coarsened
to fit cell/entry budgets). CSR cell→polygon lists built vectorized in numpy
(ragged cartesian expansion via `repeat` + rank arithmetic; stable sort keeps
per-cell lists ascending, making "first match = min polygon index"
deterministic). Query: point → one cell → candidate pairs by prefix sums →
Metal kernel refine → `np.minimum.at` reduction. Candidate expansion is
CPU-side numpy in v0.1 (int ops, subsecond at 5e7 pairs); known non-GPU stage.

## Dispatch

Below ~1e5 candidate pairs, single-threaded GEOS beats kernel launch +
transfer overhead: automatic shapely STRtree fallback with matching
first-match semantics, honoring the caller's `predicate` exactly (the GPU
kernel serves both predicates — they differ only on boundary points, inside
the fp32 tolerance band). Override per call (`backend=`) or via
`MLX_SPATIAL_BACKEND`.

## Benchmark honesty rules

End-to-end timings from host-resident inputs including ingestion; cold vs
warm-index reported separately; sweep N and M; show the CPU/GPU crossover
rather than a single headline multiple. Baselines: shapely STRtree, geopandas
sjoin, DuckDB spatial, sklearn (haversine/BallTree), numpy float64.

## The differentiated story

Array-native in unified memory: zone assignment / distance features flow
straight into an MLX training loop with zero copies and zero serialization —
something neither GeoPandas nor DuckDB can offer.
