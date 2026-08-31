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

All Metal call sites adapt among scalar, SIMD32, and TG128 traversal while
sharing identical fp32 even-odd semantics. The scalar kernels assign one
thread to each predicate. For a ring with at least 1,024 edges and a bounded
number of predicates, either one 32-lane SIMDgroup or four SIMDgroups in a
128-thread group traverse disjoint strided edge subsets. They XOR integer
crossing parity only—never floating-point partials—so edge reordering cannot
change the result; each edge retains the scalar operation order with FMA
contraction disabled. Conservative work cutoffs keep saturated or bbox-miss
heavy launches on the scalar path.

- `pip_pairs` applies this selector directly to explicit candidate pairs.
- `pip_dense` writes the final `(N, M)` boolean matrix without an `N x E`
  intermediate. Its scalar kernel uses the 2-D point/polygon grid; cooperative
  work is polygon-major for coordinate locality but writes the same row-major
  output. The composed MLX implementation remains the non-Metal fallback and
  an independent arithmetic oracle in tests.
- The indexed first- and all-match call sites use the same cooperative
  predicate implementations under the additional grid-aware rules below.

Boundary semantics: the half-open crossing rule avoids double-counting ray
crossings at vertices. Within the tolerance band — including every exact
boundary point — results are undefined, so a shared-edge point may match either
adjacent polygon, both, or neither. Valid geometry is assumed: even-odd equals
GEOS fill semantics only for valid polygons; on self-intersections or
overlapping multipolygon parts the GPU (even-odd) and the CPU fallback (GEOS)
diverge — `PolygonSet(geoms, validate=True)` rejects such input at ingestion.

## k-nearest neighbors

`haversine_knn` with `k <= 32` on Metal adaptively chooses between two fused
selectors. The scalar kernel assigns one thread to each query. When query
parallelism is scarce, the SIMDgroup kernel assigns 32 lanes to each query;
every lane scans a strided slice of the reference set and retains a local
top-k, then k shuffle reductions merge the 32 sorted list heads exactly. A
lane can therefore contribute all k global winners. Both kernels compute the
haversine `h` term inline (no asin/sqrt in the scan) and avoid both the
`(N, M)` matrix and a full reference sort.

Selected survivors get their distances recomputed through the exact `_hav`
graph, so reported distances are bit-identical to `haversine_cross`, followed
by a k-element distance sort. Exact `h` ties use the lower reference index;
distinct `h` values that round to the same reported fp32 distance retain the
documented near-tie ambiguity. Shape-aware dispatch retains the scalar kernel
once query parallelism is sufficient or SIMDgroup merge/register pressure
would dominate. The composed fallback (chunked matrix + `mx.argpartition`,
which in MLX costs a full sort) remains for `k > 32`, non-Metal systems, and
an active CPU default device.

## Grid-indexed join

Uniform grid over polygon bounds (cell ≈ median polygon bbox side, coarsened
to fit cell/entry budgets). Low-coverage construction uses NumPy's stable
integer sort, which is especially efficient when input bboxes are already
spatially ordered. More expanded grids pack every unique `(cell ID, polygon
ID)` pair into one uint64 key and sort it in place, removing the indirect
permutation array. Material high-occupancy rectangle expansions instead
derive cell counts from four-corner 2-D differences and fill CSR segments
directly with monotonic per-cell cursors while visiting polygons in ascending
order. Every route preserves the deterministic "first match = min polygon
index" contract. First-match query: float64 CPU cell assignment → one int32
cell ID per point → point-centric Metal kernel traverses the cached grid
CSR, performs bbox + parity tests, and returns on the first hit. This avoids
all per-candidate query arrays and the host `minimum.at` reduction. Whole-point
launches are split by the candidate-work budget; a cell that individually
exceeds it uses the bounded explicit-pair fallback. For `all_matches=True`,
zero-candidate points are removed and conceptual candidate ranks are split by
the same hard budget. Ordinary batches use point-centric count, prefix scan,
and ordered write passes; sparse or skewed batches use pair-parallel predicate
flags, scan, and ordered writes. Both routes avoid host candidate-pair
expansion and atomics while preserving point-then-polygon output order.

The index stores saturated per-cell maximum-ring and summed-edge-work
summaries for dispatch. First-match uses SIMD32/TG128 only when every candidate
in every used cell has a ring of at least 1,024 edges; mixed cells stay scalar
so an unvisited long polygon cannot penalize an early short-polygon hit.
All-match forces candidate-parallel compaction when a used cell contains a
long ring or at least 8,192 candidate edges, avoiding a serial point thread
and a duplicate predicate pass. Within that route, the bounded predicate-work
selector still chooses scalar, SIMD32, or TG128 independently. These choices
change scheduling only: sorted CSR traversal, stable scan/compaction, and
first-hit early return preserve the public minimum-index and output-order
contracts.

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
