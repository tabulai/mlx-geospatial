# mlx-geospatial v0.1 — design notes

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
3. **Conservative planning**: polygon cell ranges are planned in float64 and
   padded by 16 ulps of the local extent (twice the boundary tolerance): more
   than the kernel's 4.5-ulp bbox widening plus the fp32 cell-assignment
   error, which is under 8 ulps (rounded and outward-nudged grid corner,
   subtraction, and the two roundings of the multiply by the reciprocal
   cell size; 5.6 ulps found by directed search). Whichever cell the GPU
   assigns a point near a cell
   line, the list holds every polygon whose kernel bbox accepts it, so the
   answer is determined by the kernel alone and cell assignment can only add
   candidates. Tilings whose edges fall on cell lines pay with tiles listed
   on both sides of those lines: with the default quarter-median cells a
   point in a tile's central cells sees one candidate, the cells bordering a
   shared edge list two and the corner cells four, about two candidates per
   point on average (down from nine with median-sized cells). Device/host
   bit-identity and shapely parity tests are the arbiter.
4. **Device inputs keep the contract**: host float64 and ``mx.float64`` points
   are re-centered in float64 (on the MLX CPU stream, bit-identical to NumPy);
   float32 MLX inputs are re-centered on the device with a split ``hi + lo``
   origin (within 1 ulp of the float64 result) and emit ``PrecisionWarning``,
   because the raw fp32 lon/lat rounding they carry already exceeds the band.
5. Haversine: degree→radian conversion in float64 on the CPU when possible;
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
- Fifteen kernels in all (scalar, SIMD32 and TG128 variants of the pair,
  dense and first-match predicates; point-centric all-match count and write;
  pair-parallel all-match count in three variants plus its write kernel).
  `tests/test_kernel_routes.py` launches every one through its public route,
  with the expected strategy derived from the selectors and the answer checked
  against the NumPy fallback. The contraction pragma is guarded statically
  (each of the six kernel strings that evaluates the x-intersection carries
  its own `#pragma clang fp contract(off)` before it; a body pragma cannot
  protect header arithmetic) and behaviourally (`tests/test_fp_contract.py`
  drives FMA-sensitive probe points, where fused and unfused arithmetic
  disagree by a rounding, through all fourteen predicate kernels and requires
  bit-identity with the NumPy fallback).
- The indexed first- and all-match call sites use the same cooperative
  predicate implementations under the additional grid-aware rules below.

Boundary semantics: the half-open crossing rule avoids double-counting ray
crossings at vertices. Within the tolerance band — including every exact
boundary point — results are undefined, so a shared-edge point may match either
adjacent polygon, both, or neither; `predicate` (`intersects`/`within`, kept
for GeoPandas compatibility) therefore changes nothing by default, and
`exact_boundary=True` re-checks the points within the band with GEOS so the
predicate is honored exactly. GPU queries first run a conservative Metal
filter using the existing grid CSR and polygon buffers. Expanded edge boxes
and a generously padded line strip reject points far from every boundary;
uncertain arithmetic, tiny edges and cell-border points remain candidates.
The filter uses bounded batches that respect `max_pairs_per_batch` and an
additional edge-work budget. Detailed rings reuse the y-band edge index
described below. Excessive per-point work, unfavorable edge-index expansion
and unsafe numerical scales retain the full host route.

Only rows retained by the GPU filter need the exact host band test (CPU
queries test every row). A float64 uniform grid over the tolerance-widened
edge envelopes, built once per polygon set, finds exactly the rows of an
STRtree over those envelopes: the envelopes use the same float64 box
arithmetic, envelope corners and points map to cells through one monotone
float64 formula (so a point inside an envelope lands in a cell listing that
edge), each candidate pair re-applies the inclusive envelope test, and the
float64 point-to-segment arithmetic settles the result. Building it costs a
few float64 passes and a uint16 radix sort instead of one shapely box per
edge. A query whose candidate pairs far exceed its rows (a layer too
clustered for a uniform grid) falls back to that STRtree. All-match
refinement sorts only the GEOS replacement pairs and merges them into the
already sorted output; unchanged groups leave the original arrays intact.
This avoids sorting the complete join after correcting a few boundary rows.
Host work is bounded by candidate pairs: the distance test runs over pair chunks and
a point slice whose widened envelopes return too many pairs (a few large
polygons with extent-spanning edges) is retried shorter. Non-finite rows are
skipped. One
contract on every backend: the host fallback assigns cells with the device
graph's fp32 arithmetic (against the outward-nudged bounds, so a point a
fraction of an ulp outside the float64 extent is admitted on both routes) and
evaluates the same fp32 even-odd predicate in NumPy with the kernels'
operation order, so it is bit-identical to the GPU for every input, valid or
not; the single documented exception is fp32 subnormals (below ~1e-38), which
Metal flushes to zero and NumPy keeps, so a coordinate, an edge delta or an
intermediate of the crossing test that small can differ (this needs local
coordinates themselves near 1e-38 and never occurs for real-world units).
Even-odd equals GEOS fill semantics only for valid polygons; on
self-intersections or overlapping multipolygon parts it differs from GEOS,
which only `exact_boundary` consults, and only for the band —
`PolygonSet(geoms, validate=True)` rejects such input at ingestion.

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

Uniform grid over polygon bounds. The default cell is a quarter of the
median longer bbox side, which cuts candidates per point ~2.5x versus the
median for a 4-6x larger but still small CSR; it backs off to a half, then
the median, when the padded CSR would exceed 40 entries per polygon (a few
large polygons among many small ones), and is coarsened further to fit the
cell/entry budgets. Low-coverage construction uses NumPy's stable
integer sort, which is especially efficient when input bboxes are already
spatially ordered. More expanded grids pack every unique `(cell ID, polygon
ID)` pair into one uint64 key and sort it in place, removing the indirect
permutation array. Material high-occupancy rectangle expansions instead
derive cell counts from four-corner 2-D differences and fill CSR segments
directly with monotonic per-cell cursors while visiting polygons in ascending
order. Every route preserves the deterministic "first match = min polygon
index" contract. Query inputs: host arrays are re-centered in float64 on the
MLX CPU stream and return NumPy; `mx.array` and `LocalPoints` inputs stay in
unified memory and return MLX arrays (the output container follows the
input). First-match query: one compiled MLX graph assigns cells on the GPU
from the fp32 local coordinates — every per-index scalar travels as a 0-d
array so a single Metal kernel serves all indexes — and one synchronization
returns the candidate total and maximum for dispatch and batching; the
point-centric Metal kernel then traverses the cached grid CSR, performs bbox
+ parity tests, and returns on the first hit. This avoids all per-candidate
query arrays and every O(N) host pass. Whole-point launches are split by the
candidate-work budget from a device prefix sum (default 256M pairs per
launch for the fused first-match kernel, which allocates nothing per pair;
16M for all-match compaction, whose scan buffers scale with the batch); a
cell that individually exceeds it uses the bounded explicit-pair fallback
(host-planned from the device-assigned cells). For `all_matches=True`, candidate-bearing points are
compacted on the device (prefix sum + scatter) and conceptual candidate ranks
are split by the same hard budget. Ordinary batches use point-centric count, prefix scan,
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

For rings with at least 2,048 vertices (2,047 edges), sufficiently large first-match,
explicit-pair and all-match GPU queries can build a shared y-band edge
index. Each geometry has at most 512 bands. Stored fp32 cutoffs are compared
directly with binary search, so lookup introduces no coordinate arithmetic
rounding. Each original ring edge appears once in every band intersecting
its y interval, padded by four boundary tolerances. Closing-vertex jumps
between rings are excluded. The fused predicate visits just the chosen
band, retaining the exact crossing expression and disabled FMA contraction.
The boundary filter reuses these lists, including horizontal edges, and
still sends uncertain points through the original float64/GEOS oracle.

Construction is lazy and cached, including rejected builds. Expansion is
checked before allocating repeated references: at most 16 references per
original edge on average and 8 million total. Highly folded geometry with
poor pruning retains the existing traversal. Cold queries below 32,768
candidate pairs (65,536 points for first-match or boundary filtering) avoid
the build; warm queries reuse it regardless of size. The index changes the visited edge subset, not containment,
first-match selection, pair limits, or output ordering.

## Dispatch

For host-resident inputs, `auto` compares a two-term cost model: the NumPy
fallback costs `CPU_US_PER_POINT` per point plus `CPU_US_PER_EDGE` per expected
candidate polygon edge (the index's mean per-cell edge work), the GPU path a
fixed cost that dominates until kernel work shows (tens of thousands of points
for simple polygons, a few thousand for detailed ones): `GPU_LAUNCH_US` for
first match (upload, cell assignment, one launch, sync) and the larger
`GPU_ALL_MATCHES_LAUNCH_US` for all-match compaction (count, scan and write
passes with their syncs). With the shipped constants the model switches at
~250 points for 48-vertex zones, ~2,100 for an axis-aligned square tiling
(only ~9 candidate edges per point) and under ten for 2048-vertex zones for
first match, and about four times later for all matches. The constants live
in `_dispatch.py` and `bench.py --op crossover` re-fits them. The asymmetry
is bounded on both sides. The fixed GPU cost depends on the GPU's clock
state: on an M4 Max about 200 us per first-match query in a back-to-back burst
but 0.8-1.1 ms for a query issued after host-only work has let the GPU idle
(1.1 ms versus 2.5 ms for all matches), so `GPU_LAUNCH_US` (350) and
`GPU_ALL_MATCHES_LAUNCH_US` (1500) sit between the two regimes and the sweep
reports both: picking the CPU while the GPU was warm loses at most the gap to
the warm launch (~0.2 ms first match, ~0.45 ms all matches, on every layer
because the index supplies the edge estimate), picking the GPU after it has
idled loses up to ~0.6 ms / ~1 ms just above the threshold. The fallback is
the NumPy even-odd
evaluation described above (bit-identical to the kernels; GEOS is not on the
query path), so dispatch can never change an answer. It is about 1.5-2x slower
than GEOS per point on 48-vertex zones (1.5 s per million points), up to ~7x
on 2048-vertex zones and faster than GEOS on sparse axis-aligned tilings,
with ~50 us of fixed setup per call; it exists for consistency and for
Metal-less hosts, where the GPU path is unavailable, and processes points in
slices bounded by candidate pairs so its memory does not grow with the point
count (about 250-300 MB peak at 1-2M points on 260 zones). No candidate
counting is needed to
dispatch. Device-resident inputs run on the GPU whenever Metal is available,
since the fallback would need a host copy; `backend="cpu"` on them copies the
fp32 local coordinates to the host and runs the same NumPy predicate (a
parity/debug path). Override per call (`backend=`) or via
`MLX_GEOSPATIAL_BACKEND`.

## Benchmark contracts

The current [end-to-end comparison](../bench/README.md) starts from ready
point and polygon GeoDataFrames, with fresh spatial indexes for every timed
call. MLX constructs the PolygonSet, grid and any edge index, extracts point
coordinates, returns all matching pairs with `predicate="within"` and
`exact_boundary=True`, and assembles a complete point-left GeoDataFrame.
GeoPandas uses its native `sjoin(left, right, how="inner", predicate="within")`
with default indexing. Both results preserve attributes, geometry, CRS and
index labels, and both perform the same canonical sort inside the timer.
The frozen MLX adapter calls `GridIndex.query` and assembles attributes;
it does not time the public `ms.sjoin` wrapper.

Every workload receives one untimed full warmup and three measured paired
trials, alternating method order. Report medians and retain individual
samples; three trials do not establish a latency distribution. Input frame
creation/copies, imports, cold kernel compilation, I/O and validation are
excluded. Full output equality is checked outside every paired timer.
Input fingerprints, output pair hashes, installed module hashes, runtime
versions and machine metadata accompany the results. Both hosts use the
same package and dependencies, with CPU thread environment limits stated;
GeoPandas uses its native join without parallel partitioning, while MLX
uses Metal.

The six synthetic workloads cover point and polygon counts, resampled
2,048-vertex zones and one undulating 150k-vertex polygon. Resampling adds
edges without adding independent shape detail, so vertex count alone does
not predict speed. Band overlap, candidate counts and result size also
matter. These are point-in-polygon joins, not polygon overlays or arbitrary
spatial predicates.

The separate [historical component benchmarks](../bench/HISTORICAL.md)
measure warm/cold array queries, distance matrices, kNN and CPU/GPU
crossover. Their prepared-polygon orientation, output contracts and timing
statistics differ. Do not combine those ratios with the current complete
GeoDataFrame timings. The frozen end-to-end harness and reference input
fingerprints are retained under `bench/e2e/` for reproducibility.

## The differentiated story

Array-native in unified memory: `PolygonSet.to_local` re-centers a point
table once; `GridIndex.query(LocalPoints)` returns zone ids as a lazy MLX
array computed entirely on the GPU, `haversine*` return MLX distance features,
and both feed an `mlx.nn` model directly (`examples/mlx_feature_pipeline.py`).
Prepared device inputs and MLX query results can stay in the MLX pipeline
without conversion to host arrays. Host-input queries return NumPy arrays;
exact boundary refinement and DataFrame joins also perform host work.
This distinction separates the device workflow from the complete
GeoDataFrame benchmark.
