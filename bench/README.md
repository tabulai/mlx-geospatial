# Benchmarks

Methodology (the honesty rules):

- **End-to-end**: every timing starts from host-resident numpy/shapely inputs
  and includes ingestion, host→device input transfer, device result
  evaluation, and synchronization. Host-input join rows return NumPy labels
  (a zero-copy view of the MLX result, so no copy-back is timed); haversine
  rows leave results as MLX arrays. DuckDB reads from registered in-memory
  tables; conversion costs count against every framework equally.
- **Like against like**: the join baselines evaluate every (point, polygon)
  pair (shapely STRtree and GeoPandas sjoin return them; DuckDB is timed as
  `count(*)` over the join and never materializes them), so their speedups
  are computed against our `all_matches=True` row. The cheaper first-match row
  (the contract most joins want) is reported next to it, never used as the
  denominator. The kNN baseline is reported both with the BallTree built inside
  the timed region (one-off query, comparable to our end-to-end number, which
  has no index to build) and query-only on a prebuilt tree (repeated queries
  against a fixed reference set).
- **Baselines in their best configuration**: shapely prepares the geometries
  an STRtree is queried with, so the polygons are the query side
  (`STRtree(points).query(polygons)`) and the left frame of `sjoin`; prepared
  polygons carry an indexed point-in-area locator that is 8-60x faster on detailed
  polygons than the unprepared orientation, and identical in output. Point
  geometries are built inside every baseline's timed region, as our rows
  ingest raw NumPy inside theirs; polygon tables (GeoDataFrame, DuckDB table)
  are prebuilt for every framework alike.
- **Threads are stated**: shapely/GEOS, GeoPandas and scikit-learn run on one
  core; DuckDB uses its default pool (every core, printed in the row label);
  the GPU rows use the whole GPU. The machine line lists the core count.
- **Synthetic data, described honestly**: the "zones" are random star-shaped
  polygons (per-vertex radii), so they overlap and leave gaps; they are neither
  convex nor a tiling. The join report prints the measured coverage (share of
  points inside any polygon, polygons per matched point). Two detailed layers
  are benchmarked as well, 2,048-vertex zones and a single 150,000-vertex
  coastline-like polygon, because detailed geometry is where baseline
  configuration matters most and where our per-candidate ring walk meets
  GEOS's indexed locator.
- **Device-resident row**: the join also reports the array-native pattern,
  `LocalPoints` re-centered once with `PolygonSet.to_local` and queried
  many times: input already in unified memory, MLX array out, `mx.eval`
  inside the timed region. It is the number an MLX training loop sees.
- **Dense-PIP exception**: the explicitly labeled dense-PIP microbenchmark
  uses a pre-ingested `PolygonSet` and GPU-resident point array. It isolates
  the direct 2-D Metal kernel from the retained composed fallback/oracle and
  checks their outputs exactly outside the timed region.
- **Cold vs. warm**: the join reports both a cold run (PolygonSet + grid
  index built inside the timed region) and warm runs (index reused, the
  realistic repeated-query pattern). Warm rows are steady-state burst
  numbers: GPU clocks are warmed before each timing, as in the crossover
  sweep, since the fixed launch cost is 3-4x higher right after the GPU has
  idled.
- **Best-of-3** wall time after one warmup (DuckDB best of 2; GeoPandas at
  10M points a single run without warmup); DuckDB is interrupted after 300 s
  per query and reported as such.
- One asymmetry to know: the DuckDB zone table (WKB conversion + `CREATE
  TABLE`) is built once outside the timed region, so DuckDB's number is
  comparable to our *warm* rows; points are scanned from a registered
  DataFrame inside the timed query for both.
- **Crossover, not headlines**: `--op crossover` sweeps N for forced CPU (the
  NumPy even-odd fallback), forced GPU and the default auto dispatch on
  48-vertex zones, an axis-aligned square tiling and 2048-vertex zones, and
  fits the dispatch cost model's constants (the tiling's ~9 candidate edges
  per point pin the per-point term). Small inputs lose to CPU on launch
  overhead by design; the library auto-dispatches to the host fallback below
  a few hundred host points (fewer for detailed polygons, more for sparse
  tilings and all-match queries).
- Baselines that are not installed are reported `n/a`, never dropped.

Run:

```bash
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

Current numbers live in the top-level README benchmark section.
