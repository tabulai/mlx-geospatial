# Benchmarks

Methodology (the honesty rules):

- **End-to-end**: every timing starts from host-resident numpy/shapely
  inputs and includes ingestion, host↔device transfer, and materializing the
  result. DuckDB reads from registered in-memory tables; conversion costs
  count against every framework equally.
- **Dense-PIP exception**: the explicitly labeled dense-PIP microbenchmark
  uses a pre-ingested `PolygonSet` and GPU-resident point array. It isolates
  the direct 2-D Metal kernel from the retained composed fallback/oracle and
  checks their outputs exactly outside the timed region.
- **Cold vs. warm**: the join reports both a cold run (PolygonSet + grid
  index built inside the timed region) and a warm run (index reused —
  the realistic repeated-query pattern).
- **Best-of-3** wall time after one warmup (long runs use fewer repeats).
- One asymmetry to know: the DuckDB zone table (WKB conversion + `CREATE
  TABLE`) is built once outside the timed region, so DuckDB's number is
  comparable to our *warm* rows; points are scanned from a registered
  DataFrame inside the timed query for both.
- **Crossover, not headlines**: sweep N to show where the GPU starts winning.
  Small inputs lose to CPU on launch overhead by design; the library
  auto-dispatches to shapely below ~1e5 candidate pairs.
- Baselines that are not installed are reported `n/a`, never dropped.

Run:

```bash
python bench/bench.py --all
python bench/bench.py --op join --n 10000000 --m 260
python bench/bench.py --op haversine --n 20000 --m 20000
python bench/bench.py --op knn --n 200000 --m 200000 --k 10
python bench/bench.py --op dense-pip --n 100000 --m 32 --vertices 48
```

Current numbers live in the top-level README benchmark section.
