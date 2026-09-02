# Benchmarks

Methodology (the honesty rules):

- **End-to-end**: every timing starts from host-resident numpy/shapely inputs
  and includes ingestion, host→device input transfer, device result
  evaluation, and synchronization. Host-input join rows return NumPy labels
  (a zero-copy view of the MLX result, so no copy-back is timed); haversine
  rows leave results as MLX arrays. DuckDB reads from registered in-memory
  tables; conversion costs count against every framework equally.
- **Device-resident row**: the join also reports the array-native pattern,
  ``LocalPoints`` re-centered once with ``PolygonSet.to_local`` and queried
  many times: input already in unified memory, MLX array out, ``mx.eval``
  inside the timed region. It is the number an MLX training loop sees.
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
- **Crossover, not headlines**: `--op crossover` sweeps N for forced CPU,
  forced GPU and the default auto dispatch on simple and detailed zones and
  fits the dispatch cost model's constants. Small inputs lose to CPU on launch
  overhead by design; the library auto-dispatches to shapely below roughly
  500-800 host points (fewer for detailed polygons, more for all-match
  queries).
- Baselines that are not installed are reported `n/a`, never dropped.

Run:

```bash
python bench/bench.py --all
python bench/bench.py --op join --n 10000000 --m 260
python bench/bench.py --op haversine --n 20000 --m 20000
python bench/bench.py --op knn --n 1000 --m 1000000 --k 10
python bench/bench.py --op knn --n 200000 --m 200000 --k 10
python bench/bench.py --op dense-pip --n 100000 --m 32 --vertices 48
python bench/bench.py --op crossover --m 260
```

Current numbers live in the top-level README benchmark section.
