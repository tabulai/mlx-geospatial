# Speedup validation and benchmarks — 2026-10-19

Fresh measurements of commit `880522855606458869052d6ff069fbd1cbb53d1a`,
including threaded Point extraction, the host float64 edge grid,
polygon-major GEOS refinement, and chord-distance rejection in fused kNN.
The preceding [complete E2E results](../2026-10-19-full-e2e/) remain archived.

The wheel was built from a clean Git archive. Its Python modules were checked
against the checkout before measurement; the [manifest](build-manifest.json)
records the commit, wheel hash, module hashes and frozen harness hashes.
Benchmarks import the installed wheel. Tests use the identical archived source
with `MTL_DEBUG_LAYER=1` and `METAL_DEVICE_WRAPPER_TYPE=1`; timing runs disable
Metal validation. No library implementation or frozen harness was changed for
this rerun.

- [Complete join tables](tables.md), [JSON](comparison.json) and
  [CSV](comparison.csv): fresh-index, exact joins returning complete
  GeoDataFrames; median of three alternating paired trials after one warmup.
- [Component tables](COMPONENTS.md), [JSON](component-comparison.json) and
  [CSV](component-comparison.csv): warm/cold array queries, distance matrices,
  kNN, dense PIP and CPU/GPU crossover. These use the frozen component
  harness's best-of timing protocol, with the exceptions documented there.
- `m1/` and `m4/`: raw E2E results, benchmark logs, fresh test logs, and
  `components/` raw sweeps. Individual samples, source/runtime metadata,
  input fingerprints and E2E output validation are retained.
- `full-e2e-overview` and `full-e2e-m1`/`full-e2e-m4`: PNG and editable SVG
  charts generated from the same verified medians as the tables.
- [Methodology](methodology.txt): timing boundaries, thread use and validation
  totals. The [benchmark guide](../../README.md) gives reproduction commands.
- [Verification summary](verification.json): test totals, complete measurement
  coverage, documentation checks and lint scope.
- `prior/m1-component-results.json`: archived prior M1 component evidence for
  the kNN comparison; the tracked frozen reference supplies the prior M4 run.

MLX Point coordinate extraction uses up to eight host worker threads,
independently of the numerical-library thread limits set to one. GeoPandas
uses its native join without parallel partitioning. The component join
inputs are NumPy arrays and do not exercise threaded geometry extraction.

Prior/current ratios compare separate sessions on the same host, inputs and
runtime; they do not isolate code changes from background activity. Current
tables never substitute earlier measurements for a missing current run.

From the repository root, validate the evidence and regenerate the outputs:

```bash
python3 bench/results/2026-10-19-opus-speedups/analyze_comparison.py
python3 bench/results/2026-10-19-opus-speedups/analyze_components.py
python3 bench/results/2026-10-19-opus-speedups/plot_full_e2e.py \
  bench/results/2026-10-19-opus-speedups/comparison.json \
  --output-dir bench/results/2026-10-19-opus-speedups
```

Analysis uses only the Python standard library. Plotting requires Matplotlib
(3.10.3 used) and runs separately from the pinned benchmark environments.
The wheel itself is omitted from Git; when an adjacent `dist/` wheel is
available, analysis also verifies its bytes against the manifest.
