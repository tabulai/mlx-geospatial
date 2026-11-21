# M1 Ultra and M4 Max benchmarks

The November 21, 2026 runs measured source commit
`92a87cee33b48df78710506216320bde64027175`. Both hosts selected that
source through `PYTHONPATH`. The manifest records module and harness hashes.
The published measurements omit local hostnames and filesystem paths.
[Source manifest](source-manifest.json)

| Measurement | Results |
|---|---|
| Complete spatial joins with fresh indexes | [Paired medians and samples](e2e-comparison.json), [CSV](e2e-comparison.csv), [overview](full-e2e-overview.png), [editable SVG](full-e2e-overview.svg) |
| Warm and cold components | [Tables and timing contracts](COMPONENTS.md), [paired-array chart](pip-m1-m4-side-by-side.png), [editable SVG](pip-m1-m4-side-by-side.svg) |
| Public API and cold-index tuning | [Before-and-after comparison](comparison.json), [cold-query chart](cold-edge-routing.png), [editable SVG](cold-edge-routing.svg) |

The M1 Ultra and M4 Max folders contain the current exact-join, component,
and public API results, the checked test logs, and all 96 paired cutoff
comparisons per host. The component sweep includes 12 reports and 144
CPU/GPU/auto crossover timings per host. The exact harness verified all
25 input fingerprints, and every paired output matched GeoPandas. Both
hosts passed 825 tests and skipped one pandas-version-specific test using
`MTL_DEBUG_LAYER=1` and `METAL_DEVICE_WRAPPER_TYPE=1`. An additional shader
validation run aborted on M1 for both revisions.

The [pre-tuning results](before/) include exact-join and public API data from
both hosts and the preceding M4 Max component sweep. Full runs were timed in
separate sessions. The cutoff experiment alternates builds on the same
inputs and checks output equality. [Benchmark methodology](../../README.md)

## Reproduce the tables and charts

Use Python with Matplotlib 3.10.3. Run these commands from the repository root.

```bash
python3 bench/results/2026-11-21-cold-edge/summarize.py
python3 bench/results/2026-11-21-cold-edge/plot.py
```

## Reproduce the cutoff comparison

Set up the measured source and pinned dependencies as described in the
[benchmark guide](../../README.md#reproduce-the-spatial-joins).
The patch reconstructs the pre-tuning Python package from the measured
commit. Run the paired comparison with Metal validation flags unset.

```bash
RESULT_DIR="$PWD/bench/results/2026-11-21-cold-edge"
PRE_TUNE_SOURCE=$(mktemp -d /tmp/mlx-geospatial-before.XXXXXX)
git archive 92a87cee33b48df78710506216320bde64027175 | tar -x -C "$PRE_TUNE_SOURCE"
(cd "$PRE_TUNE_SOURCE" && git apply --unidiff-zero "$RESULT_DIR/before/source.patch")
PYTHONPATH="$BENCH_SOURCE:$BENCH_SOURCE/bench" \
  python "$RESULT_DIR/compare_dispatch.py" "$PRE_TUNE_SOURCE/mlx_geospatial"
```

`calibrate.py` times forced build, traversal, and cached routes with the same
input generator. The M4 Max follow-up script `exact_join_ab.py` uses the
frozen exact harness and its reference fingerprints.
