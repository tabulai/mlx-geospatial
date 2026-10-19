# Full end-to-end spatial join results — 2026-10-19

This is the complete fresh rerun with the detailed-ring optimization enabled,
using the same frozen wheel on M1 Ultra and M4 Max. See the [benchmark guide](../../README.md)
for the complete timing contract, hardware, geometry, and reproduction commands.
All six cases were remeasured on each host; no timings were reused from earlier runs.

- `comparison.json` and `comparison.csv`: medians, individual samples, ranges, ratios.
- `m1/` and `m4/`: complete raw results, metadata, input/output hashes and logs.
- `full-e2e-overview` and `full-e2e-m1`/`full-e2e-m4`: PNG and editable SVG figures.
- `build-manifest.json`: measured wheel and source identity; code is the same on both hosts.
- `previous-validation/`: 610 passing tests per host from the preceding validation
  of this unchanged wheel. The current benchmark adds 48 fresh full-output checks.
- `methodology.txt`: timing scope and limitations.

The exact geometry generator and metadata helper used in the run are frozen
with the runner under [bench/e2e](../../e2e/). The downloaded result bundle also
contains the measured wheel, source archive and cumulative patch; the repository
keeps source hashes and evidence without vendoring a wheel.

From the repository root, verify the captured results and regenerate figures:

```bash
python3 bench/results/2026-10-19-full-e2e/analyze_comparison.py
python3 bench/results/2026-10-19-full-e2e/plot_full_e2e.py \
  bench/results/2026-10-19-full-e2e/comparison.json \
  --output-dir bench/results/2026-10-19-full-e2e
```

Analysis uses only the Python standard library. Plotting requires Matplotlib
(3.10.3 used); it runs separately from the matched benchmark environments.
The analyzer verifies a wheel's bytes when an adjacent `dist/` wheel is available;
in this repository it checks raw installed module hashes against the manifest.
The chart uses the same zero-based ratio scale for both machines, with a 1×
reference. Higher GeoPandas/MLX ratios favor MLX. These are descriptive medians
from three trials on six synthetic workloads, not confidence intervals.
