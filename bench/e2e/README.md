# Frozen end-to-end spatial join harness

These files are the unchanged harness used for the current six-case comparison.
See [the benchmark guide](../README.md) for results, timing scope and commands.
`run_e2e.py` is the entry point; the adjacent `bench.py` supplies the original
geometry generator, and `run_benchmarks.py` supplies metadata and fingerprint
helpers. They are frozen together to preserve the original random-generation
sequence and recorded script hashes. The generator snapshot's module comments
describe its older array benchmark; `run_e2e.py` defines this comparison's
complete GeoDataFrame timing contract.

The reference JSON records the earlier input fingerprints; its historical
timings are not used in the current comparison. Run outputs belong in a new
`bench/results/` directory, not over the published evidence. Requirements pin
the measured Python packages; use Python 3.14.2 to reproduce the recorded
runtime. They do not pin the library itself: install the desired source or
archived wheel separately.
