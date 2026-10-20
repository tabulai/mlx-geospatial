# Component benchmark results — 2026-10-19

Fresh frozen-harness sweeps of workspace revision `880522855606`.
These tables replace the current component figures; [historical measurements](../../HISTORICAL.md) remain unchanged.
For complete exact joins from ready GeoDataFrames, use [the benchmark guide](../../README.md).

## Evidence and coverage

| Host | Status | Finished (UTC) | Evidence |
| --- | --- | --- | --- |
| M1 Ultra | Completed | 2026-10-19T23:55:33.546339+00:00 | [Raw samples](m1/components/benchmark-results.json) · [log](m1/components/benchmark.log) |
| M4 Max | Completed | 2026-10-19T23:43:17.004953+00:00 | [Raw samples](m4/components/benchmark-results.json) · [log](m4/components/benchmark.log) |

Each completed run passed checks for all 12 component reports, all 144 CPU/GPU/auto crossover timers,
complete sample counts, report minima, installed module hashes, the [build manifest](build-manifest.json),
and the frozen harness hashes. All 30 generated input fingerprints match the recorded seed-0 reference.
Completed hosts also share package source, Python/dependency versions, input fingerprints, and timer protocols.
The JSON and CSV preserve unrounded samples for the operation tables and crossover timings.
Warm launch-probe samples appear in JSON only; idle-probe medians come from rounded console output,
and the frozen runner does not save their individual samples.

The component sweep checks direct dense-PIP output against the composed oracle. It does **not** independently
validate every join, distance, or kNN output against its CPU baseline; the separate full test suite and exact
end-to-end join validation supply correctness evidence. Approximate component pair counts can differ near
boundaries from the exact join counts.

## Timing contract

- Reported elapsed times are **minimum wall times**, not medians: best of 3 after one warmup by default; DuckDB best of 2 after one warmup; GeoPandas best of 3 without warmup, except the 10M-point case uses one observation. Crossover uses best of 7 after two warmups and warms GPU clocks before each forced-GPU timer.
- Warm joins reuse a `PolygonSet` and `GridIndex`. NumPy rows include recentering/upload and materialized NumPy results. Prepared `LocalPoints` rows exclude preparation/upload and evaluate the returned MLX array. Cold first-match rows include polygon ingestion and index construction; cold kernel compilation is excluded by warmup.
- These joins use the float32 predicate without `exact_boundary=True`. CPU fallback crossover rows use the library's NumPy float32 predicate, not GeoPandas. Isolated queries after idle time may cost more than these warm bursts.
- All-pairs output contracts differ: MLX returns two NumPy index arrays; Shapely builds points and their STRtree inside the timer; GeoPandas builds point geometries/frame/index and returns a joined frame with prebuilt polygons; DuckDB prebuilds/registers inputs and only returns `count(*)`. Ratios describe elapsed times under these distinct contracts, not equivalent end-to-end speedups.
- Distance/nearest-neighbor MLX computations use float32 versus float64 CPU baselines. MLX host-input timers include conversion and evaluation. BallTree radians conversion is excluded from both CPU timers; the query-only column excludes tree construction and represents reuse of a prebuilt index.
- CPU library thread limits are one; DuckDB uses its recorded default threads, shown per host. MLX uses the GPU. All datasets are synthetic; 2,048-vertex polygons resample 48-vertex boundaries, and the 150,000-vertex case is a single generated ring.

For baseline cells, `time · R×` means **baseline elapsed time / MLX elapsed time**. A ratio below 1 means the baseline was faster. Missing baselines remain `n/a`.

## Machines and runtime

| Host | CPU cores | GPU cores | macOS | Python | MLX | DuckDB threads |
| --- | --- | --- | --- | --- | --- | --- |
| M1 Ultra | 20 | 64 | 26.5.1 | 3.14.2 | 0.32.4 | 20 |
| M4 Max | 16 | 40 | 26.6.2 | 3.14.2 | 0.32.4 | 16 |

## M1 Ultra

### All matching pair arrays

| Points × polygons; vertices | MLX all pairs | Shapely STRtree | GeoPandas | DuckDB count(*) |
| --- | --- | --- | --- | --- |
| 100k × 260; 48 vertices | 3.83 ms | 67.4 ms · 17.60× | 75.7 ms · 19.77× | 177 ms · 46.34× |
| 1M × 260; 48 vertices | 13.9 ms | 1.38 s · 99.16× | 1.39 s · 99.71× | 446 ms · 32.05× |
| 10M × 260; 48 vertices | 108 ms | 17.4 s · 160.97× | 18 s · 166.28× | 5.66 s · 52.34× |
| 1M × 10k; 48 vertices | 15.1 ms | 1.42 s · 94.07× | 1.54 s · 101.83× | 460 ms · 30.46× |
| 1M × 260; 2,048 vertices | 18.9 ms | 1.49 s · 78.70× | 1.52 s · 80.68× | 3.73 s · 197.51× |
| 1M × 1; 150,000 vertices | 29.4 ms | 2.84 s · 96.30× | 2.86 s · 96.98× | 37.2 s · 1261.75× |

### First match per point

Returns the lowest-index containing polygon or `-1`, not every matching pair.

| Points × polygons; vertices | Warm NumPy → NumPy | Prepared LocalPoints → MLX | Cold ingestion + index + query |
| --- | --- | --- | --- |
| 100k × 260; 48 vertices | 753 µs | 572 µs | 2.42 ms |
| 1M × 260; 48 vertices | 4.58 ms | 2.18 ms | 5.71 ms |
| 10M × 260; 48 vertices | 38.6 ms | 19.2 ms | 39.7 ms |
| 1M × 10k; 48 vertices | 4.75 ms | 2.69 ms | 31.8 ms |
| 1M × 260; 2,048 vertices | 8.27 ms | 6.33 ms | 50.2 ms |
| 1M × 1; 150,000 vertices | 25 ms | 22.2 ms | 57.4 ms |

### Haversine distance matrix

| Queries × references | MLX float32 | NumPy float64 | scikit-learn float64 |
| --- | --- | --- | --- |
| 20k × 20k | 10.3 ms | 4.27 s · 414.61× | 7.57 s · 735.06× |

### Haversine nearest neighbors, k=10

| Queries × references | MLX | BallTree build + query | BallTree query only (prebuilt) |
| --- | --- | --- | --- |
| 1k × 1M | 12 ms | 464 ms · 38.73× | 17.3 ms · 1.45× |
| 1M × 10k | 81.5 ms | 6.84 s · 83.97× | 6.83 s · 83.85× |
| 200k × 200k | 93.6 ms | 1.87 s · 19.97× | 1.8 s · 19.20× |
| 1M × 200k | 371 ms | 9.09 s · 24.51× | 9.01 s · 24.29× |

### Dense point-in-polygon kernel

Pre-ingested polygons and device-resident points; both outputs are evaluated and synchronized.

| Points × polygons × vertices | Direct Metal | Composed MLX oracle | Composed / direct |
| --- | --- | --- | --- |
| 100k × 32 × 48 | 879 µs | 10.9 ms | 12.43× |

### CPU/GPU crossover

All rows use the same warm index and host NumPy points. `auto` shows measured elapsed time and the backend printed by the harness; `n/a` means that log evidence is missing.

#### 48-vertex zones (260 polygons)

| Points | First CPU | First GPU | First auto / pick | All CPU | All GPU | All auto / pick |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 1.21 ms | 491 µs | 364 µs / cpu | 349 µs | 3.85 ms | 450 µs / cpu |
| 200 | 607 µs | 636 µs | 617 µs / cpu | 591 µs | 2.69 ms | 456 µs / cpu |
| 500 | 1.05 ms | 420 µs | 434 µs / gpu | 1.03 ms | 2.78 ms | 1.03 ms / cpu |
| 1,000 | 2.05 ms | 443 µs | 430 µs / gpu | 2.03 ms | 2.65 ms | 2.03 ms / cpu |
| 2,000 | 4.32 ms | 441 µs | 427 µs / gpu | 4.16 ms | 2.82 ms | 3.14 ms / gpu |
| 5,000 | 10.4 ms | 433 µs | 431 µs / gpu | 10.3 ms | 3.21 ms | 3.12 ms / gpu |
| 10,000 | 20.6 ms | 446 µs | 434 µs / gpu | 20.5 ms | 2.93 ms | 3.02 ms / gpu |
| 50,000 | 103 ms | 525 µs | 515 µs / gpu | 102 ms | 3.21 ms | 3.37 ms / gpu |

#### square tiling (256 polygons)

| Points | First CPU | First GPU | First auto / pick | All CPU | All GPU | All auto / pick |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 86.3 µs | 304 µs | 93.1 µs / cpu | 82.3 µs | 2.58 ms | 80.9 µs / cpu |
| 200 | 106 µs | 324 µs | 108 µs / cpu | 97.7 µs | 2.55 ms | 102 µs / cpu |
| 500 | 160 µs | 302 µs | 166 µs / cpu | 148 µs | 2.7 ms | 151 µs / cpu |
| 1,000 | 253 µs | 306 µs | 256 µs / cpu | 233 µs | 2.48 ms | 236 µs / cpu |
| 2,000 | 434 µs | 309 µs | 446 µs / cpu | 410 µs | 2.79 ms | 417 µs / cpu |
| 5,000 | 984 µs | 322 µs | 313 µs / gpu | 945 µs | 2.77 ms | 953 µs / cpu |
| 10,000 | 1.92 ms | 391 µs | 383 µs / gpu | 1.84 ms | 2.95 ms | 2.94 ms / gpu |
| 50,000 | 9.52 ms | 388 µs | 420 µs / gpu | 9.12 ms | 3.02 ms | 3.09 ms / gpu |

#### 2048-vertex zones (260 polygons)

| Points | First CPU | First GPU | First auto / pick | All CPU | All GPU | All auto / pick |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 7.75 ms | 669 µs | 667 µs / gpu | 7.77 ms | 3.16 ms | 3.03 ms / gpu |
| 200 | 15.9 ms | 861 µs | 779 µs / gpu | 16.1 ms | 2.9 ms | 3.07 ms / gpu |
| 500 | 39 ms | 639 µs | 636 µs / gpu | 39.1 ms | 2.82 ms | 3.07 ms / gpu |
| 1,000 | 77.2 ms | 649 µs | 649 µs / gpu | 77.3 ms | 3.16 ms | 3.2 ms / gpu |
| 2,000 | 156 ms | 718 µs | 753 µs / gpu | 156 ms | 2.99 ms | 3.11 ms / gpu |
| 5,000 | 401 ms | 3.86 ms | 3.94 ms / gpu | 401 ms | 3.63 ms | 3.94 ms / gpu |
| 10,000 | 806 ms | 3.86 ms | 3.9 ms / gpu | 804 ms | 4.09 ms | 4.23 ms / gpu |
| 50,000 | 4.05 s | 5.4 ms | 5.38 ms / gpu | 4.04 s | 3.69 ms | 3.78 ms / gpu |

Warm launch probes at 500 points use best of 7; idle probes are the median of 7 individual queries after 200 ms idle, rounded to whole microseconds by the original log. Idle raw samples are not stored by the frozen runner.

| Route | Warm probe | After 200 ms GPU idle |
| --- | --- | --- |
| first | 437 µs | 3,627 µs |
| all | 3.84 ms | 12,129 µs |

## M4 Max

### All matching pair arrays

| Points × polygons; vertices | MLX all pairs | Shapely STRtree | GeoPandas | DuckDB count(*) |
| --- | --- | --- | --- | --- |
| 100k × 260; 48 vertices | 1.89 ms | 39 ms · 20.66× | 46.3 ms · 24.49× | 129 ms · 68.53× |
| 1M × 260; 48 vertices | 9.16 ms | 698 ms · 76.13× | 714 ms · 77.89× | 537 ms · 58.62× |
| 10M × 260; 48 vertices | 69.3 ms | 8.01 s · 115.51× | 8.53 s · 123.06× | 6.57 s · 94.74× |
| 1M × 10k; 48 vertices | 7.79 ms | 807 ms · 103.53× | 837 ms · 107.34× | 569 ms · 73.06× |
| 1M × 260; 2,048 vertices | 7.95 ms | 953 ms · 119.83× | 977 ms · 122.91× | 2.23 s · 280.90× |
| 1M × 1; 150,000 vertices | 11.4 ms | 1.7 s · 148.76× | 1.73 s · 151.56× | 22.9 s · 2004.87× |

### First match per point

Returns the lowest-index containing polygon or `-1`, not every matching pair.

| Points × polygons; vertices | Warm NumPy → NumPy | Prepared LocalPoints → MLX | Cold ingestion + index + query |
| --- | --- | --- | --- |
| 100k × 260; 48 vertices | 400 µs | 336 µs | 1.14 ms |
| 1M × 260; 48 vertices | 3.02 ms | 1.42 ms | 5.91 ms |
| 10M × 260; 48 vertices | 24 ms | 12.6 ms | 25.1 ms |
| 1M × 10k; 48 vertices | 2.63 ms | 1.56 ms | 20.4 ms |
| 1M × 260; 2,048 vertices | 4.67 ms | 2.86 ms | 42.1 ms |
| 1M × 1; 150,000 vertices | 13.8 ms | 12.7 ms | 37.5 ms |

### Haversine distance matrix

| Queries × references | MLX float32 | NumPy float64 | scikit-learn float64 |
| --- | --- | --- | --- |
| 20k × 20k | 9.58 ms | 2.75 s · 286.96× | 3.93 s · 410.65× |

### Haversine nearest neighbors, k=10

| Queries × references | MLX | BallTree build + query | BallTree query only (prebuilt) |
| --- | --- | --- | --- |
| 1k × 1M | 8.52 ms | 293 ms · 34.35× | 8.06 ms · 0.95× |
| 1M × 10k | 44.4 ms | 4.47 s · 100.84× | 4.48 s · 100.89× |
| 200k × 200k | 69.5 ms | 1.18 s · 16.92× | 1.13 s · 16.21× |
| 1M × 200k | 313 ms | 5.69 s · 18.17× | 5.63 s · 17.98× |

### Dense point-in-polygon kernel

Pre-ingested polygons and device-resident points; both outputs are evaluated and synchronized.

| Points × polygons × vertices | Direct Metal | Composed MLX oracle | Composed / direct |
| --- | --- | --- | --- |
| 100k × 32 × 48 | 748 µs | 10.4 ms | 13.96× |

### CPU/GPU crossover

All rows use the same warm index and host NumPy points. `auto` shows measured elapsed time and the backend printed by the harness; `n/a` means that log evidence is missing.

#### 48-vertex zones (260 polygons)

| Points | First CPU | First GPU | First auto / pick | All CPU | All GPU | All auto / pick |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 245 µs | 557 µs | 565 µs / cpu | 231 µs | 2.08 ms | 227 µs / cpu |
| 200 | 400 µs | 471 µs | 390 µs / cpu | 378 µs | 1.46 ms | 320 µs / cpu |
| 500 | 712 µs | 247 µs | 231 µs / gpu | 693 µs | 1.23 ms | 694 µs / cpu |
| 1,000 | 1.35 ms | 234 µs | 243 µs / gpu | 1.34 ms | 1.22 ms | 1.33 ms / cpu |
| 2,000 | 2.7 ms | 204 µs | 213 µs / gpu | 2.67 ms | 1.45 ms | 1.58 ms / gpu |
| 5,000 | 6.74 ms | 221 µs | 217 µs / gpu | 6.62 ms | 1.62 ms | 1.53 ms / gpu |
| 10,000 | 13.5 ms | 220 µs | 220 µs / gpu | 13.3 ms | 1.42 ms | 1.45 ms / gpu |
| 50,000 | 68 ms | 303 µs | 302 µs / gpu | 67.4 ms | 1.71 ms | 1.76 ms / gpu |

#### square tiling (256 polygons)

| Points | First CPU | First GPU | First auto / pick | All CPU | All GPU | All auto / pick |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 61.8 µs | 180 µs | 63.6 µs / cpu | 56.6 µs | 1.18 ms | 57.9 µs / cpu |
| 200 | 76.3 µs | 147 µs | 78.1 µs / cpu | 70 µs | 1.12 ms | 72.2 µs / cpu |
| 500 | 112 µs | 155 µs | 113 µs / cpu | 104 µs | 1.08 ms | 105 µs / cpu |
| 1,000 | 172 µs | 146 µs | 177 µs / cpu | 162 µs | 1.13 ms | 163 µs / cpu |
| 2,000 | 290 µs | 163 µs | 291 µs / cpu | 274 µs | 1.19 ms | 276 µs / cpu |
| 5,000 | 638 µs | 153 µs | 155 µs / gpu | 616 µs | 1.19 ms | 612 µs / cpu |
| 10,000 | 1.23 ms | 150 µs | 164 µs / gpu | 1.19 ms | 1.17 ms | 1.28 ms / gpu |
| 50,000 | 6.25 ms | 196 µs | 201 µs / gpu | 6.03 ms | 1.27 ms | 1.24 ms / gpu |

#### 2048-vertex zones (260 polygons)

| Points | First CPU | First GPU | First auto / pick | All CPU | All GPU | All auto / pick |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 5.36 ms | 403 µs | 403 µs / gpu | 5.34 ms | 1.44 ms | 1.49 ms / gpu |
| 200 | 11 ms | 372 µs | 378 µs / gpu | 11 ms | 1.44 ms | 1.58 ms / gpu |
| 500 | 27.4 ms | 348 µs | 363 µs / gpu | 27.3 ms | 1.51 ms | 1.38 ms / gpu |
| 1,000 | 54.4 ms | 386 µs | 378 µs / gpu | 55 ms | 1.56 ms | 1.5 ms / gpu |
| 2,000 | 110 ms | 387 µs | 379 µs / gpu | 110 ms | 1.58 ms | 1.47 ms / gpu |
| 5,000 | 282 ms | 2.05 ms | 2.02 ms / gpu | 282 ms | 1.89 ms | 1.83 ms / gpu |
| 10,000 | 564 ms | 2.02 ms | 2.03 ms / gpu | 563 ms | 2.26 ms | 2.29 ms / gpu |
| 50,000 | 2.8 s | 3.25 ms | 3.28 ms / gpu | 2.8 s | 1.86 ms | 1.72 ms / gpu |

Warm launch probes at 500 points use best of 7; idle probes are the median of 7 individual queries after 200 ms idle, rounded to whole microseconds by the original log. Idle raw samples are not stored by the frozen runner.

| Route | Warm probe | After 200 ms GPU idle |
| --- | --- | --- |
| first | 241 µs | 2,211 µs |
| all | 1.82 ms | 5,590 µs |

## Prior recorded sessions: kNN context

The earlier per-host sessions below used the same older package source. Their complete module hashes, input fingerprints, harness hashes, and dependency/Python versions match the tracked M4 reference. Python compiler/build strings differ across the two hosts; each before/after comparison requires the exact same Python build on its own host.
Comparisons also require matching hardware, macOS, dependencies, CPU thread limits, and Metal settings. Source hashes and dates for both sessions are retained in the JSON.
These are **prior-session comparisons**, not an isolated causal measurement of one optimization: package changes, elapsed session time, GPU clocks, thermals, and background activity can all affect the result.

| Host | Prior session started (UTC) | Prior source manifest | Prior raw evidence |
| --- | --- | --- | --- |
| M1 Ultra | 2026-10-19T13:33:43.248449+00:00 | `cea5a55e8a97` | [Raw samples](prior/m1-component-results.json) |
| M4 Max | 2026-10-19T13:33:43.088727+00:00 | `cea5a55e8a97` | [Raw samples](../../e2e/reference-benchmark-results.json) |

| Host | Queries × references | Prior MLX | Current MLX | Prior / current |
| --- | --- | --- | --- | --- |
| M1 Ultra | 1k × 1M | 45.1 ms | 12 ms | 3.77× |
| M1 Ultra | 1M × 10k | 141 ms | 81.5 ms | 1.73× |
| M1 Ultra | 200k × 200k | 484 ms | 93.6 ms | 5.18× |
| M1 Ultra | 1M × 200k | 2.22 s | 371 ms | 5.98× |
| M4 Max | 1k × 1M | 30.6 ms | 8.52 ms | 3.60× |
| M4 Max | 1M × 10k | 147 ms | 44.4 ms | 3.31× |
| M4 Max | 200k × 200k | 543 ms | 69.5 ms | 7.82× |
| M4 Max | 1M × 200k | 2.56 s | 313 ms | 8.19× |

## Reproduce the tables

From the repository root:

```bash
python3 bench/results/2026-10-19-opus-speedups/analyze_components.py
```

The generator reads completed `m1/components/benchmark-results.json` and `m4/components/benchmark-results.json`, checks provenance and timing samples, and writes this report plus [JSON](component-comparison.json) and [CSV](component-comparison.csv). It performs no timed work and never replaces a missing current measurement with a historical value.

To collect a new sweep, install the desired wheel and pinned [runtime requirements](../../e2e/requirements.txt), use the environment settings in [the benchmark guide](../../README.md), and run the frozen entry point with a new output directory:

```bash
python -u bench/e2e/run_benchmarks.py --output /path/to/new/components --machine-label 'My Apple silicon Mac'
```
