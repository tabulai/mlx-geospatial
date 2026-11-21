# Component benchmarks

Measurements on November 21, 2026 use the source revision in the [source manifest](source-manifest.json). Both hosts completed 12 component reports and 144 CPU/GPU/auto crossover timings. The raw results retain all trial samples and 30 matched input fingerprints.

The component harness reports minimum wall times. Warm joins reuse the MLX index, and containment uses the float32 predicate with `exact_boundary=False`. The Shapely and GeoPandas rows build point geometry and their point index inside the timer, with prepared polygons on one CPU core. DuckDB queries preregistered inputs and returns a count. MLX returns pair arrays. These timing contracts describe distinct operations. The [benchmark guide](../../README.md) covers complete GeoDataFrame joins.

Distance and nearest-neighbor CPU baselines use float64. The MLX paths use float32. BallTree build-plus-query and prebuilt-tree queries have separate rows.

## M1 Ultra

[Raw samples](m1/components/benchmark-results.json) · [Console log](m1/components/benchmark.log)

### All matching pairs

| Workload | MLX pair arrays | Shapely | GeoPandas | DuckDB count |
| --- | --- | --- | --- | --- |
| 100k × 260 · 48 vertices | 2.95 ms | 68.5 ms | 81.8 ms | 177 ms |
| 1M × 260 · 48 vertices | 15.7 ms | 1.32 s | 1.42 s | 432 ms |
| 10M × 260 · 48 vertices | 106 ms | 16.59 s | 17.08 s | 5.60 s |
| 1M × 10k · 48 vertices | 15.1 ms | 1.46 s | 1.44 s | 455 ms |
| 1M × 260 · 2,048 vertices | 16.9 ms | 1.51 s | 1.51 s | 3.96 s |
| 1M × 1 · 150,000 vertices · coastline | 28.8 ms | 2.88 s | 2.93 s | 36.16 s |

### First match per point

| Workload | Cold ingestion + query | Warm NumPy | Prepared LocalPoints |
| --- | --- | --- | --- |
| 100k × 260 · 48 vertices | 2.09 ms | 775 µs | 603 µs |
| 1M × 260 · 48 vertices | 7.61 ms | 6.78 ms | 2.20 ms |
| 10M × 260 · 48 vertices | 39.9 ms | 38.8 ms | 19.3 ms |
| 1M × 10k · 48 vertices | 32.2 ms | 5.35 ms | 2.78 ms |
| 1M × 260 · 2,048 vertices | 50.6 ms | 8.43 ms | 6.29 ms |
| 1M × 1 · 150,000 vertices · coastline | 59.3 ms | 25.3 ms | 22.4 ms |

### Distance matrices, nearest neighbors and dense PIP

| Workload | Method | Elapsed |
| --- | --- | --- |
| haversine_cross 20k x 20k | mlx-geospatial haversine_cross | 10.3 ms |
| haversine_cross 20k x 20k | numpy float64 | 4.29 s |
| haversine_cross 20k x 20k | sklearn haversine_distances | 7.54 s |
| haversine_knn 1k queries, 1M refs, k=10 | mlx-geospatial haversine_knn (k=10, brute force) | 12.2 ms |
| haversine_knn 1k queries, 1M refs, k=10 | sklearn BallTree(haversine), build + query, 1 core | 467 ms |
| haversine_knn 1k queries, 1M refs, k=10 | sklearn BallTree(haversine), query only (prebuilt), 1 core | 17.3 ms |
| haversine_knn 1M queries, 10k refs, k=10 | mlx-geospatial haversine_knn (k=10, brute force) | 80.0 ms |
| haversine_knn 1M queries, 10k refs, k=10 | sklearn BallTree(haversine), build + query, 1 core | 6.79 s |
| haversine_knn 1M queries, 10k refs, k=10 | sklearn BallTree(haversine), query only (prebuilt), 1 core | 6.78 s |
| haversine_knn 200k queries, 200k refs, k=10 | mlx-geospatial haversine_knn (k=10, brute force) | 93.4 ms |
| haversine_knn 200k queries, 200k refs, k=10 | sklearn BallTree(haversine), build + query, 1 core | 1.81 s |
| haversine_knn 200k queries, 200k refs, k=10 | sklearn BallTree(haversine), query only (prebuilt), 1 core | 1.92 s |
| haversine_knn 1M queries, 200k refs, k=10 | mlx-geospatial haversine_knn (k=10, brute force) | 376 ms |
| haversine_knn 1M queries, 200k refs, k=10 | sklearn BallTree(haversine), build + query, 1 core | 8.75 s |
| haversine_knn 1M queries, 200k refs, k=10 | sklearn BallTree(haversine), query only (prebuilt), 1 core | 8.67 s |
| dense PIP 100,000 points x 32 polygons x 48 vertices | mlx-geospatial dense PIP (direct 2-D Metal) | 1.57 ms |
| dense PIP 100,000 points x 32 polygons x 48 vertices | composed MLX dense PIP oracle | 10.0 ms |

### CPU/GPU crossover

| Layer | Points | First CPU | First GPU | First auto | All CPU | All GPU | All auto |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48-vertex zones | 100 | 966 µs | 1.24 ms | 400 µs | 353 µs | 5.52 ms | 363 µs |
| 48-vertex zones | 200 | 610 µs | 668 µs | 624 µs | 605 µs | 1.91 ms | 459 µs |
| 48-vertex zones | 500 | 1.06 ms | 425 µs | 478 µs | 1.06 ms | 1.83 ms | 1.05 ms |
| 48-vertex zones | 1,000 | 2.07 ms | 489 µs | 498 µs | 2.02 ms | 2.03 ms | 2.08 ms |
| 48-vertex zones | 2,000 | 4.19 ms | 423 µs | 438 µs | 4.23 ms | 2.13 ms | 2.04 ms |
| 48-vertex zones | 5,000 | 10.6 ms | 439 µs | 437 µs | 10.5 ms | 2.08 ms | 2.09 ms |
| 48-vertex zones | 10,000 | 21.0 ms | 462 µs | 441 µs | 21.0 ms | 2.25 ms | 2.29 ms |
| 48-vertex zones | 50,000 | 104 ms | 541 µs | 543 µs | 102 ms | 2.26 ms | 2.50 ms |
| square tiling | 100 | 87 µs | 324 µs | 90 µs | 83 µs | 1.74 ms | 83 µs |
| square tiling | 200 | 109 µs | 324 µs | 109 µs | 99 µs | 1.68 ms | 101 µs |
| square tiling | 500 | 161 µs | 291 µs | 167 µs | 150 µs | 1.92 ms | 161 µs |
| square tiling | 1,000 | 253 µs | 323 µs | 264 µs | 238 µs | 1.72 ms | 244 µs |
| square tiling | 2,000 | 456 µs | 321 µs | 460 µs | 428 µs | 1.75 ms | 426 µs |
| square tiling | 5,000 | 1.01 ms | 331 µs | 330 µs | 960 µs | 1.82 ms | 967 µs |
| square tiling | 10,000 | 1.93 ms | 332 µs | 327 µs | 1.85 ms | 1.97 ms | 1.98 ms |
| square tiling | 50,000 | 9.78 ms | 413 µs | 425 µs | 9.28 ms | 2.03 ms | 2.02 ms |
| 2048-vertex zones | 100 | 7.98 ms | 714 µs | 680 µs | 7.75 ms | 1.95 ms | 2.18 ms |
| 2048-vertex zones | 200 | 15.9 ms | 666 µs | 653 µs | 15.9 ms | 2.09 ms | 2.27 ms |
| 2048-vertex zones | 500 | 39.1 ms | 744 µs | 738 µs | 40.3 ms | 2.07 ms | 2.10 ms |
| 2048-vertex zones | 1,000 | 79.5 ms | 652 µs | 739 µs | 78.3 ms | 2.16 ms | 2.20 ms |
| 2048-vertex zones | 2,000 | 160 ms | 756 µs | 651 µs | 158 ms | 2.26 ms | 2.13 ms |
| 2048-vertex zones | 5,000 | 409 ms | 898 µs | 826 µs | 408 ms | 2.73 ms | 2.79 ms |
| 2048-vertex zones | 10,000 | 823 ms | 896 µs | 945 µs | 828 ms | 3.24 ms | 3.45 ms |
| 2048-vertex zones | 50,000 | 4.08 s | 1.34 ms | 1.60 ms | 4.09 s | 2.91 ms | 2.84 ms |

## M4 Max

[Raw samples](m4/components/benchmark-results.json) · [Console log](m4/components/benchmark.log)

### All matching pairs

| Workload | MLX pair arrays | Shapely | GeoPandas | DuckDB count |
| --- | --- | --- | --- | --- |
| 100k × 260 · 48 vertices | 1.52 ms | 47.6 ms | 58.4 ms | 131 ms |
| 1M × 260 · 48 vertices | 7.05 ms | 766 ms | 771 ms | 540 ms |
| 10M × 260 · 48 vertices | 67.7 ms | 8.55 s | 8.83 s | 5.89 s |
| 1M × 10k · 48 vertices | 7.69 ms | 914 ms | 947 ms | 527 ms |
| 1M × 260 · 2,048 vertices | 7.43 ms | 1.08 s | 1.11 s | 2.27 s |
| 1M × 1 · 150,000 vertices · coastline | 11.2 ms | 1.78 s | 1.81 s | 25.62 s |

### First match per point

| Workload | Cold ingestion + query | Warm NumPy | Prepared LocalPoints |
| --- | --- | --- | --- |
| 100k × 260 · 48 vertices | 1.47 ms | 421 µs | 299 µs |
| 1M × 260 · 48 vertices | 8.25 ms | 2.67 ms | 1.44 ms |
| 10M × 260 · 48 vertices | 24.9 ms | 24.3 ms | 12.8 ms |
| 1M × 10k · 48 vertices | 26.3 ms | 2.82 ms | 1.55 ms |
| 1M × 260 · 2,048 vertices | 43.5 ms | 4.16 ms | 2.96 ms |
| 1M × 1 · 150,000 vertices · coastline | 38.5 ms | 14.0 ms | 13.1 ms |

### Distance matrices, nearest neighbors and dense PIP

| Workload | Method | Elapsed |
| --- | --- | --- |
| haversine_cross 20k x 20k | mlx-geospatial haversine_cross | 9.63 ms |
| haversine_cross 20k x 20k | numpy float64 | 2.72 s |
| haversine_cross 20k x 20k | sklearn haversine_distances | 3.95 s |
| haversine_knn 1k queries, 1M refs, k=10 | mlx-geospatial haversine_knn (k=10, brute force) | 8.76 ms |
| haversine_knn 1k queries, 1M refs, k=10 | sklearn BallTree(haversine), build + query, 1 core | 311 ms |
| haversine_knn 1k queries, 1M refs, k=10 | sklearn BallTree(haversine), query only (prebuilt), 1 core | 8.48 ms |
| haversine_knn 1M queries, 10k refs, k=10 | mlx-geospatial haversine_knn (k=10, brute force) | 45.2 ms |
| haversine_knn 1M queries, 10k refs, k=10 | sklearn BallTree(haversine), build + query, 1 core | 4.50 s |
| haversine_knn 1M queries, 10k refs, k=10 | sklearn BallTree(haversine), query only (prebuilt), 1 core | 4.50 s |
| haversine_knn 200k queries, 200k refs, k=10 | mlx-geospatial haversine_knn (k=10, brute force) | 72.6 ms |
| haversine_knn 200k queries, 200k refs, k=10 | sklearn BallTree(haversine), build + query, 1 core | 1.21 s |
| haversine_knn 200k queries, 200k refs, k=10 | sklearn BallTree(haversine), query only (prebuilt), 1 core | 1.16 s |
| haversine_knn 1M queries, 200k refs, k=10 | mlx-geospatial haversine_knn (k=10, brute force) | 322 ms |
| haversine_knn 1M queries, 200k refs, k=10 | sklearn BallTree(haversine), build + query, 1 core | 5.93 s |
| haversine_knn 1M queries, 200k refs, k=10 | sklearn BallTree(haversine), query only (prebuilt), 1 core | 5.80 s |
| dense PIP 100,000 points x 32 polygons x 48 vertices | mlx-geospatial dense PIP (direct 2-D Metal) | 789 µs |
| dense PIP 100,000 points x 32 polygons x 48 vertices | composed MLX dense PIP oracle | 11.4 ms |

### CPU/GPU crossover

| Layer | Points | First CPU | First GPU | First auto | All CPU | All GPU | All auto |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48-vertex zones | 100 | 508 µs | 411 µs | 238 µs | 235 µs | 1.14 ms | 241 µs |
| 48-vertex zones | 200 | 423 µs | 388 µs | 411 µs | 398 µs | 1.08 ms | 321 µs |
| 48-vertex zones | 500 | 719 µs | 269 µs | 270 µs | 706 µs | 783 µs | 699 µs |
| 48-vertex zones | 1,000 | 1.38 ms | 226 µs | 230 µs | 1.35 ms | 807 µs | 1.35 ms |
| 48-vertex zones | 2,000 | 2.79 ms | 200 µs | 197 µs | 2.68 ms | 905 µs | 905 µs |
| 48-vertex zones | 5,000 | 6.91 ms | 221 µs | 217 µs | 6.80 ms | 1.11 ms | 1.03 ms |
| 48-vertex zones | 10,000 | 13.7 ms | 217 µs | 221 µs | 13.7 ms | 1.04 ms | 1.03 ms |
| 48-vertex zones | 50,000 | 68.8 ms | 293 µs | 292 µs | 68.2 ms | 1.30 ms | 1.23 ms |
| square tiling | 100 | 62 µs | 151 µs | 65 µs | 57 µs | 728 µs | 62 µs |
| square tiling | 200 | 76 µs | 160 µs | 78 µs | 70 µs | 691 µs | 83 µs |
| square tiling | 500 | 112 µs | 151 µs | 117 µs | 108 µs | 720 µs | 106 µs |
| square tiling | 1,000 | 173 µs | 159 µs | 175 µs | 162 µs | 765 µs | 164 µs |
| square tiling | 2,000 | 297 µs | 146 µs | 299 µs | 274 µs | 704 µs | 277 µs |
| square tiling | 5,000 | 640 µs | 155 µs | 156 µs | 650 µs | 739 µs | 638 µs |
| square tiling | 10,000 | 1.28 ms | 156 µs | 157 µs | 1.19 ms | 851 µs | 859 µs |
| square tiling | 50,000 | 6.55 ms | 196 µs | 210 µs | 6.23 ms | 874 µs | 882 µs |
| 2048-vertex zones | 100 | 5.53 ms | 341 µs | 311 µs | 5.48 ms | 826 µs | 817 µs |
| 2048-vertex zones | 200 | 11.4 ms | 348 µs | 391 µs | 11.4 ms | 1.04 ms | 866 µs |
| 2048-vertex zones | 500 | 27.8 ms | 394 µs | 372 µs | 27.6 ms | 997 µs | 983 µs |
| 2048-vertex zones | 1,000 | 54.6 ms | 352 µs | 347 µs | 54.8 ms | 1.13 ms | 1.10 ms |
| 2048-vertex zones | 2,000 | 111 ms | 376 µs | 406 µs | 112 ms | 1.12 ms | 1.08 ms |
| 2048-vertex zones | 5,000 | 282 ms | 513 µs | 467 µs | 286 ms | 1.32 ms | 1.38 ms |
| 2048-vertex zones | 10,000 | 574 ms | 537 µs | 526 µs | 570 ms | 1.84 ms | 1.85 ms |
| 2048-vertex zones | 50,000 | 2.84 s | 594 µs | 541 µs | 2.83 s | 1.31 ms | 1.28 ms |
