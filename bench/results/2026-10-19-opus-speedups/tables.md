# Current complete spatial joins

| Points × polygons | Vertices | M1 Ultra MLX | M1 Ultra GeoPandas | Speedup | M4 Max MLX | M4 Max GeoPandas | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100k × 260 | 48 | 22.9 ms | 86.6 ms | **3.78×** | 18.6 ms | 47.5 ms | **2.55×** |
| 1M × 260 | 48 | 157 ms | 1.67 s | **10.66×** | 118 ms | 805 ms | **6.81×** |
| 10M × 260 | 48 | 2.34 s | 22.23 s | **9.52×** | 1.38 s | 11.02 s | **7.96×** |
| 1M × 10k | 48 | 263 ms | 1.60 s | **6.08×** | 191 ms | 923 ms | **4.84×** |
| 1M × 260 | 2,048 | 246 ms | 1.60 s | **6.49×** | 172 ms | 1.02 s | **5.90×** |
| 1M × 1 | 150,000 | 146 ms | 2.73 s | **18.66×** | 98.1 ms | 1.62 s | **16.51×** |

Each row uses newly measured medians; no earlier run fills a current timing.

# Change from the preceding measured build

Prior and current medians were measured in different sessions. These ratios describe
observed changes on the same host and inputs; they do not isolate the code changes
from background load or other session effects.

| Machine | Points × polygons | Vertices | Prior MLX | Current MLX | Prior/current MLX |
|---|---|---:|---:|---:|---:|
| M1 Ultra | 100k × 260 | 48 | 35.0 ms | 22.9 ms | **1.53×** |
| M4 Max | 100k × 260 | 48 | 27.9 ms | 18.6 ms | **1.50×** |
| M1 Ultra | 1M × 260 | 48 | 205 ms | 157 ms | **1.30×** |
| M4 Max | 1M × 260 | 48 | 156 ms | 118 ms | **1.32×** |
| M1 Ultra | 10M × 260 | 48 | 2.71 s | 2.34 s | **1.16×** |
| M4 Max | 10M × 260 | 48 | 1.85 s | 1.38 s | **1.33×** |
| M1 Ultra | 1M × 10k | 48 | 494 ms | 263 ms | **1.88×** |
| M4 Max | 1M × 10k | 48 | 376 ms | 191 ms | **1.97×** |
| M1 Ultra | 1M × 260 | 2,048 | 297 ms | 246 ms | **1.20×** |
| M4 Max | 1M × 260 | 2,048 | 208 ms | 172 ms | **1.20×** |
| M1 Ultra | 1M × 1 | 150,000 | 285 ms | 146 ms | **1.95×** |
| M4 Max | 1M × 1 | 150,000 | 182 ms | 98.1 ms | **1.85×** |

A prior/current ratio above 1 means the current measured MLX run was faster.
