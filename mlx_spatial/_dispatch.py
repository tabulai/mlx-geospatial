"""Backend selection: GPU kernels vs. shapely CPU fallback.

The GPU join costs a roughly fixed launch overhead (upload, cell assignment,
kernel, synchronization) that dominates until kernel work shows: tens of
thousands of points for simple polygons, a few thousand for detailed ones.
The GEOS path costs a few hundred nanoseconds per point plus about a
nanosecond per candidate polygon edge. ``auto`` compares the two with that
two-term cost model, so roughly 500-800 simple-polygon points or a few dozen
points against detailed polygons already run on the GPU; all-match queries
carry a larger fixed cost (count, scan and compaction passes) and switch
later. Being wrong toward the GPU costs at most the launch overhead, being
wrong toward the CPU can cost milliseconds on detailed polygons, so
``GPU_LAUNCH_US`` is set slightly below the 400-500 us measured on an Apple
M4 Max (long-ring layers run higher still) and the threshold errs toward the
GPU. ``python bench/bench.py --op crossover`` re-fits the constants for
another machine. Override per call (``backend=``) or globally via the
environment variable ``MLX_SPATIAL_BACKEND`` (``auto`` | ``gpu`` | ``cpu``).
"""

from __future__ import annotations

import math
import os

import mlx.core as mx

__all__ = [
    "CPU_US_PER_POINT",
    "CPU_US_PER_EDGE",
    "GPU_LAUNCH_US",
    "GPU_ALL_MATCHES_LAUNCH_US",
    "DEFAULT_EDGES_PER_POINT",
    "gpu_min_points",
    "metal_available",
    "resolve_backend",
]

# GEOS path: shapely point construction plus the STRtree query, per point ...
CPU_US_PER_POINT = 0.36
# ... plus the prepared-geometry predicate, per candidate polygon edge.
CPU_US_PER_EDGE = 0.00125
# GPU path: fixed cost of upload, cell assignment, one launch and the sync
# (first match), and of the count/scan/compaction passes plus their syncs
# (all matches).
GPU_LAUNCH_US = 300.0
GPU_ALL_MATCHES_LAUNCH_US = 1200.0
# Used when the caller has no index to estimate candidate edge work from
# (about three 48-vertex candidates per point).
DEFAULT_EDGES_PER_POINT = 150.0

_VALID = ("auto", "gpu", "cpu")


def metal_available() -> bool:
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


def gpu_min_points(
    edges_per_point: float = DEFAULT_EDGES_PER_POINT, *, all_matches: bool = False
) -> int:
    """Smallest host point count at which the cost model prefers the GPU.

    ``edges_per_point`` is the expected number of candidate polygon edges a
    point's GEOS predicate has to consider (``GridIndex`` estimates it from its
    per-cell edge work); detailed polygons lower the threshold. All-match
    queries use the larger fixed GPU cost of their compaction passes.
    """
    edges = float(edges_per_point)
    edges = edges if edges > 0.0 else 0.0  # NaN compares False: treated as 0
    per_point = CPU_US_PER_POINT + CPU_US_PER_EDGE * edges
    if per_point <= 0.0:  # a free GEOS path (only by re-fitting): never GPU
        return 2**31 - 1
    launch = GPU_ALL_MATCHES_LAUNCH_US if all_matches else GPU_LAUNCH_US
    return max(1, int(math.ceil(launch / per_point)))


def resolve_backend(
    requested: str | None,
    n_points: int | None,
    *,
    edges_per_point: float = DEFAULT_EDGES_PER_POINT,
    all_matches: bool = False,
) -> str:
    """Return ``"gpu"`` or ``"cpu"`` for this workload.

    ``n_points`` is the host-resident point count, or ``None`` for
    device-resident inputs, which ``auto`` always sends to the GPU when Metal
    is available (the fallback would need a host copy). ``all_matches``
    selects the all-match route's larger fixed GPU cost.
    """
    backend = requested or os.environ.get("MLX_SPATIAL_BACKEND", "auto")
    if not isinstance(backend, str):
        raise ValueError(f"backend must be one of {_VALID}, got {backend!r}")
    backend = backend.lower()
    if backend not in _VALID:
        raise ValueError(
            f"backend must be one of {_VALID}, got {backend!r}"
        )
    if backend == "auto":
        if not metal_available():
            return "cpu"
        if n_points is None:
            return "gpu"
        threshold = gpu_min_points(edges_per_point, all_matches=all_matches)
        return "gpu" if int(n_points) >= threshold else "cpu"
    if backend == "gpu" and not metal_available():
        raise RuntimeError("backend='gpu' requested but Metal is not available")
    return backend
