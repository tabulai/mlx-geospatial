"""Backend selection: GPU kernels vs. the host fallback.

Both backends evaluate the same fp32 even-odd predicate (the host fallback
runs it in NumPy, bit-identical to the kernels), so dispatch is purely a
speed decision. The GPU join costs a roughly fixed launch overhead (upload,
cell assignment, kernel, synchronization) that dominates until kernel work
shows: tens of thousands of points for simple polygons, a few thousand for
detailed ones. The NumPy fallback costs about 0.1 us per point plus ~9 ns per
candidate polygon edge, plus ~50 us of fixed setup that is not modelled.
``auto`` compares the two with that two-term cost model, so a few hundred
points against typical zone layers, a couple of thousand against sparse
tilings, or fewer than ten against detailed polygons already run on the GPU;
all-match queries carry a larger fixed GPU cost (count, scan and compaction
passes) and switch later. The fixed GPU cost depends on the GPU's clock state
(on an Apple M4 Max about 200 us per first-match query in a back-to-back
burst, 0.8-1.1 ms after host-only work has let the GPU idle; 1.1 versus 2.5
ms for all matches), so ``GPU_LAUNCH_US`` and ``GPU_ALL_MATCHES_LAUNCH_US``
sit between the two regimes. The cost of a wrong choice is bounded on both
sides: picking the CPU while the GPU was warm loses at most the gap between
the constant and the warm launch (~0.2 ms first match, ~0.45 ms all matches,
on every layer, since the index supplies the edge estimate), picking the GPU
after it has idled loses up to ~0.6 ms (first match) or ~1 ms (all matches)
for point counts just above the threshold. ``python bench/bench.py --op
crossover`` re-fits the constants for
another machine and prints both regimes. Override per call
(``backend=``) or globally via the environment variable
``MLX_SPATIAL_BACKEND`` (``auto`` | ``gpu`` | ``cpu``).
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

# Host fallback (NumPy even-odd): per-point/per-candidate bookkeeping ...
CPU_US_PER_POINT = 0.09
# ... plus the crossing test, per candidate polygon edge.
CPU_US_PER_EDGE = 0.0086
# GPU path: fixed cost of upload, cell assignment, one launch and the sync
# (first match), and of the count/scan/compaction passes plus their syncs
# (all matches); both between the warm-burst (~200 / 1100 us) and idle-GPU
# (~1000 / 2500 us) measurements on an M4 Max.
GPU_LAUNCH_US = 350.0
GPU_ALL_MATCHES_LAUNCH_US = 1500.0
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

    ``edges_per_point`` is the expected number of candidate polygon edges the
    host fallback tests per point (``GridIndex`` estimates it from its
    per-cell edge work); detailed polygons lower the threshold. All-match
    queries use the larger fixed GPU cost of their compaction passes.
    """
    edges = float(edges_per_point)
    edges = edges if edges > 0.0 else 0.0  # NaN compares False: treated as 0
    per_point = CPU_US_PER_POINT + CPU_US_PER_EDGE * edges
    if per_point <= 0.0:  # a free host path (only by re-fitting): never GPU
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
