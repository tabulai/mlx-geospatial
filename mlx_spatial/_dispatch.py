"""Backend selection: GPU kernels vs. shapely CPU fallback.

Small workloads lose to GEOS on kernel-launch and transfer overhead, so the
join path dispatches on the number of candidate pairs. The default can be
overridden per call (``backend=``) or globally via the environment variable
``MLX_SPATIAL_BACKEND`` (``auto`` | ``gpu`` | ``cpu``).
"""

from __future__ import annotations

import os

import mlx.core as mx

__all__ = ["GPU_MIN_CANDIDATE_PAIRS", "metal_available", "resolve_backend"]

# Below roughly this many point-polygon candidate pairs, single-threaded GEOS
# beats kernel launch + host<->device transfer overhead.
GPU_MIN_CANDIDATE_PAIRS = 100_000

_VALID = ("auto", "gpu", "cpu")


def metal_available() -> bool:
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


def resolve_backend(requested: str | None, n_candidate_pairs: int) -> str:
    """Return ``"gpu"`` or ``"cpu"`` for this workload."""
    backend = requested or os.environ.get("MLX_SPATIAL_BACKEND", "auto")
    backend = backend.lower()
    if backend not in _VALID:
        raise ValueError(
            f"backend must be one of {_VALID}, got {backend!r}"
        )
    if backend == "auto":
        if not metal_available():
            return "cpu"
        return "gpu" if n_candidate_pairs >= GPU_MIN_CANDIDATE_PAIRS else "cpu"
    if backend == "gpu" and not metal_available():
        raise RuntimeError("backend='gpu' requested but Metal is not available")
    return backend
