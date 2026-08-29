"""Coordinate precision management for fp32 GPU compute.

Metal is fp32-only. Raw lon/lat degrees stored in fp32 carry roughly 1e-5
degree spacing at continental longitudes (about a meter on the ground) —
enough to flip containment decisions near polygon edges. All GPU work in
mlx-spatial therefore runs in *locally re-centered* coordinates: a float64
origin (the dataset bbox center) is subtracted on the CPU before casting to
float32. At city/state extents this restores millimeter-scale resolution.

Points and polygons taking part in the same predicate must share one origin;
:class:`mlx_spatial.PolygonSet` owns it and re-centers query points with it.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "compute_origin",
    "recenter",
    "float32_resolution",
    "boundary_tolerance",
]


def compute_origin(*coord_arrays: np.ndarray) -> np.ndarray:
    """Bbox center of one or more ``(..., 2)`` coordinate arrays, float64 ``(2,)``.

    NaN coordinates (e.g. bounds of empty geometries) are ignored.
    """
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    for arr in coord_arrays:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
        if arr.size == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            lo = np.fmin(lo, np.nanmin(arr, axis=0))
            hi = np.fmax(hi, np.nanmax(arr, axis=0))
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        return np.zeros(2)
    return (lo + hi) / 2.0


def recenter(coords: np.ndarray, origin: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Subtract ``origin`` in float64, then cast (default: to float32)."""
    coords = np.asarray(coords, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    return np.ascontiguousarray(coords - origin, dtype=dtype)


def float32_resolution(magnitude: float) -> float:
    """Spacing (ulp) of float32 at the given coordinate magnitude."""
    magnitude = abs(float(magnitude))
    if magnitude == 0.0 or not np.isfinite(magnitude):
        magnitude = 1e-30
    return float(np.spacing(np.float32(magnitude)))


def boundary_tolerance(max_abs_coord: float) -> float:
    """Half-width of the undefined band around polygon boundaries.

    Points within this distance of an edge (in re-centered coordinate units,
    which equal the input units — re-centering is a pure translation) may be
    classified either way by the fp32 crossing-number test. Golden tests
    exclude such points; callers needing exact boundary semantics should
    refine near-boundary hits with shapely.
    """
    return 8.0 * float32_resolution(max_abs_coord)
