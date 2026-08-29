"""Great-circle distances on the sphere.

All functions take coordinates as ``(..., 2)`` arrays in ``(lon, lat)`` order
— matching shapely/GeoArrow ``(x, y)`` convention — in degrees by default.
Distances are returned in the units of ``radius`` (meters by default).

Accuracy: computation is fp32 on the GPU. Differencing nearby coordinates in
fp32 radians bounds absolute error at roughly 1-2 m away from the antipode,
so relative error shrinks with distance (~1e-4 at 10 km) and is in any case
far below the ~0.5% sphere-vs-ellipsoid modeling error inherent to
haversine. Exception: within ~2 degrees of the exact antipode the arcsin
amplifies fp32 rounding, growing absolute error to 10^2-10^4 m (relative
error stays below ~2e-4). Degree-to-radian conversion happens in float64
whenever the input is a CPU array. Chunked paths peak at ~2x the result size
(chunks are concatenated at the end).
"""

from __future__ import annotations

import math

import numpy as np
import mlx.core as mx

__all__ = ["R_EARTH_M", "haversine", "haversine_cross", "haversine_knn"]

R_EARTH_M = 6371008.8  # IUGG mean Earth radius, meters

_DEG2RAD_F32 = np.float32(math.pi / 180.0)


def _prep(a, degrees: bool) -> mx.array:
    """Normalize input to an mx float32 ``(..., 2)`` array in radians."""
    if isinstance(a, mx.array):
        arr = a.astype(mx.float32)
        if arr.ndim == 0 or arr.shape[-1] != 2:
            raise ValueError(f"expected (..., 2) (lon, lat) coordinates, got shape {arr.shape}")
        if degrees:
            arr = arr * _DEG2RAD_F32
        return arr
    np_arr = np.asarray(a)
    if np_arr.dtype == object:
        from ._ingest import as_point_array

        np_arr = as_point_array(a)
    np_arr = np.asarray(np_arr, dtype=np.float64)
    if np_arr.ndim == 0 or np_arr.shape[-1] != 2:
        raise ValueError(f"expected (..., 2) (lon, lat) coordinates, got shape {np_arr.shape}")
    if degrees:
        np_arr = np.radians(np_arr)
    return mx.array(np.ascontiguousarray(np_arr, dtype=np.float32))


def _hav_graph(lon1, lat1, lon2, lat2, radius):
    sin_dlat = mx.sin((lat2 - lat1) * 0.5)
    sin_dlon = mx.sin((lon2 - lon1) * 0.5)
    h = sin_dlat * sin_dlat + mx.cos(lat1) * mx.cos(lat2) * sin_dlon * sin_dlon
    h = mx.clip(h, 0.0, 1.0)
    return (2.0 * radius) * mx.arcsin(mx.sqrt(h))


# Fused via mx.compile: the unfused broadcast chain materializes ~8 (N, M)
# temporaries and is memory-bound; fusion turns it into one kernel.
_hav_fused = mx.compile(_hav_graph)


def _hav(lon1, lat1, lon2, lat2, radius: float) -> mx.array:
    return _hav_fused(lon1, lat1, lon2, lat2, mx.array(radius, dtype=mx.float32))


def haversine(a, b, *, radius: float = R_EARTH_M, degrees: bool = True) -> mx.array:
    """Elementwise (broadcasting) great-circle distance between ``a`` and ``b``."""
    a = _prep(a, degrees)
    b = _prep(b, degrees)
    return _hav(a[..., 0], a[..., 1], b[..., 0], b[..., 1], radius)


def haversine_cross(
    a,
    b,
    *,
    radius: float = R_EARTH_M,
    degrees: bool = True,
    chunk_bytes: int = 512 * 1024 * 1024,
) -> mx.array:
    """All-pairs distance matrix: ``(N, 2) x (M, 2) -> (N, M)``."""
    a = _prep(a, degrees)
    b = _prep(b, degrees)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("haversine_cross expects 2-D inputs of shape (N, 2) and (M, 2)")
    n, m = a.shape[0], b.shape[0]
    lon2 = b[:, 0][None, :]
    lat2 = b[:, 1][None, :]
    rows_per_chunk = max(1, chunk_bytes // max(m * 4 * 4, 1))
    if rows_per_chunk >= n:
        return _hav(a[:, 0:1], a[:, 1:2], lon2, lat2, radius)
    chunks = []
    for start in range(0, n, rows_per_chunk):
        stop = min(start + rows_per_chunk, n)
        d = _hav(a[start:stop, 0:1], a[start:stop, 1:2], lon2, lat2, radius)
        mx.eval(d)
        chunks.append(d)
    return mx.concatenate(chunks, axis=0)


def haversine_knn(
    query,
    reference,
    k: int,
    *,
    radius: float = R_EARTH_M,
    degrees: bool = True,
    chunk_bytes: int = 512 * 1024 * 1024,
) -> tuple[mx.array, mx.array]:
    """k nearest reference points for each query point, by great-circle distance.

    Returns ``(distances, indices)`` of shape ``(N, k)``, sorted ascending per
    row. Ties near the k-th neighbor may resolve either way at fp32
    (~meter-scale) distance resolution.

    Brute force, O(N·M): fastest when the reference set is moderate (up to
    roughly 10-20k references). For very large reference sets a CPU tree
    (e.g. sklearn ``BallTree(metric="haversine")``) is algorithmically better.
    """
    q = _prep(query, degrees)
    r = _prep(reference, degrees)
    if q.ndim != 2 or r.ndim != 2:
        raise ValueError("haversine_knn expects 2-D inputs of shape (N, 2) and (M, 2)")
    n, m = q.shape[0], r.shape[0]
    if not 1 <= k <= m:
        raise ValueError(f"k must be in [1, {m}], got {k}")
    if n == 0:
        return (
            mx.zeros((0, k), dtype=mx.float32),
            mx.zeros((0, k), dtype=mx.int32),
        )
    lon2 = r[:, 0][None, :]
    lat2 = r[:, 1][None, :]
    rows_per_chunk = max(1, chunk_bytes // max(m * 4 * 4, 1))
    dist_chunks, idx_chunks = [], []
    for start in range(0, n, rows_per_chunk):
        stop = min(start + rows_per_chunk, n)
        d = _hav(q[start:stop, 0:1], q[start:stop, 1:2], lon2, lat2, radius)
        if k < m:
            idx = mx.argpartition(d, kth=k - 1, axis=1)[:, :k]
            dk = mx.take_along_axis(d, idx, axis=1)
        else:
            idx = mx.broadcast_to(mx.arange(m)[None, :], d.shape)
            dk = d
        order = mx.argsort(dk, axis=1)
        dk = mx.take_along_axis(dk, order, axis=1)
        idx = mx.take_along_axis(idx, order, axis=1).astype(mx.int32)
        mx.eval(dk, idx)
        dist_chunks.append(dk)
        idx_chunks.append(idx)
    if len(dist_chunks) == 1:
        return dist_chunks[0], idx_chunks[0]
    return mx.concatenate(dist_chunks, axis=0), mx.concatenate(idx_chunks, axis=0)
