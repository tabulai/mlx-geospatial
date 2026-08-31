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
import operator

import numpy as np
import mlx.core as mx

__all__ = ["R_EARTH_M", "haversine", "haversine_cross", "haversine_knn"]

R_EARTH_M = 6371008.8  # IUGG mean Earth radius, meters

_DEG2RAD_F32 = np.float32(math.pi / 180.0)


# Keep radius inside the normal float32 range used by MLX/Metal.  Below the
# lower bound the scalar itself can underflow (or be flushed) to zero, which
# makes the composed kNN path rank every reference as an exact tie.  The upper
# bound keeps pi * radius (the largest representable distance) well below
# float32 max, so distances cannot collide with kNN's inf "no neighbor" pad.
_MIN_RADIUS = float(np.finfo(np.float32).tiny)
_MAX_RADIUS = 1e37


def _check_radius(radius) -> float:
    radius = float(radius)
    if not (_MIN_RADIUS <= radius < _MAX_RADIUS):
        raise ValueError(
            "radius must be finite and within the normal float32 computation "
            f"range [{_MIN_RADIUS:g}, {_MAX_RADIUS:g}), got {radius}"
        )
    return radius


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


# ---------------------------------------------------------------------------
# Fused k-nearest-neighbor selection kernel
#
# The composed path (distance matrix + mx.argpartition) sorts all N*M
# distances to keep k of them — mx.argpartition costs a full sort. This
# kernel gives each query one thread that streams the reference set and
# maintains its k best in registers: no (N, M) matrix, no sort, one compare
# per candidate. Techniques borrowed from mlxlearn's neighbor index: rank by
# a cheap monotone surrogate (the haversine `h` term — no asin/sqrt in the
# inner loop, as mlxlearn ranks by squared distance), then recompute the
# survivors' distances with the exact _hav arithmetic so reported distances
# are bit-identical to haversine_cross.
# ---------------------------------------------------------------------------

_KNN_KERNEL_MAX_K = 32  # k register slots per thread; beyond this, occupancy suffers

_KNN_SELECT_SOURCE = """
    uint tid = thread_position_in_grid.x;
    int n_q = shape[0];
    if (tid >= (uint)n_q) return;
    int m = shape[1];
    float qlon = q[2 * (size_t)tid + 0];
    float qlat = q[2 * (size_t)tid + 1];
    float qc = qcos[tid];

    float best_h[K];
    int best_i[K];
    for (int s = 0; s < K; ++s) { best_h[s] = INFINITY; best_i[s] = -1; }

    for (int j = 0; j < m; ++j) {
        float sdlat = metal::precise::sin((r[2 * (size_t)j + 1] - qlat) * 0.5f);
        float sdlon = metal::precise::sin((r[2 * (size_t)j + 0] - qlon) * 0.5f);
        float h = sdlat * sdlat + qc * rcos[j] * sdlon * sdlon;
        // Strict < plus strict shift keeps earlier reference indices on
        // exact ties (matching a stable sort's tie-break).
        if (h < best_h[K - 1]) {
            int s = K - 1;
            while (s > 0 && best_h[s - 1] > h) {
                best_h[s] = best_h[s - 1];
                best_i[s] = best_i[s - 1];
                --s;
            }
            best_h[s] = h;
            best_i[s] = j;
        }
    }
    for (int s = 0; s < K; ++s) {
        out_h[(size_t)tid * K + s] = best_h[s];
        out_i[(size_t)tid * K + s] = best_i[s];
    }
"""

_knn_select_kernel = None


def _get_knn_select_kernel():
    global _knn_select_kernel
    if _knn_select_kernel is None:
        _knn_select_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_knn_select",
            input_names=["q", "qcos", "r", "rcos", "shape"],
            output_names=["out_h", "out_i"],
            source=_KNN_SELECT_SOURCE,
        )
    return _knn_select_kernel


def _knn_fused(q: mx.array, r: mx.array, k: int, radius: float, *, threadgroup: int = 128):
    """Fused-select path: q/r are (N, 2)/(M, 2) float32 radians."""
    radius = _check_radius(radius)
    n, m = int(q.shape[0]), int(r.shape[0])
    kernel = _get_knn_select_kernel()
    qcos = mx.cos(q[:, 1])
    rcos = mx.cos(r[:, 1])
    grid_x = ((n + threadgroup - 1) // threadgroup) * threadgroup
    out_h, out_i = kernel(
        inputs=[q, qcos, r, rcos, mx.array([n, m], dtype=mx.int32)],
        template=[("K", k)],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n, k), (n, k)],
        output_dtypes=[mx.float32, mx.int32],
    )
    # Recompute the survivors with the exact _hav arithmetic (mlxlearn's
    # refine step): reported distances match haversine_cross bit-for-bit.
    # A slot can stay -1 only when fewer than k references have finite
    # coordinates (NaN rows never win a comparison); keep it -1 with inf
    # distance rather than letting the negative index wrap the gather.
    safe_i = mx.maximum(out_i, 0)
    ref_sel = mx.take(r, safe_i, axis=0)  # (n, k, 2)
    d = _hav(q[:, 0:1], q[:, 1:2], ref_sel[..., 0], ref_sel[..., 1], radius)
    d = mx.where(out_i < 0, mx.array(np.inf, dtype=mx.float32), d)
    order = mx.argsort(d, axis=1)
    d = mx.take_along_axis(d, order, axis=1)
    idx = mx.take_along_axis(out_i, order, axis=1)
    return d, idx


def haversine(a, b, *, radius: float = R_EARTH_M, degrees: bool = True) -> mx.array:
    """Elementwise (broadcasting) great-circle distance between ``a`` and ``b``."""
    radius = _check_radius(radius)
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
    radius = _check_radius(radius)
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

    NaN coordinates never match: a reference row with NaN coordinates is
    never selected, and any slot that no finite-coordinate reference can fill
    — including every slot of a query row with NaN coordinates — reports
    index ``-1`` with distance ``inf``. Both execution paths below share this
    contract.

    Exact brute force. On Metal with ``k <= 32`` this runs a fused
    selection kernel (one thread per query keeps its k best in registers —
    no ``(N, M)`` matrix, no sort), which stays competitive with tree methods
    even for reference sets in the hundreds of thousands. Otherwise a
    chunked distance-matrix + ``argpartition`` fallback runs; that fallback
    is sort-bound, so for large reference sets with ``k > 32`` a CPU tree
    (sklearn ``BallTree(metric="haversine")``) may win.
    """
    from ._dispatch import metal_available

    radius = _check_radius(radius)
    if isinstance(k, (bool, np.bool_)):
        # bool is a valid int per Python's numeric model but is not a valid
        # cardinality in k-nearest-neighbor semantics.
        raise TypeError("k must be an integer, got bool")
    try:
        # Normalize numpy/other integer types: Metal template arguments and
        # range checks both need a plain Python int; reject floats cleanly.
        k = operator.index(k)
    except TypeError:
        raise TypeError(f"k must be an integer, got {type(k).__name__}") from None
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
    if metal_available() and k <= _KNN_KERNEL_MAX_K:
        d, idx = _knn_fused(q, r, k, radius)
        mx.eval(d, idx)
        return d, idx
    return _knn_composed(q, r, k, radius, chunk_bytes)


def _knn_composed(q: mx.array, r: mx.array, k: int, radius: float, chunk_bytes: int):
    """Chunked distance-matrix + argpartition fallback (no Metal, or k > 32)."""
    radius = _check_radius(radius)
    n, m = q.shape[0], r.shape[0]
    lon2 = r[:, 0][None, :]
    lat2 = r[:, 1][None, :]
    rows_per_chunk = max(1, chunk_bytes // max(m * 4 * 4, 1))
    dist_chunks, idx_chunks = [], []
    for start in range(0, n, rows_per_chunk):
        stop = min(start + rows_per_chunk, n)
        d = _hav(q[start:stop, 0:1], q[start:stop, 1:2], lon2, lat2, radius)
        # Match the fused kernel's NaN contract: NaN coordinates (query or
        # reference) become +inf distances that never win selection, and any
        # slot only inf could fill reports index -1. _hav itself never
        # produces inf (h is clipped and _check_radius bounds pi * radius
        # below float32 max), so inf is unambiguous.
        d = mx.where(mx.isnan(d), mx.array(np.inf, dtype=mx.float32), d)
        if k < m:
            idx = mx.argpartition(d, kth=k - 1, axis=1)[:, :k]
            dk = mx.take_along_axis(d, idx, axis=1)
        else:
            idx = mx.broadcast_to(mx.arange(m)[None, :], d.shape)
            dk = d
        order = mx.argsort(dk, axis=1)
        dk = mx.take_along_axis(dk, order, axis=1)
        idx = mx.take_along_axis(idx, order, axis=1).astype(mx.int32)
        idx = mx.where(mx.isinf(dk), mx.array(-1, dtype=mx.int32), idx)
        mx.eval(dk, idx)
        dist_chunks.append(dk)
        idx_chunks.append(idx)
    if len(dist_chunks) == 1:
        return dist_chunks[0], idx_chunks[0]
    return mx.concatenate(dist_chunks, axis=0), mx.concatenate(idx_chunks, axis=0)
