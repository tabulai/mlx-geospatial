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
whenever the input is a CPU array. ``haversine_cross`` runs as one fused
launch by default (the result is the only allocation); an explicit
``chunk_bytes`` bounds the rows per launch and then peaks at ~2x the result
size while chunks are concatenated. An explicit budget must be a positive
integer; ``None`` selects the single-launch default.
"""

from __future__ import annotations

import math
import operator
import textwrap

import numpy as np
import mlx.core as mx

from ._validation import check_chunk_bytes

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
    """Normalize input to an mx float32 ``(..., 2)`` array in radians.

    ``mx.float64`` arrays (CPU-only in MLX) take the float64 NumPy conversion,
    so their degree-to-radian step matches the NumPy path bit for bit on every
    supported MLX version; float32 MLX arrays convert on the device.
    """
    from ._ingest import LocalPoints

    if isinstance(a, LocalPoints):
        raise TypeError(
            "LocalPoints hold re-centered polygon-frame coordinates, not lon/lat; "
            "pass the original coordinates to the haversine functions"
        )
    if isinstance(a, mx.array) and a.dtype == mx.float64:
        a = np.asarray(a)
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
# Fused k-nearest-neighbor selection kernels
#
# The composed path (distance matrix + mx.argpartition) sorts all N*M
# distances to keep k of them — mx.argpartition costs a full sort. The scalar
# kernel gives each query one thread. The SIMDgroup kernel gives each query 32
# lanes, lets every lane maintain a local top-k, then performs an exact 32-way
# merge with shuffle reductions. Both stream the reference set without an
# (N, M) matrix and rank by the cheap monotone haversine `h` term (no
# asin/sqrt in the inner loop). The survivors are recomputed with the exact
# _hav arithmetic so reported distances are bit-identical to haversine_cross.
# ---------------------------------------------------------------------------

_KNN_KERNEL_MAX_K = 32  # k register slots per thread; beyond this, occupancy suffers
_KNN_SIMDGROUP_WIDTH = 32
# Maximum query counts for which 32 lanes/query beat one thread/query on an
# M4-class GPU. The limits deliberately stay on the fast side of measured
# crossovers; larger k consumes more registers and performs a longer merge.
_KNN_SIMDGROUP_QUERY_LIMITS = (
    (1, 32_768),
    (5, 16_384),
    (10, 8_192),
    (16, 4_096),
    (32, 2_048),
)
_KNN_SIMDGROUP_LARGE_REF_QUERY_LIMITS = (
    (1, 65_536),
    (5, 32_768),
    (10, 16_384),
    (16, 8_192),
    (32, 2_048),
)

_KNN_SELECT_SOURCE = """
    // Keep this expression tree bit-identical to _hav_graph.  Contracting
    // the final multiply-add changes h by an ulp for some pairs; close to an
    // antipode, that is enough to reverse neighbors after asin amplification.
    #pragma clang fp contract(off)

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
        float h_lat = sdlat * sdlat;
        float h_lon = qc * rcos[j] * sdlon * sdlon;
        float h = h_lat + h_lon;
        // Match mx.clip while preserving NaN: comparisons with NaN are
        // false, so non-finite coordinate rows still never enter the list.
        if (h < 0.0f) h = 0.0f;
        else if (h > 1.0f) h = 1.0f;
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
        out_i[(size_t)tid * K + s] = best_i[s];
    }
"""


_KNN_SIMDGROUP_SELECT_SOURCE = """
    // See the scalar selector: contraction here can disagree with the public
    // distance graph by an ulp and produce a wrong ordering near antipodes.
    #pragma clang fp contract(off)

    uint tid = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint qid = tid / 32;
    int n_q = shape[0];
    // The launch is rounded to whole threadgroups. Since qid changes only at
    // SIMDgroup boundaries, this return is uniform across every SIMDgroup.
    if (qid >= (uint)n_q) return;
    int m = shape[1];
    float qlon = q[2 * (size_t)qid + 0];
    float qlat = q[2 * (size_t)qid + 1];
    float qc = qcos[qid];

    // Each lane sees monotonically increasing reference indices. Strict
    // insertion therefore keeps its local list ordered by (h, index).
    float best_h[K];
    int best_i[K];
    for (int s = 0; s < K; ++s) { best_h[s] = INFINITY; best_i[s] = -1; }

    for (int j = (int)lane; j < m; j += 32) {
        float sdlat = metal::precise::sin(
            (r[2 * (size_t)j + 1] - qlat) * 0.5f);
        float sdlon = metal::precise::sin(
            (r[2 * (size_t)j + 0] - qlon) * 0.5f);
        float h_lat = sdlat * sdlat;
        float h_lon = qc * rcos[j] * sdlon * sdlon;
        float h = h_lat + h_lon;
        if (h < 0.0f) h = 0.0f;
        else if (h > 1.0f) h = 1.0f;
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

    // The global top-k is contained in the union of the 32 local top-k
    // lists. Merge their heads k times. The lexicographic comparator makes
    // exact ties deterministic and identical to the scalar selector.
    int cursor = 0;
    for (int s = 0; s < K; ++s) {
        float winner_h = cursor < K ? best_h[cursor] : INFINITY;
        int winner_i = cursor < K ? best_i[cursor] : -1;
        uint winner_lane = lane;
        for (ushort delta = 16; delta > 0; delta >>= 1) {
            float other_h = metal::simd_shuffle_down(winner_h, delta);
            int other_i = metal::simd_shuffle_down(winner_i, delta);
            uint other_lane = metal::simd_shuffle_down(winner_lane, delta);
            bool other_wins = other_h < winner_h ||
                (other_h == winner_h && other_i >= 0 &&
                 (winner_i < 0 || other_i < winner_i));
            if (other_wins) {
                winner_h = other_h;
                winner_i = other_i;
                winner_lane = other_lane;
            }
        }
        uint chosen_lane = metal::simd_broadcast(winner_lane, 0);
        int chosen_i = metal::simd_broadcast(winner_i, 0);
        if (lane == 0) out_i[(size_t)qid * K + s] = chosen_i;
        if (lane == chosen_lane && cursor < K) ++cursor;
    }
"""

_knn_select_kernel = None
_knn_simdgroup_select_kernel = None


def _get_knn_select_kernel():
    global _knn_select_kernel
    if _knn_select_kernel is None:
        _knn_select_kernel = mx.fast.metal_kernel(
            name="mlx_geospatial_knn_select",
            input_names=["q", "qcos", "r", "rcos", "shape"],
            output_names=["out_i"],
            source=_KNN_SELECT_SOURCE,
        )
    return _knn_select_kernel


def _get_knn_simdgroup_select_kernel():
    global _knn_simdgroup_select_kernel
    if _knn_simdgroup_select_kernel is None:
        _knn_simdgroup_select_kernel = mx.fast.metal_kernel(
            name="mlx_geospatial_knn_simdgroup_select",
            input_names=["q", "qcos", "r", "rcos", "shape"],
            output_names=["out_i"],
            source=_KNN_SIMDGROUP_SELECT_SOURCE,
        )
    return _knn_simdgroup_select_kernel


def _knn_refine(
    q: mx.array, r: mx.array, out_i: mx.array, radius: float
) -> tuple[mx.array, mx.array]:
    """Recompute and stably order selected survivors with public arithmetic."""
    # A slot can stay -1 only when fewer than k references have finite
    # coordinates (NaN rows never win a comparison); keep it -1 with inf
    # distance rather than letting the negative index wrap the gather.
    safe_i = mx.maximum(out_i, 0)
    ref_sel = mx.take(r, safe_i, axis=0)
    d = _hav(q[:, 0:1], q[:, 1:2], ref_sel[..., 0], ref_sel[..., 1], radius)
    d = mx.where(out_i < 0, mx.array(np.inf, dtype=mx.float32), d)
    order = mx.argsort(d, axis=1)
    d = mx.take_along_axis(d, order, axis=1)
    idx = mx.take_along_axis(out_i, order, axis=1)
    return d, idx


def _knn_fused(q: mx.array, r: mx.array, k: int, radius: float, *, threadgroup: int = 128):
    """Scalar fused-select path: one Metal thread per query."""
    radius = _check_radius(radius)
    n, m = int(q.shape[0]), int(r.shape[0])
    kernel = _get_knn_select_kernel()
    qcos = mx.cos(q[:, 1])
    rcos = mx.cos(r[:, 1])
    grid_x = ((n + threadgroup - 1) // threadgroup) * threadgroup
    (out_i,) = kernel(
        inputs=[q, qcos, r, rcos, mx.array([n, m], dtype=mx.int32)],
        template=[("K", k)],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n, k)],
        output_dtypes=[mx.int32],
        stream=mx.gpu,
    )
    return _knn_refine(q, r, out_i, radius)


def _knn_simdgroup(
    q: mx.array, r: mx.array, k: int, radius: float, *, threadgroup: int = 128
):
    """Exact fused-select path: one 32-lane SIMDgroup per query."""
    radius = _check_radius(radius)
    if not _KNN_SIMDGROUP_WIDTH <= threadgroup <= 1024 or (
        threadgroup % _KNN_SIMDGROUP_WIDTH
    ):
        raise ValueError(
            "threadgroup must be a multiple of the SIMDgroup width (32) "
            "between 32 and 1024"
        )
    n, m = int(q.shape[0]), int(r.shape[0])
    kernel = _get_knn_simdgroup_select_kernel()
    qcos = mx.cos(q[:, 1])
    rcos = mx.cos(r[:, 1])
    threads = n * _KNN_SIMDGROUP_WIDTH
    grid_x = ((threads + threadgroup - 1) // threadgroup) * threadgroup
    (out_i,) = kernel(
        inputs=[q, qcos, r, rcos, mx.array([n, m], dtype=mx.int32)],
        template=[("K", k)],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n, k)],
        output_dtypes=[mx.int32],
        stream=mx.gpu,
    )
    return _knn_refine(q, r, out_i, radius)


# ---------------------------------------------------------------------------
# Chord-filtered kNN selectors
#
# The selectors above spend two precise sines on every (query, reference)
# pair. On the unit sphere the squared chord between two points is exactly
# 4 * h, so a reference whose squared chord clearly exceeds 4 * w, with w the
# current k-th best h, cannot be inserted. These selectors compute the squared
# chord from precomputed unit vectors (three loads from one 16-byte row, three
# subtractions, three multiply-adds) and evaluate the unchanged exact h only
# when the chord passes a conservative threshold
#
#     thr(w) = (1 + rho) * ((1 + rho) * (2 * sqrt(w) + slack) + eta)^2,
#
# recomputed on insertion only. A skipped reference has an fp32 h above w, so
# it could not have been inserted: the selected set, its order and every
# output bit match the selectors above. The error budget:
#
# - eta (1e-5, in chord units) covers the absolute error of the fp32 unit
#   vectors and their squared difference, about 1e-6 at worst;
# - rho (1e-4) covers relative rounding in sin, the products, sqrt and the
#   threshold arithmetic itself, a few 1e-7;
# - slack = 16 * 2^-24 * B, with B the largest finite |coordinate| in
#   radians, covers the kernel's rounded coordinate differences: with
#   unwrapped longitudes (540 degrees, say) the half-angle passed to sin is off
#   by up to 2^-24 * B, which moves 2 * sqrt(h) by up to 2 * 2^-24 * B, and
#   the latitude difference adds at most as much again.
#
# Two input conditions break that budget, and the device-side slack then
# switches pruning off (slack < 0), so the kernel runs the plain loop of the
# selectors above: a latitude beyond +-pi/2 makes cos(lat) negative, h_lat and
# h_lon then cancel and h carries ~1e-8 absolute error (1e-4 after the sqrt);
# coordinates beyond 1e3 rad (~57,000 degrees) leave the validated range. A
# NaN or inf coordinate gives a NaN chord, which fails the test, but such a
# pair has a NaN h that never inserts either.
#
# In the SIMDgroup selector each lane keeps its own top-k, and any lane's k-th
# best is an upper bound on the query's global k-th best. The lanes therefore
# share the smallest threshold: a reference it skips is worse than k others
# and cannot reach the merged top-k, even if it would have entered its lane's
# list.
#
# Both kernels also fuse _knn_refine: the k distances use the _hav_graph
# arithmetic (MLX's arcsin and sqrt are metal::precise), and a stable sort by
# distance replaces its argsort, which is stable as well.
# ---------------------------------------------------------------------------

_KNN_CHORD_RHO = 1e-4
_KNN_CHORD_ETA = 1e-5
_KNN_CHORD_SLACK_PER_RADIAN = 16.0 * 2.0**-24
_KNN_CHORD_MAX_COORD = 1e3  # radians
_HALF_PI_F32 = float(np.float32(math.pi / 2))
_KNN_CHORD_MAX_K = 16

# Exact h of reference REF, with the expression tree of _KNN_SELECT_SOURCE,
# and strict insertion. COS_LAT is the reference's cos(lat) (rcos[REF], or the
# same value from the unit-vector row just loaded); TIGHTEN marks where an
# insertion lowers the threshold.
_KNN_CHORD_VISIT = """\
float sdlat = metal::precise::sin((r[2 * (size_t)(REF) + 1] - qlat) * 0.5f);
float sdlon = metal::precise::sin((r[2 * (size_t)(REF) + 0] - qlon) * 0.5f);
float h_lat = sdlat * sdlat;
float h_lon = qc * COS_LAT * sdlon * sdlon;
float h = h_lat + h_lon;
if (h < 0.0f) h = 0.0f;
else if (h > 1.0f) h = 1.0f;
if (h < best_h[K - 1]) {
    int s = K - 1;
    while (s > 0 && best_h[s - 1] > h) {
        best_h[s] = best_h[s - 1];
        best_i[s] = best_i[s - 1];
        --s;
    }
    best_h[s] = h;
    best_i[s] = (REF);
TIGHTEN}
"""

_KNN_CHORD_TIGHTEN = """\
    float t = RHO1 * (2.0f * metal::precise::sqrt(best_h[K - 1]) + slack) + ETA;
    thr = metal::min(thr, t * t * RHO1);
"""

# Squared chord to reference REF. The fma never reaches a result: chordU only
# decides whether the exact h is evaluated.
_KNN_CHORD_CHORD = """\
float exU = rv[4 * (size_t)(REF) + 0] - qx;
float eyU = rv[4 * (size_t)(REF) + 1] - qy;
float ezU = rv[4 * (size_t)(REF) + 2] - qz;
float chordU = metal::fma(exU, exU, metal::fma(eyU, eyU, ezU * ezU));
"""


def _knn_chord_scan(simdgroup: bool) -> str:
    """Reference loop: four chords per iteration, all four skipped together
    when none passes; the plain loop when the slack disables pruning."""
    start, step = ("(int)lane", "32") if simdgroup else ("0", "1")

    def visit(ref, tighten=True):
        # The plain scan reads the contiguous rcos; the filtered scan reuses
        # the unit-vector row whose x, y, z the chord just loaded.
        cos_lat = "rv[4 * (size_t)(REF) + 3]" if tighten else "rcos[REF]"
        code = _KNN_CHORD_VISIT.replace("COS_LAT", cos_lat).replace("REF", ref)
        return code.replace("TIGHTEN", _KNN_CHORD_TIGHTEN if tighten else "")

    def chord(ref, u):
        return _KNN_CHORD_CHORD.replace("REF", ref).replace("U", str(u))

    def block(test, code):
        return f"if ({test}) {{\n{textwrap.indent(code, '    ')}}}\n"

    refs = [f"j + {u} * {step}" for u in range(4)]
    passes = " | ".join(f"(chord{u} <= thr)" for u in range(4))
    group = "thr = metal::simd_min(thr);\n" if simdgroup else ""
    group += "".join(chord(ref, u) for u, ref in enumerate(refs))
    group += f"if (!({passes})) continue;\n"
    group += "".join(block(f"chord{u} <= thr", visit(ref)) for u, ref in enumerate(refs))
    tail = chord("j", 0) + block("chord0 <= thr", visit("j"))
    plain = visit("j", tighten=False)
    scan = f"""\
float thr = INFINITY;
if (slack < 0.0f) {{
    // Pruning is off for these inputs: the plain scan of the selectors above.
    for (int j = {start}; j < m; j += {step}) {{
{textwrap.indent(plain, ' ' * 8)}    }}
}} else {{
    int j = {start};
    for (; j < m - 3 * {step}; j += 4 * {step}) {{
{textwrap.indent(group, ' ' * 8)}    }}
    for (; j < m; j += {step}) {{
{textwrap.indent(tail, ' ' * 8)}    }}
}}
"""
    return textwrap.indent(scan, " " * 4)


def _knn_chord_source(simdgroup: bool) -> str:
    head = """
    // See _KNN_SELECT_SOURCE: h must keep the _hav_graph expression tree.
    #pragma clang fp contract(off)

    uint tid = thread_position_in_grid.x;
"""
    if simdgroup:
        head += """\
    uint lane = thread_index_in_simdgroup;
    uint qid = tid / 32;
    int n_q = shape[0];
    // Uniform per SIMDgroup, as in _KNN_SIMDGROUP_SELECT_SOURCE.
    if (qid >= (uint)n_q) return;
"""
        tail = """
    // The exact merge of _KNN_SIMDGROUP_SELECT_SOURCE; lane s keeps slot s.
    int cursor = 0;
    float my_h = INFINITY;
    int my_i = -1;
    for (int s = 0; s < K; ++s) {
        float winner_h = cursor < K ? best_h[cursor] : INFINITY;
        int winner_i = cursor < K ? best_i[cursor] : -1;
        uint winner_lane = lane;
        for (ushort delta = 16; delta > 0; delta >>= 1) {
            float other_h = metal::simd_shuffle_down(winner_h, delta);
            int other_i = metal::simd_shuffle_down(winner_i, delta);
            uint other_lane = metal::simd_shuffle_down(winner_lane, delta);
            bool other_wins = other_h < winner_h ||
                (other_h == winner_h && other_i >= 0 &&
                 (winner_i < 0 || other_i < winner_i));
            if (other_wins) {
                winner_h = other_h;
                winner_i = other_i;
                winner_lane = other_lane;
            }
        }
        uint chosen_lane = metal::simd_broadcast(winner_lane, 0);
        int chosen_i = metal::simd_broadcast(winner_i, 0);
        float chosen_h = metal::simd_broadcast(winner_h, 0);
        if (lane == (uint)s) {
            my_h = chosen_h;
            my_i = chosen_i;
        }
        if (lane == chosen_lane && cursor < K) ++cursor;
    }
    // Rank by distance; ties keep the merged (h, index) order (stable).
    float my_d = my_i < 0 ? INFINITY
        : rad2[0] * metal::precise::asin(metal::precise::sqrt(my_h));
    int rank = 0;
    for (int t = 0; t < K; ++t) {
        float other_d = metal::simd_shuffle(my_d, (ushort)t);
        rank += (other_d < my_d || (other_d == my_d && t < (int)lane)) ? 1 : 0;
    }
    if (lane < (uint)K) {
        out_d[(size_t)qid * K + rank] = my_d;
        out_i[(size_t)qid * K + rank] = my_i;
    }
"""
    else:
        head += """\
    uint qid = tid;
    int n_q = shape[0];
    if (qid >= (uint)n_q) return;
"""
        tail = """
    // Distances, then a stable insertion sort by distance: ties keep the
    // (h, index) order. In place on purpose: separate private dd[K] / ii[K]
    // arrays miscompile (column K - 2 comes back 0) for K >= 24.
    for (int s = 0; s < K; ++s) {
        int bi = best_i[s];
        float d = bi < 0 ? INFINITY
            : rad2[0] * metal::precise::asin(metal::precise::sqrt(best_h[s]));
        int t = s;
        while (t > 0 && best_h[t - 1] > d) {
            best_h[t] = best_h[t - 1];
            best_i[t] = best_i[t - 1];
            --t;
        }
        best_h[t] = d;
        best_i[t] = bi;
    }
    for (int s = 0; s < K; ++s) {
        out_d[(size_t)qid * K + s] = best_h[s];
        out_i[(size_t)qid * K + s] = best_i[s];
    }
"""
    body = """
    int m = shape[1];
    float qlon = q[2 * (size_t)qid + 0];
    float qlat = q[2 * (size_t)qid + 1];
    float qx = qv[4 * (size_t)qid + 0];
    float qy = qv[4 * (size_t)qid + 1];
    float qz = qv[4 * (size_t)qid + 2];
    float qc = qv[4 * (size_t)qid + 3];
    float slack = bound[0];

    float best_h[K];
    int best_i[K];
    for (int s = 0; s < K; ++s) { best_h[s] = INFINITY; best_i[s] = -1; }

"""
    source = head + body + _knn_chord_scan(simdgroup) + tail
    return (
        source.replace("RHO1", _f32_literal(1.0 + _KNN_CHORD_RHO))
        .replace("ETA", _f32_literal(_KNN_CHORD_ETA))
    )


def _f32_literal(value: float) -> str:
    return repr(float(np.float32(value))) + "f"


_KNN_CHORD_SELECT_SOURCE = _knn_chord_source(simdgroup=False)
_KNN_CHORD_SIMDGROUP_SELECT_SOURCE = _knn_chord_source(simdgroup=True)

_knn_chord_kernels: dict[str, object] = {}


def _get_knn_chord_kernel(strategy: str):
    kernel = _knn_chord_kernels.get(strategy)
    if kernel is None:
        if strategy == "simdgroup":
            kernel = mx.fast.metal_kernel(
                name="mlx_geospatial_knn_chord_simdgroup_select",
                input_names=["q", "qv", "r", "rv", "rcos", "bound", "rad2", "shape"],
                output_names=["out_d", "out_i"],
                source=_KNN_CHORD_SIMDGROUP_SELECT_SOURCE,
            )
        else:
            kernel = mx.fast.metal_kernel(
                name="mlx_geospatial_knn_chord_select",
                input_names=["q", "qv", "r", "rv", "rcos", "bound", "rad2", "shape"],
                output_names=["out_d", "out_i"],
                source=_KNN_CHORD_SELECT_SOURCE,
            )
        _knn_chord_kernels[strategy] = kernel
    return kernel


def _knn_unit_vectors_graph(p: mx.array) -> tuple[mx.array, mx.array]:
    lon, lat = p[:, 0], p[:, 1]
    cos_lat = mx.cos(lat)
    rows = mx.stack(
        [cos_lat * mx.cos(lon), cos_lat * mx.sin(lon), mx.sin(lat), cos_lat], axis=1
    )
    return rows, cos_lat


# ``(x, y, z, cos lat)`` rows and the ``cos lat`` column on its own. The cosine
# is metal::precise::cos, compiled or not, so it equals the selectors'
# mx.cos(lat) bit for bit. Compiled to halve the passes over large inputs.
_knn_unit_vectors = mx.compile(_knn_unit_vectors_graph)


# One pass instead of three. Only exact constants belong in compiled graphs:
# mx.compile inlines float constants into the kernel source with too few
# digits (fp32 pi/2 compares as 1.570796), so the bounds below stay eager.
_finite_abs = mx.compile(lambda p: mx.where(mx.isfinite(p), mx.abs(p), 0.0))


def _knn_chord_slack(q: mx.array, r: mx.array) -> mx.array:
    """Device-side threshold slack ``16 * 2^-24 * B``, or -1 to disable pruning.

    Lazy on the GPU, so choosing the route never waits for a host sync.
    """
    lon_lat = mx.maximum(
        mx.max(_finite_abs(q), axis=0), mx.max(_finite_abs(r), axis=0)
    )
    b = mx.max(lon_lat)
    safe = (lon_lat[1] <= _HALF_PI_F32) & (b <= _KNN_CHORD_MAX_COORD)
    slack = mx.where(safe, b * _KNN_CHORD_SLACK_PER_RADIAN, -1.0)
    return slack.astype(mx.float32).reshape(1)


def _knn_chord(
    q: mx.array, r: mx.array, k: int, radius: float, strategy: str, *, threadgroup: int = 128
) -> tuple[mx.array, mx.array]:
    """Chord-filtered selection with the refine fused in; ``strategy`` is
    ``"scalar"`` (one thread per query) or ``"simdgroup"`` (32 lanes)."""
    radius = _check_radius(radius)
    if strategy not in ("scalar", "simdgroup"):
        raise ValueError(f"unknown kNN strategy {strategy!r}")
    if not _KNN_SIMDGROUP_WIDTH <= threadgroup <= 1024 or (
        threadgroup % _KNN_SIMDGROUP_WIDTH
    ):
        raise ValueError(
            "threadgroup must be a multiple of the SIMDgroup width (32) "
            "between 32 and 1024"
        )
    n, m = int(q.shape[0]), int(r.shape[0])
    lanes = _KNN_SIMDGROUP_WIDTH if strategy == "simdgroup" else 1
    grid_x = ((n * lanes + threadgroup - 1) // threadgroup) * threadgroup
    # The float32 value _hav uses for (2.0 * radius).
    rad2 = mx.array(np.array([2.0 * np.float32(radius)], dtype=np.float32))
    rv, rcos = _knn_unit_vectors(r)
    out_d, out_i = _get_knn_chord_kernel(strategy)(
        inputs=[
            q,
            _knn_unit_vectors(q)[0],
            r,
            rv,
            rcos,
            _knn_chord_slack(q, r),
            rad2,
            mx.array([n, m], dtype=mx.int32),
        ],
        template=[("K", k)],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n, k), (n, k)],
        output_dtypes=[mx.float32, mx.int32],
        stream=mx.gpu,
    )
    return out_d, out_i


# SIMDgroup query limits for the chord-filtered selectors, as (minimum m,
# limits by k). Pruning shortens the scans, so one thread per query fills the
# GPU from about half the query counts of _KNN_SIMDGROUP_QUERY_LIMITS.
# Measured on an M4 Max, like the routing rule in _knn_chord_route.
_KNN_CHORD_SIMDGROUP_QUERY_LIMITS = (
    (131_072, ((1, 16_384), (5, 8_192), (10, 8_192), (16, 4_096))),
    (4_096, ((1, 16_384), (5, 8_192), (10, 4_096), (16, 2_048))),
    (0, ((1, 8_192), (5, 4_096), (10, 4_096), (16, 4_096))),
)


def _knn_chord_strategy(n: int, m: int, k: int) -> str:
    """The faster chord-filtered selector for ``k <= 16``."""
    for min_m, limits in _KNN_CHORD_SIMDGROUP_QUERY_LIMITS:
        if m >= min_m:
            break
    for max_k, max_queries in limits:
        if k <= max_k:
            return "simdgroup" if n <= max_queries else "scalar"
    return "scalar"


def _knn_chord_route(n: int, m: int, k: int, strategy: str) -> bool:
    """Whether the chord-filtered ``strategy`` beats today's selectors.

    Over n in [256, 262144], m in [128, 1e6] and k in {1, 5, 10, 16} this rule
    is never more than 3% slower than the plain selectors and gains 1.8x in
    geometric mean. Beyond k = 16 the larger top-k loosens the threshold and
    the extra registers cost occupancy (0.7x at k = 32 with small m), and tiny
    shapes cannot repay the unit-vector and slack passes (~0.05 ms).
    """
    if k > _KNN_CHORD_MAX_K:
        return False
    if strategy == "scalar":
        # Large query batches gain from the fused refine alone.
        return n >= 16_384 or n * m >= 1 << 24
    # A lane scans m / 32 references; pruning pays on long scans only.
    return n * m >= 1 << 24 and (m >= 32_768 or (k <= 5 and m >= 8_192))


def _knn_metal_strategy(n: int, m: int, k: int) -> str:
    """Choose the faster exact Metal selector for an ``(n, m, k)`` shape."""
    # A SIMDgroup also wins launch-bound workloads, but multiplying every
    # query by 32 lanes stops paying once many short reference scans already
    # fill the GPU. Keep that small-M/high-N corner on the scalar selector.
    if n <= 1024:
        return "simdgroup"
    if m < 128:
        return "scalar"
    limits = (
        _KNN_SIMDGROUP_LARGE_REF_QUERY_LIMITS
        if m >= 65_536
        else _KNN_SIMDGROUP_QUERY_LIMITS
    )
    for max_k, max_queries in limits:
        if k <= max_k:
            return "simdgroup" if n <= max_queries else "scalar"
    return "scalar"


def haversine(a, b, *, radius: float = R_EARTH_M, degrees: bool = True) -> mx.array:
    """Elementwise (broadcasting) great-circle distance between ``a`` and ``b``."""
    radius = _check_radius(radius)
    a = _prep(a, degrees)
    b = _prep(b, degrees)
    if a.size == 0 or b.size == 0:
        # MLX's compiled graph can dispatch a zero-sized Metal grid for empty
        # broadcasts. Validate the coordinate axes before returning an empty
        # result, without slicing or evaluating either input's lazy graph.
        shape = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        return mx.zeros(shape, dtype=mx.float32)
    return _hav(a[..., 0], a[..., 1], b[..., 0], b[..., 1], radius)


def haversine_cross(
    a,
    b,
    *,
    radius: float = R_EARTH_M,
    degrees: bool = True,
    chunk_bytes: int | None = None,
) -> mx.array:
    """All-pairs distance matrix: ``(N, 2) x (M, 2) -> (N, M)``.

    The distance graph is compiled into one fused kernel, so by default the
    whole matrix is a single launch and the result is the only allocation
    (MLX handles results beyond 2^31 elements). Pass ``chunk_bytes`` to
    bound the rows per launch instead; the chunks are concatenated at the
    end, so peak memory then reaches about twice the result size.
    """
    radius = _check_radius(radius)
    if chunk_bytes is not None:
        chunk_bytes = check_chunk_bytes(chunk_bytes)
    a = _prep(a, degrees)
    b = _prep(b, degrees)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("haversine_cross expects 2-D inputs of shape (N, 2) and (M, 2)")
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        # Avoid a zero-work dispatch in the compiled distance graph. All
        # public argument and shape validation above still applies.
        return mx.zeros((n, m), dtype=mx.float32)
    lon2 = b[:, 0][None, :]
    lat2 = b[:, 1][None, :]
    if chunk_bytes is None:
        return _hav(a[:, 0:1], a[:, 1:2], lon2, lat2, radius)
    rows_per_chunk = max(1, chunk_bytes // max(m * 4 * 4, 1))
    if rows_per_chunk >= n:
        return _hav(a[:, 0:1], a[:, 1:2], lon2, lat2, radius)
    chunks = [
        _hav(a[start : start + rows_per_chunk, 0:1], a[start : start + rows_per_chunk, 1:2], lon2, lat2, radius)
        for start in range(0, n, rows_per_chunk)
    ]
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

    Exact brute force. On Metal with ``k <= 32`` this adaptively runs one of
    two fused selection kernels: a 32-lane SIMDgroup cooperates on each query
    when query parallelism is scarce, while one thread per query wins for
    large query batches. Neither materializes an ``(N, M)`` matrix or sorts
    the reference set. For ``k <= 16`` on large inputs both skip most exact
    distance evaluations with a conservative chord-distance bound; results
    are bit-identical either way. Otherwise a chunked distance-matrix +
    ``argpartition`` fallback runs; that fallback is sort-bound, so for large
    reference sets with ``k > 32`` a CPU tree (sklearn
    ``BallTree(metric="haversine")``) may win. ``chunk_bytes`` must be a
    positive integer even when the selected fused route does not consume it.
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
    # Validate before selecting a Metal or composed route. The fused kernels
    # do not need this budget, but accepting invalid values only on that route
    # would make the public API depend on hardware, k, and active MLX device.
    chunk_bytes = check_chunk_bytes(chunk_bytes)
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
    if (
        metal_available()
        and mx.default_device() == mx.gpu
        and k <= _KNN_KERNEL_MAX_K
    ):
        strategy = _knn_chord_strategy(n, m, k)
        if _knn_chord_route(n, m, k, strategy):
            d, idx = _knn_chord(q, r, k, radius, strategy)
        elif _knn_metal_strategy(n, m, k) == "simdgroup":
            d, idx = _knn_simdgroup(q, r, k, radius)
        else:
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
