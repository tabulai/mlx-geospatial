"""Point-in-polygon predicates (even-odd / crossing-number rule).

Four GPU paths share identical semantics:

- :func:`pip_pairs` — adaptive explicit-pair kernels. Ordinary saturated work
  uses one thread per pair; long rings use one SIMDgroup or four-SIMDgroup
  threadgroup per pair to distribute edges.
- :func:`pip_first_match` — adaptive point-centric kernels. Candidates remain
  sequential for minimum-index semantics, while long rings are distributed
  across 32 or 128 cooperating lanes.
- :func:`pip_all_matches` — an adaptive ordered GPU compactor. Ordinary
  candidate lists use point-centric count/scan/write passes; sparse or skewed
  lists use a pair-parallel flag/scan/write path so one long list cannot
  serialize a thread.
- :func:`pip_dense` — an ordinary 2-D kernel plus poly-major cooperative
  long-ring kernels, selected by shape. Used for dense ``(N, M)`` containment
  matrices.

The former composed-MLX dense implementation remains as a private fallback
for non-Metal systems and as an independent correctness oracle in tests.

Boundary semantics: the crossing-number test uses the half-open rule
``(y1 > py) != (y2 > py)`` and ``px < x_intersection`` to avoid double-counting
ray crossings at vertices. Within ``PolygonSet.boundary_tolerance`` of an
edge — including every exact boundary point — the fp32 result is undefined.
Consequently a point on a shared edge may match either adjacent polygon, both,
or neither; there is no general uniqueness guarantee inside the band.
"""

from __future__ import annotations

import mlx.core as mx

from ._dispatch import metal_available
from ._ingest import PolygonSet

__all__ = [
    "pip_pairs",
    "pip_first_match",
    "pip_all_matches",
    "pip_dense",
    "contains_matrix",
]

_PAIR_KERNEL_SOURCE = """
    // Keep every float op individually rounded: with contraction enabled the
    // Metal compiler fuses the x-intersection's multiply-add into an FMA,
    // while the composed dense oracle rounds each op separately. All paths
    // must agree bit-for-bit, including inside the tolerance band.
    #pragma clang fp contract(off)
    uint i = thread_position_in_grid.x;
    if (i >= (uint)n_pairs[0]) return;
    // Index math in 64-bit: 2 * pt overflows int32 from pt = 2^30, silently
    // reading out of bounds for point/vertex indices in [2^30, 2^31).
    long pt = pair_point[i];
    long pg = pair_poly[i];
    float px = points[2 * pt + 0];
    float py = points[2 * pt + 1];
    if (px < bboxes[4 * pg + 0] || py < bboxes[4 * pg + 1] ||
        px > bboxes[4 * pg + 2] || py > bboxes[4 * pg + 3]) {
        inside[i] = 0;
        return;
    }
    int crossings = 0;
    long r_begin = geom_ring_offsets[pg];
    long r_end = geom_ring_offsets[pg + 1];
    for (long r = r_begin; r < r_end; ++r) {
        long v_begin = ring_offsets[r];
        long v_end = ring_offsets[r + 1];
        for (long v = v_begin; v + 1 < v_end; ++v) {
            float y1 = coords[2 * v + 1];
            float y2 = coords[2 * v + 3];
            if ((y1 > py) != (y2 > py)) {
                float x1 = coords[2 * v + 0];
                float x2 = coords[2 * v + 2];
                // Divide before multiplying: the straddle test bounds the
                // ratio to [0, 1], so huge coordinates cannot overflow the
                // intermediate product to inf (which would misclassify
                // points far outside the tolerance band).
                float xint = x1 + (py - y1) / (y2 - y1) * (x2 - x1);
                if (px < xint) {
                    crossings += 1;
                }
            }
        }
    }
    inside[i] = (uint8_t)(crossings & 1);
"""

_PIP_SIMDGROUP_HEADER = """
    // Cooperative long-polygon predicate shared by every specialized path.
    // Edge order does not affect parity: each lane evaluates a disjoint
    // strided subset with scalar-identical arithmetic, then the SIMDgroup
    // XOR-reduces the 32 partial parities.
    #pragma clang fp contract(off)
    template <typename Coords, typename RingOffsets,
              typename GeomRingOffsets, typename Bboxes>
    METAL_FUNC uint mlx_spatial_pip_contains_simdgroup(
        float px, float py, long pg, uint lane,
        Coords coords, RingOffsets ring_offsets,
        GeomRingOffsets geom_ring_offsets, Bboxes bboxes) {
        uint parity = 0;
        bool in_bbox = !(
            px < bboxes[4 * pg + 0] || py < bboxes[4 * pg + 1] ||
            px > bboxes[4 * pg + 2] || py > bboxes[4 * pg + 3]);
        if (!in_bbox) return 0;
        long r_begin = geom_ring_offsets[pg];
        long r_end = geom_ring_offsets[pg + 1];
        for (long r = r_begin; r < r_end; ++r) {
            long v_begin = ring_offsets[r];
            long v_end = ring_offsets[r + 1];
            for (long v = v_begin + (long)lane;
                 v + 1 < v_end; v += 32) {
                float y1 = coords[2 * v + 1];
                float y2 = coords[2 * v + 3];
                if ((y1 > py) != (y2 > py)) {
                    float x1 = coords[2 * v + 0];
                    float x2 = coords[2 * v + 2];
                    float xint = x1 +
                        (py - y1) / (y2 - y1) * (x2 - x1);
                    parity ^= (uint)(px < xint);
                }
            }
        }
        for (ushort delta = 16; delta > 0; delta >>= 1) {
            parity ^= metal::simd_shuffle_down(parity, delta);
        }
        return metal::simd_broadcast(parity, 0) & 1;
    }
"""

_PIP_THREADGROUP_HEADER = """
    // Four SIMDgroups cooperatively evaluate one point/polygon predicate.
    // Lane zero receives the final parity in partial[0]. Callers that need
    // every lane to observe it add one more threadgroup barrier. Callers
    // that reuse `partial` for several predicates in a loop must also
    // barrier again after reading partial[0], before the next call: the next
    // predicate writes partial[0] before its own reduction barrier, so a
    // lagging SIMDgroup would otherwise read the next verdict as this one.
    #pragma clang fp contract(off)
    template <typename Coords, typename RingOffsets,
              typename GeomRingOffsets, typename Bboxes>
    METAL_FUNC void mlx_spatial_pip_contains_threadgroup(
        float px, float py, long pg, uint local, uint lane, uint simd,
        Coords coords, RingOffsets ring_offsets,
        GeomRingOffsets geom_ring_offsets, Bboxes bboxes,
        threadgroup uint* partial) {
        uint parity = 0;
        bool in_bbox = !(
            px < bboxes[4 * pg + 0] || py < bboxes[4 * pg + 1] ||
            px > bboxes[4 * pg + 2] || py > bboxes[4 * pg + 3]);
        if (!in_bbox) {
            if (local == 0) partial[0] = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return;
        }
        long r_begin = geom_ring_offsets[pg];
        long r_end = geom_ring_offsets[pg + 1];
        for (long r = r_begin; r < r_end; ++r) {
            long v_begin = ring_offsets[r];
            long v_end = ring_offsets[r + 1];
            for (long v = v_begin + (long)local;
                 v + 1 < v_end; v += 128) {
                float y1 = coords[2 * v + 1];
                float y2 = coords[2 * v + 3];
                if ((y1 > py) != (y2 > py)) {
                    float x1 = coords[2 * v + 0];
                    float x2 = coords[2 * v + 2];
                    float xint = x1 +
                        (py - y1) / (y2 - y1) * (x2 - x1);
                    parity ^= (uint)(px < xint);
                }
            }
        }
        for (ushort delta = 16; delta > 0; delta >>= 1) {
            parity ^= metal::simd_shuffle_down(parity, delta);
        }
        if (lane == 0) partial[simd] = parity;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            uint group_parity = lane < 4 ? partial[lane] : 0;
            for (ushort delta = 16; delta > 0; delta >>= 1) {
                group_parity ^= metal::simd_shuffle_down(group_parity, delta);
            }
            if (lane == 0) partial[0] = group_parity & 1;
        }
    }
"""

_PAIR_SIMDGROUP_KERNEL_SOURCE = """
    // One SIMDgroup cooperates on every long candidate polygon.
    uint tid = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint pair = tid / 32;
    int count = n_pairs[0];
    // pair changes only at SIMDgroup boundaries, so the tail return is
    // uniform and no live lane can be stranded in a shuffle reduction.
    if (pair >= (uint)count) return;

    long pt = pair_point[pair];
    long pg = pair_poly[pair];
    float px = points[2 * pt + 0];
    float py = points[2 * pt + 1];
    uint hit = mlx_spatial_pip_contains_simdgroup(
        px, py, pg, lane, coords, ring_offsets, geom_ring_offsets, bboxes);
    if (lane == 0) inside[pair] = (uint8_t)hit;
"""

_PAIR_THREADGROUP_KERNEL_SOURCE = """
    // Four SIMDgroups cooperate on one very long polygon. Each first lane
    // publishes its SIMDgroup parity, then SIMDgroup zero performs the final
    // XOR. A whole threadgroup maps to one pair, so every barrier is uniform.
    #pragma clang fp contract(off)
    uint pair = threadgroup_position_in_grid.x;
    uint local = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd = simdgroup_index_in_threadgroup;
    int count = n_pairs[0];
    if (pair >= (uint)count) return;

    long pt = pair_point[pair];
    long pg = pair_poly[pair];
    float px = points[2 * pt + 0];
    float py = points[2 * pt + 1];
    threadgroup uint partial[4];
    mlx_spatial_pip_contains_threadgroup(
        px, py, pg, local, lane, simd, coords, ring_offsets,
        geom_ring_offsets, bboxes, partial);
    if (local == 0) inside[pair] = (uint8_t)partial[0];
"""

_FIRST_MATCH_SIMDGROUP_KERNEL_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint pt = tid / 32;
    int n_points = shape[0];
    if (pt >= (uint)n_points) return;

    if (lane == 0) match[pt] = -1;
    int cell = point_cell[pt];
    if (cell < 0) return;
    float px = points[2 * (size_t)pt + 0];
    float py = points[2 * (size_t)pt + 1];
    long candidate_begin = cell_offsets[cell];
    long candidate_end = cell_offsets[cell + 1];
    for (long candidate = candidate_begin;
         candidate < candidate_end; ++candidate) {
        int pg = cell_poly[candidate];
        uint hit = mlx_spatial_pip_contains_simdgroup(
            px, py, pg, lane, coords, ring_offsets,
            geom_ring_offsets, bboxes);
        if (hit) {
            if (lane == 0) match[pt] = pg;
            return;
        }
    }
"""

_FIRST_MATCH_THREADGROUP_KERNEL_SOURCE = """
    uint pt = threadgroup_position_in_grid.x;
    uint local = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd = simdgroup_index_in_threadgroup;
    int n_points = shape[0];
    if (pt >= (uint)n_points) return;

    if (local == 0) match[pt] = -1;
    int cell = point_cell[pt];
    if (cell < 0) return;
    float px = points[2 * (size_t)pt + 0];
    float py = points[2 * (size_t)pt + 1];
    threadgroup uint partial[4];
    long candidate_begin = cell_offsets[cell];
    long candidate_end = cell_offsets[cell + 1];
    for (long candidate = candidate_begin;
         candidate < candidate_end; ++candidate) {
        int pg = cell_poly[candidate];
        mlx_spatial_pip_contains_threadgroup(
            px, py, (long)pg, local, lane, simd, coords, ring_offsets,
            geom_ring_offsets, bboxes, partial);
        // Unlike the pair kernel, every SIMDgroup must observe the reduced
        // predicate before it can uniformly continue or return.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint hit = partial[0];
        // The next candidate's predicate overwrites partial[0] (SIMDgroup 0
        // publishes its own parity there before the reduction barrier, and a
        // bbox miss writes 0 immediately). Without this second barrier a
        // lagging SIMDgroup could read that later value as this candidate's
        // verdict, return early, and leave its partial[simd] stale for every
        // remaining candidate: a write-after-read hazard on threadgroup
        // memory that produced false matches at low polygon indices.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (hit) {
            if (local == 0) match[pt] = pg;
            return;
        }
    }
"""

_DENSE_KERNEL_SOURCE = """
    // Match pip_pairs and the composed oracle operation-for-operation.
    #pragma clang fp contract(off)
    uint pt = thread_position_in_grid.x;
    uint pg = thread_position_in_grid.y;
    int n_points = shape[0];
    int n_geoms = shape[1];
    if (pt >= (uint)n_points || pg >= (uint)n_geoms) return;

    size_t out_idx = (size_t)pt * (size_t)n_geoms + (size_t)pg;
    float px = points[2 * (size_t)pt + 0];
    float py = points[2 * (size_t)pt + 1];
    if (px < bboxes[4 * (size_t)pg + 0] ||
        py < bboxes[4 * (size_t)pg + 1] ||
        px > bboxes[4 * (size_t)pg + 2] ||
        py > bboxes[4 * (size_t)pg + 3]) {
        inside[out_idx] = false;
        return;
    }

    int crossings = 0;
    long r_begin = geom_ring_offsets[pg];
    long r_end = geom_ring_offsets[pg + 1];
    for (long r = r_begin; r < r_end; ++r) {
        long v_begin = ring_offsets[r];
        long v_end = ring_offsets[r + 1];
        for (long v = v_begin; v + 1 < v_end; ++v) {
            float y1 = coords[2 * v + 1];
            float y2 = coords[2 * v + 3];
            if ((y1 > py) != (y2 > py)) {
                float x1 = coords[2 * v + 0];
                float x2 = coords[2 * v + 2];
                float xint = x1 + (py - y1) / (y2 - y1) * (x2 - x1);
                if (px < xint) {
                    crossings += 1;
                }
            }
        }
    }
    inside[out_idx] = (crossings & 1) != 0;
"""

_DENSE_SIMDGROUP_KERNEL_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    size_t pair = (size_t)tid / 32;
    int n_points = shape[0];
    int n_geoms = shape[1];
    size_t n_pairs = (size_t)n_points * (size_t)n_geoms;
    if (pair >= n_pairs) return;

    // Poly-major work order keeps adjacent cooperative groups on the same
    // long coordinate block; the required result remains row-major.
    size_t pg = pair / (size_t)n_points;
    size_t pt = pair - pg * (size_t)n_points;
    size_t out = pt * (size_t)n_geoms + pg;
    float px = points[2 * (size_t)pt + 0];
    float py = points[2 * (size_t)pt + 1];
    uint hit = mlx_spatial_pip_contains_simdgroup(
        px, py, (long)pg, lane, coords, ring_offsets,
        geom_ring_offsets, bboxes);
    if (lane == 0) inside[out] = hit != 0;
"""

_DENSE_THREADGROUP_KERNEL_SOURCE = """
    size_t pair = (size_t)threadgroup_position_in_grid.x;
    uint local = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd = simdgroup_index_in_threadgroup;
    int n_points = shape[0];
    int n_geoms = shape[1];
    size_t n_pairs = (size_t)n_points * (size_t)n_geoms;
    if (pair >= n_pairs) return;

    size_t pg = pair / (size_t)n_points;
    size_t pt = pair - pg * (size_t)n_points;
    size_t out = pt * (size_t)n_geoms + pg;
    float px = points[2 * (size_t)pt + 0];
    float py = points[2 * (size_t)pt + 1];
    threadgroup uint partial[4];
    mlx_spatial_pip_contains_threadgroup(
        px, py, (long)pg, local, lane, simd, coords, ring_offsets,
        geom_ring_offsets, bboxes, partial);
    if (local == 0) inside[out] = partial[0] != 0;
"""

_FIRST_MATCH_KERNEL_SOURCE = """
    // Match pip_pairs operation-for-operation, but traverse one grid cell's
    // sorted polygon list per point and stop at the minimum-index hit.
    #pragma clang fp contract(off)
    uint pt = thread_position_in_grid.x;
    int n_points = shape[0];
    if (pt >= (uint)n_points) return;

    match[pt] = -1;
    int cell = point_cell[pt];
    if (cell < 0) return;

    float px = points[2 * (size_t)pt + 0];
    float py = points[2 * (size_t)pt + 1];
    long candidate_begin = cell_offsets[cell];
    long candidate_end = cell_offsets[cell + 1];
    for (long candidate = candidate_begin; candidate < candidate_end; ++candidate) {
        int pg = cell_poly[candidate];
        if (px < bboxes[4 * (size_t)pg + 0] ||
            py < bboxes[4 * (size_t)pg + 1] ||
            px > bboxes[4 * (size_t)pg + 2] ||
            py > bboxes[4 * (size_t)pg + 3]) {
            continue;
        }

        int crossings = 0;
        long r_begin = geom_ring_offsets[pg];
        long r_end = geom_ring_offsets[pg + 1];
        for (long r = r_begin; r < r_end; ++r) {
            long v_begin = ring_offsets[r];
            long v_end = ring_offsets[r + 1];
            for (long v = v_begin; v + 1 < v_end; ++v) {
                float y1 = coords[2 * v + 1];
                float y2 = coords[2 * v + 3];
                if ((y1 > py) != (y2 > py)) {
                    float x1 = coords[2 * v + 0];
                    float x2 = coords[2 * v + 2];
                    float xint = x1 + (py - y1) / (y2 - y1) * (x2 - x1);
                    if (px < xint) {
                        crossings += 1;
                    }
                }
            }
        }
        if (crossings & 1) {
            match[pt] = pg;
            return;
        }
    }
"""

_ALL_MATCHES_HEADER = """
    // Keep the count and write passes operation-for-operation identical to
    // pip_pairs, including individually rounded x-intersection arithmetic.
    #pragma clang fp contract(off)
    template <typename Coords, typename RingOffsets,
              typename GeomRingOffsets, typename Bboxes>
    METAL_FUNC bool mlx_spatial_pip_contains(
        float px, float py, int pg,
        Coords coords, RingOffsets ring_offsets,
        GeomRingOffsets geom_ring_offsets, Bboxes bboxes) {
        if (px < bboxes[4 * (size_t)pg + 0] ||
            py < bboxes[4 * (size_t)pg + 1] ||
            px > bboxes[4 * (size_t)pg + 2] ||
            py > bboxes[4 * (size_t)pg + 3]) {
            return false;
        }

        int crossings = 0;
        long r_begin = geom_ring_offsets[pg];
        long r_end = geom_ring_offsets[pg + 1];
        for (long r = r_begin; r < r_end; ++r) {
            long v_begin = ring_offsets[r];
            long v_end = ring_offsets[r + 1];
            for (long v = v_begin; v + 1 < v_end; ++v) {
                float y1 = coords[2 * (size_t)v + 1];
                float y2 = coords[2 * (size_t)v + 3];
                if ((y1 > py) != (y2 > py)) {
                    float x1 = coords[2 * (size_t)v + 0];
                    float x2 = coords[2 * (size_t)v + 2];
                    float xint = x1 + (py - y1) / (y2 - y1) * (x2 - x1);
                    if (px < xint) {
                        crossings += 1;
                    }
                }
            }
        }
        return (crossings & 1) != 0;
    }
"""

_ALL_MATCHES_COUNT_SOURCE = """
    uint active = thread_position_in_grid.x;
    int n_active = shape[0];
    if (active >= (uint)n_active) return;

    int cell = point_cell[active];
    if (cell < 0) {
        match_count[active] = 0;
        return;
    }
    int pt = point_id[active];
    float px = points[2 * (size_t)pt + 0];
    float py = points[2 * (size_t)pt + 1];
    long point_begin = candidate_offsets[active];
    long candidate_begin = point_begin;
    long candidate_end = candidate_offsets[active + 1];
    if (candidate_begin < limits[0]) candidate_begin = limits[0];
    if (candidate_end > limits[1]) candidate_end = limits[1];
    long cell_begin = cell_offsets[cell];

    int count = 0;
    for (long candidate = candidate_begin;
         candidate < candidate_end; ++candidate) {
        int pg = cell_poly[cell_begin + candidate - point_begin];
        if (mlx_spatial_pip_contains(
                px, py, pg, coords, ring_offsets,
                geom_ring_offsets, bboxes)) {
            count += 1;
        }
    }
    match_count[active] = count;
"""

_ALL_MATCHES_WRITE_SOURCE = """
    uint active = thread_position_in_grid.x;
    int n_active = shape[0];
    if (active >= (uint)n_active) return;

    int cell = point_cell[active];
    if (cell < 0) return;
    int pt = point_id[active];
    float px = points[2 * (size_t)pt + 0];
    float py = points[2 * (size_t)pt + 1];
    long point_begin = candidate_offsets[active];
    long candidate_begin = point_begin;
    long candidate_end = candidate_offsets[active + 1];
    if (candidate_begin < limits[0]) candidate_begin = limits[0];
    if (candidate_end > limits[1]) candidate_end = limits[1];
    long cell_begin = cell_offsets[cell];
    int out = match_offsets[active];

    for (long candidate = candidate_begin;
         candidate < candidate_end; ++candidate) {
        int pg = cell_poly[cell_begin + candidate - point_begin];
        if (mlx_spatial_pip_contains(
                px, py, pg, coords, ring_offsets,
                geom_ring_offsets, bboxes)) {
            out_point[out] = (long)pt;
            out_poly[out] = (long)pg;
            out += 1;
        }
    }
"""

_ALL_MATCHES_PAIR_COUNT_SOURCE = """
    uint pair = thread_position_in_grid.x;
    int n_pairs = shape[0];
    int n_active = shape[1];
    if (pair >= (uint)n_pairs) return;

    long candidate = limits[0] + (long)pair;
    int lo = 0;
    int hi = n_active;
    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;
        if (candidate_offsets[mid] <= candidate) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    int pt = point_id[lo];
    int cell = point_cell[lo];
    long cell_candidate = cell_offsets[cell] +
        candidate - candidate_offsets[lo];
    int pg = cell_poly[cell_candidate];
    float px = points[2 * (size_t)pt + 0];
    float py = points[2 * (size_t)pt + 1];
    match_flag[pair] = (int)mlx_spatial_pip_contains(
        px, py, pg, coords, ring_offsets, geom_ring_offsets, bboxes);
"""

_ALL_MATCHES_PAIR_SIMDGROUP_COUNT_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint pair = tid / 32;
    int n_pairs = shape[0];
    int n_active = shape[1];
    if (pair >= (uint)n_pairs) return;

    long candidate = limits[0] + (long)pair;
    int lo = 0;
    if (lane == 0) {
        int hi = n_active;
        while (lo + 1 < hi) {
            int mid = lo + (hi - lo) / 2;
            if (candidate_offsets[mid] <= candidate) lo = mid;
            else hi = mid;
        }
    }
    lo = metal::simd_broadcast(lo, 0);
    int pt = point_id[lo];
    int cell = point_cell[lo];
    long cell_candidate = cell_offsets[cell] +
        candidate - candidate_offsets[lo];
    int pg = cell_poly[cell_candidate];
    float px = points[2 * (size_t)pt + 0];
    float py = points[2 * (size_t)pt + 1];
    uint hit = mlx_spatial_pip_contains_simdgroup(
        px, py, (long)pg, lane, coords, ring_offsets,
        geom_ring_offsets, bboxes);
    if (lane == 0) match_flag[pair] = (int)hit;
"""

_ALL_MATCHES_PAIR_THREADGROUP_COUNT_SOURCE = """
    uint pair = threadgroup_position_in_grid.x;
    uint local = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd = simdgroup_index_in_threadgroup;
    int n_pairs = shape[0];
    int n_active = shape[1];
    if (pair >= (uint)n_pairs) return;

    long candidate = limits[0] + (long)pair;
    int lo = 0;
    int hi = n_active;
    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;
        if (candidate_offsets[mid] <= candidate) lo = mid;
        else hi = mid;
    }
    int pt = point_id[lo];
    int cell = point_cell[lo];
    long cell_candidate = cell_offsets[cell] +
        candidate - candidate_offsets[lo];
    int pg = cell_poly[cell_candidate];
    float px = points[2 * (size_t)pt + 0];
    float py = points[2 * (size_t)pt + 1];
    threadgroup uint partial[4];
    mlx_spatial_pip_contains_threadgroup(
        px, py, (long)pg, local, lane, simd, coords, ring_offsets,
        geom_ring_offsets, bboxes, partial);
    if (local == 0) match_flag[pair] = (int)partial[0];
"""

_ALL_MATCHES_PAIR_WRITE_SOURCE = """
    uint pair = thread_position_in_grid.x;
    int n_pairs = shape[0];
    int n_active = shape[1];
    if (pair >= (uint)n_pairs || match_flag[pair] == 0) return;

    long candidate = limits[0] + (long)pair;
    int lo = 0;
    int hi = n_active;
    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;
        if (candidate_offsets[mid] <= candidate) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    int pt = point_id[lo];
    int cell = point_cell[lo];
    long cell_candidate = cell_offsets[cell] +
        candidate - candidate_offsets[lo];
    int pg = cell_poly[cell_candidate];
    int out = inclusive[pair] - 1;
    out_point[out] = (long)pt;
    out_poly[out] = (long)pg;
"""

_pair_kernel = None
_pair_simdgroup_kernel = None
_pair_threadgroup_kernel = None
_dense_kernel = None
_dense_simdgroup_kernel = None
_dense_threadgroup_kernel = None
_first_match_kernel = None
_first_match_simdgroup_kernel = None
_first_match_threadgroup_kernel = None
_all_matches_count_kernel = None
_all_matches_write_kernel = None
_all_matches_pair_count_kernel = None
_all_matches_pair_simdgroup_count_kernel = None
_all_matches_pair_threadgroup_count_kernel = None
_all_matches_pair_write_kernel = None

_ALL_MATCHES_PAIR_ACTIVE_THRESHOLD = 1024
_ALL_MATCHES_PAIR_POINT_THRESHOLD = 256
_ALL_MATCHES_LONG_POLYGON_THRESHOLD = 1024
_ALL_MATCHES_EDGE_WORK_THRESHOLD = 8192

_PIP_LONG_RING_THRESHOLD = 1024
_PIP_LONG_SIMDGROUP_MAX_WORK = 16_384
_PIP_LONG_THREADGROUP_MAX_WORK = 4096


def _pip_long_polygon_strategy(
    n_work: int,
    max_ring_edges: int,
    *,
    max_work: int = _PIP_LONG_SIMDGROUP_MAX_WORK,
) -> str:
    """Choose scalar, SIMDgroup, or threadgroup traversal for long rings."""
    n_work = int(n_work)
    max_ring_edges = int(max_ring_edges)
    if n_work <= 0 or max_ring_edges < _PIP_LONG_RING_THRESHOLD:
        return "scalar"

    # Four SIMDgroups amortize their barriers only while each work item has
    # enough edges and the launch does not create excessive lane pressure.
    threadgroup_limit = min(
        _PIP_LONG_THREADGROUP_MAX_WORK,
        max_ring_edges,
    )
    if n_work <= min(threadgroup_limit, max_work):
        return "threadgroup"

    # SIMD32 remains profitable at a much wider launch. Scale the crossover
    # with ring length, capped before 32x lane expansion hurts saturated work.
    simdgroup_limit = min(
        _PIP_LONG_SIMDGROUP_MAX_WORK,
        max(4096, max_ring_edges * 4),
        max_work,
    )
    if n_work <= simdgroup_limit:
        return "simdgroup"
    return "scalar"


def _all_matches_strategy(
    n_active: int,
    n_pairs: int,
    max_point_pairs: int,
    max_candidate_edges: int = 0,
    max_ring_edges: int = 0,
) -> str:
    """Choose point-centric or pair-parallel ordered GPU compaction."""
    if n_pairs == 0:
        return "point"
    # Candidate-parallel evaluates the PIP predicate once. Point-centric
    # count/write evaluates it twice and serializes every candidate list,
    # which is catastrophic when even one candidate polygon is expensive.
    if (
        max_ring_edges >= _ALL_MATCHES_LONG_POLYGON_THRESHOLD
        or max_candidate_edges >= _ALL_MATCHES_EDGE_WORK_THRESHOLD
    ):
        return "pair"
    if n_active < _ALL_MATCHES_PAIR_ACTIVE_THRESHOLD:
        return "pair"
    # The average is not a safe skew signal: one point can own nearly all
    # candidates while tens of thousands of other points own one each. Such
    # a chunk would leave the point-centric launch waiting on one serial
    # thread, whereas the pair path distributes that list across the grid.
    if max_point_pairs >= _ALL_MATCHES_PAIR_POINT_THRESHOLD:
        return "pair"
    return "point"


def _crossed_graph(px, py, x1, y1, x2, y2):
    straddles = (y1 > py) != (y2 > py)
    # Horizontal edges (y1 == y2) never straddle; the resulting inf/nan
    # x-intersections are discarded by the logical_and. Divide before
    # multiplying (identically to the pair kernel): the ratio is bounded
    # for straddling edges, so huge coordinates cannot overflow to inf.
    xint = x1 + (py - y1) / (y2 - y1) * (x2 - x1)
    return mx.logical_and(straddles, px < xint).astype(mx.int32)


# Fused: the unfused broadcast chain materializes ~6 (N, E) temporaries.
_crossed_fused = mx.compile(_crossed_graph)


def _get_pair_kernel():
    global _pair_kernel
    if _pair_kernel is None:
        _pair_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_pairs",
            input_names=[
                "points",
                "pair_point",
                "pair_poly",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "n_pairs",
            ],
            output_names=["inside"],
            source=_PAIR_KERNEL_SOURCE,
        )
    return _pair_kernel


def _get_pair_simdgroup_kernel():
    global _pair_simdgroup_kernel
    if _pair_simdgroup_kernel is None:
        _pair_simdgroup_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_pairs_simdgroup",
            input_names=[
                "points",
                "pair_point",
                "pair_poly",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "n_pairs",
            ],
            output_names=["inside"],
            header=_PIP_SIMDGROUP_HEADER,
            source=_PAIR_SIMDGROUP_KERNEL_SOURCE,
        )
    return _pair_simdgroup_kernel


def _get_pair_threadgroup_kernel():
    global _pair_threadgroup_kernel
    if _pair_threadgroup_kernel is None:
        _pair_threadgroup_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_pairs_threadgroup",
            input_names=[
                "points",
                "pair_point",
                "pair_poly",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "n_pairs",
            ],
            output_names=["inside"],
            header=_PIP_THREADGROUP_HEADER,
            source=_PAIR_THREADGROUP_KERNEL_SOURCE,
        )
    return _pair_threadgroup_kernel


def _pip_pairs_simdgroup(
    points_local: mx.array,
    polygons: PolygonSet,
    pair_point: mx.array,
    pair_poly: mx.array,
    *,
    threadgroup: int = 128,
) -> mx.array:
    """Cooperative long-polygon PIP: one SIMDgroup per candidate pair."""
    if threadgroup < 32 or threadgroup > 1024 or threadgroup % 32:
        raise ValueError("threadgroup must be a multiple of 32 in [32, 1024]")
    n_pairs = int(pair_point.shape[0])
    if n_pairs == 0:
        return mx.zeros((0,), dtype=mx.uint8)
    threads = n_pairs * 32
    grid_x = ((threads + threadgroup - 1) // threadgroup) * threadgroup
    kernel = _get_pair_simdgroup_kernel()
    (inside,) = kernel(
        inputs=[
            points_local,
            pair_point,
            pair_poly,
            polygons.coords,
            polygons.ring_offsets,
            polygons.geom_ring_offsets,
            polygons.bboxes_local,
            mx.array([n_pairs], dtype=mx.int32),
        ],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n_pairs,)],
        output_dtypes=[mx.uint8],
        stream=mx.gpu,
    )
    return inside


def _pip_pairs_threadgroup(
    points_local: mx.array,
    polygons: PolygonSet,
    pair_point: mx.array,
    pair_poly: mx.array,
) -> mx.array:
    """Cooperative very-long PIP: one 128-lane threadgroup per pair."""
    n_pairs = int(pair_point.shape[0])
    if n_pairs == 0:
        return mx.zeros((0,), dtype=mx.uint8)
    kernel = _get_pair_threadgroup_kernel()
    (inside,) = kernel(
        inputs=[
            points_local,
            pair_point,
            pair_poly,
            polygons.coords,
            polygons.ring_offsets,
            polygons.geom_ring_offsets,
            polygons.bboxes_local,
            mx.array([n_pairs], dtype=mx.int32),
        ],
        grid=(n_pairs * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n_pairs,)],
        output_dtypes=[mx.uint8],
        stream=mx.gpu,
    )
    return inside


def _get_first_match_simdgroup_kernel():
    global _first_match_simdgroup_kernel
    if _first_match_simdgroup_kernel is None:
        _first_match_simdgroup_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_first_match_simdgroup",
            input_names=[
                "points",
                "point_cell",
                "cell_offsets",
                "cell_poly",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "shape",
            ],
            output_names=["match"],
            header=_PIP_SIMDGROUP_HEADER,
            source=_FIRST_MATCH_SIMDGROUP_KERNEL_SOURCE,
        )
    return _first_match_simdgroup_kernel


def _get_first_match_threadgroup_kernel():
    global _first_match_threadgroup_kernel
    if _first_match_threadgroup_kernel is None:
        _first_match_threadgroup_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_first_match_threadgroup",
            input_names=[
                "points",
                "point_cell",
                "cell_offsets",
                "cell_poly",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "shape",
            ],
            output_names=["match"],
            header=_PIP_THREADGROUP_HEADER,
            source=_FIRST_MATCH_THREADGROUP_KERNEL_SOURCE,
        )
    return _first_match_threadgroup_kernel


def _pip_first_match_simdgroup(
    points_local: mx.array,
    polygons: PolygonSet,
    point_cell: mx.array,
    cell_offsets: mx.array,
    cell_poly: mx.array,
    *,
    threadgroup: int = 128,
) -> mx.array:
    """Cooperative long-polygon first match: one SIMDgroup per point."""
    if threadgroup < 32 or threadgroup > 1024 or threadgroup % 32:
        raise ValueError("threadgroup must be a multiple of 32 in [32, 1024]")
    n_points = int(points_local.shape[0])
    if n_points == 0:
        return mx.zeros((0,), dtype=mx.int32)
    threads = n_points * 32
    grid_x = ((threads + threadgroup - 1) // threadgroup) * threadgroup
    kernel = _get_first_match_simdgroup_kernel()
    (match,) = kernel(
        inputs=[
            points_local,
            point_cell,
            cell_offsets,
            cell_poly,
            polygons.coords,
            polygons.ring_offsets,
            polygons.geom_ring_offsets,
            polygons.bboxes_local,
            mx.array([n_points], dtype=mx.int32),
        ],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n_points,)],
        output_dtypes=[mx.int32],
        stream=mx.gpu,
    )
    return match


def _pip_first_match_threadgroup(
    points_local: mx.array,
    polygons: PolygonSet,
    point_cell: mx.array,
    cell_offsets: mx.array,
    cell_poly: mx.array,
) -> mx.array:
    """Cooperative very-long first match: one 128-lane group per point."""
    n_points = int(points_local.shape[0])
    if n_points == 0:
        return mx.zeros((0,), dtype=mx.int32)
    kernel = _get_first_match_threadgroup_kernel()
    (match,) = kernel(
        inputs=[
            points_local,
            point_cell,
            cell_offsets,
            cell_poly,
            polygons.coords,
            polygons.ring_offsets,
            polygons.geom_ring_offsets,
            polygons.bboxes_local,
            mx.array([n_points], dtype=mx.int32),
        ],
        grid=(n_points * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n_points,)],
        output_dtypes=[mx.int32],
        stream=mx.gpu,
    )
    return match


def _get_dense_kernel():
    global _dense_kernel
    if _dense_kernel is None:
        _dense_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_dense",
            input_names=[
                "points",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "shape",
            ],
            output_names=["inside"],
            source=_DENSE_KERNEL_SOURCE,
        )
    return _dense_kernel


def _get_dense_simdgroup_kernel():
    global _dense_simdgroup_kernel
    if _dense_simdgroup_kernel is None:
        _dense_simdgroup_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_dense_simdgroup",
            input_names=[
                "points",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "shape",
            ],
            output_names=["inside"],
            header=_PIP_SIMDGROUP_HEADER,
            source=_DENSE_SIMDGROUP_KERNEL_SOURCE,
        )
    return _dense_simdgroup_kernel


def _get_dense_threadgroup_kernel():
    global _dense_threadgroup_kernel
    if _dense_threadgroup_kernel is None:
        _dense_threadgroup_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_dense_threadgroup",
            input_names=[
                "points",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "shape",
            ],
            output_names=["inside"],
            header=_PIP_THREADGROUP_HEADER,
            source=_DENSE_THREADGROUP_KERNEL_SOURCE,
        )
    return _dense_threadgroup_kernel


def _get_first_match_kernel():
    global _first_match_kernel
    if _first_match_kernel is None:
        _first_match_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_first_match",
            input_names=[
                "points",
                "point_cell",
                "cell_offsets",
                "cell_poly",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "shape",
            ],
            output_names=["match"],
            source=_FIRST_MATCH_KERNEL_SOURCE,
        )
    return _first_match_kernel


def _get_all_matches_count_kernel():
    global _all_matches_count_kernel
    if _all_matches_count_kernel is None:
        _all_matches_count_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_all_matches_count",
            input_names=[
                "points",
                "point_id",
                "point_cell",
                "candidate_offsets",
                "cell_offsets",
                "cell_poly",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "limits",
                "shape",
            ],
            output_names=["match_count"],
            header=_ALL_MATCHES_HEADER,
            source=_ALL_MATCHES_COUNT_SOURCE,
        )
    return _all_matches_count_kernel


def _get_all_matches_write_kernel():
    global _all_matches_write_kernel
    if _all_matches_write_kernel is None:
        _all_matches_write_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_all_matches_write",
            input_names=[
                "points",
                "point_id",
                "point_cell",
                "candidate_offsets",
                "cell_offsets",
                "cell_poly",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "limits",
                "match_offsets",
                "shape",
            ],
            output_names=["out_point", "out_poly"],
            header=_ALL_MATCHES_HEADER,
            source=_ALL_MATCHES_WRITE_SOURCE,
        )
    return _all_matches_write_kernel


def _get_all_matches_pair_count_kernel():
    global _all_matches_pair_count_kernel
    if _all_matches_pair_count_kernel is None:
        _all_matches_pair_count_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_all_matches_pair_count",
            input_names=[
                "points",
                "point_id",
                "point_cell",
                "candidate_offsets",
                "cell_offsets",
                "cell_poly",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "limits",
                "shape",
            ],
            output_names=["match_flag"],
            header=_ALL_MATCHES_HEADER,
            source=_ALL_MATCHES_PAIR_COUNT_SOURCE,
        )
    return _all_matches_pair_count_kernel


def _get_all_matches_pair_simdgroup_count_kernel():
    global _all_matches_pair_simdgroup_count_kernel
    if _all_matches_pair_simdgroup_count_kernel is None:
        _all_matches_pair_simdgroup_count_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_all_matches_pair_simdgroup_count",
            input_names=[
                "points",
                "point_id",
                "point_cell",
                "candidate_offsets",
                "cell_offsets",
                "cell_poly",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "limits",
                "shape",
            ],
            output_names=["match_flag"],
            header=_PIP_SIMDGROUP_HEADER,
            source=_ALL_MATCHES_PAIR_SIMDGROUP_COUNT_SOURCE,
        )
    return _all_matches_pair_simdgroup_count_kernel


def _get_all_matches_pair_threadgroup_count_kernel():
    global _all_matches_pair_threadgroup_count_kernel
    if _all_matches_pair_threadgroup_count_kernel is None:
        _all_matches_pair_threadgroup_count_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_all_matches_pair_threadgroup_count",
            input_names=[
                "points",
                "point_id",
                "point_cell",
                "candidate_offsets",
                "cell_offsets",
                "cell_poly",
                "coords",
                "ring_offsets",
                "geom_ring_offsets",
                "bboxes",
                "limits",
                "shape",
            ],
            output_names=["match_flag"],
            header=_PIP_THREADGROUP_HEADER,
            source=_ALL_MATCHES_PAIR_THREADGROUP_COUNT_SOURCE,
        )
    return _all_matches_pair_threadgroup_count_kernel


def _get_all_matches_pair_write_kernel():
    global _all_matches_pair_write_kernel
    if _all_matches_pair_write_kernel is None:
        _all_matches_pair_write_kernel = mx.fast.metal_kernel(
            name="mlx_spatial_pip_all_matches_pair_write",
            input_names=[
                "point_id",
                "point_cell",
                "candidate_offsets",
                "cell_offsets",
                "cell_poly",
                "limits",
                "match_flag",
                "inclusive",
                "shape",
            ],
            output_names=["out_point", "out_poly"],
            source=_ALL_MATCHES_PAIR_WRITE_SOURCE,
        )
    return _all_matches_pair_write_kernel


def pip_pairs(
    points_local: mx.array,
    polygons: PolygonSet,
    pair_point: mx.array,
    pair_poly: mx.array,
    *,
    max_ring_edges: int | None = None,
    threadgroup: int = 256,
) -> mx.array:
    """Containment flags for explicit (point, polygon) candidate pairs.

    Parameters
    ----------
    points_local : mx.array (N, 2) float32
        Points already re-centered with ``polygons.origin``.
    pair_point, pair_poly : mx.array (P,) int32
        Candidate pair indices into ``points_local`` / ``polygons``.

    Returns
    -------
    mx.array (P,) uint8 — 1 where the point is inside the polygon.
    """
    n_pairs = int(pair_point.shape[0])
    if n_pairs == 0:
        return mx.zeros((0,), dtype=mx.uint8)
    if max_ring_edges is None:
        max_ring_edges = polygons.max_edges_per_ring
    strategy = _pip_long_polygon_strategy(n_pairs, max_ring_edges)
    if strategy == "threadgroup":
        return _pip_pairs_threadgroup(
            points_local, polygons, pair_point, pair_poly
        )
    if strategy == "simdgroup":
        return _pip_pairs_simdgroup(
            points_local, polygons, pair_point, pair_poly
        )
    kernel = _get_pair_kernel()
    grid_x = ((n_pairs + threadgroup - 1) // threadgroup) * threadgroup
    (inside,) = kernel(
        inputs=[
            points_local,
            pair_point,
            pair_poly,
            polygons.coords,
            polygons.ring_offsets,
            polygons.geom_ring_offsets,
            polygons.bboxes_local,
            mx.array([n_pairs], dtype=mx.int32),
        ],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n_pairs,)],
        output_dtypes=[mx.uint8],
        stream=mx.gpu,
    )
    return inside


def pip_first_match(
    points_local: mx.array,
    polygons: PolygonSet,
    point_cell: mx.array,
    cell_offsets: mx.array,
    cell_poly: mx.array,
    *,
    max_ring_edges: int | None = None,
    threadgroup: int = 256,
) -> mx.array:
    """First containing polygon for each point from grid-cell CSR candidates.

    ``point_cell`` contains one grid cell ID per point, or ``-1`` for points
    outside the grid. Each cell's polygon list must be sorted ascending; the
    kernel's early return then preserves the public minimum-index semantics.
    ``cell_offsets`` is int64 and ``cell_poly`` is int32.
    """
    n_points = int(points_local.shape[0])
    if n_points == 0:
        return mx.zeros((0,), dtype=mx.int32)
    if max_ring_edges is None:
        max_ring_edges = polygons.max_edges_per_ring
    strategy = _pip_long_polygon_strategy(
        n_points, max_ring_edges, max_work=4096
    )
    if strategy == "threadgroup":
        return _pip_first_match_threadgroup(
            points_local, polygons, point_cell, cell_offsets, cell_poly
        )
    if strategy == "simdgroup":
        return _pip_first_match_simdgroup(
            points_local, polygons, point_cell, cell_offsets, cell_poly
        )
    kernel = _get_first_match_kernel()
    threadgroup = min(threadgroup, n_points)
    (match,) = kernel(
        inputs=[
            points_local,
            point_cell,
            cell_offsets,
            cell_poly,
            polygons.coords,
            polygons.ring_offsets,
            polygons.geom_ring_offsets,
            polygons.bboxes_local,
            mx.array([n_points], dtype=mx.int32),
        ],
        grid=(n_points, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n_points,)],
        output_dtypes=[mx.int32],
        stream=mx.gpu,
    )
    return match


def pip_all_matches(
    points_local: mx.array,
    polygons: PolygonSet,
    point_id: mx.array,
    point_cell: mx.array,
    candidate_offsets: mx.array,
    cell_offsets: mx.array,
    cell_poly: mx.array,
    *,
    pair_begin: int,
    pair_end: int,
    max_point_pairs: int | None = None,
    max_ring_edges: int | None = None,
    max_candidate_edges: int | None = None,
    threadgroup: int = 256,
) -> tuple[mx.array, mx.array]:
    """Compact every CSR containment hit for one bounded candidate range.

    ``point_id`` and ``point_cell`` contain candidate-bearing points in
    ascending point order. ``candidate_offsets`` is their int64 prefix sum in
    the conceptual point-major candidate array. ``[pair_begin, pair_end)``
    may split the first or last point, which keeps the caller's candidate-work
    cap exact even when one grid cell alone exceeds it.

    The caller may supply the maximum number of candidates contributed by any
    point after clipping to this batch. It lets the adaptive implementation
    avoid serializing a highly skewed list in one point thread. If omitted,
    the whole batch is used as a conservative bound.

    ``max_ring_edges`` controls cooperative edge traversal. The maximum total
    candidate-edge work contributed by one point, ``max_candidate_edges``,
    lets expensive candidate lists use the pair-parallel one-predicate path
    rather than the point-centric two-predicate path.

    Both GPU strategies preserve point-major then polygon-major order without
    materializing explicit candidate-index arrays. Returned arrays use the
    public int64 pair-index dtype.
    """
    n_active = int(point_id.shape[0])
    pair_begin = int(pair_begin)
    pair_end = int(pair_end)
    if pair_begin < 0 or pair_end < pair_begin:
        raise ValueError("candidate range must satisfy 0 <= pair_begin <= pair_end")
    if pair_end - pair_begin >= 2**31:
        raise ValueError("one all-matches GPU batch must contain fewer than 2**31 pairs")
    if (
        int(point_cell.shape[0]) != n_active
        or int(candidate_offsets.shape[0]) != n_active + 1
    ):
        raise ValueError("point arrays and candidate_offsets have inconsistent shapes")
    if n_active == 0 or pair_begin == pair_end:
        return (
            mx.zeros((0,), dtype=mx.int64, stream=mx.gpu),
            mx.zeros((0,), dtype=mx.int64, stream=mx.gpu),
        )
    if not 1 <= threadgroup <= 1024:
        raise ValueError("threadgroup must be in [1, 1024]")

    n_pairs = pair_end - pair_begin
    if max_point_pairs is None:
        max_point_pairs = n_pairs
    max_point_pairs = int(max_point_pairs)
    if not 0 <= max_point_pairs <= n_pairs:
        raise ValueError("max_point_pairs must be in [0, pair_end - pair_begin]")
    if max_ring_edges is None:
        max_ring_edges = polygons.max_edges_per_ring
    if max_candidate_edges is None:
        max_candidate_edges = int(polygons._geom_edge_counts.sum())
    max_ring_edges = int(max_ring_edges)
    max_candidate_edges = int(max_candidate_edges)
    if max_ring_edges < 0 or max_candidate_edges < 0:
        raise ValueError("edge-count bounds must be non-negative")
    limits = mx.array([pair_begin, pair_end], dtype=mx.int64)
    common_inputs = [
        points_local,
        point_id,
        point_cell,
        candidate_offsets,
        cell_offsets,
        cell_poly,
        polygons.coords,
        polygons.ring_offsets,
        polygons.geom_ring_offsets,
        polygons.bboxes_local,
        limits,
    ]

    if (
        _all_matches_strategy(
            n_active,
            n_pairs,
            max_point_pairs,
            max_candidate_edges,
            max_ring_edges,
        )
        == "pair"
    ):
        shape = mx.array([n_pairs, n_active], dtype=mx.int32)
        pair_threadgroup = min(threadgroup, n_pairs)
        long_strategy = _pip_long_polygon_strategy(n_pairs, max_ring_edges)
        if long_strategy == "threadgroup":
            count_kernel = _get_all_matches_pair_threadgroup_count_kernel()
            count_grid = (n_pairs * 128, 1, 1)
            count_threadgroup = (128, 1, 1)
        elif long_strategy == "simdgroup":
            count_kernel = _get_all_matches_pair_simdgroup_count_kernel()
            count_threads = n_pairs * 32
            count_grid_x = ((count_threads + 127) // 128) * 128
            count_grid = (count_grid_x, 1, 1)
            count_threadgroup = (128, 1, 1)
        else:
            count_kernel = _get_all_matches_pair_count_kernel()
            count_grid = (n_pairs, 1, 1)
            count_threadgroup = (pair_threadgroup, 1, 1)
        (match_flag,) = count_kernel(
            inputs=[*common_inputs, shape],
            grid=count_grid,
            threadgroup=count_threadgroup,
            output_shapes=[(n_pairs,)],
            output_dtypes=[mx.int32],
            stream=mx.gpu,
        )
        inclusive = mx.cumsum(match_flag, stream=mx.gpu)
        n_matches = int(inclusive[-1].item())
        if n_matches == 0:
            return (
                mx.zeros((0,), dtype=mx.int64, stream=mx.gpu),
                mx.zeros((0,), dtype=mx.int64, stream=mx.gpu),
            )
        write_kernel = _get_all_matches_pair_write_kernel()
        out_point, out_poly = write_kernel(
            inputs=[
                point_id,
                point_cell,
                candidate_offsets,
                cell_offsets,
                cell_poly,
                limits,
                match_flag,
                inclusive,
                shape,
            ],
            grid=(n_pairs, 1, 1),
            threadgroup=(pair_threadgroup, 1, 1),
            output_shapes=[(n_matches,), (n_matches,)],
            output_dtypes=[mx.int64, mx.int64],
            stream=mx.gpu,
        )
        return out_point, out_poly

    shape = mx.array([n_active], dtype=mx.int32)
    point_threadgroup = min(threadgroup, n_active)
    count_kernel = _get_all_matches_count_kernel()
    (match_count,) = count_kernel(
        inputs=[*common_inputs, shape],
        grid=(n_active, 1, 1),
        threadgroup=(point_threadgroup, 1, 1),
        output_shapes=[(n_active,)],
        output_dtypes=[mx.int32],
        stream=mx.gpu,
    )
    inclusive = mx.cumsum(match_count, stream=mx.gpu)
    match_offsets = mx.concatenate(
        [mx.zeros((1,), dtype=mx.int32, stream=mx.gpu), inclusive],
        stream=mx.gpu,
    )
    n_matches = int(match_offsets[-1].item())
    if n_matches == 0:
        return (
            mx.zeros((0,), dtype=mx.int64, stream=mx.gpu),
            mx.zeros((0,), dtype=mx.int64, stream=mx.gpu),
        )

    write_kernel = _get_all_matches_write_kernel()
    out_point, out_poly = write_kernel(
        inputs=[*common_inputs, match_offsets, shape],
        grid=(n_active, 1, 1),
        threadgroup=(point_threadgroup, 1, 1),
        output_shapes=[(n_matches,), (n_matches,)],
        output_dtypes=[mx.int64, mx.int64],
        stream=mx.gpu,
    )
    return out_point, out_poly


def _pip_dense_composed(
    points_local: mx.array,
    polygons: PolygonSet,
    *,
    chunk_bytes: int = 256 * 1024 * 1024,
) -> mx.array:
    """Composed-MLX dense containment fallback and correctness oracle.

    Crossing counts per geometry are recovered from a cumulative sum over the
    geometry-grouped edge axis, differenced at geometry boundaries — no
    scatter needed. ``chunk_bytes`` bounds the per-chunk temporaries; the
    peak still includes the full ``(N, M)`` result plus its concatenation
    copy (~2x the result size).
    """
    edge_data = polygons.edges
    edges = edge_data.edges
    n_edges = int(edges.shape[0])
    n_points = int(points_local.shape[0])
    n_geoms = polygons.n_polygons
    if n_points == 0 or n_edges == 0:
        return mx.zeros((n_points, n_geoms), dtype=mx.bool_)

    x1 = edges[:, 0][None, :]
    y1 = edges[:, 1][None, :]
    x2 = edges[:, 2][None, :]
    y2 = edges[:, 3][None, :]

    # ~5 live (chunk, E) fp32/int32 temporaries at peak.
    rows_per_chunk = max(1, chunk_bytes // max(n_edges * 4 * 5, 1))
    out_chunks = []
    for start in range(0, n_points, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_points)
        px = points_local[start:stop, 0:1]
        py = points_local[start:stop, 1:2]
        crossed = _crossed_fused(px, py, x1, y1, x2, y2)
        csum = mx.cumsum(crossed, axis=1)
        padded = mx.concatenate(
            [mx.zeros((stop - start, 1), dtype=mx.int32), csum], axis=1
        )
        ends = mx.take(padded, edge_data.geom_offsets_mx, axis=1)
        counts = ends[:, 1:] - ends[:, :-1]
        inside = (counts % 2).astype(mx.bool_)
        mx.eval(inside)
        out_chunks.append(inside)
    if len(out_chunks) == 1:
        return out_chunks[0]
    return mx.concatenate(out_chunks, axis=0)


def _pip_dense_simdgroup(
    points_local: mx.array,
    polygons: PolygonSet,
    *,
    threadgroup: int = 128,
) -> mx.array:
    """Dense long-ring PIP with one SIMDgroup per matrix cell."""
    if threadgroup < 32 or threadgroup > 1024 or threadgroup % 32:
        raise ValueError("threadgroup must be a multiple of 32 in [32, 1024]")
    n_points = int(points_local.shape[0])
    n_geoms = polygons.n_polygons
    n_pairs = n_points * n_geoms
    threads = n_pairs * 32
    grid_x = ((threads + threadgroup - 1) // threadgroup) * threadgroup
    kernel = _get_dense_simdgroup_kernel()
    (inside,) = kernel(
        inputs=[
            points_local,
            polygons.coords,
            polygons.ring_offsets,
            polygons.geom_ring_offsets,
            polygons.bboxes_local,
            mx.array([n_points, n_geoms], dtype=mx.int32),
        ],
        grid=(grid_x, 1, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n_points, n_geoms)],
        output_dtypes=[mx.bool_],
        stream=mx.gpu,
    )
    return inside


def _pip_dense_threadgroup(
    points_local: mx.array, polygons: PolygonSet
) -> mx.array:
    """Dense very-long PIP with one 128-lane group per matrix cell."""
    n_points = int(points_local.shape[0])
    n_geoms = polygons.n_polygons
    n_pairs = n_points * n_geoms
    kernel = _get_dense_threadgroup_kernel()
    (inside,) = kernel(
        inputs=[
            points_local,
            polygons.coords,
            polygons.ring_offsets,
            polygons.geom_ring_offsets,
            polygons.bboxes_local,
            mx.array([n_points, n_geoms], dtype=mx.int32),
        ],
        grid=(n_pairs * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n_points, n_geoms)],
        output_dtypes=[mx.bool_],
        stream=mx.gpu,
    )
    return inside


def pip_dense(
    points_local: mx.array,
    polygons: PolygonSet,
    *,
    chunk_bytes: int = 256 * 1024 * 1024,
) -> mx.array:
    """Dense ``(N, M)`` bool containment matrix.

    On Metal, ordinary shapes use a 2-D custom kernel with one thread per
    ``(point, polygon)`` output. For long rings and bounded output counts, a
    poly-major kernel assigns either 32 or 128 lanes to each output and
    XOR-reduces their exact integer crossing parities. Widened geometry bboxes
    reject pairs before ring traversal, and every route writes the result
    directly without an ``(N, E)`` crossing matrix or cumulative sum.

    ``chunk_bytes`` is retained for API compatibility and controls the
    composed fallback on non-Metal systems. The Metal path has no dense
    intermediate to chunk; its only size-dependent allocation is the required
    ``(N, M)`` boolean result.
    """
    n_points = int(points_local.shape[0])
    n_geoms = polygons.n_polygons
    if n_points == 0 or polygons.n_vertices == 0:
        return mx.zeros((n_points, n_geoms), dtype=mx.bool_)
    # mx.fast.metal_kernel requires the active default device to be the GPU;
    # Metal can still be physically available while a caller has selected
    # mx.cpu for composed MLX execution.
    if not metal_available() or mx.default_device() != mx.gpu:
        return _pip_dense_composed(
            points_local, polygons, chunk_bytes=chunk_bytes
        )

    strategy = _pip_long_polygon_strategy(
        n_points * n_geoms, polygons.max_edges_per_ring
    )
    if strategy == "threadgroup":
        return _pip_dense_threadgroup(points_local, polygons)
    if strategy == "simdgroup":
        return _pip_dense_simdgroup(points_local, polygons)

    # MLX dispatches partial threadgroups, so the exact x extent avoids idle
    # tail lanes and cannot overflow by rounding an INT_MAX-sized axis up.
    threadgroup = min(256, n_points)
    kernel = _get_dense_kernel()
    (inside,) = kernel(
        inputs=[
            points_local,
            polygons.coords,
            polygons.ring_offsets,
            polygons.geom_ring_offsets,
            polygons.bboxes_local,
            mx.array([n_points, n_geoms], dtype=mx.int32),
        ],
        grid=(n_points, n_geoms, 1),
        threadgroup=(threadgroup, 1, 1),
        output_shapes=[(n_points, n_geoms)],
        output_dtypes=[mx.bool_],
    )
    return inside


def contains_matrix(points, polygons, *, chunk_bytes: int = 256 * 1024 * 1024) -> mx.array:
    """Public dense containment matrix: ``result[i, j]`` iff point i is in polygon j.

    Intended for small-to-moderate polygon counts (the ``N x M`` matrix is
    materialized). For large polygon sets use :func:`mlx_spatial.point_in_polygon`,
    which goes through the grid index.
    """
    polyset = polygons if isinstance(polygons, PolygonSet) else PolygonSet(polygons)
    _, points_local = polyset.recenter_points(points)
    return pip_dense(points_local, polyset, chunk_bytes=chunk_bytes)
