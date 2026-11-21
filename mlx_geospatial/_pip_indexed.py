"""Fused containment over y-band edge lists for very detailed polygons."""

from __future__ import annotations

import mlx.core as mx

from ._edge_index import EDGE_BIN_HEADER, cold_edge_index_eligible, get_edge_index

_CONTAINS_HEADER = EDGE_BIN_HEADER + """
    #pragma clang fp contract(off)
    template <typename Coords, typename Bboxes, typename Cuts,
              typename GeomBins, typename Offsets, typename EdgeIds>
    METAL_FUNC uint mlx_geospatial_pip_indexed(
        float px, float py, int pg, Coords coords, Bboxes bboxes,
        Cuts cuts, GeomBins geom_bins, Offsets offsets, EdgeIds edge_ids) {
        if (!isfinite(px) || !isfinite(py) ||
            px < bboxes[4 * (size_t)pg] || py < bboxes[4 * (size_t)pg + 1] ||
            px > bboxes[4 * (size_t)pg + 2] || py > bboxes[4 * (size_t)pg + 3]) return 0;
        int bin = mlx_geospatial_edge_bin(py, pg, cuts, geom_bins);
        uint parity = 0;
        for (int e = offsets[bin]; e < offsets[bin + 1]; ++e) {
            long v = edge_ids[e];
            float y1 = coords[2 * v + 1];
            float y2 = coords[2 * v + 3];
            if ((y1 > py) != (y2 > py)) {
                float x1 = coords[2 * v];
                float x2 = coords[2 * v + 2];
                float xint = x1 + (py - y1) / (y2 - y1) * (x2 - x1);
                parity ^= (uint)(px < xint);
            }
        }
        return parity;
    }
"""

_PAIR_SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)shape[0]) return;
    long pt = pair_point[i];
    int pg = pair_poly[i];
    result[i] = (uint8_t)mlx_geospatial_pip_indexed(
        points[2 * pt], points[2 * pt + 1], pg,
        coords, bboxes, cuts, geom_bins, offsets, edge_ids);
"""

_FIRST_SOURCE = """
    uint pt = thread_position_in_grid.x;
    if (pt >= (uint)shape[0]) return;
    result[pt] = -1;
    int cell = point_cell[pt];
    if (cell < 0) return;
    float px = points[2 * (size_t)pt];
    float py = points[2 * (size_t)pt + 1];
    for (long c = cell_offsets[cell]; c < cell_offsets[cell + 1]; ++c) {
        int pg = cell_poly[c];
        if (mlx_geospatial_pip_indexed(px, py, pg,
            coords, bboxes, cuts, geom_bins, offsets, edge_ids)) {
            result[pt] = pg;
            return;
        }
    }
"""

_ALL_SOURCE = """
    uint pair = thread_position_in_grid.x;
    if (pair >= (uint)shape[0]) return;
    long candidate = limits[0] + (long)pair;
    int lo = 0;
    int hi = shape[1];
    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;
        if (candidate_offsets[mid] <= candidate) lo = mid;
        else hi = mid;
    }
    int pt = point_id[lo];
    int cell = point_cell[lo];
    long c = cell_offsets[cell] + candidate - candidate_offsets[lo];
    int pg = cell_poly[c];
    result[pair] = (int)mlx_geospatial_pip_indexed(
        points[2 * (size_t)pt], points[2 * (size_t)pt + 1], pg,
        coords, bboxes, cuts, geom_bins, offsets, edge_ids);
"""

_KERNELS = {}
_SOURCES = {"pairs": _PAIR_SOURCE, "first": _FIRST_SOURCE, "all": _ALL_SOURCE}
_INPUTS = {
    "pairs": ["pair_point", "pair_poly"],
    "first": ["point_cell", "cell_offsets", "cell_poly"],
    "all": ["point_id", "point_cell", "candidate_offsets", "cell_offsets",
            "cell_poly", "limits"],
}


# The index covers the whole layer. Estimate the work avoided in the current
# query from its candidate edges, using a higher break-even when cooperative
# traversal is available. These ratios cover the M1 Ultra and M4 Max sweeps
# across dispersed zones, overlap, out-of-grid points and a long coastline.
_COLD_MIN_EDGE_WORK = 64_000_000
_COLD_FIRST_EDGES_MULTIPLIER = 6_000
_COLD_COOPERATIVE_EDGES_MULTIPLIER = 14_000
_COLD_PAIR_EDGES_MULTIPLIER = 4_800
# Repeated medium batches amortize the one-time edge-index build only when each
# batch is large enough for indexed traversal to beat cooperative SIMD32/TG128
# (~16k points on 2,048-vertex zones, ~8k on 4,096-vertex zones, ~256 on 150k).
_REPEAT_MIN_EDGE_WORK = 32_000_000
# Break-even cumulative point-edge work to repay edge-index construction (~40 ns
# per total edge vs ~2-4 ps saved per point-edge on Apple silicon).
_REPEAT_BUILD_EDGES_MULTIPLIER = 10_000
_REPEAT_BUILD_MIN_EDGE_WORK = 5_000_000_000


def indexed_pip(
    kind, points, polygons, extra_inputs, shape, *,
    threadgroup=256, candidate_edge_work=None, cooperative=False,
):
    """Return flags/first matches, or None to use ordinary edge traversal.

    Candidate batching stays with the caller. The all-matches flags feed the
    existing ordered scan/write path; no point/polygon pairs are added here.
    Small cold queries keep cooperative traversal to avoid index build cost;
    repeated medium batches accumulate work and build once amortized.
    """
    n = int(shape[0])
    if polygons._y_edge_index is False:
        if not cold_edge_index_eligible(polygons):
            return None
        # A pair-range batch may include only part of the first and last cell's
        # candidates. Cap the cell estimate by the number of pairs evaluated.
        work_cap = n * polygons.max_edges_per_polygon
        edge_work = work_cap if candidate_edge_work is None else candidate_edge_work
        if kind == "first":
            min_size = 16_384
            multiplier = (
                _COLD_COOPERATIVE_EDGES_MULTIPLIER if cooperative else _COLD_FIRST_EDGES_MULTIPLIER
            )
        else:
            min_size = 32_768
            multiplier = _COLD_PAIR_EDGES_MULTIPLIER
            edge_work = min(edge_work, work_cap)
        build_now = n >= min_size and edge_work >= max(
            _COLD_MIN_EDGE_WORK, multiplier * polygons._total_edges
        )
        if not build_now:
            # First-match can stop early. Preserve the per-batch work limit
            # while removing credit for points that never reach any polygon.
            repeat_work = min(edge_work, n * polygons.max_edges_per_ring)
            if repeat_work < _REPEAT_MIN_EDGE_WORK:
                return None
            polygons._y_edge_index_work += repeat_work
            threshold = max(
                _REPEAT_BUILD_MIN_EDGE_WORK,
                _REPEAT_BUILD_EDGES_MULTIPLIER * polygons._total_edges,
            )
            if polygons._y_edge_index_work < threshold:
                return None
    index = get_edge_index(polygons)
    if index is None:
        return None
    if kind not in _KERNELS:
        _KERNELS[kind] = mx.fast.metal_kernel(
            name=f"mlx_geospatial_indexed_{kind}",
            input_names=["points", *_INPUTS[kind], "coords", "bboxes", "cuts",
                         "geom_bins", "offsets", "edge_ids", "shape"],
            output_names=["result"],
            header=_CONTAINS_HEADER,
            source=_SOURCES[kind],
        )
    (result,) = _KERNELS[kind](
        inputs=[points, *extra_inputs, polygons.coords, polygons.bboxes_local,
                index.cuts, index.geom_bins, index.offsets, index.edge_ids,
                mx.array(shape, dtype=mx.int32)],
        grid=(n, 1, 1),
        threadgroup=(min(threadgroup, n), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.uint8 if kind == "pairs" else mx.int32],
        stream=mx.gpu,
    )
    return result
