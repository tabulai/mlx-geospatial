"""Fused containment over y-band edge lists for very detailed polygons."""

from __future__ import annotations

import mlx.core as mx

from ._edge_index import EDGE_BIN_HEADER, get_edge_index

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


def indexed_pip(kind, points, polygons, extra_inputs, shape, *, threadgroup=256):
    """Return flags/first matches, or None to use ordinary edge traversal.

    Candidate batching stays with the caller. The all-matches flags feed the
    existing ordered scan/write path; no point/polygon pairs are added here.
    Small cold queries keep cooperative traversal to avoid index build cost.
    """
    n = int(shape[0])
    # Cold index construction can dominate small detailed-polygon queries.
    # First-match sees every point, including rows outside the grid; pair
    # routes have already compacted away zero-candidate rows.
    cold_minimum = 65_536 if kind == "first" else 32_768
    if n < cold_minimum and polygons._y_edge_index is False:
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
