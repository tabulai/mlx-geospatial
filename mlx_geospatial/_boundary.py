"""Conservative Metal broad phase for exact host boundary refinement.

This does not decide whether a point is inside the exact tolerance band. It
only discards points that cannot be there; the original float64 host test
settles every retained row. Expanded segment envelopes and a conservatively
widened line strip avoid squared distances and endpoint projections.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


_MAX_RING_EDGES = 4096
_MAX_POINT_EDGE_WORK = 32_768
# The maximum per-cell work is a conservative bound, often much higher than
# the average. Small budgets turn detailed layers into hundreds of launches
# and host synchronizations; one billion bounds the work without that cost.
_MAX_BATCH_EDGE_WORK = 1_000_000_000
_MAX_BATCH_POINTS = 1_000_000
_candidate_kernel = None
_long_candidate_kernel = None

_CANDIDATE_SOURCE = """
    #pragma clang fp contract(off)
    uint pt = thread_position_in_grid.x;
    if (pt >= (uint)shape[0]) return;
    float px = points[2 * (size_t)pt];
    float py = points[2 * (size_t)pt + 1];
    int cell = point_cell[pt];
    // Outside rows and non-finite local coordinates are retained. The host
    // oracle excludes non-finite originals and catches outer-edge points.
    candidate[pt] = 1;
    if (cell < 0 || !isfinite(px) || !isfinite(py)) return;

    int ix = cell % shape[1];
    int iy = cell / shape[1];
    float gx = (px - grid[0]) * grid[2];
    float gy = (py - grid[1]) * grid[2];
    float fx = gx - (float)ix;
    float fy = gy - (float)iy;
    float cell_margin = grid[4];
    // Do not rely on the CSR padding for points near a cell line. These
    // rows also cover uncertainty in the fp32 cell-assignment arithmetic.
    if (!isfinite(gx) || !isfinite(gy) ||
        fx <= cell_margin || fx >= 1.0f - cell_margin ||
        fy <= cell_margin || fy >= 1.0f - cell_margin) return;

    float margin = grid[3];
    for (long c = cell_offsets[cell]; c < cell_offsets[cell + 1]; ++c) {
        int pg = cell_poly[c];
        for (long r = geom_ring_offsets[pg];
             r < geom_ring_offsets[pg + 1]; ++r) {
            for (long v = ring_offsets[r]; v + 1 < ring_offsets[r + 1]; ++v) {
                float x1 = coords[2 * (size_t)v];
                float y1 = coords[2 * (size_t)v + 1];
                float x2 = coords[2 * (size_t)v + 2];
                float y2 = coords[2 * (size_t)v + 3];
                if (px >= min(x1, x2) - margin &&
                    px <= max(x1, x2) + margin &&
                    py >= min(y1, y2) - margin &&
                    py <= max(y1, y2) + margin) {
                    float dx = x2 - x1;
                    float dy = y2 - y1;
                    float length = max(abs(dx), abs(dy));
                    // Tiny/degenerate edges retain the envelope result:
                    // quantization can otherwise dominate their direction.
                    if (length <= grid[5]) return;
                    float sx = dx / length;
                    float sy = dy / length;
                    float cross = (px - x1) * sy - (py - y1) * sx;
                    // This is an infinite-line strip, not a distance or an
                    // endpoint projection. Its generous width absorbs all
                    // fp32 direction, coordinate and arithmetic rounding.
                    if (!isfinite(cross) || abs(cross) <= grid[6]) return;
                }
            }
        }
    }
    candidate[pt] = 0;
"""

_LONG_CANDIDATE_SOURCE = """
    #pragma clang fp contract(off)
    uint pt = thread_position_in_grid.x;
    if (pt >= (uint)shape[0]) return;
    float px = points[2 * (size_t)pt];
    float py = points[2 * (size_t)pt + 1];
    candidate[pt] = 1;
    if (!isfinite(px) || !isfinite(py)) return;
    float margin = grid[3];
    int cell = point_cell[pt];
    if (cell < 0) {
        // A detailed polygon can occupy a small part of the query extent.
        // Outer-edge uncertainty is retained, but distant finite points
        // cannot reach any segment's expanded envelope.
        if (px < grid[0] - margin || py < grid[1] - margin ||
            px > grid[7] + margin || py > grid[8] + margin) candidate[pt] = 0;
        return;
    }
    int ix = cell % shape[1];
    int iy = cell / shape[1];
    float gx = (px - grid[0]) * grid[2];
    float gy = (py - grid[1]) * grid[2];
    float fx = gx - (float)ix;
    float fy = gy - (float)iy;
    float cell_margin = grid[4];
    if (!isfinite(gx) || !isfinite(gy) ||
        fx <= cell_margin || fx >= 1.0f - cell_margin ||
        fy <= cell_margin || fy >= 1.0f - cell_margin) return;

    for (long c = cell_offsets[cell]; c < cell_offsets[cell + 1]; ++c) {
        int pg = cell_poly[c];
        if (px < bboxes[4 * (size_t)pg] - margin ||
            py < bboxes[4 * (size_t)pg + 1] - margin ||
            px > bboxes[4 * (size_t)pg + 2] + margin ||
            py > bboxes[4 * (size_t)pg + 3] + margin) continue;
        int bin = mlx_geospatial_edge_bin(py, pg, cuts, geom_bins);
        for (long e = edge_offsets[bin]; e < edge_offsets[bin + 1]; ++e) {
            long v = edge_ids[e];
            float x1 = coords[2 * (size_t)v];
            float y1 = coords[2 * (size_t)v + 1];
            float x2 = coords[2 * (size_t)v + 2];
            float y2 = coords[2 * (size_t)v + 3];
            if (px >= min(x1, x2) - margin &&
                px <= max(x1, x2) + margin &&
                py >= min(y1, y2) - margin &&
                py <= max(y1, y2) + margin) {
                float dx = x2 - x1;
                float dy = y2 - y1;
                float length = max(abs(dx), abs(dy));
                if (length <= grid[5]) return;
                float sx = dx / length;
                float sy = dy / length;
                float cross = (px - x1) * sy - (py - y1) * sx;
                if (!isfinite(cross) || abs(cross) <= grid[6]) return;
            }
        }
    }
    candidate[pt] = 0;
"""


def _get_candidate_kernel():
    global _candidate_kernel
    if _candidate_kernel is None:
        _candidate_kernel = mx.fast.metal_kernel(
            name="mlx_geospatial_boundary_candidates",
            input_names=[
                "points", "point_cell", "cell_offsets", "cell_poly", "coords",
                "ring_offsets", "geom_ring_offsets", "grid", "shape",
            ],
            output_names=["candidate"],
            source=_CANDIDATE_SOURCE,
        )
    return _candidate_kernel


def _get_long_candidate_kernel():
    from ._edge_index import EDGE_BIN_HEADER

    global _long_candidate_kernel
    if _long_candidate_kernel is None:
        _long_candidate_kernel = mx.fast.metal_kernel(
            name="mlx_geospatial_long_boundary_candidates",
            input_names=[
                "points", "point_cell", "cell_offsets", "cell_poly", "coords",
                "bboxes", "cuts", "geom_bins", "edge_offsets", "edge_ids",
                "grid", "shape",
            ],
            output_names=["candidate"],
            header=EDGE_BIN_HEADER,
            source=_LONG_CANDIDATE_SOURCE,
        )
    return _long_candidate_kernel


def _long_boundary_candidate_rows(index, local, cell, *, max_pairs_per_batch, tol):
    """Reuse the containment edge bands for a detailed polygon's boundary.

    Band insertion includes each edge's fp32 y envelope expanded by four
    tolerances. Band assignment is an upper-bound search against the exact
    same fp32 cutoffs, so it introduces no coordinate arithmetic error and
    covers every segment envelope used by this two-tolerance filter.
    """
    from ._edge_index import get_edge_index

    if int(local.shape[0]) < 65_536 and index.polygons._y_edge_index is False:
        return None
    edge_index = get_edge_index(index.polygons)
    if edge_index is None:
        return None
    max_edges = index._max_cell_count * edge_index.max_bin_edges
    if max_edges > _MAX_POINT_EDGE_WORK:
        return None
    x0, y0, x1, y1, inv_cell = index._host_grid()
    cell_margin = 4.0 * tol * float(inv_cell)
    if not np.isfinite(cell_margin) or not 0.0 < cell_margin < 0.125:
        return None
    if not np.isfinite([x0, y0, x1, y1, inv_cell]).all():
        return None
    cell_offsets, cell_poly = index._device_csr()
    if int(cell_poly.shape[0]) == 0:
        return None
    grid = mx.array(np.asarray(
        [x0, y0, inv_cell, 2.0 * tol, cell_margin, 32.0 * tol, 8.0 * tol, x1, y1],
        dtype=np.float32,
    ))
    step = min(
        _MAX_BATCH_POINTS,
        max_pairs_per_batch // max(1, index._max_cell_count),
        _MAX_BATCH_EDGE_WORK // max(1, max_edges),
    )
    kernel = _get_long_candidate_kernel()
    polygons = index.polygons
    n = int(local.shape[0])
    rows = []
    for start in range(0, n, step):
        stop = min(start + step, n)
        size = stop - start
        shape = mx.array([size, index.nx], dtype=mx.int32)
        with mx.stream(mx.gpu):
            (candidate,) = kernel(
                inputs=[
                    local[start:stop], cell[start:stop], cell_offsets, cell_poly,
                    polygons.coords, polygons.bboxes_local, edge_index.cuts,
                    edge_index.geom_bins, edge_index.offsets, edge_index.edge_ids,
                    grid, shape,
                ],
                grid=(size, 1, 1),
                threadgroup=(min(size, 256), 1, 1),
                output_shapes=[(size,)],
                output_dtypes=[mx.uint8],
                stream=mx.gpu,
            )
        batch_rows = np.flatnonzero(np.asarray(candidate))
        if batch_rows.size:
            rows.append(batch_rows + start)
    if not rows:
        return np.empty(0, dtype=np.int64)
    return rows[0] if len(rows) == 1 else np.concatenate(rows)


def boundary_candidate_rows(
    index,
    local: mx.array,
    cell: mx.array,
    *,
    max_pairs_per_batch: int = 16_000_000,
) -> np.ndarray | None:
    """Return a conservative subset of rows, or ``None`` for the host route.

    The caller supplies the already-computed local points and cell IDs. A
    point within ``tol`` of an original segment is inside its axis-aligned
    envelope expanded by ``tol``. Rounding either endpoint and the nearby
    point contributes at most a few ulps of the polygon extent; ``2 * tol``
    provides 16 ulps, twice the documented 8-ulp band. A second filter uses
    the segment's infinite line with direction normalized by its largest
    component. A true band point has cross product at most ``sqrt(2)*tol``.
    Its strip width is ``8*tol`` (64 global ulps), allowing coordinate,
    direction and arithmetic rounding. Edges shorter than ``32*tol`` keep
    the envelope result, avoiding ill-conditioned directions; the envelope
    bounds the other point-edge offsets by the edge length plus padding.
    Large coordinates and underflow-sensitive scales use the host route.

    Grid-border rows are retained with a four-tolerance margin. Away from
    this band, a segment within one tolerance must intersect the point's
    cell even accounting for the cell graph's under-eight-ulp error; its
    polygon is therefore in the CSR. The extra padding already present in
    that CSR is not needed for this argument. Very fine grids, where the
    border margin would cover much of a cell, keep the host route.

    Short rings traverse their polygon candidates directly; long rings reuse
    the containment index's padded y bands. Both routes limit aggregate
    per-point edge work. Whole-point batches respect the caller's candidate
    pair cap and an additional edge-work budget. If one point's conservative
    bound exceeds the cap, refinement retains the host route.
    """
    polygons = index.polygons
    n = int(local.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.int64)
    max_count = index._max_cell_count
    max_edges = index._max_cell_edge_work
    if max_count > max_pairs_per_batch:
        return None
    tol = float(polygons.boundary_tolerance)
    f32 = np.finfo(np.float32)
    if not np.isfinite(tol) or tol < 32.0 * f32.tiny or 4.0 * tol > f32.max:
        return None
    # Keep edge differences, normalized-cross products and their difference
    # far below overflow. This also bounds any local coordinate rounding.
    if polygons.max_abs_coord > f32.max / 16.0:
        return None
    # The original host oracle forms segment projections in original f64
    # coordinates. At a huge offset with a very small local extent, adding
    # a projection back to its endpoint can round onto an unrelated point.
    # Preserve that oracle (including invalid-geometry parity semantics)
    # rather than filtering its numerical false positives on the GPU.
    finite_bounds = polygons.bboxes[np.isfinite(polygons.bboxes)]
    if finite_bounds.size:
        original_scale = float(np.abs(finite_bounds).max())
        with np.errstate(over="ignore", invalid="ignore"):
            original_ulp = np.spacing(original_scale)
        if not np.isfinite(original_ulp) or original_ulp > tol / 16.0:
            return None
    from ._edge_index import _MIN_RING_EDGES

    if polygons.max_edges_per_ring >= _MIN_RING_EDGES:
        indexed = _long_boundary_candidate_rows(
            index, local, cell, max_pairs_per_batch=max_pairs_per_batch, tol=tol
        )
        if indexed is not None:
            return indexed
    if polygons.max_edges_per_ring > _MAX_RING_EDGES:
        return None
    if max_edges > _MAX_POINT_EDGE_WORK:
        return None
    x0, y0, _, _, inv_cell = index._host_grid()
    cell_margin = 4.0 * tol * float(inv_cell)
    if not np.isfinite(cell_margin) or not 0.0 < cell_margin < 0.125:
        return None
    if not np.isfinite([x0, y0, inv_cell]).all():
        return None
    cell_offsets, cell_poly = index._device_csr()
    if int(cell_poly.shape[0]) == 0:
        return None
    grid = mx.array(np.asarray(
        [x0, y0, inv_cell, 2.0 * tol, cell_margin, 32.0 * tol, 8.0 * tol],
        dtype=np.float32,
    ))
    step = min(
        _MAX_BATCH_POINTS,
        max_pairs_per_batch // max(1, max_count),
        _MAX_BATCH_EDGE_WORK // max(1, max_edges),
    )
    kernel = _get_candidate_kernel()
    rows = []
    for start in range(0, n, step):
        stop = min(start + step, n)
        size = stop - start
        shape = mx.array([size, index.nx], dtype=mx.int32)
        with mx.stream(mx.gpu):
            (candidate,) = kernel(
                inputs=[
                    local[start:stop], cell[start:stop], cell_offsets, cell_poly,
                    polygons.coords, polygons.ring_offsets,
                    polygons.geom_ring_offsets, grid, shape,
                ],
                grid=(size, 1, 1),
                threadgroup=(min(size, 256), 1, 1),
                output_shapes=[(size,)],
                output_dtypes=[mx.uint8],
                stream=mx.gpu,
            )
        batch_rows = np.flatnonzero(np.asarray(candidate))
        if batch_rows.size:
            rows.append(batch_rows + start)
    if not rows:
        return np.empty(0, dtype=np.int64)
    return rows[0] if len(rows) == 1 else np.concatenate(rows)
