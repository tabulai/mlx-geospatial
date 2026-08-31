"""Point-in-polygon predicates (even-odd / crossing-number rule).

Two GPU paths share identical semantics:

- :func:`pip_pairs` — a custom Metal kernel, one thread per (point, polygon)
  candidate pair, looping that polygon's rings serially. Used by the
  grid-indexed join, where candidates are ragged.
- :func:`pip_dense` — a custom 2-D Metal kernel, one thread per output cell,
  with adjacent lanes sharing a polygon while evaluating contiguous points.
  Used for dense ``(N, M)`` containment matrices.

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

__all__ = ["pip_pairs", "pip_dense", "contains_matrix"]

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

_pair_kernel = None
_dense_kernel = None


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


def pip_pairs(
    points_local: mx.array,
    polygons: PolygonSet,
    pair_point: mx.array,
    pair_poly: mx.array,
    *,
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
    )
    return inside


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


def pip_dense(
    points_local: mx.array,
    polygons: PolygonSet,
    *,
    chunk_bytes: int = 256 * 1024 * 1024,
) -> mx.array:
    """Dense ``(N, M)`` bool containment matrix.

    On Metal, a 2-D custom kernel evaluates one ``(point, polygon)`` output
    per thread. The x-axis is points and the y-axis is polygons, so adjacent
    lanes read contiguous points while sharing polygon metadata and edge
    addresses. Widened geometry bboxes reject most pairs before the ring
    traversal, and the result is written directly without an ``(N, E)``
    crossing matrix or cumulative sum.

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
