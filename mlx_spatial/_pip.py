"""Point-in-polygon predicates (even-odd / crossing-number rule).

Two GPU paths share identical semantics:

- :func:`pip_pairs` — a custom Metal kernel, one thread per (point, polygon)
  candidate pair, looping that polygon's rings serially. Used by the
  grid-indexed join, where candidates are ragged.
- :func:`pip_dense` — a composed-MLX dense ``(N, M)`` evaluation over the flat
  edge list. Used for small polygon sets and as an independent
  cross-validation oracle for the kernel in tests.

Boundary semantics: the crossing-number test with the half-open rule
``(y1 > py) != (y2 > py)`` and ``px < x_intersection`` classifies points on
edges to exactly one side; a point on the shared edge of two adjacent
polygons lands in exactly one of them. Within
``PolygonSet.boundary_tolerance`` of an edge the fp32 result is undefined —
matching the documented tolerance policy, not GEOS bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from ._ingest import PolygonSet

__all__ = ["pip_pairs", "pip_dense", "contains_matrix"]

_PAIR_KERNEL_SOURCE = """
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
                float xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1);
                if (px < xint) {
                    crossings += 1;
                }
            }
        }
    }
    inside[i] = (uint8_t)(crossings & 1);
"""

_pair_kernel = None


def _crossed_graph(px, py, x1, y1, x2, y2):
    straddles = (y1 > py) != (y2 > py)
    # Horizontal edges (y1 == y2) never straddle; the resulting inf/nan
    # x-intersections are discarded by the logical_and.
    xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
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


def pip_dense(
    points_local: mx.array,
    polygons: PolygonSet,
    *,
    chunk_bytes: int = 256 * 1024 * 1024,
) -> mx.array:
    """Dense ``(N, M)`` bool containment matrix via the flat edge list.

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


def contains_matrix(points, polygons, *, chunk_bytes: int = 256 * 1024 * 1024) -> mx.array:
    """Public dense containment matrix: ``result[i, j]`` iff point i is in polygon j.

    Intended for small-to-moderate polygon counts (the ``N x M`` matrix is
    materialized). For large polygon sets use :func:`mlx_spatial.point_in_polygon`,
    which goes through the grid index.
    """
    polyset = polygons if isinstance(polygons, PolygonSet) else PolygonSet(polygons)
    _, points_local = polyset.recenter_points(points)
    return pip_dense(points_local, polyset, chunk_bytes=chunk_bytes)
