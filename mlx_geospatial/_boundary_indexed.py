"""Sparse float64 boundary verification using an already-built edge index.

This preserves the edge-envelope and projection arithmetic of the original
STRtree oracle while avoiding a second index over every individual edge.
It is intentionally optional: dense work or uncertain numerical conditions
return ``None`` so the caller can retain the original bounded host search.
"""

from __future__ import annotations

import numpy as np

from ._edge_index import EdgeIndex


_MAX_POINTS = 32_768
_MAX_BOUND_COMPARISONS = 4_000_000
_MAX_EDGE_PAIRS = 2_000_000
_PAIR_CHUNK = 200_000


def indexed_near_boundary(index, pts64: np.ndarray, tol: float) -> np.ndarray | None:
    """Return exactly the host boundary-oracle rows, or ``None`` to fall back.

    Only an accepted cached y-band index is reused; this helper never builds
    one. All-ring polygon bounds shortlist point/polygon pairs, including
    invalid holes lying outside their shells. Each shared y band includes
    the edge's fp32 envelope padded by four tolerances, which covers the
    original float64 edge envelope under the numerical guards below.

    Before the projection test, each candidate pair must pass the original
    float64 edge envelope expanded by exactly ``tol``. Projection operations
    then follow ``GridIndex._near_boundary`` in the same order and dtype.
    """
    polygons = index.polygons
    cached = polygons._y_edge_index
    if not isinstance(cached, EdgeIndex):
        return None
    n = len(pts64)
    if n > _MAX_POINTS or n * polygons.n_polygons > _MAX_BOUND_COMPARISONS:
        return None
    tol = float(tol)
    if tol != polygons.boundary_tolerance:
        return None
    f32 = np.finfo(np.float32)
    if not np.isfinite(tol) or tol < 32.0 * f32.tiny or 4.0 * tol > f32.max:
        return None
    if polygons.max_abs_coord > f32.max / 16.0:
        return None
    bounds = polygons.bboxes
    finite_bounds = bounds[np.isfinite(bounds)]
    if finite_bounds.size:
        with np.errstate(over="ignore", invalid="ignore"):
            original_ulp = np.spacing(float(np.abs(finite_bounds).max()))
        if not np.isfinite(original_ulp) or original_ulp > tol / 16.0:
            return None
    rows = np.flatnonzero(np.isfinite(pts64).all(axis=1))
    if rows.size == 0:
        return np.empty(0, dtype=np.int64)

    cuts = np.asarray(cached.cuts)
    geom_bins = np.asarray(cached.geom_bins)
    offsets = np.asarray(cached.offsets)
    px, py = pts64[rows, 0], pts64[rows, 1]
    point_parts, bin_parts, count_parts = [], [], []
    total = 0
    for pg, box in enumerate(bounds):
        # Use bounds of every ragged ring, not a GEOS polygon envelope: an
        # invalid hole outside its shell still contributes oracle edges.
        picked = np.flatnonzero(
            (px >= box[0] - 2.0 * tol) & (py >= box[1] - 2.0 * tol)
            & (px <= box[2] + 2.0 * tol) & (py <= box[3] + 2.0 * tol)
        )
        if picked.size == 0:
            continue
        first, stop = int(geom_bins[pg]), int(geom_bins[pg + 1])
        cut_start = first - pg
        local_y = (py[picked] - polygons.origin[1]).astype(np.float32)
        bins = first + np.searchsorted(
            cuts[cut_start:cut_start + stop - first - 1], local_y, side="right"
        )
        counts = offsets[bins + 1].astype(np.int64) - offsets[bins]
        total += int(counts.sum())
        # Check before reading the large edge-reference array or allocating
        # any expanded point/edge arrays or the original float64 edge table.
        if total > _MAX_EDGE_PAIRS:
            return None
        nonempty = counts > 0
        if nonempty.any():
            point_parts.append(rows[picked[nonempty]])
            bin_parts.append(bins[nonempty])
            count_parts.append(counts[nonempty])
    if total == 0:
        return np.empty(0, dtype=np.int64)

    points = np.concatenate(point_parts)
    bins = np.concatenate(bin_parts)
    counts = np.concatenate(count_parts)
    starts = np.cumsum(counts) - counts
    reference = np.arange(total, dtype=np.int64)
    reference -= np.repeat(starts, counts)
    reference += np.repeat(offsets[bins], counts)
    vertices = np.asarray(cached.edge_ids)[reference]
    del reference
    pair_points = np.repeat(points, counts)
    # Dense edge rows exclude each ring's closing vertex. Empty rings can
    # repeat closing offsets, so only distinct nonnegative closers count.
    closers = polygons._ring_offsets[1:] - 1
    closers = np.unique(closers[closers >= 0])
    edge_rows = vertices.astype(np.int64) - np.searchsorted(closers, vertices, side="left")
    edges = polygons._edges_f64()
    tol2 = tol * tol
    near = []
    for start in range(0, total, _PAIR_CHUNK):
        stop = min(start + _PAIR_CHUNK, total)
        pi = pair_points[start:stop]
        e = edges[edge_rows[start:stop]]
        x, y = pts64[pi, 0], pts64[pi, 1]
        envelope = (
            (x >= np.minimum(e[:, 0], e[:, 2]) - tol)
            & (x <= np.maximum(e[:, 0], e[:, 2]) + tol)
            & (y >= np.minimum(e[:, 1], e[:, 3]) - tol)
            & (y <= np.maximum(e[:, 1], e[:, 3]) + tol)
        )
        if not envelope.any():
            continue
        pi, e, x, y = pi[envelope], e[envelope], x[envelope], y[envelope]
        dx, dy = e[:, 2] - e[:, 0], e[:, 3] - e[:, 1]
        l2 = dx * dx + dy * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((x - e[:, 0]) * dx + (y - e[:, 1]) * dy) / l2
        t = np.where(l2 > 0.0, np.clip(t, 0.0, 1.0), 0.0)
        qx = e[:, 0] + t * dx
        qy = e[:, 1] + t * dy
        d2 = (x - qx) ** 2 + (y - qy) ** 2
        hit = pi[d2 <= tol2]
        if hit.size:
            near.append(np.unique(hit))
    if not near:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(near))
