"""Bounded y-band edge index shared by long-ring Metal predicates.

Band lookup compares coordinates to stored fp32 cuts, without arithmetic.
An edge is listed once in every band intersecting its padded y interval.
Thus every straddling edge is visited, independent of bin rounding, ring
orientation, or validity. Padding also supports exact-boundary broad phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

# A closed ring with 2,048 vertices has 2,047 edges. Detailed zone layers
# benefit from the same edge bands as large individual polygons.
_MIN_RING_EDGES = 2047
_MAX_BINS = 512
_MAX_REFERENCES = 8_000_000
_MAX_EXPANSION = 16


@dataclass(frozen=True)
class EdgeIndex:
    cuts: mx.array
    geom_bins: mx.array
    offsets: mx.array
    edge_ids: mx.array
    max_bin_edges: int


EDGE_BIN_HEADER = """
    template <typename Cuts, typename GeomBins>
    METAL_FUNC int mlx_geospatial_edge_bin(
        float py, int pg, Cuts cuts, GeomBins geom_bins) {
        int start = geom_bins[pg];
        int lo = 0;
        int hi = geom_bins[pg + 1] - start - 1;
        int cut_start = start - pg;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (py >= cuts[cut_start + mid]) lo = mid + 1;
            else hi = mid;
        }
        return start + lo;
    }
"""


def get_edge_index(polygons) -> EdgeIndex | None:
    """Lazily build an index, or retain traversal when expansion is excessive.

    The reference cap is checked before any repeated edge arrays are made.
    Ordinary short-ring workloads never construct this index. Both accepted
    and rejected builds are cached for the lifetime of the immutable layer.
    """
    if polygons.max_edges_per_ring < _MIN_RING_EDGES:
        return None
    if polygons._y_edge_index is not False:
        return polygons._y_edge_index
    polygons._y_edge_index = _build_edge_index(polygons)
    return polygons._y_edge_index


def _build_edge_index(polygons) -> EdgeIndex | None:
    # No need to duplicate all the geometry metadata for unusually large
    # layers with just one long ring. Existing traversal remains bounded.
    if polygons.n_polygons > 1024:
        return None
    # Every original edge needs at least one reference. Reject impossible
    # builds before reading coordinates or allocating per-vertex masks.
    if int(polygons._geom_edge_counts.sum()) > _MAX_REFERENCES:
        return None
    coords = np.asarray(polygons.coords)
    keep = np.ones(polygons.n_vertices, dtype=bool)
    ends = polygons._ring_offsets[1:] - 1
    keep[ends[ends >= 0]] = False
    vertices = np.flatnonzero(keep).astype(np.int32)
    total_edges = len(vertices)
    budget = min(_MAX_REFERENCES, _MAX_EXPANSION * total_edges)
    padding = 4.0 * float(polygons.boundary_tolerance)
    if not np.isfinite(padding):
        return None
    cuts_parts, bin_parts, vertex_parts = [], [], []
    geom_bins = [0]
    total_refs = 0
    for pg in range(polygons.n_polygons):
        v = vertices[polygons._geom_edge_offsets[pg]:polygons._geom_edge_offsets[pg + 1]]
        bins = max(1, min(_MAX_BINS, len(v) // 64))
        if len(v):
            y1 = coords[v, 1].astype(np.float64)
            y2 = coords[v + 1, 1].astype(np.float64)
            # Float64 construction avoids overflow in the span of finite
            # fp32 endpoints. Duplicate cuts are safe under upper_bound.
            cuts = np.linspace(min(y1.min(), y2.min()), max(y1.max(), y2.max()),
                               bins + 1)[1:-1].astype(np.float32)
            first = np.searchsorted(cuts, np.minimum(y1, y2) - padding, side="right")
            last = np.searchsorted(cuts, np.maximum(y1, y2) + padding, side="right")
            counts = last - first + 1
            refs = int(counts.sum())
            total_refs += refs
            if total_refs > budget:
                return None
            repeats = np.repeat(np.arange(len(v)), counts)
            starts = np.cumsum(counts) - counts
            bin_ids = (first[repeats] + np.arange(refs) - starts[repeats]
                       + geom_bins[-1]).astype(np.int32)
            bin_parts.append(bin_ids)
            vertex_parts.append(v[repeats])
        else:
            cuts = np.empty(0, dtype=np.float32)
        cuts_parts.append(cuts)
        geom_bins.append(geom_bins[-1] + bins)
    if not total_refs:
        return None
    bin_ids = np.concatenate(bin_parts)
    edge_ids = np.concatenate(vertex_parts)
    counts = np.bincount(bin_ids, minlength=geom_bins[-1])
    # A highly folded layer can still fit the memory cap yet provide no
    # meaningful pruning. Avoid introducing an expensive redundant index.
    if counts.max() >= total_edges // 2:
        return None
    order = np.argsort(bin_ids, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return EdgeIndex(
        cuts=mx.array(np.concatenate(cuts_parts)),
        geom_bins=mx.array(geom_bins, dtype=mx.int32),
        offsets=mx.array(offsets),
        edge_ids=mx.array(edge_ids[order]),
        max_bin_edges=int(counts.max()),
    )
