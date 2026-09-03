"""Grid-indexed point-to-polygon spatial join.

The index is a uniform grid over the polygon set's bounds. Each cell holds
the (sorted) list of polygons whose bbox intersects it, in CSR form. A query
point maps to exactly one cell; its candidate polygons are that cell's list,
so the first-match Metal kernel walks that list directly and returns on its
first containment hit. All-match queries count, prefix-scan, and compact hits
directly on the GPU without explicit candidate-pair arrays.

Cell assignment runs on the GPU in float32 local coordinates through one
compiled graph. Polygon cell ranges are planned in float64 and padded by more
than the fp32 assignment error plus the kernel's bbox widening, so every
polygon whose kernel bbox accepts a point is in that point's cell list: cell
assignment can only add candidates, never lose a hit, and the answer is
determined by the kernel alone. Device-resident inputs (``mx.array``,
:class:`~mlx_geospatial.LocalPoints`) therefore never touch the host, and the
output container follows the input container.

One containment contract: the fp32 even-odd rule in re-centered coordinates,
on every backend. The host fallback assigns cells with the device graph's fp32
arithmetic and evaluates the same predicate in NumPy with the kernels'
operation order, so CPU and GPU results are bit-identical for every input
(valid geometry or not; the one exception is fp32 subnormals, below ~1e-38,
which Metal flushes to zero and NumPy keeps: a coordinate, an edge delta or an
intermediate of the crossing test that small can differ, which needs local
coordinates themselves near 1e-38 and never occurs for real-world units) and
auto dispatch can never change an answer. GEOS enters only through the
opt-in ``exact_boundary=True`` re-check of near-boundary points. Small
host-resident workloads take the fallback: ``mlx_geospatial._dispatch`` compares
its cost (per point plus per candidate edge, estimated from the index's
per-cell edge work) with the fixed GPU launch cost. Device-resident inputs run
on the GPU whenever Metal is available.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import shapely

from ._dispatch import resolve_backend
from ._ingest import (
    LocalPoints,
    PolygonSet,
    _as_geometry_array,
    _crs_of,
    as_point_array,
)
from ._precision import recenter
from ._pip import (
    _ALL_MATCHES_EDGE_WORK_THRESHOLD,
    _PIP_LONG_RING_THRESHOLD,
    pip_all_matches,
    pip_first_match,
    pip_pairs,
)
from ._validation import check_positive_int

__all__ = ["GridIndex", "point_in_polygon", "sjoin"]

_INT32_MAX = np.iinfo(np.int32).max
_UINT32_MASK = np.uint64(np.iinfo(np.uint32).max)

# Direct polygon-order filling avoids sorting entirely, but its Python loop and
# two grid-wide cumulative sums are worthwhile only for material, dense
# rectangle expansion. Low-coverage grids retain NumPy's fast stable integer
# sort; moderately expanded grids use the lower-memory packed-key builder.
_GRID_DIRECT_MIN_ENTRIES = 65_536
_GRID_DIRECT_MIN_CELLS_PER_POLYGON = 256
_GRID_PACKED_MIN_ENTRIES = 65_536
_GRID_PACKED_MIN_CELLS_PER_POLYGON = 8

# Default per-launch candidate-pair caps. The fused first-match kernel
# allocates nothing per pair, so its cap only bounds launch duration (~70-110
# ms for 256M pairs on an M4 Max, depending on candidates per point);
# all-match compaction scans per pair, so its
# batches stay small enough to keep those buffers around 100 MB.
_FIRST_MATCH_MAX_PAIRS_PER_BATCH = 256_000_000
_ALL_MATCHES_MAX_PAIRS_PER_BATCH = 16_000_000

# The default grid refines below the median bbox side only while the padded
# CSR stays this dense: uniform zone layers need ~24 entries per polygon at a
# quarter of the median and edge-aligned tilings 36, whereas a few large
# polygons among many small ones would need hundreds and inflate the index
# 10x or more, so those layers back off to a half, then the median.
_GRID_REFINE_MAX_ENTRIES_PER_POLYGON = 40

# Host fallback chunking. Pair-level arrays cost ~100 bytes per candidate pair
# and the (pair, edge) crossing rows ~50 bytes each, so points are processed
# in slices holding at most _CPU_PAIRS_PER_CHUNK candidate pairs and each
# slice's crossing rows _CPU_ROWS_PER_CHUNK at a time: peak memory stays
# around 250-300 MB whatever the point count (the fallback is the only route
# on Metal-less hosts). Chunking changes nothing in the arithmetic.
_CPU_PAIRS_PER_CHUNK = 1_000_000
_CPU_ROWS_PER_CHUNK = 2_000_000
# exact_boundary's near-boundary search: points per STRtree query, and the
# candidate (point, edge) pair count above which a slice is retried shorter
# (the tree output costs 16 bytes per pair; the distance test itself runs over
# _CPU_ROWS_PER_CHUNK pairs at a time).
_NEAR_BOUNDARY_POINTS_PER_SLICE = 1_000_000
_NEAR_BOUNDARY_PAIRS_PER_SLICE = 8_000_000
_NEAR_BOUNDARY_PROBE_POINTS = 4096


def _cpu_crossed_rows(pxr: np.ndarray, pyr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The kernel's fp32 crossing predicate for one bounded edge-row batch."""
    x1, y1, x2, y2 = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]
    straddles = (y1 > pyr) != (y2 > pyr)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x1 + (pyr - y1) / (y2 - y1) * (x2 - x1)
    return straddles & (pxr < xint)


def _plan_cells(cell_size, fb, x0, y0, width, height, pad, max_cells, max_entries):
    """Coarsen ``cell_size`` until grid and padded CSR fit the budgets.

    Returns ``(cell_size, nx, ny, ix0, iy0, ix1, iy1, cells_per_poly, total)``
    or ``None`` when no doubling within 64 steps fits.
    """
    for _ in range(64):
        nx = max(1, int(np.ceil(width / cell_size)) if width > 0 else 1)
        ny = max(1, int(np.ceil(height / cell_size)) if height > 0 else 1)
        # Cell ids are part of the public/internal int32 query contract.
        # Coarsen even if a caller supplies a larger max_cells budget.
        if nx * ny > max_cells or nx * ny > _INT32_MAX:
            cell_size *= 2.0
            continue
        ix0 = np.clip(np.floor((fb[:, 0] - pad - x0) / cell_size), 0, nx - 1).astype(np.int64)
        iy0 = np.clip(np.floor((fb[:, 1] - pad - y0) / cell_size), 0, ny - 1).astype(np.int64)
        ix1 = np.clip(np.floor((fb[:, 2] + pad - x0) / cell_size), 0, nx - 1).astype(np.int64)
        iy1 = np.clip(np.floor((fb[:, 3] + pad - y0) / cell_size), 0, ny - 1).astype(np.int64)
        cells_per_poly = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
        total = int(cells_per_poly.sum())
        if total > max_entries:
            cell_size *= 2.0
            continue
        return cell_size, nx, ny, ix0, iy0, ix1, iy1, cells_per_poly, total
    return None


def _cell_graph(x, y, x0, y0, x1, y1, inv_cell, nx, nx_m1, ny_m1, cell_offsets):
    """fp32 cell assignment for local-frame points (see GridIndex docstring).

    Every scalar arrives as a 0-d array so one compiled Metal kernel serves
    every index; ``x``/``y`` are split by the caller because a shapeless
    compile cannot slice inside the graph. NaN/inf coordinates fail the
    in-grid test and get cell ``-1`` with zero candidates.
    """
    in_grid = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    ix = mx.clip(mx.floor((x - x0) * inv_cell), 0, nx_m1).astype(mx.int32)
    iy = mx.clip(mx.floor((y - y0) * inv_cell), 0, ny_m1).astype(mx.int32)
    cell = mx.where(in_grid, iy * nx + ix, -1)
    safe = mx.where(in_grid, cell, 0)
    counts = mx.where(
        in_grid, mx.take(cell_offsets, safe + 1) - mx.take(cell_offsets, safe), 0
    )
    return cell, counts


_device_cells = mx.compile(_cell_graph, shapeless=True)


def _grid_csr_strategy(total: int, n_polygons: int, n_cells: int) -> str:
    """Choose stable, packed-key, or direct monotonic CSR construction."""
    if (
        total >= _GRID_DIRECT_MIN_ENTRIES
        and total >= _GRID_DIRECT_MIN_CELLS_PER_POLYGON * n_polygons
        and total * 2 >= n_cells
    ):
        return "direct"
    if (
        total >= _GRID_PACKED_MIN_ENTRIES
        and total >= _GRID_PACKED_MIN_CELLS_PER_POLYGON * n_polygons
    ):
        return "packed"
    return "stable"


def _build_grid_csr_stable(
    finite_idx: np.ndarray,
    ix0: np.ndarray,
    iy0: np.ndarray,
    ix1: np.ndarray,
    iy1: np.ndarray,
    cells_per_poly: np.ndarray,
    total: int,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build low-coverage CSR with NumPy's stable integer argsort."""
    poly_rep = np.repeat(finite_idx.astype(np.int32), cells_per_poly)
    starts = np.empty(cells_per_poly.size, dtype=np.int64)
    starts[0] = 0
    if starts.size > 1:
        np.cumsum(cells_per_poly[:-1], out=starts[1:])
    rank = np.arange(total, dtype=np.int64) - np.repeat(
        starts, cells_per_poly
    )
    dy_rep = np.repeat(iy1 - iy0 + 1, cells_per_poly)
    local_x = rank // dy_rep
    local_y = rank % dy_rep
    cell_id = (
        (np.repeat(iy0, cells_per_poly) + local_y) * nx
        + np.repeat(ix0, cells_per_poly)
        + local_x
    ).astype(np.int32)

    order = np.argsort(cell_id, kind="stable")
    cell_poly = poly_rep[order]
    counts = np.bincount(cell_id, minlength=nx * ny)
    return cell_poly, counts, _offsets_from_counts(counts)


def _offsets_from_counts(counts: np.ndarray) -> np.ndarray:
    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets


def _build_grid_csr_packed(
    finite_idx: np.ndarray,
    ix0: np.ndarray,
    iy0: np.ndarray,
    ix1: np.ndarray,
    iy1: np.ndarray,
    cells_per_poly: np.ndarray,
    total: int,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sorted cell CSR through unique packed ``(cell, polygon)`` keys."""
    # Expand each bbox rectangle in original polygon order. Reuse the two
    # entry-sized int64 work arrays in place to keep peak memory bounded.
    polygon_starts = np.empty(cells_per_poly.size, dtype=np.int64)
    polygon_starts[0] = 0
    if polygon_starts.size > 1:
        np.cumsum(cells_per_poly[:-1], out=polygon_starts[1:])
    rank = np.arange(total, dtype=np.int64)
    rank -= np.repeat(polygon_starts, cells_per_poly)
    dy = np.repeat(iy1 - iy0 + 1, cells_per_poly)
    local_y = np.remainder(rank, dy)
    np.floor_divide(rank, dy, out=rank)
    del dy

    rank += np.repeat(ix0, cells_per_poly)
    local_y += np.repeat(iy0, cells_per_poly)
    local_y *= nx
    local_y += rank
    del rank

    cell_id = local_y.astype(np.int32)
    del local_y
    counts = np.bincount(cell_id, minlength=nx * ny)
    offsets = _offsets_from_counts(counts)

    # Every polygon contributes at most one entry to a cell, so these keys are
    # unique. An in-place unstable quicksort therefore still produces the exact
    # required cell-major, ascending-polygon order without an int64 argsort
    # permutation or a separate repeated polygon array.
    keys = cell_id.astype(np.uint64)
    del cell_id
    keys <<= np.uint64(32)
    keys |= np.repeat(finite_idx.astype(np.uint32), cells_per_poly)
    keys.sort(kind="quicksort")
    cell_poly = (keys & _UINT32_MASK).astype(np.int32)
    return cell_poly, counts, offsets


def _build_grid_csr_direct(
    finite_idx: np.ndarray,
    ix0: np.ndarray,
    iy0: np.ndarray,
    ix1: np.ndarray,
    iy1: np.ndarray,
    total: int,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dense rectangle CSR without sorting, preserving polygon order."""
    # Four-corner range updates recover exact per-cell candidate counts in
    # O(polygons + cells), without materializing one cell id per entry.
    count_grid = np.zeros((ny, nx), dtype=np.int64)
    np.add.at(count_grid, (iy0, ix0), 1)
    x_after = ix1 + 1
    y_after = iy1 + 1
    has_x_after = x_after < nx
    has_y_after = y_after < ny
    np.add.at(
        count_grid,
        (iy0[has_x_after], x_after[has_x_after]),
        -1,
    )
    np.add.at(
        count_grid,
        (y_after[has_y_after], ix0[has_y_after]),
        -1,
    )
    has_corner = has_x_after & has_y_after
    np.add.at(
        count_grid,
        (y_after[has_corner], x_after[has_corner]),
        1,
    )
    np.cumsum(count_grid, axis=0, out=count_grid)
    np.cumsum(count_grid, axis=1, out=count_grid)
    counts = count_grid.reshape(-1)
    offsets = _offsets_from_counts(counts)

    # Polygon ids arrive in ascending original order. Each polygon touches a
    # cell only once, so advancing that cell's cursor writes the exact sorted
    # segment directly and deterministically—no atomics or post-sort needed.
    cell_poly = np.empty(total, dtype=np.int32)
    cursor = offsets[:-1].copy()
    for p in range(finite_idx.size):
        xs = np.arange(ix0[p], ix1[p] + 1, dtype=np.int64)
        row_starts = np.arange(iy0[p], iy1[p] + 1, dtype=np.int64) * nx
        cell_id = (row_starts[:, None] + xs).reshape(-1)
        destination = cursor[cell_id]
        cell_poly[destination] = finite_idx[p]
        cursor[cell_id] = destination + 1
    return cell_poly, counts, offsets


class GridIndex:
    """Uniform-grid spatial index over a :class:`PolygonSet`."""

    def __init__(
        self,
        polygons,
        *,
        cell_size: float | None = None,
        max_cells: int = 4_194_304,
        max_entries: int = 50_000_000,
    ):
        polyset = polygons if isinstance(polygons, PolygonSet) else PolygonSet(polygons)
        self.polygons = polyset
        self._cell_poly_mx = None
        self._cell_offsets_mx = None
        self._grid_params_mx = None
        self._grid_params_np = None
        self._tables_mx = None
        self._max_cell_count = 0

        bboxes = polyset.bboxes  # (M, 4) float64, original coordinates
        finite = np.isfinite(bboxes).all(axis=1)

        # Validate caller-supplied parameters before the all-empty shortcut so
        # their contract does not depend on whether the geometry happens to
        # contribute a finite bbox.  Keep the ordinary construction order:
        # cell_size is normalized first, then the two resource budgets are
        # considered by the planner.
        if cell_size is not None:
            cell_size = float(cell_size)
            if cell_size <= 0.0 or not np.isfinite(cell_size):
                raise ValueError("cell_size must be positive and finite")
        max_cells = check_positive_int(max_cells, "max_cells")
        max_entries = check_positive_int(max_entries, "max_entries")

        if not finite.any():
            # Every geometry is empty: a degenerate grid whose bounds exclude
            # all points, so every query yields zero candidates on either
            # backend.
            self.x0 = self.y0 = 0.0
            self.x1 = self.y1 = -1.0
            self.cell_size = 1.0
            self.nx = self.ny = 1
            self._cell_poly = np.empty(0, dtype=np.int32)
            self._cell_offsets = np.zeros(2, dtype=np.int64)
            self._cell_max_ring_edges = None
            self._cell_edge_work = np.zeros(1, dtype=np.uint16)
            self._cell_all_long_rings = None
            self._max_cell_ring_edges = 0
            self._max_cell_edge_work = 0
            self._edges_per_point = 0.0
            return
        fb = bboxes[finite]
        self.x0 = float(fb[:, 0].min())
        self.y0 = float(fb[:, 1].min())
        x1 = float(fb[:, 2].max())
        y1 = float(fb[:, 3].max())
        self.x1, self.y1 = x1, y1
        width = max(x1 - self.x0, 0.0)
        height = max(y1 - self.y0, 0.0)

        # Polygon cell ranges are padded by twice the boundary tolerance
        # (16 ulps of the local extent): more than the kernel's 4.5-ulp bbox
        # widening plus the fp32 cell-assignment error on the device (under
        # 8 ulps: the rounded, outward-nudged grid corner, the subtraction,
        # and the multiply by inv_cell). Whichever cell a point near a cell
        # line lands in therefore lists every polygon whose kernel bbox
        # accepts it, and the answer is determined by the kernel alone. Bboxes
        # that end exactly on a cell line (edge-aligned tilings) are listed on
        # both sides.
        pad = 2.0 * polyset.boundary_tolerance
        finite_idx = np.flatnonzero(finite)
        # Guard the coarsening loop against float overflow (width / cell_size
        # -> inf) when the cell is absurdly small relative to the extent.
        min_cell = max(width, height, 1.0) * 1e-12

        if cell_size is None:
            # Default: a quarter of the median (longer) bbox side. Against the
            # median itself it cuts candidates per point ~2.5x and kernel time
            # ~40% on random zones and takes edge-aligned tilings from 9
            # candidates per point to about 2, for a 4-6x larger but still
            # small CSR. Refinement backs off (half, then the median) while the
            # CSR would exceed _GRID_REFINE_MAX_ENTRIES_PER_POLYGON, which is
            # what a few large polygons among many small ones would cause.
            spans = np.maximum(fb[:, 2] - fb[:, 0], fb[:, 3] - fb[:, 1])
            base = float(np.median(spans))
            if base <= 0.0:
                base = max(width, height) / 64.0
            if base <= 0.0:
                base = 1.0
            ladder = [base / 4.0, base / 2.0, base]
        else:
            ladder = [cell_size]

        plan = None
        for step, candidate in enumerate(ladder):
            plan = _plan_cells(
                max(candidate, min_cell), fb, self.x0, self.y0, width, height,
                pad, max_cells, max_entries,
            )
            if plan is None:
                continue
            refining = step < len(ladder) - 1
            if not refining or plan[8] <= _GRID_REFINE_MAX_ENTRIES_PER_POLYGON * finite_idx.size:
                break
        if plan is None:
            raise ValueError("could not build a grid within the memory budget")
        cell_size, nx, ny, ix0, iy0, ix1, iy1, cells_per_poly, total = plan

        self.cell_size = cell_size
        self.nx, self.ny = nx, ny

        # Keep low-coverage input on stable integer sort, pack moderately
        # expanded unique keys in place, and remove sorting entirely when bbox
        # rectangles densely cover a material fraction of the grid.
        strategy = _grid_csr_strategy(total, finite_idx.size, nx * ny)
        if strategy == "direct":
            self._cell_poly, counts, self._cell_offsets = _build_grid_csr_direct(
                finite_idx, ix0, iy0, ix1, iy1, total, nx, ny
            )
        elif strategy == "packed":
            self._cell_poly, counts, self._cell_offsets = _build_grid_csr_packed(
                finite_idx,
                ix0,
                iy0,
                ix1,
                iy1,
                cells_per_poly,
                total,
                nx,
                ny,
            )
        else:
            self._cell_poly, counts, self._cell_offsets = _build_grid_csr_stable(
                finite_idx,
                ix0,
                iy0,
                ix1,
                iy1,
                cells_per_poly,
                total,
                nx,
                ny,
            )

        # Largest per-cell candidate count: n * _max_cell_count bounds a
        # query's candidate pairs for the single-launch first-match fast path.
        self._max_cell_count = int(counts.max(initial=0))

        # Host selectors need the actual candidate complexity of each cell.
        # This avoids routing a query through a cooperative long-ring kernel
        # merely because some unrelated polygon elsewhere in the set is long.
        nonempty_cell = counts > 0
        starts = self._cell_offsets[:-1][nonempty_cell]
        # Selectors saturate well below 65k edges. Narrow the small geometry
        # tables *before* expanding by CSR entry so max_entries-sized builds
        # never create temporary int64 entry arrays.
        u16_max = np.iinfo(np.uint16).max
        geom_edge_counts = np.minimum(
            polyset._geom_edge_counts, u16_max
        ).astype(np.uint16)
        polygon_edges = geom_edge_counts[self._cell_poly]
        self._cell_edge_work = np.zeros(nx * ny, dtype=np.uint16)
        edge_work = np.add.reduceat(
            polygon_edges, starts, dtype=np.uint64
        )
        self._cell_edge_work[nonempty_cell] = np.minimum(
            edge_work, u16_max
        ).astype(np.uint16)
        self._max_cell_edge_work = int(self._cell_edge_work.max(initial=0))
        # Expected candidate edge work per query point for the CPU/GPU
        # dispatch cost model: a per-cell mean over candidate-bearing cells,
        # not a point-density mean, so skewed layers err toward the GPU.
        self._edges_per_point = (
            float(self._cell_edge_work[nonempty_cell].mean()) if nonempty_cell.any() else 0.0
        )

        if polyset.max_edges_per_ring >= _PIP_LONG_RING_THRESHOLD:
            geom_ring_edges = np.minimum(
                polyset._geom_max_ring_edges, u16_max
            ).astype(np.uint16)
            geom_has_long_ring = (
                polyset._geom_max_ring_edges >= _PIP_LONG_RING_THRESHOLD
            ).astype(np.uint8)
            ring_edges = geom_ring_edges[self._cell_poly]
            long_ring = geom_has_long_ring[self._cell_poly]
            self._cell_max_ring_edges = np.zeros(nx * ny, dtype=np.uint16)
            self._cell_all_long_rings = np.zeros(nx * ny, dtype=bool)
            self._cell_max_ring_edges[nonempty_cell] = np.maximum.reduceat(
                ring_edges, starts
            )
            self._cell_all_long_rings[nonempty_cell] = np.minimum.reduceat(
                long_ring, starts
            ).astype(bool)
            self._max_cell_ring_edges = int(
                self._cell_max_ring_edges.max(initial=0)
            )
        else:
            self._cell_max_ring_edges = None
            self._cell_all_long_rings = None
            self._max_cell_ring_edges = 0

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def __repr__(self) -> str:
        return (
            f"GridIndex(nx={self.nx}, ny={self.ny}, cell_size={self.cell_size:g}, "
            f"entries={len(self._cell_poly)}, polygons={self.polygons.n_polygons})"
        )

    def _candidates(self, pts64: np.ndarray, *, include_starts: bool = True):
        """Float64 host cell lookup: candidate counts, CSR starts, total, cell IDs.

        This is the exact-bounds reference cell assignment (tests compare the
        device graph against it) and feeds nothing on the query path: GPU
        queries assign cells through :meth:`_device_candidates` and the host
        fallback through :meth:`_host_candidates`, both in fp32 against the
        outward-nudged bounds, so a point a fraction of an ulp outside the
        float64 extent gets the same answer on every route.
        """
        x, y = pts64[:, 0], pts64[:, 1]
        # NaN coordinates (empty points) compare False everywhere -> not in grid.
        in_grid = (x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1)
        xs = np.where(in_grid, x, self.x0)
        ys = np.where(in_grid, y, self.y0)
        ix = np.clip(np.floor((xs - self.x0) / self.cell_size), 0, self.nx - 1).astype(np.int64)
        iy = np.clip(np.floor((ys - self.y0) / self.cell_size), 0, self.ny - 1).astype(np.int64)
        cell_id = np.where(in_grid, iy * self.nx + ix, 0)
        counts = np.where(
            in_grid,
            self._cell_offsets[cell_id + 1] - self._cell_offsets[cell_id],
            0,
        )
        point_cell = np.where(in_grid, cell_id, -1).astype(np.int32)
        cand_starts = self._cell_offsets[cell_id] if include_starts else None
        return counts, cand_starts, int(counts.sum()), point_cell

    def _device_csr(self) -> tuple[mx.array, mx.array]:
        """Lazily cache the grid CSR arrays for point-centric GPU queries."""
        if self._cell_offsets_mx is None:
            # Offsets remain int64: max_entries is caller-adjustable, so an
            # unconditional int32 cast would silently narrow the index domain.
            self._cell_offsets_mx = mx.array(self._cell_offsets)
            self._cell_poly_mx = mx.array(self._cell_poly)
        return self._cell_offsets_mx, self._cell_poly_mx

    def _host_grid(self) -> tuple:
        """fp32 local-frame grid bounds and reciprocal cell size (np.float32).

        The outer bounds are rounded outward so the fp32 in-grid test admits
        every point the float64 test admits. Both the device graph and the
        host fallback assign cells from exactly these five scalars.
        """
        if self._grid_params_np is None:
            ox, oy = self.polygons.origin
            f32 = np.float32
            neg_inf, pos_inf = f32(-np.inf), f32(np.inf)
            self._grid_params_np = (
                np.nextafter(f32(self.x0 - ox), neg_inf),
                np.nextafter(f32(self.y0 - oy), neg_inf),
                np.nextafter(f32(self.x1 - ox), pos_inf),
                np.nextafter(f32(self.y1 - oy), pos_inf),
                f32(1.0 / self.cell_size),
            )
        return self._grid_params_np

    def _device_grid(self) -> tuple:
        """Lazily cache the fp32 local-frame grid parameters as 0-d arrays.

        Every per-index scalar travels as an ``mx.array``: Python constants
        would be baked into the compiled graph's Metal source and force a
        fresh JIT for every index. Bounds come from :meth:`_host_grid`.
        """
        if self._grid_params_mx is None:
            x0, y0, x1, y1, inv_cell = self._host_grid()
            cell_offsets_mx, _ = self._device_csr()
            self._grid_params_mx = (
                mx.array(x0),
                mx.array(y0),
                mx.array(x1),
                mx.array(y1),
                mx.array(inv_cell),
                mx.array(self.nx, dtype=mx.int32),
                mx.array(self.nx - 1, dtype=mx.int32),
                mx.array(self.ny - 1, dtype=mx.int32),
                cell_offsets_mx,
            )
        return self._grid_params_mx

    def _host_candidates(self, pts32: np.ndarray):
        """Host twin of :func:`_cell_graph`: candidate counts, CSR starts, total.

        ``pts32`` is the ``(N, 2)`` float32 local-frame array the kernels see.
        The in-grid test and the cell arithmetic run in float32 against the
        same outward-nudged bounds and reciprocal cell size as the device
        graph, so the fallback admits exactly the points the GPU admits. (A
        one-ulp difference in the cell itself would not matter: padded cell
        ranges make the answer independent of which cell a point lands in.)
        NaN/inf coordinates fail the in-grid test and get zero candidates.
        """
        x0, y0, x1, y1, inv_cell = self._host_grid()
        x, y = pts32[:, 0], pts32[:, 1]
        in_grid = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        xs = np.where(in_grid, x, x0)
        ys = np.where(in_grid, y, y0)
        ix = np.clip(np.floor((xs - x0) * inv_cell), 0, self.nx - 1).astype(np.int64)
        iy = np.clip(np.floor((ys - y0) * inv_cell), 0, self.ny - 1).astype(np.int64)
        cell_id = np.where(in_grid, iy * self.nx + ix, 0)
        counts = np.where(
            in_grid,
            self._cell_offsets[cell_id + 1] - self._cell_offsets[cell_id],
            0,
        )
        return counts, self._cell_offsets[cell_id], int(counts.sum())

    def _device_tables(self) -> tuple:
        """Lazily cache the per-cell selector summaries on device."""
        if self._tables_mx is None:
            self._tables_mx = (
                None
                if self._cell_max_ring_edges is None
                else mx.array(self._cell_max_ring_edges),
                None
                if self._cell_all_long_rings is None
                else mx.array(self._cell_all_long_rings),
                mx.array(self._cell_edge_work),
            )
        return self._tables_mx

    def _device_candidates(self, local: mx.array):
        """GPU cell lookup for fp32 local points: cell IDs, counts, total, max.

        One synchronization returns the two scalars the dispatcher and the
        batcher need. Polygon cell ranges were padded at build time by more
        than the fp32 assignment error, so whichever cell a point near a cell
        line lands in lists every polygon whose kernel bbox accepts it.
        """
        with mx.stream(mx.gpu):
            cell, counts = _device_cells(local[:, 0], local[:, 1], *self._device_grid())
            total = counts.sum()
            cmax = counts.max()
            mx.eval(cell, counts, total, cmax)
        return cell, counts, int(total.item()), int(cmax.item())

    def _candidate_edge_bounds(
        self, point_cell: np.ndarray
    ) -> tuple[int, int]:
        """Maximum ring size and aggregate edge work in referenced cells."""
        # The ordinary short-polygon case cannot change either adaptive
        # selector, so avoid an O(N) point-cell gather on every query.
        if (
            self._max_cell_ring_edges < _PIP_LONG_RING_THRESHOLD
            and self._max_cell_edge_work < _ALL_MATCHES_EDGE_WORK_THRESHOLD
        ):
            return self._max_cell_ring_edges, self._max_cell_edge_work
        cells = point_cell[point_cell >= 0]
        if cells.size:
            cells = cells[
                self._cell_offsets[cells + 1] > self._cell_offsets[cells]
            ]
        if cells.size == 0:
            return 0, 0
        max_ring_edges = (
            0
            if self._cell_max_ring_edges is None
            else int(self._cell_max_ring_edges[cells].max(initial=0))
        )
        return max_ring_edges, int(self._cell_edge_work[cells].max(initial=0))

    def _first_match_ring_bound(self, point_cell: np.ndarray) -> int:
        """Cooperate only when every candidate in every used cell is long."""
        if self._max_cell_ring_edges < _PIP_LONG_RING_THRESHOLD:
            return 0
        cells = point_cell[point_cell >= 0]
        if cells.size:
            cells = cells[
                self._cell_offsets[cells + 1] > self._cell_offsets[cells]
            ]
        if cells.size == 0 or not self._cell_all_long_rings[cells].all():
            return 0
        return int(self._cell_max_ring_edges[cells].max(initial=0))

    def _device_first_match_bound(self, cell: mx.array, counts: mx.array) -> int:
        """Device twin of :meth:`_first_match_ring_bound` (one sync if needed)."""
        if self._max_cell_ring_edges < _PIP_LONG_RING_THRESHOLD:
            return 0
        max_ring_mx, all_long_mx, _ = self._device_tables()
        with mx.stream(mx.gpu):
            used = counts > 0
            safe = mx.maximum(cell, 0)
            all_long = mx.where(used, mx.take(all_long_mx, safe), True).all()
            max_ring = mx.where(used, mx.take(max_ring_mx, safe), 0).max()
            mx.eval(all_long, max_ring)
        if not bool(all_long.item()):
            return 0
        return int(max_ring.item())

    def _device_edge_bounds(self, cells: mx.array) -> tuple[int, int]:
        """Device twin of :meth:`_candidate_edge_bounds` for candidate-bearing cells."""
        if (
            self._max_cell_ring_edges < _PIP_LONG_RING_THRESHOLD
            and self._max_cell_edge_work < _ALL_MATCHES_EDGE_WORK_THRESHOLD
        ):
            return self._max_cell_ring_edges, self._max_cell_edge_work
        if int(cells.shape[0]) == 0:
            return 0, 0
        max_ring_mx, _, edge_work_mx = self._device_tables()
        with mx.stream(mx.gpu):
            edge_work = mx.take(edge_work_mx, cells).max()
            if max_ring_mx is None:
                mx.eval(edge_work)
                return 0, int(edge_work.item())
            max_ring = mx.take(max_ring_mx, cells).max()
            mx.eval(max_ring, edge_work)
        return int(max_ring.item()), int(edge_work.item())

    def query(
        self,
        points,
        *,
        all_matches: bool = False,
        backend: str | None = None,
        predicate: str = "intersects",
        exact_boundary: bool = False,
        max_pairs_per_batch: int | None = None,
        _stacklevel: int = 4,
    ):
        """Assign each point to a containing polygon.

        Returns the lowest index of a containing polygon per point (``-1``
        for none) as an ``(N,)`` int32 array, or with ``all_matches=True``
        the ``(point_indices, polygon_indices)`` int64 arrays of every
        containment pair, sorted by point then polygon.

        The output container follows the input: numpy arrays, sequences,
        shapely points and GeoSeries return NumPy arrays; ``mx.array`` and
        :class:`~mlx_geospatial.LocalPoints` inputs return MLX arrays and the
        whole query, cell assignment included, runs on the GPU without a host
        round trip. Device-resident first-match results are lazy when one
        kernel launch suffices (the call itself does not synchronize); call
        ``mx.eval`` when you need the values.

        Backend: host inputs dispatch on a cost model (the NumPy fallback's
        cost per point and per candidate edge versus the fixed GPU cost of the
        route), so a few hundred points against typical zone layers, a couple
        of thousand against sparse tilings or fewer than ten against detailed
        polygons already run on the GPU for first match, and all-match queries
        switch about four times later; device inputs run on the GPU whenever
        Metal is available, since the fallback would need a host copy.
        ``backend=`` and ``MLX_GEOSPATIAL_BACKEND`` override both; ``backend="cpu"``
        on device inputs copies the fp32 local coordinates to the host and
        runs the same predicate there.

        ``max_pairs_per_batch`` is a hard upper bound on each GPU kernel
        launch, including when one point alone has more candidates than the
        requested batch size. ``None`` selects the route default: 256M
        candidate pairs for first match, whose fused kernel allocates nothing
        per pair, and 16M for all-match compaction, whose scan buffers scale
        with the batch.

        One containment contract on every backend: the fp32 even-odd rule in
        re-centered coordinates. The CPU fallback assigns cells with the
        device graph's fp32 arithmetic and evaluates the same predicate in
        NumPy with the kernels' operation order, so ``backend="cpu"`` and
        ``backend="gpu"`` are bit-identical for every input, valid geometry or
        not, and auto dispatch can never change an answer. (Only fp32
        subnormals, below ~1e-38, escape this: Metal flushes them to zero and
        NumPy keeps them, so a coordinate, an edge delta or an intermediate of
        the crossing test that small can differ; that needs local coordinates
        themselves near 1e-38 and never occurs for real-world units.)

        ``predicate`` is ``"intersects"`` (boundary counts as a match) or
        ``"within"`` (it does not), accepted for GeoPandas compatibility. The
        two differ only for points exactly on a polygon boundary, inside the
        fp32 tolerance band where the even-odd result is undefined, so by
        default both return the same result. ``exact_boundary=True`` re-checks
        every point within ``PolygonSet.boundary_tolerance`` of a boundary with
        GEOS honoring ``predicate``. Finding those points is a host pass over
        all points (an STRtree over tolerance-widened edge boxes plus a
        float64 distance test: roughly 0.3 s per million points on 48-vertex
        zones, 0.5 s on 10k zones), cheaper than a full shapely join but far
        from GPU speed, so reserve it for boundary-sensitive work. Only the
        band is re-checked: interior points of invalid geometry keep the
        even-odd answer (valid geometry assumed; ``validate=True`` rejects the
        rest). It needs host coordinates, so it is not available for
        device-resident inputs.
        """
        if predicate not in ("within", "intersects"):
            raise ValueError(
                f"predicate must be 'within' or 'intersects', got {predicate!r}"
            )
        if exact_boundary and isinstance(points, (mx.array, LocalPoints)):
            raise ValueError(
                "exact_boundary=True needs host coordinates (GEOS re-checks the "
                "near-boundary points); pass the original NumPy/GeoSeries points"
            )
        if max_pairs_per_batch is None:
            max_pairs_per_batch = (
                _ALL_MATCHES_MAX_PAIRS_PER_BATCH
                if all_matches
                else _FIRST_MATCH_MAX_PAIRS_PER_BATCH
            )
        max_pairs_per_batch = check_positive_int(
            max_pairs_per_batch, "max_pairs_per_batch"
        )
        if max_pairs_per_batch >= 2**31:
            raise ValueError(
                f"max_pairs_per_batch must be in [1, 2**31), got {max_pairs_per_batch}"
            )
        polyset = self.polygons
        # Host inputs defer the MLX local array: the fallback re-centers in
        # NumPy (bit-identical) and never needs it.
        pts64, local, on_device = polyset._ingest_points(
            points, stacklevel=_stacklevel, defer_local=True
        )
        n = int(local.shape[0]) if on_device else len(pts64)
        if n >= 2**31:
            raise ValueError("more than 2^31 points per query is not supported")
        if n == 0:
            resolve_backend(backend, 0)
            return _empty_result(all_matches, on_device, 0)

        # Dispatch needs no candidate count: the cost model compares the
        # fallback's cost for n points (per point plus per expected candidate
        # edge) with the fixed GPU launch cost; device inputs stay on the GPU.
        chosen = resolve_backend(
            backend,
            None if on_device else n,
            edges_per_point=self._edges_per_point,
            all_matches=all_matches,
        )
        def finish(out):
            if exact_boundary:
                out = self._refine_boundary(
                    pts64, out, all_matches=all_matches, predicate=predicate
                )
            return _to_container(out, on_device)

        if chosen == "cpu":
            # The fallback works on the very float32 local array the kernels
            # would see: device inputs are copied out as they are (no lossy
            # float64 round trip through the origin), host inputs re-centered
            # in NumPy, bit for bit what the MLX CPU stream produces.
            pts32 = np.asarray(local) if on_device else recenter(pts64, polyset.origin)
            return finish(self._query_cpu(pts32, all_matches=all_matches, predicate=predicate))
        if local is None:
            local = polyset._local_from_f64(pts64)

        bound = n * self._max_cell_count  # >= the candidate-pair total
        if not all_matches and bound <= max_pairs_per_batch:
            # Settled GPU first match that fits one launch: assign cells lazily
            # and hand back the lazy kernel result without a device round trip.
            # An empty candidate list simply yields -1 inside the kernel.
            with mx.stream(mx.gpu):
                cell, counts = _device_cells(local[:, 0], local[:, 1], *self._device_grid())
            out = self._query_gpu_first_match(
                local,
                cell,
                counts,
                total_pairs=bound,
                max_pairs_per_batch=max_pairs_per_batch,
            )
            return finish(out)

        cell, counts, total_pairs, max_count = self._device_candidates(local)
        if total_pairs == 0:
            return finish(_empty_result(all_matches, False, n))
        if all_matches:
            out = self._query_gpu_all_matches(
                local,
                cell,
                counts,
                total_pairs=total_pairs,
                max_pairs_per_batch=max_pairs_per_batch,
            )
        elif max_count <= max_pairs_per_batch:
            out = self._query_gpu_first_match(
                local,
                cell,
                counts,
                total_pairs=total_pairs,
                max_pairs_per_batch=max_pairs_per_batch,
            )
        else:
            # One point alone has more candidates than the cap, so it cannot
            # fit in a whole-point fused launch: use the splittable pair path.
            out = self._query_gpu_pairs(
                local,
                cell,
                counts,
                total_pairs=total_pairs,
                max_pairs_per_batch=max_pairs_per_batch,
            )
        return finish(out)

    def _query_gpu_all_matches(
        self,
        local: mx.array,
        cell: mx.array,
        counts: mx.array,
        *,
        total_pairs: int,
        max_pairs_per_batch: int,
    ) -> tuple[mx.array, mx.array]:
        """GPU count/scan/write compaction, batched by candidate rank."""
        cell_offsets_mx, cell_poly_mx = self._device_csr()
        n = int(local.shape[0])
        empty = (
            mx.zeros((0,), dtype=mx.int64),
            mx.zeros((0,), dtype=mx.int64),
        )
        with mx.stream(mx.gpu):
            # Zero-candidate points cannot contribute output. Compact the
            # candidate-bearing points on device (their ascending order is the
            # output order) so millions of outside rows cost no host work.
            active = counts > 0
            n_active = int(active.sum().item())
            if n_active == 0:
                return empty
            dest = mx.cumsum(active.astype(mx.int32)) - 1
            active_point = mx.zeros((n_active + 1,), dtype=mx.int32)
            # Inactive rows all scatter into the spare last slot, dropped below.
            active_point[mx.where(active, dest, n_active)] = mx.arange(n, dtype=mx.int32)
            active_point = active_point[:n_active]
            active_cell = mx.take(cell, active_point)
            active_count = mx.take(counts, active_point)
            candidate_offsets = mx.concatenate(
                [mx.zeros((1,), dtype=mx.int64), mx.cumsum(active_count)]
            )
            mx.eval(active_point, active_cell, candidate_offsets)
        if int(candidate_offsets[-1].item()) != total_pairs:
            raise RuntimeError("active candidate offsets do not cover all pairs")

        matches_pt: list[mx.array] = []
        matches_pg: list[mx.array] = []
        pair0 = 0
        while pair0 < total_pairs:
            pair1 = min(pair0 + max_pairs_per_batch, total_pairs)
            with mx.stream(mx.gpu):
                # searchsorted equivalents: counts of offsets <= a rank.
                active0 = int((candidate_offsets <= pair0).sum().item()) - 1
                active1 = int((candidate_offsets <= pair1 - 1).sum().item())
                # Clip the active prefix offsets to this exact pair-rank batch.
                # The largest difference detects a single skewed candidate
                # list even when its cost is hidden by a tiny chunk average.
                clipped = mx.clip(candidate_offsets[active0 : active1 + 1], pair0, pair1)
                max_point_pairs = int((clipped[1:] - clipped[:-1]).max().item())
            max_ring_edges, max_candidate_edges = self._device_edge_bounds(
                active_cell[active0:active1]
            )
            out_point, out_poly = pip_all_matches(
                local,
                self.polygons,
                active_point[active0:active1],
                active_cell[active0:active1],
                candidate_offsets[active0 : active1 + 1],
                cell_offsets_mx,
                cell_poly_mx,
                pair_begin=pair0,
                pair_end=pair1,
                max_point_pairs=max_point_pairs,
                max_ring_edges=max_ring_edges,
                max_candidate_edges=max_candidate_edges,
            )
            if int(out_point.shape[0]):
                matches_pt.append(out_point)
                matches_pg.append(out_poly)
            pair0 = pair1

        if not matches_pt:
            return empty
        if len(matches_pt) == 1:
            return matches_pt[0], matches_pg[0]
        with mx.stream(mx.gpu):
            return mx.concatenate(matches_pt), mx.concatenate(matches_pg)

    def _query_gpu_first_match(
        self,
        local: mx.array,
        cell: mx.array,
        counts: mx.array,
        *,
        total_pairs: int,
        max_pairs_per_batch: int,
    ) -> mx.array:
        """Point-centric CSR traversal, batched by candidate-work budget."""
        cell_offsets_mx, cell_poly_mx = self._device_csr()
        n = int(local.shape[0])

        if total_pairs <= max_pairs_per_batch:
            max_ring_edges = self._device_first_match_bound(cell, counts)
            return pip_first_match(
                local,
                self.polygons,
                cell,
                cell_offsets_mx,
                cell_poly_mx,
                max_ring_edges=max_ring_edges,
            )

        # Whole-point batches preserve the public hard cap on candidate work
        # per launch. The caller has already routed any individually oversized
        # point to the splittable explicit-pair path. Batch boundaries come
        # from the device prefix sum; each needs one small synchronization.
        with mx.stream(mx.gpu):
            cum = mx.cumsum(counts)  # inclusive, int64
            mx.eval(cum)
        results: list[mx.array] = []
        point0 = 0
        base = 0
        while point0 < n:
            with mx.stream(mx.gpu):
                point1 = int((cum <= base + max_pairs_per_batch).sum().item())
            if point1 <= point0:  # defensive: counts.max() was checked above
                raise RuntimeError("could not form a bounded fused point batch")
            max_ring_edges = self._device_first_match_bound(
                cell[point0:point1], counts[point0:point1]
            )
            results.append(
                pip_first_match(
                    local[point0:point1],
                    self.polygons,
                    cell[point0:point1],
                    cell_offsets_mx,
                    cell_poly_mx,
                    max_ring_edges=max_ring_edges,
                )
            )
            base = int(cum[point1 - 1].item())
            point0 = point1
        with mx.stream(mx.gpu):
            return mx.concatenate(results)

    def _query_gpu_pairs(
        self,
        local: mx.array,
        cell: mx.array,
        counts: mx.array,
        *,
        total_pairs: int,
        max_pairs_per_batch: int,
    ) -> mx.array:
        """Bounded explicit-pair fallback for oversized first-match cells.

        Rare (one cell holds more polygons than the per-launch cap), so the
        pair bookkeeping runs on the host from the device-assigned cells.
        """
        counts = np.asarray(counts).astype(np.int64, copy=False)
        cand_starts = self._cell_offsets[np.maximum(np.asarray(cell), 0)]
        n = len(counts)

        # Expand candidate pairs in contiguous ranges of the conceptual flat
        # pair array. Chunking by pair rank (rather than only by point ranges)
        # keeps max_pairs_per_batch a hard bound even when one point's cell
        # contains more polygons than the budget.
        cum = np.concatenate([[0], np.cumsum(counts)])  # (n + 1,)
        result = np.full(n, _INT32_MAX, dtype=np.int32)
        pair0 = 0
        while pair0 < total_pairs:
            pair1 = min(pair0 + max_pairs_per_batch, total_pairs)
            point0 = int(np.searchsorted(cum, pair0, side="right")) - 1
            point1 = int(np.searchsorted(cum, pair1 - 1, side="right")) - 1

            # Keep only candidate-bearing points in this span. Long runs of
            # outside/empty points therefore cost one scan, not pair storage;
            # every retained point contributes at least one pair, so these
            # arrays remain bounded by the pair budget.
            span_counts = counts[point0 : point1 + 1]
            nonzero = np.flatnonzero(span_counts)
            point_ids = point0 + nonzero
            chunk_counts = span_counts[nonzero].copy()
            chunk_counts[0] -= pair0 - cum[point_ids[0]]
            chunk_counts[-1] -= cum[point_ids[-1] + 1] - pair1

            pair_point = np.repeat(point_ids, chunk_counts)
            pair_rank = np.arange(pair0, pair1, dtype=np.int64)
            rank_in_point = pair_rank - cum[pair_point]
            pair_poly = self._cell_poly[cand_starts[pair_point] + rank_in_point]
            max_ring_edges = int(
                self.polygons._geom_max_ring_edges[pair_poly].max(initial=0)
            )
            flags = np.array(
                pip_pairs(
                    local,
                    self.polygons,
                    mx.array(pair_point.astype(np.int32)),
                    mx.array(pair_poly.astype(np.int32)),
                    max_ring_edges=max_ring_edges,
                )
            )
            hit = flags.astype(bool)
            np.minimum.at(result, pair_point[hit], pair_poly[hit].astype(np.int32))
            pair0 = pair1

        result[result == _INT32_MAX] = -1
        return mx.array(result)

    def _query_cpu(
        self,
        pts32: np.ndarray,
        *,
        all_matches: bool,
        predicate: str = "intersects",
    ):
        """Host fallback: the kernels' fp32 even-odd predicate, in NumPy.

        ``pts32`` is the ``(N, 2)`` float32 local-frame array the kernels
        would see. Same cell assignment (:meth:`_host_candidates`, the device
        graph's fp32 arithmetic against the outward-nudged bounds), same
        widened float32 kernel bboxes, and the same float32 operation order as
        the Metal kernels and the composed oracle (every op rounded on its
        own, divide before multiply), so the result is bit-identical to the
        GPU path for every input outside the fp32 subnormal range, valid
        geometry or not. Points are processed
        in slices bounded by candidate pairs and crossing rows
        (``_CPU_PAIRS_PER_CHUNK``, ``_CPU_ROWS_PER_CHUNK``) so memory does not
        grow with the point count. ``predicate`` is accepted for signature
        compatibility; both predicates evaluate the same rule here as on the
        GPU (see :meth:`query`).
        """
        pts32 = np.ascontiguousarray(pts32, dtype=np.float32)
        n = len(pts32)
        counts, starts, total = self._host_candidates(pts32)
        if total == 0:
            return _empty_result(all_matches, False, n)

        # Iterate over the conceptual flat candidate-pair array, rather than
        # whole points.  One heavily occupied cell can give a single point
        # more candidates than the budget; pair-rank slices keep the limit
        # hard in that case too.
        cum = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
        result = None if all_matches else np.full(n, _INT32_MAX, dtype=np.int32)
        pairs_pt: list[np.ndarray] = []
        pairs_pg: list[np.ndarray] = []
        pair0 = 0
        while pair0 < total:
            pair1 = min(pair0 + _CPU_PAIRS_PER_CHUNK, total)
            hit_pt, hit_pg = self._cpu_hits(pts32, counts, cum, starts, pair0, pair1)
            if hit_pt.size:
                if all_matches:
                    pairs_pt.append(hit_pt.astype(np.int64))
                    pairs_pg.append(hit_pg.astype(np.int64))
                else:
                    # A point can span chunks now.  Accumulating the minimum
                    # preserves first-match semantics across every split.
                    first = np.unique(hit_pt, return_index=True)[1]
                    point_hit = hit_pt[first]
                    result[point_hit] = np.minimum(result[point_hit], hit_pg[first])
            pair0 = pair1

        if not all_matches:
            result[result == _INT32_MAX] = -1
            return result
        if not pairs_pt:
            return _empty_result(True, False, n)
        return np.concatenate(pairs_pt), np.concatenate(pairs_pg)

    def _cpu_hits(self, pts32, counts, cum, starts, pair0: int, pair1: int):
        """Containment pairs ``(point, polygon)`` for one flat-pair slice.

        Explicit candidate pairs from the CSR lists (sorted, so pairs are
        point-major and ascending in polygon), the kernel's bbox reject, then
        the crossing test over ``(pair, edge)`` rows in bounded chunks. Both
        this slice and every crossing-row slice are hard-capped even when a
        single point or polygon exceeds the corresponding budget.
        """
        polyset = self.polygons
        empty = np.empty(0, dtype=np.int32)
        total = pair1 - pair0
        if total <= 0:
            return empty, empty

        # Only two searches per chunk: derive the per-point segment lengths,
        # then expand them with repeat.  This avoids a searchsorted per pair
        # on the fallback's hot path.
        point0 = int(np.searchsorted(cum, pair0, side="right")) - 1
        point1 = int(np.searchsorted(cum, pair1 - 1, side="right")) - 1
        span_counts = counts[point0 : point1 + 1]
        nonzero = np.flatnonzero(span_counts)
        point_ids = point0 + nonzero
        chunk_counts = span_counts[nonzero].astype(np.int64, copy=True)
        chunk_counts[0] -= pair0 - cum[point_ids[0]]
        chunk_counts[-1] -= cum[point_ids[-1] + 1] - pair1
        pair_point = np.repeat(point_ids, chunk_counts)
        pair_rank = np.arange(pair0, pair1, dtype=np.int64)
        rank_in_point = pair_rank - cum[pair_point]
        pair_poly = self._cell_poly[starts[pair_point] + rank_in_point]
        px = pts32[pair_point, 0]
        py = pts32[pair_point, 1]
        bb = polyset._bboxes_local_np()[pair_poly]
        # The kernel rejects on any failing comparison; NaN coordinates fail
        # none of them, reach the edge loop and never straddle an edge.
        rejected = (px < bb[:, 0]) | (py < bb[:, 1]) | (px > bb[:, 2]) | (py > bb[:, 3])

        edges = polyset._edges_np()
        edge_offsets = polyset._geom_edge_offsets
        n_edges = edge_offsets[pair_poly + 1] - edge_offsets[pair_poly]
        n_edges = np.where(rejected, 0, n_edges)
        parity = np.zeros(total, dtype=bool)
        cum_edges = np.concatenate([[0], np.cumsum(n_edges, dtype=np.int64)])
        total_rows = int(cum_edges[-1])
        row0 = 0
        while row0 < total_rows:
            row1 = min(row0 + _CPU_ROWS_PER_CHUNK, total_rows)

            # As above, two boundary searches plus repeat expand the CSR
            # segment without a search per crossing row.  Adjusting the first
            # and last counts lets a single polygon continue in the next
            # chunk while no materialized row array exceeds the hard cap.
            edge_pair0 = int(np.searchsorted(cum_edges, row0, side="right")) - 1
            edge_pair1 = int(np.searchsorted(cum_edges, row1 - 1, side="right")) - 1
            span_reps = n_edges[edge_pair0 : edge_pair1 + 1]
            nonzero = np.flatnonzero(span_reps)
            edge_pair_ids = edge_pair0 + nonzero
            chunk_reps = span_reps[nonzero].astype(np.int64, copy=True)
            chunk_reps[0] -= row0 - cum_edges[edge_pair_ids[0]]
            chunk_reps[-1] -= cum_edges[edge_pair_ids[-1] + 1] - row1

            row_pair = np.repeat(edge_pair_ids, chunk_reps)
            row_rank = np.arange(row0, row1, dtype=np.int64)
            within = row_rank - cum_edges[row_pair]
            e = edges[edge_offsets[pair_poly[row_pair]] + within]
            crossed = _cpu_crossed_rows(px[row_pair], py[row_pair], e)

            # row_pair is sorted, so reduce each nonempty run to one parity
            # bit. XOR accumulation handles a pair split across row chunks.
            group_start = np.concatenate(
                [[0], np.flatnonzero(row_pair[1:] != row_pair[:-1]) + 1]
            )
            parity[row_pair[group_start]] ^= np.logical_xor.reduceat(crossed, group_start)
            row0 = row1
        hit = parity & ~rejected
        return pair_point[hit].astype(np.int32), pair_poly[hit].astype(np.int32)

    def _refine_boundary(self, pts64: np.ndarray, out, *, all_matches: bool, predicate: str):
        """Re-check points within the tolerance band of a boundary with GEOS.

        Turns the fp32 even-odd answer into exact GEOS ``predicate`` semantics
        for valid geometry: only points within ``boundary_tolerance`` of some
        polygon boundary can differ, and those are re-evaluated with an
        STRtree query honoring ``predicate``. Non-finite rows (empty or
        missing points) are never near a boundary and keep their ``-1``.
        """
        polyset = self.polygons
        near = self._near_boundary(pts64, polyset.boundary_tolerance)
        if near.size == 0:
            return out
        pt_idx, poly_idx = polyset.strtree.query(
            shapely.points(pts64[near]), predicate=predicate
        )
        if not all_matches:
            fixed = np.full(near.size, _INT32_MAX, dtype=np.int32)
            np.minimum.at(fixed, pt_idx, poly_idx.astype(np.int32))
            fixed[fixed == _INT32_MAX] = -1
            result = np.array(out, dtype=np.int32, copy=True)
            result[near] = fixed
            return result
        pt_all, pg_all = (np.asarray(out[0], dtype=np.int64), np.asarray(out[1], dtype=np.int64))
        keep = ~np.isin(pt_all, near)
        new_pt = near[pt_idx].astype(np.int64)
        new_pg = poly_idx.astype(np.int64)
        pt_all = np.concatenate([pt_all[keep], new_pt])
        pg_all = np.concatenate([pg_all[keep], new_pg])
        order = np.lexsort((pg_all, pt_all))
        return pt_all[order], pg_all[order]

    def _near_boundary(self, pts64: np.ndarray, tol: float) -> np.ndarray:
        """Sorted row indices of the finite points within ``tol`` of an edge.

        An STRtree over the edge envelopes widened by ``tol`` (cached on the
        polygon set) yields the candidate (point, edge) pairs by envelope
        alone; a float64 point-to-segment distance test in NumPy settles them.
        Every point within ``tol`` of an edge lies in that edge's widened
        envelope, so nothing is missed, and GEOS is never asked for a distance:
        3-5x cheaper than ``dwithin`` against whole boundaries, which has to
        evaluate a GEOS distance for nearly every point on zone layers.

        Work is bounded by candidate pairs, not points: a probe on a sample of
        the slice estimates pairs per point and sizes the slice handed to the
        tree to about ``_NEAR_BOUNDARY_PAIRS_PER_SLICE`` pairs (a slice that
        still overshoots is retried shorter), and the distance test runs over
        pair chunks of ``_CPU_ROWS_PER_CHUNK``, so layers whose edges span the
        extent (a few large polygons with long diagonal edges put nearly every
        point in many widened envelopes) stay near a few hundred MB instead of
        scaling with points x long edges.
        """
        polyset = self.polygons
        edges = polyset._edges_f64()
        rows = np.flatnonzero(np.isfinite(pts64).all(axis=1))
        if rows.size == 0 or edges.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        tree = polyset._edge_box_tree()
        tol2 = float(tol) * float(tol)
        near: list[np.ndarray] = []
        budget = _NEAR_BOUNDARY_PAIRS_PER_SLICE
        step = min(rows.size, _NEAR_BOUNDARY_POINTS_PER_SLICE)
        if rows.size > _NEAR_BOUNDARY_PROBE_POINTS:
            # Size the slices from a sample's pairs per point so the tree's
            # 16-byte-per-pair output never has to be discarded and redone.
            sample = rows[:: max(1, rows.size // _NEAR_BOUNDARY_PROBE_POINTS)]
            density = tree.query(shapely.points(pts64[sample]))[0].size / sample.size
            step = int(min(step, max(1.0, budget / max(density, 1e-3))))
        i = 0
        while i < rows.size:
            r = rows[i : i + step]
            p = pts64[r]
            pt_i, seg_i = tree.query(shapely.points(p))
            if pt_i.size > 2 * budget and r.size > 1:
                # Far denser than the probe suggested: retry the slice shorter.
                step = max(1, r.size // 2)
                continue
            for c0 in range(0, pt_i.size, _CPU_ROWS_PER_CHUNK):
                sl = slice(c0, c0 + _CPU_ROWS_PER_CHUNK)
                pi, e = pt_i[sl], edges[seg_i[sl]]
                px, py = p[pi, 0], p[pi, 1]
                dx, dy = e[:, 2] - e[:, 0], e[:, 3] - e[:, 1]
                l2 = dx * dx + dy * dy
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = ((px - e[:, 0]) * dx + (py - e[:, 1]) * dy) / l2
                t = np.where(l2 > 0.0, np.clip(t, 0.0, 1.0), 0.0)
                qx = e[:, 0] + t * dx
                qy = e[:, 1] + t * dy
                d2 = (px - qx) ** 2 + (py - qy) ** 2
                hit = pi[d2 <= tol2]
                if hit.size:
                    near.append(r[np.unique(hit)])
            i += r.size
        if not near:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(near))


def _empty_result(all_matches: bool, on_device: bool, n: int):
    """Typed empty/no-match results in the caller's container."""
    if all_matches:
        if on_device:
            return (mx.zeros((0,), dtype=mx.int64), mx.zeros((0,), dtype=mx.int64))
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if on_device:
        return mx.full((n,), -1, dtype=mx.int32)
    return np.full(n, -1, dtype=np.int32)


def _to_container(out, on_device: bool):
    """Output container follows input: MLX for device inputs, NumPy otherwise."""
    if isinstance(out, tuple):
        return tuple(_to_container(o, on_device) for o in out)
    if on_device:
        return out if isinstance(out, mx.array) else mx.array(out)
    return out if isinstance(out, np.ndarray) else np.asarray(out)


def _point_count(points) -> int:
    if isinstance(points, LocalPoints):
        return len(points)
    if isinstance(points, mx.array):
        if points.ndim == 1 and points.shape[0] == 2:
            return 1
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"expected points of shape (N, 2), got {tuple(points.shape)}"
            )
        return int(points.shape[0])
    return len(as_point_array(points))


def point_in_polygon(
    points,
    polygons=None,
    *,
    index: GridIndex | None = None,
    all_matches: bool = False,
    backend: str | None = None,
    predicate: str = "intersects",
    exact_boundary: bool = False,
):
    """First containing polygon per point (``-1`` for none), or all pairs.

    ``points`` may be a numpy array or sequence of ``(lon, lat)`` pairs,
    shapely Points, a GeoSeries/GeoDataFrame, an ``mx.array``, or
    :class:`~mlx_geospatial.LocalPoints` from :meth:`PolygonSet.to_local`. The
    output container follows the input: host inputs return NumPy (an int32
    label array, or two int64 pair arrays with ``all_matches=True``); MLX
    inputs return MLX arrays computed entirely on the GPU.

    ``polygons`` may be shapely geometries, a GeoSeries/GeoDataFrame, or a
    prebuilt :class:`~mlx_geospatial.PolygonSet` (reused across queries). The
    grid index is built lazily and cached on the polygon set. Alternatively
    pass a prebuilt ``index=`` (and then either omit ``polygons`` or pass the
    exact PolygonSet the index wraps). Containment is the fp32 even-odd rule
    on every backend; see :meth:`GridIndex.query` for ``predicate`` (accepted
    for GeoPandas compatibility; both values agree except for points exactly on
    a boundary, inside the fp32 tolerance band) and ``exact_boundary`` (re-check
    near-boundary points with GEOS so ``predicate`` is honored exactly).
    """
    if index is not None:
        if polygons is not None and polygons is not index.polygons:
            raise ValueError(
                "point_in_polygon got both polygons and index=, and the index "
                "wraps a different PolygonSet; pass one or the other"
            )
        return index.query(
            points,
            all_matches=all_matches,
            backend=backend,
            predicate=predicate,
            exact_boundary=exact_boundary,
            _stacklevel=5,
        )
    if polygons is None:
        raise ValueError("point_in_polygon requires polygons (or a prebuilt index=)")
    if not isinstance(polygons, PolygonSet):
        polygon_values = (
            polygons.geometry
            if hasattr(polygons, "geometry") and not isinstance(polygons, shapely.Geometry)
            else polygons
        )
        try:
            n_polygons = len(polygon_values)
        except TypeError:
            n_polygons = None
        if n_polygons == 0:
            if predicate not in ("within", "intersects"):
                raise ValueError(
                    f"predicate must be 'within' or 'intersects', got {predicate!r}"
                )
            if exact_boundary and isinstance(points, (mx.array, LocalPoints)):
                raise ValueError(
                    "exact_boundary=True needs host coordinates (GEOS re-checks the "
                    "near-boundary points); pass the original NumPy/GeoSeries points"
                )
            point_crs = _crs_of(points)
            # Match PolygonSet's non-empty path: CRS belongs to the original
            # GeoSeries/GeoDataFrame-like container, while polygon_values is
            # only the normalized object used for the zero-length check.
            polygon_crs = _crs_of(polygons)
            if (
                point_crs is not None
                and polygon_crs is not None
                and point_crs != polygon_crs
            ):
                raise ValueError(
                    f"CRS mismatch: points={point_crs}, polygons={polygon_crs}"
                )
            # A zero length alone does not make an object a polygon
            # collection: strings, mappings and malformed empty ndarray
            # shapes also report len() == 0.  Apply the same container-shape
            # normalization as PolygonSet while retaining the original object
            # above for CRS discovery.
            _as_geometry_array(polygons)
            n_points = _point_count(points)
            resolve_backend(backend, 0)
            return _empty_result(
                all_matches, isinstance(points, (mx.array, LocalPoints)), n_points
            )
    polyset = polygons if isinstance(polygons, PolygonSet) else PolygonSet(polygons)
    if polyset._grid_index is None:
        polyset._grid_index = GridIndex(polyset)
    return polyset._grid_index.query(
        points,
        all_matches=all_matches,
        backend=backend,
        predicate=predicate,
        exact_boundary=exact_boundary,
        _stacklevel=5,
    )


def sjoin(
    left,
    right,
    *,
    how: str = "inner",
    predicate: str = "intersects",
    exact_boundary: bool = False,
    backend: str | None = None,
):
    """Spatial join: points in ``left`` matched to containing polygons in ``right``.

    A GeoPandas-flavored convenience wrapper around :func:`point_in_polygon`.
    ``left`` is a (Geo)DataFrame/Series of points, ``right`` of polygons; a
    plain pandas DataFrame with a ``geometry`` column of shapely objects works
    too. Returns the left rows with an ``index_right`` column plus the right
    table's non-geometry columns (suffixed ``_right`` on collision). Each
    point matches at most one polygon (the lowest polygon index wins when
    polygons overlap). ``predicate`` defaults to ``"intersects"`` like
    GeoPandas; ``"within"`` would exclude points exactly on a polygon
    boundary, a distinction inside the fp32 tolerance band that the default
    even-odd rule leaves undefined. Pass ``exact_boundary=True`` to re-check
    near-boundary points with GEOS so the predicate is honored exactly (see
    :meth:`GridIndex.query`). ``backend`` forces the execution path as in
    :func:`point_in_polygon`.
    """
    import pandas as pd

    if how not in ("inner", "left"):
        raise ValueError(f"how must be 'inner' or 'left', got {how!r}")
    if predicate not in ("within", "intersects"):
        raise ValueError(f"predicate must be 'within' or 'intersects', got {predicate!r}")
    if isinstance(left, pd.DataFrame) and "index_right" in left.columns:
        # Assigning the output column would silently destroy the user's data.
        raise ValueError(
            "'index_right' cannot be a column in the left frame; rename it "
            "before joining (sjoin stores the matched right labels there)"
        )

    left_geom = left.geometry if hasattr(left, "geometry") else left
    right_geom = right.geometry if hasattr(right, "geometry") else right
    left_crs = getattr(left_geom, "crs", None)
    right_crs = getattr(right_geom, "crs", None)
    if left_crs is not None and right_crs is not None and left_crs != right_crs:
        raise ValueError(f"CRS mismatch: left={left_crs}, right={right_crs}")

    n_right = len(right_geom) if hasattr(right_geom, "__len__") else 1
    if n_right == 0:
        # PolygonSet requires at least one geometry; a zero-polygon join is
        # simply a join with no matches. Argument validation must not depend
        # on data shape, so backend= still goes through resolve_backend.
        resolve_backend(backend, 0)
        _as_geometry_array(right)
        assigned = np.full(len(as_point_array(left_geom)), -1, dtype=np.int32)
    else:
        assigned = np.asarray(
            point_in_polygon(
                left_geom,
                right_geom,
                predicate=predicate,
                exact_boundary=exact_boundary,
                backend=backend,
            )
        )
    matched = assigned >= 0

    if isinstance(left, pd.DataFrame):
        base = left
    else:
        candidate_index = getattr(left, "index", None)
        index = (
            candidate_index
            if callable(getattr(candidate_index, "to_numpy", None))
            else pd.RangeIndex(len(assigned))
        )
        base = pd.DataFrame({"geometry": np.asarray(left_geom, dtype=object)}, index=index)

    candidate_index = getattr(right, "index", None)
    right_index = (
        candidate_index
        if callable(getattr(candidate_index, "to_numpy", None))
        else pd.RangeIndex(len(right_geom))
    )
    right_labels = right_index.to_numpy()

    if how == "inner":
        out = base.iloc[np.flatnonzero(matched)].copy()
        out["index_right"] = right_labels[assigned[matched]]
    else:
        out = base.copy()
        col = np.full(len(assigned), np.nan, dtype=object)
        col[matched] = right_labels[assigned[matched]]
        out["index_right"] = col

    if isinstance(right, pd.DataFrame):
        right_cols = right
        geom_name = getattr(getattr(right, "geometry", None), "name", None)
        if geom_name is None and "geometry" in right.columns:
            geom_name = "geometry"
        if geom_name is not None and geom_name in right_cols.columns:
            right_cols = right_cols.drop(columns=[geom_name])
        if len(right_cols.columns):
            # Attach right columns POSITIONALLY. A label-based join on
            # index_right fans rows out when right index labels repeat and
            # attaches attributes from polygons that don't contain the point.
            if how == "inner":
                pos = assigned[matched]
                keep = None
            else:
                pos = np.where(matched, assigned, 0)
                keep = matched
            existing = set(out.columns)
            for col in right_cols.columns:
                name = col
                while name in existing:  # keep suffixing until unique
                    name = f"{name}_right"
                existing.add(name)
                rvals = right_cols[col].to_numpy()
                if len(rvals):
                    vals = rvals[pos]
                    if keep is not None:
                        vals = pd.Series(vals).where(pd.Series(keep)).to_numpy()
                else:  # zero right rows: nothing matched, only NaN to carry
                    vals = np.full(len(pos), np.nan, dtype=object)
                out[name] = vals
    return out
