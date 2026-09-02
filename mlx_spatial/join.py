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
:class:`~mlx_spatial.LocalPoints`) therefore never touch the host, and the
output container follows the input container.

Small host-resident workloads (fewer candidate pairs than
``mlx_spatial._dispatch.GPU_MIN_CANDIDATE_PAIRS``) automatically fall back to
a shapely STRtree, which beats kernel-launch overhead there; device-resident
inputs run on the GPU whenever Metal is available.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import shapely

from ._dispatch import GPU_MIN_CANDIDATE_PAIRS, resolve_backend
from ._ingest import LocalPoints, PolygonSet, as_point_array
from ._pip import (
    _ALL_MATCHES_EDGE_WORK_THRESHOLD,
    _PIP_LONG_RING_THRESHOLD,
    pip_all_matches,
    pip_first_match,
    pip_pairs,
)

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

# Host queries small enough that settling an undecided auto dispatch with the
# float64 host cell lookup is cheaper than a device round trip.
_HOST_DISPATCH_MAX_POINTS = 65_536


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
        self._tables_mx = None
        self._max_cell_count = 0

        bboxes = polyset.bboxes  # (M, 4) float64, original coordinates
        finite = np.isfinite(bboxes).all(axis=1)
        if not finite.any():
            # Every geometry is empty: a degenerate grid whose bounds exclude
            # all points, so every query yields zero candidates (and the CPU
            # fallback's STRtree finds nothing either).
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
            return
        fb = bboxes[finite]
        self.x0 = float(fb[:, 0].min())
        self.y0 = float(fb[:, 1].min())
        x1 = float(fb[:, 2].max())
        y1 = float(fb[:, 3].max())
        self.x1, self.y1 = x1, y1
        width = max(x1 - self.x0, 0.0)
        height = max(y1 - self.y0, 0.0)

        if cell_size is None:
            spans = np.maximum(fb[:, 2] - fb[:, 0], fb[:, 3] - fb[:, 1])
            cell_size = float(np.median(spans))
            if cell_size <= 0.0:
                cell_size = max(width, height) / 64.0
            if cell_size <= 0.0:
                cell_size = 1.0
        cell_size = float(cell_size)
        if cell_size <= 0.0 or not np.isfinite(cell_size):
            raise ValueError("cell_size must be positive and finite")
        # Guard the coarsening loop against float overflow (width / cell_size
        # -> inf) when the cell is absurdly small relative to the extent.
        min_cell = max(width, height, 1.0) * 1e-12
        cell_size = max(cell_size, min_cell)

        # Coarsen until the grid and its CSR entry count are within budget.
        # Polygon cell ranges are padded by twice the boundary tolerance
        # (16 ulps of the local extent): more than the kernel's 4.5-ulp bbox
        # widening plus the fp32 cell-assignment error on the device (under
        # 8 ulps: the rounded, outward-nudged grid corner, the subtraction,
        # and the multiply by inv_cell). Whichever cell a point near a cell
        # line lands in therefore
        # lists every polygon whose kernel bbox accepts it, and the answer is
        # determined by the kernel alone. Bboxes that end exactly on a cell
        # line (adjacent tilings) are listed on both sides.
        pad = 2.0 * polyset.boundary_tolerance
        finite_idx = np.flatnonzero(finite)
        for _ in range(64):
            nx = max(1, int(np.ceil(width / cell_size)) if width > 0 else 1)
            ny = max(1, int(np.ceil(height / cell_size)) if height > 0 else 1)
            # Cell ids are part of the public/internal int32 query contract.
            # Coarsen even if a caller supplies a larger max_cells budget.
            if nx * ny > max_cells or nx * ny > _INT32_MAX:
                cell_size *= 2.0
                continue
            ix0 = np.clip(np.floor((fb[:, 0] - pad - self.x0) / cell_size), 0, nx - 1).astype(np.int64)
            iy0 = np.clip(np.floor((fb[:, 1] - pad - self.y0) / cell_size), 0, ny - 1).astype(np.int64)
            ix1 = np.clip(np.floor((fb[:, 2] + pad - self.x0) / cell_size), 0, nx - 1).astype(np.int64)
            iy1 = np.clip(np.floor((fb[:, 3] + pad - self.y0) / cell_size), 0, ny - 1).astype(np.int64)
            cells_per_poly = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
            total = int(cells_per_poly.sum())
            if total > max_entries:
                cell_size *= 2.0
                continue
            break
        else:
            raise ValueError("could not build a grid within the memory budget")

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

        # Cheap upper bound on candidate pairs for small host queries.
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

        This is the reference cell assignment (tests compare the device graph
        against it) and feeds nothing on the query path any more: queries
        assign cells on the GPU through :meth:`_device_candidates`.
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

    def _device_grid(self) -> tuple:
        """Lazily cache the fp32 local-frame grid parameters as 0-d arrays.

        Every per-index scalar travels as an ``mx.array``: Python constants
        would be baked into the compiled graph's Metal source and force a
        fresh JIT for every index. The outer bounds are rounded outward so the
        fp32 in-grid test admits every point the float64 test admits.
        """
        if self._grid_params_mx is None:
            ox, oy = self.polygons.origin
            f32 = np.float32
            neg_inf, pos_inf = f32(-np.inf), f32(np.inf)
            cell_offsets_mx, _ = self._device_csr()
            self._grid_params_mx = (
                mx.array(np.nextafter(f32(self.x0 - ox), neg_inf)),
                mx.array(np.nextafter(f32(self.y0 - oy), neg_inf)),
                mx.array(np.nextafter(f32(self.x1 - ox), pos_inf)),
                mx.array(np.nextafter(f32(self.y1 - oy), pos_inf)),
                mx.array(f32(1.0 / self.cell_size)),
                mx.array(self.nx, dtype=mx.int32),
                mx.array(self.nx - 1, dtype=mx.int32),
                mx.array(self.ny - 1, dtype=mx.int32),
                cell_offsets_mx,
            )
        return self._grid_params_mx

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
        max_pairs_per_batch: int = 16_000_000,
        _stacklevel: int = 4,
    ):
        """Assign each point to a containing polygon.

        Returns the lowest index of a containing polygon per point (``-1``
        for none) as an ``(N,)`` int32 array, or with ``all_matches=True``
        the ``(point_indices, polygon_indices)`` int64 arrays of every
        containment pair, sorted by point then polygon.

        The output container follows the input: numpy arrays, sequences,
        shapely points and GeoSeries return NumPy arrays; ``mx.array`` and
        :class:`~mlx_spatial.LocalPoints` inputs return MLX arrays and the
        whole query, cell assignment included, runs on the GPU without a host
        round trip. Device-resident first-match results are lazy when one
        kernel launch suffices (the call itself does not synchronize); call
        ``mx.eval`` when you need the values.

        Backend: host inputs dispatch on the candidate count (small workloads
        go to shapely); device inputs run on the GPU whenever Metal is
        available, since the fallback would need a host copy. ``backend=`` and
        ``MLX_SPATIAL_BACKEND`` override both.

        ``max_pairs_per_batch`` is a hard upper bound on each GPU kernel
        launch, including when one point alone has more candidates than the
        requested batch size.

        ``predicate`` is ``"intersects"`` (boundary counts as a match) or
        ``"within"`` (it does not). The two differ only for points exactly on
        a polygon boundary — inside the documented fp32 tolerance band, where
        the GPU result is undefined anyway — so the GPU kernel serves both;
        the CPU fallback honors the predicate exactly via GEOS. For
        device-resident inputs the CPU path sees float32 local coordinates
        plus the origin, i.e. it is exact only up to that same band.
        """
        if predicate not in ("within", "intersects"):
            raise ValueError(
                f"predicate must be 'within' or 'intersects', got {predicate!r}"
            )
        max_pairs_per_batch = int(max_pairs_per_batch)
        if not 1 <= max_pairs_per_batch < 2**31:
            raise ValueError(
                f"max_pairs_per_batch must be in [1, 2**31), got {max_pairs_per_batch}"
            )
        polyset = self.polygons
        pts64, local, on_device = polyset._ingest_points(points, stacklevel=_stacklevel)
        n = int(local.shape[0])
        if n >= 2**31:
            raise ValueError("more than 2^31 points per query is not supported")
        if n == 0:
            resolve_backend(backend, 0)
            return _empty_result(all_matches, on_device, 0)

        # Resolve the backend before touching the device where the answer does
        # not depend on the candidate count: explicit or env requests, and
        # Metal-less hosts, give the same result for 0 and threshold pairs.
        low = resolve_backend(backend, 0)
        high = resolve_backend(backend, GPU_MIN_CANDIDATE_PAIRS)
        bound = n * self._max_cell_count  # >= the candidate-pair total
        if low == high:
            chosen = low
        elif on_device:
            # auto with device-resident input: the shapely fallback would need
            # a host copy, so stay on the GPU.
            chosen = "gpu"
        elif bound < GPU_MIN_CANDIDATE_PAIRS:
            # auto: the cheap upper bound already settles small host queries.
            chosen = "cpu"
        elif n <= _HOST_DISPATCH_MAX_POINTS:
            # auto, undecided by the bound, but small: the float64 host lookup
            # is cheaper than a device round trip, so settle the count there.
            chosen = resolve_backend(
                backend, self._candidates(pts64, include_starts=False)[2]
            )
        else:
            chosen = None

        if chosen == "cpu":
            if pts64 is None:
                pts64 = polyset._host_coords(local)
            out = self._query_cpu(pts64, all_matches=all_matches, predicate=predicate)
            return _to_container(out, on_device)

        if chosen == "gpu" and not all_matches and bound <= max_pairs_per_batch:
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
            return _to_container(out, on_device)

        cells = self._device_candidates(local)
        if chosen is None:
            chosen = resolve_backend(backend, cells[2])
            if chosen == "cpu":
                out = self._query_cpu(pts64, all_matches=all_matches, predicate=predicate)
                return _to_container(out, on_device)

        cell, counts, total_pairs, max_count = cells
        if total_pairs == 0:
            return _empty_result(all_matches, on_device, n)
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
        return _to_container(out, on_device)

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
        self, pts64: np.ndarray, *, all_matches: bool, predicate: str = "intersects"
    ):
        """shapely STRtree fallback with matching first-match semantics."""
        tree = self.polygons.strtree
        pt_idx, poly_idx = tree.query(shapely.points(pts64), predicate=predicate)
        if all_matches:
            order = np.lexsort((poly_idx, pt_idx))
            return pt_idx[order].astype(np.int64), poly_idx[order].astype(np.int64)
        result = np.full(len(pts64), _INT32_MAX, dtype=np.int32)
        np.minimum.at(result, pt_idx, poly_idx.astype(np.int32))
        result[result == _INT32_MAX] = -1
        return result


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
):
    """First containing polygon per point (``-1`` for none), or all pairs.

    ``points`` may be a numpy array or sequence of ``(lon, lat)`` pairs,
    shapely Points, a GeoSeries/GeoDataFrame, an ``mx.array``, or
    :class:`~mlx_spatial.LocalPoints` from :meth:`PolygonSet.to_local`. The
    output container follows the input: host inputs return NumPy (an int32
    label array, or two int64 pair arrays with ``all_matches=True``); MLX
    inputs return MLX arrays computed entirely on the GPU.

    ``polygons`` may be shapely geometries, a GeoSeries/GeoDataFrame, or a
    prebuilt :class:`~mlx_spatial.PolygonSet` (reused across queries). The
    grid index is built lazily and cached on the polygon set. Alternatively
    pass a prebuilt ``index=`` (and then either omit ``polygons`` or pass the
    exact PolygonSet the index wraps). See :meth:`GridIndex.query` for the
    ``predicate`` semantics (they differ only on boundary points, inside the
    fp32 tolerance band).
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
        _stacklevel=5,
    )


def sjoin(
    left,
    right,
    *,
    how: str = "inner",
    predicate: str = "within",
    backend: str | None = None,
):
    """Spatial join: points in ``left`` matched to containing polygons in ``right``.

    A GeoPandas-flavored convenience wrapper around :func:`point_in_polygon`.
    ``left`` is a (Geo)DataFrame/Series of points, ``right`` of polygons; a
    plain pandas DataFrame with a ``geometry`` column of shapely objects works
    too. Returns the left rows with an ``index_right`` column plus the right
    table's non-geometry columns (suffixed ``_right`` on collision). Each
    point matches at most one polygon (the lowest polygon index wins when
    polygons overlap). ``predicate`` follows GeoPandas: ``"within"`` excludes
    points exactly on a polygon boundary, ``"intersects"`` includes them —
    a distinction inside the fp32 tolerance band, honored exactly only on the
    CPU path (see :meth:`GridIndex.query`). ``backend`` forces the execution
    path as in :func:`point_in_polygon`.
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
        assigned = np.full(len(as_point_array(left_geom)), -1, dtype=np.int32)
    else:
        assigned = np.asarray(
            point_in_polygon(left_geom, right_geom, predicate=predicate, backend=backend)
        )
    matched = assigned >= 0

    if isinstance(left, pd.DataFrame):
        base = left
    else:
        index = getattr(left, "index", pd.RangeIndex(len(assigned)))
        base = pd.DataFrame({"geometry": np.asarray(left_geom, dtype=object)}, index=index)

    right_index = right.index if hasattr(right, "index") else pd.RangeIndex(len(right_geom))
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
