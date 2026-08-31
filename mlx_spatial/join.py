"""Grid-indexed point-to-polygon spatial join.

The index is a uniform grid over the polygon set's bounds. Each cell holds
the (sorted) list of polygons whose bbox intersects it, in CSR form. A query
point maps to exactly one cell; its candidate polygons are that cell's list,
so the first-match Metal kernel walks that list directly and returns on its
first containment hit. All-match queries retain bounded explicit candidate
pairs. Grid planning runs in float64 on original coordinates, so cell
assignment can never disagree with the (widened-bbox) kernel test.

Small workloads (fewer candidate pairs than
``mlx_spatial._dispatch.GPU_MIN_CANDIDATE_PAIRS``) automatically fall back to
a shapely STRtree, which beats kernel-launch overhead there.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import shapely

from ._dispatch import resolve_backend
from ._ingest import PolygonSet, as_point_array
from ._pip import pip_first_match, pip_pairs

__all__ = ["GridIndex", "point_in_polygon", "sjoin"]

_INT32_MAX = np.iinfo(np.int32).max


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
        finite_idx = np.flatnonzero(finite)
        for _ in range(64):
            nx = max(1, int(np.ceil(width / cell_size)) if width > 0 else 1)
            ny = max(1, int(np.ceil(height / cell_size)) if height > 0 else 1)
            if nx * ny > max_cells:
                cell_size *= 2.0
                continue
            ix0 = np.clip(np.floor((fb[:, 0] - self.x0) / cell_size), 0, nx - 1).astype(np.int64)
            iy0 = np.clip(np.floor((fb[:, 1] - self.y0) / cell_size), 0, ny - 1).astype(np.int64)
            ix1 = np.clip(np.floor((fb[:, 2] - self.x0) / cell_size), 0, nx - 1).astype(np.int64)
            iy1 = np.clip(np.floor((fb[:, 3] - self.y0) / cell_size), 0, ny - 1).astype(np.int64)
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

        # Ragged cartesian expansion: every (polygon, covered-cell) entry.
        poly_rep = np.repeat(finite_idx, cells_per_poly)
        starts = np.concatenate([[0], np.cumsum(cells_per_poly)[:-1]])
        rank = np.arange(total, dtype=np.int64) - np.repeat(starts, cells_per_poly)
        dy = iy1 - iy0 + 1
        dy_rep = np.repeat(dy, cells_per_poly)
        local_x = rank // dy_rep
        local_y = rank % dy_rep
        cell_id = (np.repeat(iy0, cells_per_poly) + local_y) * nx + (
            np.repeat(ix0, cells_per_poly) + local_x
        )

        # Stable sort by cell keeps polygon ids ascending within each cell.
        order = np.argsort(cell_id, kind="stable")
        self._cell_poly = poly_rep[order].astype(np.int32)
        counts = np.bincount(cell_id[order], minlength=nx * ny)
        self._cell_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def __repr__(self) -> str:
        return (
            f"GridIndex(nx={self.nx}, ny={self.ny}, cell_size={self.cell_size:g}, "
            f"entries={len(self._cell_poly)}, polygons={self.polygons.n_polygons})"
        )

    def _candidates(self, pts64: np.ndarray, *, include_starts: bool = True):
        """Cell lookup: candidate counts, CSR starts, total, and cell IDs."""
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

    def query(
        self,
        points,
        *,
        all_matches: bool = False,
        backend: str | None = None,
        predicate: str = "intersects",
        max_pairs_per_batch: int = 16_000_000,
    ):
        """Assign each point to a containing polygon.

        Returns an ``(N,)`` int32 array with the lowest index of a containing
        polygon, or ``-1``. With ``all_matches=True`` returns
        ``(point_indices, polygon_indices)`` for every containment pair,
        sorted by point then polygon.

        ``max_pairs_per_batch`` is a hard upper bound on each GPU kernel
        launch, including when one point alone has more candidates than the
        requested batch size.

        ``predicate`` is ``"intersects"`` (boundary counts as a match) or
        ``"within"`` (it does not). The two differ only for points exactly on
        a polygon boundary — inside the documented fp32 tolerance band, where
        the GPU result is undefined anyway — so the GPU kernel serves both;
        the CPU fallback honors the predicate exactly via GEOS.
        """
        if predicate not in ("within", "intersects"):
            raise ValueError(
                f"predicate must be 'within' or 'intersects', got {predicate!r}"
            )
        polyset = self.polygons
        pts64, pts_local = polyset.recenter_points(points)
        n = len(pts64)
        if n >= 2**31:
            raise ValueError("more than 2^31 points per query is not supported")

        max_pairs_per_batch = int(max_pairs_per_batch)
        if not 1 <= max_pairs_per_batch < 2**31:
            raise ValueError(
                f"max_pairs_per_batch must be in [1, 2**31), got {max_pairs_per_batch}"
            )

        counts, cand_starts, total_pairs, point_cell = self._candidates(
            pts64, include_starts=all_matches
        )
        chosen = resolve_backend(backend, total_pairs)
        if chosen == "cpu":
            del point_cell
            return self._query_cpu(pts64, all_matches=all_matches, predicate=predicate)

        if total_pairs == 0:
            if all_matches:
                return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
            return np.full(n, -1, dtype=np.int32)

        if not all_matches and int(counts.max(initial=0)) <= max_pairs_per_batch:
            return self._query_gpu_first_match(
                pts_local,
                point_cell,
                counts,
                total_pairs=total_pairs,
                max_pairs_per_batch=max_pairs_per_batch,
            )

        # all_matches requires compaction of every hit. A first-match query
        # also uses this bounded legacy path when one point alone has more
        # candidates than max_pairs_per_batch and therefore cannot fit in a
        # whole-point fused launch.
        if cand_starts is None:
            cand_starts = self._cell_offsets[np.maximum(point_cell, 0)]
        return self._query_gpu_pairs(
            pts_local,
            counts,
            cand_starts,
            total_pairs=total_pairs,
            all_matches=all_matches,
            max_pairs_per_batch=max_pairs_per_batch,
        )

    def _query_gpu_first_match(
        self,
        pts_local: mx.array,
        point_cell: np.ndarray,
        counts: np.ndarray,
        *,
        total_pairs: int,
        max_pairs_per_batch: int,
    ) -> np.ndarray:
        """Point-centric CSR traversal, batched by candidate-work budget."""
        cell_offsets_mx, cell_poly_mx = self._device_csr()
        point_cell_mx = mx.array(point_cell)
        n = len(point_cell)

        if total_pairs <= max_pairs_per_batch:
            return np.array(
                pip_first_match(
                    pts_local,
                    self.polygons,
                    point_cell_mx,
                    cell_offsets_mx,
                    cell_poly_mx,
                )
            )

        # Whole-point batches preserve the public hard cap on candidate work
        # per launch. The caller has already routed any individually oversized
        # point to the splittable explicit-pair path.
        cum = np.concatenate([[0], np.cumsum(counts)])
        result = np.empty(n, dtype=np.int32)
        point0 = 0
        while point0 < n:
            pair_limit = int(cum[point0]) + max_pairs_per_batch
            point1 = int(np.searchsorted(cum, pair_limit, side="right")) - 1
            if point1 <= point0:  # defensive: counts.max() was checked above
                raise RuntimeError("could not form a bounded fused point batch")
            result[point0:point1] = np.array(
                pip_first_match(
                    pts_local[point0:point1],
                    self.polygons,
                    point_cell_mx[point0:point1],
                    cell_offsets_mx,
                    cell_poly_mx,
                )
            )
            point0 = point1
        return result

    def _query_gpu_pairs(
        self,
        pts_local: mx.array,
        counts: np.ndarray,
        cand_starts: np.ndarray,
        *,
        total_pairs: int,
        all_matches: bool,
        max_pairs_per_batch: int,
    ):
        """Bounded explicit-pair GPU path for all hits and heavy cells."""
        n = len(counts)

        # Expand candidate pairs in contiguous ranges of the conceptual flat
        # pair array. Chunking by pair rank (rather than only by point ranges)
        # keeps max_pairs_per_batch a hard bound even when one point's cell
        # contains more polygons than the budget. Global pair order is point
        # then polygon, so concatenating per-chunk hits preserves the documented
        # all_matches ordering.
        cum = np.concatenate([[0], np.cumsum(counts)])  # (n + 1,)
        result = np.full(n, _INT32_MAX, dtype=np.int32)
        matches_pt: list[np.ndarray] = []
        matches_pg: list[np.ndarray] = []
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
            flags = np.array(
                pip_pairs(
                    pts_local,
                    self.polygons,
                    mx.array(pair_point.astype(np.int32)),
                    mx.array(pair_poly.astype(np.int32)),
                )
            )
            hit = flags.astype(bool)
            if all_matches:
                # pair_point ascending by construction; polygon ids ascending
                # within each point because cell lists are sorted.
                matches_pt.append(pair_point[hit])
                matches_pg.append(pair_poly[hit].astype(np.int64))
            else:
                np.minimum.at(result, pair_point[hit], pair_poly[hit].astype(np.int32))
            pair0 = pair1

        if all_matches:
            if matches_pt:
                return np.concatenate(matches_pt), np.concatenate(matches_pg)
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        result[result == _INT32_MAX] = -1
        return result

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
            points, all_matches=all_matches, backend=backend, predicate=predicate
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
            pts64 = as_point_array(points)
            resolve_backend(backend, 0)
            if all_matches:
                return (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                )
            return np.full(len(pts64), -1, dtype=np.int32)
    polyset = polygons if isinstance(polygons, PolygonSet) else PolygonSet(polygons)
    if polyset._grid_index is None:
        polyset._grid_index = GridIndex(polyset)
    return polyset._grid_index.query(
        points, all_matches=all_matches, backend=backend, predicate=predicate
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
        assigned = point_in_polygon(
            left_geom, right_geom, predicate=predicate, backend=backend
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
