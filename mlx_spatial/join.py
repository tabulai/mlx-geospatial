"""Grid-indexed point-to-polygon spatial join.

The index is a uniform grid over the polygon set's bounds. Each cell holds
the (sorted) list of polygons whose bbox intersects it, in CSR form. A query
point maps to exactly one cell; its candidate polygons are that cell's list,
so candidate pairs expand with plain prefix sums and the Metal crossing-number
kernel refines them. Grid planning runs in float64 on original coordinates,
so cell assignment can never disagree with the (widened-bbox) kernel test.

Small workloads (fewer candidate pairs than
``mlx_spatial._dispatch.GPU_MIN_CANDIDATE_PAIRS``) automatically fall back to
a shapely STRtree, which beats kernel-launch overhead there.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import shapely

from ._dispatch import resolve_backend
from ._ingest import PolygonSet
from ._pip import pip_pairs

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

        bboxes = polyset.bboxes  # (M, 4) float64, original coordinates
        finite = np.isfinite(bboxes).all(axis=1)
        if not finite.any():
            raise ValueError("GridIndex requires at least one non-empty polygon")
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

    def _candidates(self, pts64: np.ndarray):
        """Cell lookup: per-point candidate counts and CSR start offsets."""
        n = len(pts64)
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
        return counts, self._cell_offsets[cell_id], int(counts.sum()), n

    def query(
        self,
        points,
        *,
        all_matches: bool = False,
        backend: str | None = None,
        max_pairs_per_batch: int = 16_000_000,
    ):
        """Assign each point to a containing polygon.

        Returns an ``(N,)`` int32 array with the lowest index of a containing
        polygon, or ``-1``. With ``all_matches=True`` returns
        ``(point_indices, polygon_indices)`` for every containment pair,
        sorted by point then polygon.
        """
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

        counts, cand_starts, total_pairs, _ = self._candidates(pts64)
        chosen = resolve_backend(backend, total_pairs)
        if chosen == "cpu":
            return self._query_cpu(pts64, all_matches=all_matches)

        if total_pairs == 0:
            if all_matches:
                return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
            return np.full(n, -1, dtype=np.int32)

        # Expand candidate pairs slice-by-slice over point ranges so HOST
        # memory is bounded by the same knob as the GPU batch — a full upfront
        # expansion costs ~35 bytes per candidate pair. A single point whose
        # candidate list alone exceeds the budget still runs as one batch
        # (its candidates are at most one cell's polygon list).
        cum = np.concatenate([[0], np.cumsum(counts)])  # (n + 1,)
        result = np.full(n, _INT32_MAX, dtype=np.int32)
        matches_pt: list[np.ndarray] = []
        matches_pg: list[np.ndarray] = []
        p0 = 0
        while p0 < n:
            p1 = int(np.searchsorted(cum, cum[p0] + max_pairs_per_batch, side="right")) - 1
            p1 = min(max(p1, p0 + 1), n)
            slice_counts = counts[p0:p1]
            slice_total = int(cum[p1] - cum[p0])
            if slice_total == 0:
                p0 = p1
                continue
            pair_point = np.repeat(np.arange(p0, p1, dtype=np.int64), slice_counts)
            local_starts = cum[p0:p1] - cum[p0]
            rank = np.arange(slice_total, dtype=np.int64) - np.repeat(
                local_starts, slice_counts
            )
            pair_poly = self._cell_poly[
                np.repeat(cand_starts[p0:p1], slice_counts) + rank
            ]
            flags = np.array(
                pip_pairs(
                    pts_local,
                    polyset,
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
            p0 = p1

        if all_matches:
            if matches_pt:
                return np.concatenate(matches_pt), np.concatenate(matches_pg)
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        result[result == _INT32_MAX] = -1
        return result

    def _query_cpu(self, pts64: np.ndarray, *, all_matches: bool):
        """shapely STRtree fallback with matching first-match semantics."""
        tree = self.polygons.strtree
        pt_idx, poly_idx = tree.query(shapely.points(pts64), predicate="intersects")
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
):
    """First containing polygon per point (``-1`` for none), or all pairs.

    ``polygons`` may be shapely geometries, a GeoSeries/GeoDataFrame, or a
    prebuilt :class:`~mlx_spatial.PolygonSet` (reused across queries). The
    grid index is built lazily and cached on the polygon set. Alternatively
    pass a prebuilt ``index=`` (and then either omit ``polygons`` or pass the
    exact PolygonSet the index wraps).
    """
    if index is not None:
        if polygons is not None and polygons is not index.polygons:
            raise ValueError(
                "point_in_polygon got both polygons and index=, and the index "
                "wraps a different PolygonSet; pass one or the other"
            )
        return index.query(points, all_matches=all_matches, backend=backend)
    if polygons is None:
        raise ValueError("point_in_polygon requires polygons (or a prebuilt index=)")
    polyset = polygons if isinstance(polygons, PolygonSet) else PolygonSet(polygons)
    if polyset._grid_index is None:
        polyset._grid_index = GridIndex(polyset)
    return polyset._grid_index.query(points, all_matches=all_matches, backend=backend)


def sjoin(left, right, *, how: str = "inner", predicate: str = "within"):
    """Spatial join: points in ``left`` matched to containing polygons in ``right``.

    A GeoPandas-flavored convenience wrapper around :func:`point_in_polygon`.
    ``left`` is a (Geo)DataFrame/Series of points, ``right`` of polygons; a
    plain pandas DataFrame with a ``geometry`` column of shapely objects works
    too. Returns the left rows with an ``index_right`` column plus the right
    table's non-geometry columns (suffixed ``_right`` on collision). Each
    point matches at most one polygon (the lowest polygon index wins when
    polygons overlap).
    """
    import pandas as pd

    if how not in ("inner", "left"):
        raise ValueError(f"how must be 'inner' or 'left', got {how!r}")
    if predicate not in ("within", "intersects"):
        raise ValueError(f"predicate must be 'within' or 'intersects', got {predicate!r}")

    left_geom = left.geometry if hasattr(left, "geometry") else left
    right_geom = right.geometry if hasattr(right, "geometry") else right
    left_crs = getattr(left_geom, "crs", None)
    right_crs = getattr(right_geom, "crs", None)
    if left_crs is not None and right_crs is not None and left_crs != right_crs:
        raise ValueError(f"CRS mismatch: left={left_crs}, right={right_crs}")

    assigned = point_in_polygon(left_geom, right_geom)
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
                vals = right_cols[col].to_numpy()[pos]
                if keep is not None:
                    vals = pd.Series(vals).where(pd.Series(keep)).to_numpy()
                out[name] = vals
    return out
