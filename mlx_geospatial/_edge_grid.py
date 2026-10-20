"""Host float64 uniform grid over the tolerance-widened edge envelopes.

``exact_boundary=True`` re-checks with GEOS every finite point within
``PolygonSet.boundary_tolerance`` of an edge. :class:`HostEdgeGrid` finds
exactly the row set of the edge-box STRtree oracle
(``GridIndex._near_boundary``), not merely a superset:

* each edge's envelope uses the oracle's own float64 box expressions,
  ``min(x1, x2) - tol`` .. ``max(y1, y2) + tol``;
* envelope corners and query points map to cells through one float64
  formula, ``min(trunc((v - X0) * inv), n - 1)`` for ``v >= X0``. Subtracting
  a constant, multiplying by a positive constant, truncating and ``min`` are
  all monotone non-decreasing, so a point inside an edge's inclusive envelope
  lands in a cell between the envelope's corner cells, which lists the edge;
* each candidate pair re-applies the inclusive envelope test (GEOS envelope
  intersection), which reduces the candidates to exactly the tree's pairs,
  and the oracle's float64 projection then runs on them unchanged.

Nothing here depends on an fp32 argument. A query whose candidate pairs
exceed the caller's budget (strongly clustered layers, where a uniform grid
cannot separate the edges) returns ``None`` so the caller keeps the STRtree
oracle.
"""

from __future__ import annotations

import numpy as np

# Cell budget: ``sqrt(W * H / min(_TARGET_CELLS, E / 2))`` sets the side. The
# CSR keys sort in one uint16 radix pass up to 65,536 cells and in two beyond
# (thin layers whose extent is far longer than it is wide).
_TARGET_CELLS = 65_000
_MAX_CELLS = 1 << 20
# A cell side of three median envelope sides keeps most edges in at most
# 2 x 2 cells; coarsen while the expanded entries exceed this budget.
_SIDE_PER_MEDIAN = 3.0
_ENTRIES_PER_EDGE = 8
_ENTRIES_BASE = 65_536
_SIDE_SAMPLES = 4096
# Query: (point, edge) candidates per chunk (~120 bytes each at peak).
_PAIR_CHUNK = 2_000_000


def _cells(values: np.ndarray, origin: float, inv: float, n: int) -> np.ndarray:
    """Clipped cell coordinate of ``values >= origin`` (monotone in value)."""
    cell = ((values - origin) * inv).astype(np.int64)
    np.minimum(cell, n - 1, out=cell)
    return cell


def _stable_cell_order(cells: np.ndarray, n_cells: int) -> np.ndarray:
    """Stable argsort of nonnegative cell keys by uint16 LSD radix passes."""
    if n_cells <= 1 << 16:
        return np.argsort(cells.astype(np.uint16), kind="stable")
    order = np.argsort((cells & 0xFFFF).astype(np.uint16), kind="stable")
    high = (cells[order] >> 16).astype(np.uint16)
    return order[np.argsort(high, kind="stable")]


class HostEdgeGrid:
    """Uniform float64 grid of edge envelopes widened by ``tol`` (CSR form).

    ``edges`` is the ``(E, 4)`` float64 ``(x1, y1, x2, y2)`` table of
    ``PolygonSet._edges_f64`` and ``env`` the matching ``(xmin, ymin, xmax,
    ymax)`` envelopes. Cell ``c`` lists ``edge_ids[offsets[c]:offsets[c + 1]]``.
    """

    __slots__ = (
        "edges", "env", "tol", "x0", "y0", "x1", "y1", "inv", "nx", "ny",
        "offsets", "edge_ids",
    )

    @classmethod
    def build(cls, edges: np.ndarray, tol: float) -> HostEdgeGrid | None:
        """Build the grid, or ``None`` when no finite positive cell size fits."""
        tol = float(tol)
        n_edges = len(edges)
        if n_edges == 0 or not np.isfinite(tol):
            return None
        # The oracle's box arithmetic, verbatim (PolygonSet._edge_box_tree).
        env = np.empty((n_edges, 4), dtype=np.float64)
        np.subtract(np.minimum(edges[:, 0], edges[:, 2]), tol, out=env[:, 0])
        np.subtract(np.minimum(edges[:, 1], edges[:, 3]), tol, out=env[:, 1])
        np.add(np.maximum(edges[:, 0], edges[:, 2]), tol, out=env[:, 2])
        np.add(np.maximum(edges[:, 1], edges[:, 3]), tol, out=env[:, 3])
        x0, y0 = float(env[:, 0].min()), float(env[:, 1].min())
        x1, y1 = float(env[:, 2].max()), float(env[:, 3].max())
        width, height = x1 - x0, y1 - y0
        if not (np.isfinite(width) and np.isfinite(height)):
            return None

        sample = env[:: max(1, n_edges // _SIDE_SAMPLES)]
        side = np.maximum(sample[:, 2] - sample[:, 0], sample[:, 3] - sample[:, 1])
        target = min(_TARGET_CELLS, max(1.0, n_edges / 2))
        h = max(_SIDE_PER_MEDIAN * float(np.median(side)),
                float(np.sqrt(width * height / target)))
        budget = _ENTRIES_PER_EDGE * n_edges + _ENTRIES_BASE
        for _ in range(2100):  # a doubling sweep of the float64 range
            if not np.isfinite(h) or h <= 0.0:
                return None
            inv = 1.0 / h
            gx, gy = width * inv, height * inv
            if not (np.isfinite(inv) and inv > 0.0 and np.isfinite(gx) and np.isfinite(gy)):
                return None
            nx, ny = int(gx) + 1, int(gy) + 1
            if nx * ny > _MAX_CELLS:
                h *= 2.0
                continue
            cx0 = _cells(env[:, 0], x0, inv, nx)
            cy0 = _cells(env[:, 1], y0, inv, ny)
            sx = _cells(env[:, 2], x0, inv, nx)
            sx -= cx0
            sx += 1
            sy = _cells(env[:, 3], y0, inv, ny)
            sy -= cy0
            sy += 1
            total = int(np.dot(sx, sy))
            if total <= budget:
                break
            h *= 2.0
        else:
            return None

        base = cy0
        base *= nx
        base += cx0
        del cx0, cy0
        # Edges spanning at most 2 x 2 cells emit their cells directly; only
        # larger ranges expand through repeat/divmod.
        right = (sx == 2) & (sy <= 2)
        up = (sy == 2) & (sx <= 2)
        cells = [base]
        ids = [np.arange(n_edges, dtype=np.int32)]
        for mask, offset in ((right, 1), (up, nx), (right & up, nx + 1)):
            picked = np.flatnonzero(mask)
            if picked.size:
                cells.append(base[picked] + offset)
                ids.append(picked.astype(np.int32))
        large = np.flatnonzero((sx > 2) | (sy > 2))
        if large.size:
            extra = sx[large] * sy[large] - 1
            rep = np.repeat(large, extra)
            rank = np.arange(len(rep), dtype=np.int64)
            rank -= np.repeat(np.cumsum(extra) - extra, extra)
            rank += 1  # rank 0 is the base cell, already listed
            width_cells = sx[rep]
            row, col = np.divmod(rank, width_cells)
            row *= nx
            row += col
            row += base[rep]
            cells.append(row)
            ids.append(rep.astype(np.int32))
        cells = np.concatenate(cells) if len(cells) > 1 else cells[0]
        ids = np.concatenate(ids) if len(ids) > 1 else ids[0]
        n_cells = nx * ny
        order = _stable_cell_order(cells, n_cells)
        offsets = np.zeros(n_cells + 1, dtype=np.int64)
        np.cumsum(np.bincount(cells, minlength=n_cells), out=offsets[1:])

        self = cls.__new__(cls)
        self.edges, self.env, self.tol = edges, env, tol
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.inv, self.nx, self.ny = inv, nx, ny
        self.offsets, self.edge_ids = offsets, ids[order]
        return self

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def near_rows(self, pts64: np.ndarray, rows: np.ndarray | None = None,
                  max_pairs: int | None = None):
        """Sorted unique positions within ``tol`` of an edge, or ``None``.

        Equals ``GridIndex._near_boundary(pts64, tol)`` with ``rows=None`` and
        ``GridIndex._near_boundary(pts64[rows], tol)`` otherwise (positions
        into ``rows``). ``None`` means the candidate (point, edge) pairs
        exceed ``max_pairs``, counted before any pair is expanded.
        """
        if rows is None:
            x, y = pts64[:, 0], pts64[:, 1]
        else:
            x, y = pts64[rows, 0], pts64[rows, 1]
        # Non-finite rows fail these comparisons; points outside the extent
        # lie outside every envelope.
        ok = np.flatnonzero((x >= self.x0) & (x <= self.x1)
                            & (y >= self.y0) & (y <= self.y1))
        if ok.size == 0:
            return np.empty(0, dtype=np.int64)
        px, py = x[ok], y[ok]
        cell = _cells(py, self.y0, self.inv, self.ny)
        cell *= self.nx
        cell += _cells(px, self.x0, self.inv, self.nx)
        first = self.offsets[cell]
        count = self.offsets[cell + 1] - first
        keep = np.flatnonzero(count)
        if keep.size == 0:
            return np.empty(0, dtype=np.int64)
        if keep.size < ok.size:
            ok, px, py = ok[keep], px[keep], py[keep]
            first, count = first[keep], count[keep]
        end = np.cumsum(count)
        total = int(end[-1])
        if max_pairs is not None and total > max_pairs:
            return None
        # Pair g of point i reads CSR slot g + delta[i].
        delta = first - (end - count)
        edge_ids, env, edges = self.edge_ids, self.env, self.edges
        tol = self.tol
        tol2 = tol * tol
        near = []
        for c0 in range(0, total, _PAIR_CHUNK):
            c1 = min(c0 + _PAIR_CHUNK, total)
            a = int(np.searchsorted(end, c0, side="right"))
            b = int(np.searchsorted(end, c1 - 1, side="right")) + 1
            start = end[a:b] - count[a:b]
            span = np.minimum(end[a:b], c1) - np.maximum(start, c0)
            ref = np.arange(c0, c1, dtype=np.int64)
            ref += np.repeat(delta[a:b], span)
            e = edge_ids[ref]
            del ref
            box = env[e]
            qx = np.repeat(px[a:b], span)
            qy = np.repeat(py[a:b], span)
            inside = ((qx >= box[:, 0]) & (qx <= box[:, 2])
                      & (qy >= box[:, 1]) & (qy <= box[:, 3]))
            del box
            sel = np.flatnonzero(inside)
            if sel.size == 0:
                continue
            pi = np.repeat(np.arange(a, b, dtype=np.int64), span)[sel]
            ex, ey = qx[sel], qy[sel]
            seg = edges[e[sel]]
            # GridIndex._near_boundary's projection, in the same order and dtype.
            dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
            l2 = dx * dx + dy * dy
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((ex - seg[:, 0]) * dx + (ey - seg[:, 1]) * dy) / l2
            t = np.where(l2 > 0.0, np.clip(t, 0.0, 1.0), 0.0)
            qx2 = seg[:, 0] + t * dx
            qy2 = seg[:, 1] + t * dy
            d2 = (ex - qx2) ** 2 + (ey - qy2) ** 2
            hit = pi[d2 <= tol2]
            if hit.size:
                near.append(np.unique(hit))
        if not near:
            return np.empty(0, dtype=np.int64)
        return ok[np.unique(np.concatenate(near))]
