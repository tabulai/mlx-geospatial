"""Bridge between shapely/GeoPandas geometries and flat MLX arrays.

mlx-spatial never builds a geometry engine. Polygons are ingested through
``shapely.to_ragged_array`` into flat coordinate + offset arrays (the
GeoArrow layout), re-centered to a local float64 origin, and uploaded to
unified memory as float32. Holes and multipolygons are handled naturally by
the even-odd crossing-number rule: every ring of a geometry contributes to
one parity count.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import mlx.core as mx
import shapely
from shapely import GeometryType

from ._precision import (
    boundary_tolerance,
    compute_origin,
    float32_resolution,
    recenter,
)

__all__ = ["PolygonSet", "from_shapely", "as_point_array"]

EdgeData = namedtuple("EdgeData", ["edges", "geom_offsets_np", "geom_offsets_mx"])


def _as_geometry_array(geoms) -> np.ndarray:
    """Normalize GeoDataFrame / GeoSeries / sequence-of-geometries input."""
    if hasattr(geoms, "geometry") and not isinstance(geoms, shapely.Geometry):
        geoms = geoms.geometry  # GeoDataFrame -> GeoSeries
    if hasattr(geoms, "values"):
        geoms = geoms.values  # GeoSeries / pandas Series -> array
    if isinstance(geoms, shapely.Geometry):
        geoms = [geoms]
    # Own snapshot (np.array copies): PolygonSet must not alias caller storage,
    # or mutating the source array desynchronizes the CPU fallback from the
    # GPU-resident data.
    arr = np.array(geoms, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D sequence of geometries, got shape {arr.shape}")
    return arr


def as_point_array(points) -> np.ndarray:
    """Normalize point input to a float64 ``(N, 2)`` array of ``(x, y)``.

    Accepts numpy arrays, ``mx.array``, nested sequences, shapely Point
    arrays, GeoSeries, and GeoDataFrames. Coordinate order is always
    ``(x, y)`` — i.e. ``(lon, lat)`` for geographic data.

    Empty Point geometries and missing geometries (``None``) become NaN
    rows, preserving row alignment: NaN coordinates never match any polygon
    and yield NaN distances.
    """
    if isinstance(points, mx.array):
        points = np.array(points)
    if hasattr(points, "geometry") and not isinstance(points, shapely.Geometry):
        points = points.geometry
    if hasattr(points, "values"):
        points = points.values
    if isinstance(points, shapely.Geometry):
        points = [points]
    arr = np.asarray(points)
    if arr.dtype == object:
        arr = np.asarray(arr, dtype=object)
        if arr.ndim != 1:  # e.g. None -> a 0-d object array
            raise ValueError(
                f"expected a 1-D sequence of points, got shape {arr.shape}"
            )
        type_ids = shapely.get_type_id(arr)
        # Missing geometries (None) carry type id -1 and are treated like
        # empty points below.
        if not np.all((type_ids == GeometryType.POINT.value) | (type_ids == -1)):
            raise ValueError("object arrays must contain only Point geometries")
        # get_coordinates emits no row for empty or missing points; preserve
        # alignment with NaN rows instead of silently shifting every later row.
        out = np.full((len(arr), 2), np.nan, dtype=np.float64)
        present = (type_ids == GeometryType.POINT.value) & ~shapely.is_empty(arr)
        out[present] = shapely.get_coordinates(arr[present])
        return out
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim == 1 and pts.size == 2:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected points of shape (N, 2), got {pts.shape}")
    return np.ascontiguousarray(pts)


class PolygonSet:
    """Flat, GPU-resident ragged representation of a set of (multi)polygons.

    Containment follows the even-odd (crossing-number) rule over all rings of
    a geometry. For *valid* geometries this matches GEOS fill semantics; for
    invalid ones (self-intersections, overlapping multipolygon parts) even-odd
    differs from GEOS — and the CPU fallback, which uses GEOS, will then
    disagree with the GPU. Pass ``validate=True`` to reject invalid input at
    ingestion.

    Attributes
    ----------
    coords : mx.array (C, 2) float32
        Re-centered ring vertices (rings are closed: first == last vertex).
    ring_offsets : mx.array (R + 1,) int32
        Vertex-range boundaries per ring, indexing rows of ``coords``.
    geom_ring_offsets : mx.array (M + 1,) int32
        Ring-range boundaries per geometry (multipolygon parts are collapsed:
        even-odd parity over all rings of a geometry is containment).
    bboxes_local : mx.array (M, 4) float32
        Re-centered per-geometry bounds, slightly widened so fp32 rounding can
        only add candidates, never drop true hits.
    bboxes : np.ndarray (M, 4) float64
        Original-coordinate bounds, used for CPU-side grid planning.
    origin : np.ndarray (2,) float64
        The shared re-centering origin. Query points must be re-centered with
        this same origin; the array is read-only so it cannot drift from the
        already-recentered GPU coordinates.
    """

    def __init__(self, geoms, *, origin=None, validate=False):
        arr = _as_geometry_array(geoms)
        if len(arr) == 0:
            raise ValueError("PolygonSet requires at least one geometry")
        type_ids = shapely.get_type_id(arr)
        missing = type_ids == -1
        if missing.any():
            # Missing geometries (None) become empty polygons: they stay
            # row-aligned and can never contain a point on either backend.
            # arr is our own snapshot, so in-place replacement is safe.
            arr[missing] = shapely.Polygon()
            type_ids = shapely.get_type_id(arr)
        valid = {GeometryType.POLYGON.value, GeometryType.MULTIPOLYGON.value}
        if not np.all(np.isin(type_ids, list(valid))):
            bad = set(np.unique(type_ids)) - valid
            raise ValueError(
                "PolygonSet accepts Polygon and MultiPolygon geometries only; "
                f"got shapely type ids {sorted(bad)}"
            )
        if validate:
            invalid = np.flatnonzero(~shapely.is_valid(arr))
            if invalid.size:
                raise ValueError(
                    "invalid geometries at indices "
                    f"{invalid[:10].tolist()}{'...' if invalid.size > 10 else ''}: "
                    "GPU containment uses the even-odd rule, which differs from "
                    "GEOS on self-intersections and overlapping multipolygon parts"
                )

        geom_type, coords, offsets = shapely.to_ragged_array(arr)
        if coords.ndim == 2 and coords.shape[1] > 2:
            coords = coords[:, :2]  # drop Z/M; predicates are 2-D, as in GEOS
        if not np.isfinite(coords).all():
            raise ValueError(
                "polygon coordinates must be finite; a NaN/inf vertex would "
                "silently poison the shared origin and boundary tolerance"
            )
        if geom_type == GeometryType.POLYGON:
            ring_offsets = np.asarray(offsets[0], dtype=np.int64)
            geom_ring_offsets = np.asarray(offsets[1], dtype=np.int64)
        elif geom_type == GeometryType.MULTIPOLYGON:
            ring_offsets = np.asarray(offsets[0], dtype=np.int64)
            part_ring_offsets = np.asarray(offsets[1], dtype=np.int64)
            geom_part_offsets = np.asarray(offsets[2], dtype=np.int64)
            geom_ring_offsets = part_ring_offsets[geom_part_offsets]
        else:  # pragma: no cover - excluded by the type check above
            raise ValueError(f"unsupported ragged geometry type: {geom_type}")

        if len(coords) >= 2**31:
            raise ValueError("polygon sets beyond 2^31 vertices are not supported")

        self._shapely = arr
        self.n_polygons = len(arr)
        self.n_rings = len(ring_offsets) - 1
        self.n_vertices = len(coords)
        self.bboxes = shapely.bounds(arr).astype(np.float64)

        if origin is None:
            origin = compute_origin(coords)
        # Own, frozen copy: aliasing a caller-owned (writable) array would let
        # a later mutation desynchronize the CPU fallback (which re-centers
        # query points with self.origin) from the already-uploaded GPU coords.
        origin = np.array(origin, dtype=np.float64).reshape(2).copy()
        if not np.isfinite(origin).all():
            raise ValueError(f"origin must be finite, got {origin.tolist()}")
        origin.flags.writeable = False
        self.origin = origin

        with np.errstate(over="ignore"):
            coords_local = recenter(coords, self.origin)
        if len(coords_local) and not np.isfinite(coords_local).all():
            raise ValueError(
                "re-centered coordinates overflow float32; origin="
                f"{self.origin.tolist()} is too far from the data or the "
                "data extent exceeds the float32 coordinate range"
            )
        if len(coords_local) > 1:
            # The crossing test subtracts consecutive edge endpoints.
            # Individually finite fp32 coordinates are not sufficient: e.g.
            # -3e38 and +3e38 both fit, but their difference overflows to inf.
            # Do not reject merely distant rings/geometries, though: adjacent
            # rows at ring boundaries are not edges and are never subtracted
            # by either PIP implementation. The cheap global-span test avoids
            # allocating a full delta array for ordinary coordinate ranges;
            # only a globally extreme set needs the edge-precise check.
            local_lo = coords_local.min(axis=0).astype(np.float64)
            local_hi = coords_local.max(axis=0).astype(np.float64)
            if np.any(local_hi - local_lo > float(np.finfo(np.float32).max)):
                with np.errstate(over="ignore", invalid="ignore"):
                    consecutive_delta = coords_local[1:] - coords_local[:-1]
                safe_delta = np.isfinite(consecutive_delta).all(axis=1)
                cross_ring = ring_offsets[1:-1] - 1
                cross_ring = cross_ring[
                    (cross_ring >= 0) & (cross_ring < len(safe_delta))
                ]
                safe_delta[cross_ring] = True
                if not safe_delta.all():
                    raise ValueError(
                        "re-centered edge endpoint difference overflows "
                        "float32; rescale the coordinates or subdivide the "
                        "affected edge"
                    )
        self.coords = mx.array(coords_local)
        self.ring_offsets = mx.array(ring_offsets.astype(np.int32))
        self.geom_ring_offsets = mx.array(geom_ring_offsets.astype(np.int32))

        self.max_abs_coord = (
            float(np.abs(coords_local).max()) if len(coords_local) else 0.0
        )
        self.boundary_tolerance = boundary_tolerance(self.max_abs_coord)

        # Local-frame bboxes for the kernel's pre-test. Empty geometries have
        # NaN bounds; replace with an impossible box so every point is
        # rejected. Widen finite boxes by a few ulps: a false bbox positive
        # just costs a parity run, a false negative would be a wrong answer.
        bboxes_local = self.bboxes.copy()
        bboxes_local[:, 0] -= self.origin[0]
        bboxes_local[:, 2] -= self.origin[0]
        bboxes_local[:, 1] -= self.origin[1]
        bboxes_local[:, 3] -= self.origin[1]
        empty = ~np.isfinite(bboxes_local).all(axis=1)
        bboxes_local[empty] = [1.0, 1.0, -1.0, -1.0]
        pad = 4.0 * float32_resolution(self.max_abs_coord)
        bboxes_local[~empty, 0:2] -= pad
        bboxes_local[~empty, 2:4] += pad
        self.bboxes_local = mx.array(bboxes_local.astype(np.float32))

        # CPU-side planning copies.
        self._ring_offsets = ring_offsets
        self._geom_ring_offsets = geom_ring_offsets

        # Lazy caches.
        self._edge_data: EdgeData | None = None
        self._strtree = None
        self._grid_index = None

    def __len__(self) -> int:
        return self.n_polygons

    def __repr__(self) -> str:
        return (
            f"PolygonSet(n_polygons={self.n_polygons}, n_rings={self.n_rings}, "
            f"n_vertices={self.n_vertices}, origin={self.origin.tolist()})"
        )

    @property
    def edges(self) -> EdgeData:
        """Dense edge representation for the vectorized point-in-polygon path.

        Returns ``EdgeData(edges, geom_offsets_np, geom_offsets_mx)`` where
        ``edges`` is an ``(E, 4) float32 mx.array`` of ``(x1, y1, x2, y2)``
        segments grouped by geometry, and the offset arrays give the edge
        range of each geometry (length ``M + 1``).
        """
        if self._edge_data is None:
            ring_offsets = self._ring_offsets
            coords = np.array(self.coords)  # (C, 2) float32, local frame
            keep = np.ones(self.n_vertices, dtype=bool)
            keep[ring_offsets[1:] - 1] = False  # each ring's closing vertex
            start_idx = np.flatnonzero(keep)
            edges = np.concatenate(
                [coords[start_idx], coords[start_idx + 1]], axis=1
            ).astype(np.float32)
            edges_per_ring = np.maximum(np.diff(ring_offsets) - 1, 0)
            cum = np.concatenate([[0], np.cumsum(edges_per_ring)])
            geom_offsets = cum[self._geom_ring_offsets].astype(np.int64)
            self._edge_data = EdgeData(
                edges=mx.array(edges),
                geom_offsets_np=geom_offsets,
                geom_offsets_mx=mx.array(geom_offsets.astype(np.int32)),
            )
        return self._edge_data

    @property
    def strtree(self) -> shapely.STRtree:
        """Cached shapely STRtree over the original geometries (CPU fallback)."""
        if self._strtree is None:
            self._strtree = shapely.STRtree(self._shapely)
        return self._strtree

    def recenter_points(self, points) -> tuple[np.ndarray, mx.array]:
        """Normalize + re-center query points with this set's origin.

        Returns ``(points_f64, points_local_mx)``: the float64 originals for
        CPU planning and the float32 local-frame array for kernels.
        """
        pts64 = as_point_array(points)
        local = recenter(pts64, self.origin)
        return pts64, mx.array(local)


def from_shapely(geoms, *, origin=None, validate=False) -> PolygonSet:
    """Build a :class:`PolygonSet` from shapely geometries / GeoSeries / GeoDataFrame."""
    return PolygonSet(geoms, origin=origin, validate=validate)
