"""Bridge between shapely/GeoPandas geometries and flat MLX arrays.

mlx-geospatial never builds a geometry engine. Polygons are ingested through
``shapely.to_ragged_array`` into flat coordinate + offset arrays (the
GeoArrow layout), re-centered to a local float64 origin, and uploaded to
unified memory as float32. Holes and multipolygons are handled naturally by
the even-odd crossing-number rule: every ring of a geometry contributes to
one parity count.
"""

from __future__ import annotations

import warnings
from collections import namedtuple

import numpy as np
import mlx.core as mx
import shapely
from shapely import GeometryType

from ._precision import (
    PrecisionWarning,
    boundary_tolerance,
    compute_origin,
    float32_resolution,
    recenter,
    split_origin_fp32,
)

__all__ = ["PolygonSet", "LocalPoints", "from_shapely", "as_point_array"]

EdgeData = namedtuple("EdgeData", ["edges", "geom_offsets_np", "geom_offsets_mx"])

_MX_INT_DTYPES = (
    mx.int8,
    mx.int16,
    mx.int32,
    mx.int64,
    mx.uint8,
    mx.uint16,
    mx.uint32,
    mx.uint64,
)
_MX_UINT_DTYPES = (mx.uint8, mx.uint16, mx.uint32, mx.uint64)
_MX_SMALL_FLOAT_DTYPES = (mx.float32, mx.float16, mx.bfloat16)

_FP32_INPUT_MESSAGE = (
    "query points are a {dtype} MLX array: raw lon/lat in float32 already "
    "carries ~0.5 ulp of rounding (about 0.4 m at longitude 74), wider than the "
    "local tolerance band at city or state extents, so results near polygon "
    "edges cannot match GEOS outside the documented band. Re-center from the "
    "float64 host array once "
    "with polyset.to_local(np_points), or build the array with "
    "mx.array(x, dtype=mx.float64). Silence with "
    'warnings.filterwarnings("ignore", category=mlx_geospatial.PrecisionWarning).'
)


def _crs_of(obj):
    """CRS attribute of GeoSeries/GeoDataFrame-like inputs, else None."""
    if isinstance(obj, (np.ndarray, mx.array, list, tuple, shapely.Geometry)):
        return None
    return getattr(obj, "crs", None)


class LocalPoints:
    """Query points already re-centered into a :class:`PolygonSet`'s frame.

    Create with :meth:`PolygonSet.to_local`. ``coords`` is an ``(N, 2)``
    float32 ``mx.array`` in unified memory and ``origin`` the float64 origin
    that was subtracted. :func:`point_in_polygon`, :meth:`GridIndex.query`
    and :func:`contains_matrix` accept it, run entirely on the GPU, and
    return MLX arrays: no host round trip, so labels flow straight into an
    MLX model. Re-center once, slice per batch (``local[i:j]`` or
    ``local[index_array]``), query many times.

    The wrapper is a safety device: local coordinates look like ordinary
    small numbers, so every other entry point (``haversine``,
    ``as_point_array``, ``PolygonSet``) rejects it, and a query against a
    PolygonSet with a different origin raises. To share one frame across
    several polygon layers, build them with
    ``PolygonSet(geoms, origin=first.origin)``.
    """

    __slots__ = ("coords", "origin", "crs")

    def __init__(self, coords, origin, crs=None):
        if not isinstance(coords, mx.array):
            raise TypeError(
                "LocalPoints.coords must be an mx.array; create LocalPoints with "
                "PolygonSet.to_local(points)"
            )
        if coords.dtype != mx.float32 or coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                "LocalPoints.coords must be an (N, 2) float32 array, got shape "
                f"{tuple(coords.shape)} and dtype {coords.dtype}"
            )
        origin = np.array(origin, dtype=np.float64).reshape(2).copy()
        if not np.isfinite(origin).all():
            raise ValueError(f"origin must be finite, got {origin.tolist()}")
        origin.flags.writeable = False
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "crs", crs)

    def __setattr__(self, name, value):
        raise AttributeError("LocalPoints is immutable; build a new one")

    def __delattr__(self, name):
        raise AttributeError("LocalPoints is immutable; build a new one")

    def __reduce__(self):
        return (LocalPoints, (self.coords, self.origin, self.crs))

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.coords.shape)

    def __getitem__(self, key) -> LocalPoints:
        """Slice or gather rows (int index arrays); the origin travels along."""
        if isinstance(key, slice):
            return LocalPoints(self.coords[key], self.origin, self.crs)
        if isinstance(key, tuple):
            raise TypeError(
                "LocalPoints index over points only; read .coords[:, 0] for a "
                "coordinate column"
            )
        if isinstance(key, (bool, np.bool_)) or (
            isinstance(key, (int, np.integer)) and not isinstance(key, np.ndarray)
        ):
            raise TypeError(
                "index LocalPoints with a slice or an integer index array, not a "
                "single integer; read .coords for the raw array"
            )
        if isinstance(key, mx.array):
            idx = key
        else:
            arr = np.asarray(key)
            if arr.size == 0:
                arr = arr.astype(np.int64)  # an empty list defaults to float64
            try:
                idx = mx.array(arr)
            except Exception as exc:
                raise TypeError(
                    "LocalPoints index arrays must be 1-D integer arrays"
                ) from exc
        if idx.ndim != 1 or idx.dtype not in _MX_INT_DTYPES:
            raise TypeError(
                "LocalPoints index arrays must be 1-D integer arrays (boolean "
                "masks are not supported by MLX indexing; select indices first)"
            )
        n = len(self)
        if int(idx.shape[0]):
            # mx.take does not bounds-check: out-of-range rows would silently
            # read other memory (or crash on the CPU device).
            # Widen before comparing so a large ``n`` is not coerced into a
            # narrow index dtype.  uint64 needs its upper-bound check before
            # widening: values above int64 max wrap to negative on conversion.
            is_unsigned = idx.dtype in _MX_UINT_DTYPES
            if idx.dtype == mx.uint64:
                out_of_range = idx >= mx.array(n, dtype=mx.uint64)
            else:
                idx = idx.astype(mx.int64)
                out_of_range = idx >= n
                if not is_unsigned:
                    out_of_range = out_of_range | (idx < -n)
            if bool(out_of_range.any().item()):
                raise IndexError(f"LocalPoints index out of range for {n} points")
        # mx.take accepts several integer dtypes, but using one signed dtype
        # gives negative indexing consistent semantics on every MLX device.
        idx = idx.astype(mx.int64)
        return LocalPoints(mx.take(self.coords, idx, axis=0), self.origin, self.crs)

    def __iter__(self):
        raise TypeError("LocalPoints is not iterable; slice it or read .coords")

    def __repr__(self) -> str:
        return (
            f"LocalPoints(n={len(self)}, origin={self.origin.tolist()}, "
            f"dtype=float32)"
        )


def _as_geometry_array(geoms) -> np.ndarray:
    """Normalize GeoDataFrame / GeoSeries / sequence-of-geometries input."""
    if isinstance(geoms, (LocalPoints, mx.array)):
        raise TypeError(
            "PolygonSet takes polygon geometries (shapely, GeoSeries, "
            "GeoDataFrame), not point arrays"
        )
    if hasattr(geoms, "geometry") and not isinstance(geoms, shapely.Geometry):
        geoms = geoms.geometry  # GeoDataFrame -> GeoSeries
    if hasattr(geoms, "values"):
        geoms = geoms.values  # GeoSeries / pandas Series -> array
    if isinstance(geoms, shapely.Geometry):
        geoms = [geoms]
    # Own snapshot (np.array copies): PolygonSet must not alias caller storage,
    # or mutating the source array desynchronizes the exact_boundary STRtrees
    # (built lazily from this snapshot) from the GPU-resident data.
    arr = np.array(geoms, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D sequence of geometries, got shape {arr.shape}")
    return arr


def _ragged_geometry_bounds(
    coords: np.ndarray,
    ring_offsets: np.ndarray,
    geom_ring_offsets: np.ndarray,
) -> np.ndarray:
    """Per-geometry bounds over every coordinate in every ingested ring.

    GEOS envelopes are shell-derived for polygons. That is equivalent to a
    vertex reduction for valid geometry, but not for an invalid polygon whose
    interior ring extends outside its shell. Containment deliberately accepts
    such geometry when ``validate=False`` and applies even-odd parity over *all*
    rings, so its cheap bbox rejection must cover those same rings too.

    Empty geometries have duplicate vertex offsets and retain an all-NaN row.
    Filtering them also makes the ``reduceat`` starts strictly increasing.
    """
    bounds = np.full((len(geom_ring_offsets) - 1, 4), np.nan, dtype=np.float64)
    if not len(coords):
        return bounds

    geom_vertex_offsets = ring_offsets[geom_ring_offsets]
    nonempty = geom_vertex_offsets[1:] > geom_vertex_offsets[:-1]
    starts = geom_vertex_offsets[:-1][nonempty]
    # Coordinates are grouped by geometry in the ragged layout. Skipped empty
    # rows contribute no coordinates, so consecutive starts still delimit each
    # non-empty geometry exactly even when empties occur between them.
    bounds[nonempty, 0:2] = np.minimum.reduceat(coords, starts, axis=0)
    bounds[nonempty, 2:4] = np.maximum.reduceat(coords, starts, axis=0)
    return bounds


def as_point_array(points) -> np.ndarray:
    """Normalize point input to a float64 ``(N, 2)`` array of ``(x, y)``.

    Accepts numpy arrays, ``mx.array``, nested sequences, shapely Point
    arrays, GeoSeries, and GeoDataFrames. Coordinate order is always
    ``(x, y)`` — i.e. ``(lon, lat)`` for geographic data.

    Empty Point geometries and missing geometries (``None``) become NaN
    rows, preserving row alignment: NaN coordinates never match any polygon
    and yield NaN distances.
    """
    if isinstance(points, LocalPoints):
        raise TypeError(
            "LocalPoints hold re-centered polygon-frame coordinates, not lon/lat; "
            "pass the original coordinates (or LocalPoints only to the join "
            "functions)"
        )
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
        coords = shapely.get_coordinates(arr)
        if len(coords) == len(arr):
            # Every geometry produced one row: no empty or missing points, so
            # the is_empty pass and the masked copy are unnecessary.
            return np.ascontiguousarray(coords, dtype=np.float64)
        # get_coordinates emits no row for empty or missing points; preserve
        # alignment with NaN rows instead of silently shifting every later row.
        out = np.full((len(arr), 2), np.nan, dtype=np.float64)
        present = (type_ids == GeometryType.POINT.value) & ~shapely.is_empty(arr)
        out[present] = coords
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
    a geometry, on every backend (the CPU fallback evaluates the same fp32
    rule, bit for bit). For *valid* geometries this matches GEOS fill
    semantics; for invalid ones (self-intersections, overlapping multipolygon
    parts) even-odd differs from GEOS, which you meet only through
    ``exact_boundary=True`` (it re-checks near-boundary points with GEOS and
    assumes valid geometry). Pass ``validate=True`` to reject invalid input at
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
        Original-coordinate bounds over every ingested ring, used for CPU-side
        grid planning. For invalid geometry these can be wider than the GEOS
        envelope, matching the all-ring even-odd containment contract. The
        array is read-only so it cannot drift from lazily built grid indexes.
    origin : np.ndarray (2,) float64
        The shared re-centering origin. Query points must be re-centered with
        this same origin; the array is read-only so it cannot drift from the
        already-recentered GPU coordinates.
    crs : optional
        The CRS of the input GeoSeries/GeoDataFrame, if it had one. Only used
        to reject query points that carry a different CRS.
    """

    def __init__(self, geoms, *, origin=None, validate=False):
        self.crs = _crs_of(geoms)
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
        # Original float64 vertices in ragged order, for the exact-boundary
        # host edge table (no second to_ragged_array).
        coords.flags.writeable = False
        self._coords_f64 = coords
        self.n_polygons = len(arr)
        self.n_rings = len(ring_offsets) - 1
        self.n_vertices = len(coords)
        bboxes = _ragged_geometry_bounds(
            coords, ring_offsets, geom_ring_offsets
        )
        # GridIndex instances are built lazily from these bounds. A caller
        # mutation before that first query would otherwise make the CPU grid
        # disagree with the already-uploaded coordinates and kernel bboxes.
        bboxes.flags.writeable = False
        self.bboxes = bboxes

        if origin is None:
            # These all-ring bounds have the same extrema as the ragged vertex
            # array, including for invalid polygons whose interior rings poke
            # outside their shells. The origin, grid and kernel bbox test must
            # all describe the same ring set as even-odd containment.
            origin = compute_origin(self.bboxes[:, 0:2], self.bboxes[:, 2:4])
        # Own, frozen copy: aliasing a caller-owned (writable) array would let
        # a later mutation desynchronize query re-centering (host points are
        # re-centered with self.origin) from the already-uploaded GPU coords.
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
            # Column-wise reductions: numpy's axis-0 reduction over a (C, 2)
            # array walks row by row and is ~30x slower.
            local_lo = np.array(
                [coords_local[:, 0].min(), coords_local[:, 1].min()], dtype=np.float64
            )
            local_hi = np.array(
                [coords_local[:, 0].max(), coords_local[:, 1].max()], dtype=np.float64
            )
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
        edges_per_ring = np.maximum(np.diff(ring_offsets) - 1, 0)
        self.max_edges_per_ring = int(edges_per_ring.max(initial=0))
        geom_max_ring_edges = np.zeros(self.n_polygons, dtype=np.int64)
        nonempty_geom = geom_ring_offsets[1:] > geom_ring_offsets[:-1]
        if nonempty_geom.any():
            # Filtering empty geometries before reduceat is essential: their
            # duplicate ring offsets must not borrow the following geometry's
            # first ring length.
            geom_max_ring_edges[nonempty_geom] = np.maximum.reduceat(
                edges_per_ring, geom_ring_offsets[:-1][nonempty_geom]
            )
        self._geom_max_ring_edges = geom_max_ring_edges
        edge_cumulative = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(edges_per_ring)]
        )
        self._geom_edge_offsets = edge_cumulative[geom_ring_offsets]
        self._geom_edge_counts = np.diff(self._geom_edge_offsets)
        self.max_edges_per_polygon = int(self._geom_edge_counts.max(initial=0))

        # Lazy caches.
        self._edge_data: EdgeData | None = None
        self._strtree = None
        self._edge_box_strtree = None
        self._edges_f64_cache = None
        self._host_edge_grid_cache = False  # False: unbuilt; None: unbuildable.
        self._edges_np_cache = None
        self._bboxes_local_np_cache = None
        self._grid_index = None
        self._origin_f64_mx = None
        self._origin_hi_lo_mx = None
        self._y_edge_index = False  # False: unbuilt; None: traversal is preferable.

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
            geom_offsets = self._geom_edge_offsets
            self._edge_data = EdgeData(
                edges=mx.array(edges),
                geom_offsets_np=geom_offsets,
                geom_offsets_mx=mx.array(geom_offsets.astype(np.int32)),
            )
        return self._edge_data

    @property
    def strtree(self) -> shapely.STRtree:
        """Cached shapely STRtree over the original geometries.

        Used only by ``exact_boundary=True`` refinement (and as a test oracle);
        the query path itself never calls GEOS.
        """
        if self._strtree is None:
            self._strtree = shapely.STRtree(self._shapely)
        return self._strtree

    def _edges_f64(self) -> np.ndarray:
        """``(E, 4)`` float64 edge table in original coordinates.

        Same edge order as :attr:`edges` (rings in ragged order, the closing
        vertex of each ring dropped), rebuilt from the geometry snapshot so
        the float64 values are exact. Used by ``exact_boundary=True`` to
        measure point-to-edge distances on the host.
        """
        if self._edges_f64_cache is None:
            # One 16-byte (x, y) item per vertex: boolean selection of the
            # edge starts and of their successors (the vertex after each kept
            # one) is several times faster than a (C, 2) fancy-index gather.
            vertices = np.ascontiguousarray(self._coords_f64, dtype=np.float64)
            vertices = vertices.view(np.complex128).reshape(-1)
            keep = np.ones(len(vertices), dtype=bool)
            keep[self._ring_offsets[1:] - 1] = False
            edges = np.empty((int(keep.sum()), 2), dtype=np.complex128)
            edges[:, 0] = vertices[keep]
            # Vertex C - 1 always closes the last ring, so rolling ``keep``
            # marks exactly the successors.
            edges[:, 1] = vertices[np.roll(keep, 1)]
            self._edges_f64_cache = edges.view(np.float64).reshape(-1, 4)
        return self._edges_f64_cache

    def _host_edge_grid(self):
        """Cached :class:`~mlx_geospatial._edge_grid.HostEdgeGrid`, or ``None``.

        The float64 uniform grid over the same widened edge envelopes as
        :meth:`_edge_box_tree`; it answers the same near-boundary rows
        without building one shapely box per edge.
        """
        if self._host_edge_grid_cache is False:
            from ._edge_grid import HostEdgeGrid

            self._host_edge_grid_cache = HostEdgeGrid.build(
                self._edges_f64(), self.boundary_tolerance
            )
        return self._host_edge_grid_cache

    def _edge_box_tree(self) -> shapely.STRtree:
        """Cached STRtree over every edge's envelope widened by the tolerance.

        A point within ``boundary_tolerance`` of an edge lies inside that
        edge's widened envelope, so an envelope-only query against this tree
        yields a superset of the near-boundary (point, edge) pairs that a
        NumPy distance test then settles (see ``GridIndex._near_boundary``).
        """
        if self._edge_box_strtree is None:
            e = self._edges_f64()
            tol = self.boundary_tolerance
            self._edge_box_strtree = shapely.STRtree(
                shapely.box(
                    np.minimum(e[:, 0], e[:, 2]) - tol,
                    np.minimum(e[:, 1], e[:, 3]) - tol,
                    np.maximum(e[:, 0], e[:, 2]) + tol,
                    np.maximum(e[:, 1], e[:, 3]) + tol,
                )
            )
        return self._edge_box_strtree

    def _edges_np(self) -> np.ndarray:
        """Host view of the ``(E, 4)`` float32 edge table (see :attr:`edges`)."""
        if self._edges_np_cache is None:
            self._edges_np_cache = np.asarray(self.edges.edges)
        return self._edges_np_cache

    def _bboxes_local_np(self) -> np.ndarray:
        """Host view of the widened float32 kernel bboxes."""
        if self._bboxes_local_np_cache is None:
            self._bboxes_local_np_cache = np.asarray(self.bboxes_local)
        return self._bboxes_local_np_cache

    def to_local(self, points) -> LocalPoints:
        """Re-center query points into this set's frame, once, for GPU queries.

        Accepts everything :func:`as_point_array` accepts plus ``mx.array``
        inputs. Host float64 coordinates and ``mx.float64`` arrays are
        re-centered in float64 (numpy inputs on the MLX CPU stream, bit for
        bit the same values as NumPy) and cast to float32, so the documented
        precision contract holds. float32 (or 16-bit) MLX arrays are
        re-centered on the device with a split ``hi + lo`` origin and emit a
        :class:`PrecisionWarning`, because raw lon/lat in float32 has already
        lost more than the tolerance band. The result carries the input's CRS,
        or this set's CRS when the input has none, so a later query against a
        layer with another CRS raises; it is also rejected by every polygon set
        with a different origin.
        """
        if isinstance(points, LocalPoints):
            raise TypeError(
                "points are already LocalPoints; pass the original lon/lat "
                "coordinates to to_local"
            )
        crs = _crs_of(points)
        self._check_crs(crs)
        local = self._local_coords(points, stacklevel=3)
        return LocalPoints(local, self.origin, crs=crs if crs is not None else self.crs)

    def recenter_points(self, points) -> tuple[np.ndarray, mx.array]:
        """Normalize + re-center query points with this set's origin.

        Returns ``(points_f64, points_local_mx)``: float64 coordinates for
        CPU planning and the float32 local-frame array for kernels. For
        device-resident inputs (``mx.array``, :class:`LocalPoints`) the
        float64 coordinates are reconstructed from the float32 local frame,
        so prefer :meth:`to_local` for GPU pipelines.
        """
        pts64, local, _ = self._ingest_points(points, stacklevel=4)
        if pts64 is None:
            pts64 = self._host_coords(local)
        return pts64, local

    def _ingest_points(self, points, *, stacklevel: int = 4, defer_local: bool = False):
        """Normalize any point input for the kernels.

        Returns ``(points_f64_or_None, points_local_mx, on_device)``:
        host inputs (numpy, sequences, shapely, GeoSeries) provide float64
        coordinates and a float32 local-frame array; ``mx.array`` and
        :class:`LocalPoints` inputs stay on device (``points_f64`` is None)
        and mark the result container as MLX. With ``defer_local=True`` host
        inputs return ``None`` for the local array, for callers that may not
        need it (the host fallback re-centers in NumPy instead).
        """
        if isinstance(points, LocalPoints):
            self._check_local(points)
            return None, points.coords, True
        if isinstance(points, mx.array):
            return None, self._local_coords(points, stacklevel=stacklevel), True
        self._check_crs(_crs_of(points))
        pts64 = as_point_array(points)
        return pts64, (None if defer_local else self._local_from_f64(pts64)), False

    def _check_crs(self, crs) -> None:
        if crs is not None and self.crs is not None and crs != self.crs:
            raise ValueError(f"CRS mismatch: points={crs}, polygons={self.crs}")

    def _check_local(self, local: LocalPoints) -> None:
        if not np.array_equal(local.origin, self.origin):
            raise ValueError(
                f"LocalPoints origin {local.origin.tolist()} differs from this "
                f"PolygonSet's origin {self.origin.tolist()}; re-center with this "
                "set (polyset.to_local(points)) or build both layers with "
                "PolygonSet(geoms, origin=shared_origin)"
            )
        self._check_crs(local.crs)

    def _origin_f64(self) -> mx.array:
        if self._origin_f64_mx is None:
            self._origin_f64_mx = mx.array(self.origin, dtype=mx.float64)
        return self._origin_f64_mx

    def _local_from_f64(self, pts64: np.ndarray) -> mx.array:
        """float64 -> float32 local frame on the MLX CPU stream.

        Bit-identical to ``recenter(pts64, origin)`` (one correctly rounded
        float64 subtraction, one cast) and about three times faster than the
        NumPy version plus upload; float64 is CPU-only in MLX, hence the
        explicit stream.
        """
        p64 = mx.array(pts64, dtype=mx.float64)
        diff = mx.subtract(p64, self._origin_f64(), stream=mx.cpu)
        return diff.astype(mx.float32, stream=mx.cpu)

    def _local_coords(self, points, *, stacklevel: int) -> mx.array:
        """Local-frame float32 coordinates for host or device inputs."""
        if not isinstance(points, mx.array):
            return self._local_from_f64(as_point_array(points))
        if points.ndim == 1 and points.shape[0] == 2:
            points = mx.reshape(
                points,
                (1, 2),
                stream=mx.cpu if points.dtype == mx.float64 else None,
            )
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"expected points of shape (N, 2), got {tuple(points.shape)}"
            )
        if points.dtype == mx.float64 or points.dtype in _MX_INT_DTYPES:
            # Exact: float64 (and integer) values re-centered in float64 on the
            # CPU stream, exactly like the host path.
            p64 = (
                points
                if points.dtype == mx.float64
                else points.astype(mx.float64, stream=mx.cpu)
            )
            diff = mx.subtract(p64, self._origin_f64(), stream=mx.cpu)
            return diff.astype(mx.float32, stream=mx.cpu)
        if points.dtype in _MX_SMALL_FLOAT_DTYPES:
            warnings.warn(
                PrecisionWarning(_FP32_INPUT_MESSAGE.format(dtype=points.dtype)),
                stacklevel=stacklevel,
            )
            p32 = points if points.dtype == mx.float32 else points.astype(mx.float32)
            if self._origin_hi_lo_mx is None:
                hi, lo = split_origin_fp32(self.origin)
                self._origin_hi_lo_mx = (mx.array(hi), mx.array(lo))
            hi, lo = self._origin_hi_lo_mx
            return (p32 - hi) - lo
        raise TypeError(
            f"unsupported point dtype {points.dtype}; use float64, float32, or "
            "integer coordinates"
        )

    def _host_coords(self, local: mx.array) -> np.ndarray:
        """float64 host coordinates reconstructed from a local-frame array."""
        return np.asarray(local, dtype=np.float64) + self.origin


def from_shapely(geoms, *, origin=None, validate=False) -> PolygonSet:
    """Build a :class:`PolygonSet` from shapely geometries / GeoSeries / GeoDataFrame."""
    return PolygonSet(geoms, origin=origin, validate=validate)
