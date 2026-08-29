"""mlx-spatial: GPU-accelerated spatial operations for Apple silicon via MLX.

Coordinate convention: ``(x, y)`` == ``(lon, lat)``, degrees, everywhere.

- :func:`haversine`, :func:`haversine_cross`, :func:`haversine_knn` —
  great-circle distances (spherical model).
- :func:`point_in_polygon`, :func:`contains_matrix`, :class:`GridIndex` —
  planar even-odd containment with a grid-indexed candidate filter.
- :func:`sjoin` — GeoPandas-flavored point-in-polygon join wrapper.
- :class:`PolygonSet` — ingested, re-centered, GPU-resident polygons; build
  once and reuse across queries.
"""

from ._version import __version__
from ._ingest import PolygonSet, as_point_array, from_shapely
from ._pip import contains_matrix
from .haversine import R_EARTH_M, haversine, haversine_cross, haversine_knn
from .join import GridIndex, point_in_polygon, sjoin

__all__ = [
    "__version__",
    "PolygonSet",
    "as_point_array",
    "from_shapely",
    "contains_matrix",
    "R_EARTH_M",
    "haversine",
    "haversine_cross",
    "haversine_knn",
    "GridIndex",
    "point_in_polygon",
    "sjoin",
]
