"""mlx-spatial: GPU-accelerated spatial operations for Apple silicon via MLX.

Coordinate convention: ``(x, y)`` == ``(lon, lat)``, degrees, everywhere.

- :func:`haversine`, :func:`haversine_cross`, :func:`haversine_knn` —
  great-circle distances (spherical model).
- :func:`point_in_polygon`, :func:`contains_matrix`, :class:`GridIndex` —
  planar even-odd containment with a grid-indexed candidate filter.
- :func:`sjoin` — GeoPandas-flavored point-in-polygon join wrapper.
- :class:`PolygonSet` — ingested, re-centered, GPU-resident polygons; build
  once and reuse across queries.
- :class:`LocalPoints` — query points re-centered once with
  :meth:`PolygonSet.to_local`; joins on them run entirely on the GPU and
  return MLX arrays.

Output containers: the predicates (:func:`point_in_polygon`,
:meth:`GridIndex.query`, :func:`contains_matrix`) return the container you
passed in — NumPy for host inputs, MLX arrays for ``mx.array`` /
:class:`LocalPoints` inputs; the distance functions always return MLX arrays;
:func:`sjoin` returns pandas. :class:`PrecisionWarning` flags float32 MLX
inputs, whose raw lon/lat rounding already exceeds the tolerance band.
"""

from ._version import __version__
from ._ingest import LocalPoints, PolygonSet, as_point_array, from_shapely
from ._pip import contains_matrix
from ._precision import PrecisionWarning
from .haversine import R_EARTH_M, haversine, haversine_cross, haversine_knn
from .join import GridIndex, point_in_polygon, sjoin

__all__ = [
    "__version__",
    "PolygonSet",
    "LocalPoints",
    "PrecisionWarning",
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
