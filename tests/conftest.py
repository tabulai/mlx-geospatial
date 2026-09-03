"""Shared fixtures and geometry generators for the mlx-geospatial test suite."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon


def make_convex_polygons(
    m: int,
    rng: np.random.Generator,
    *,
    extent=(-1.0, -1.0, 1.0, 1.0),
    n_vertices=(5, 12),
    radius_frac: float = 0.05,
) -> np.ndarray:
    """Random star-convex polygons scattered over ``extent``."""
    x0, y0, x1, y1 = extent
    scale = max(x1 - x0, y1 - y0)
    polys = []
    for _ in range(m):
        cx = rng.uniform(x0, x1)
        cy = rng.uniform(y0, y1)
        r = rng.uniform(0.3, 1.0) * radius_frac * scale
        k = int(rng.integers(n_vertices[0], n_vertices[1] + 1))
        angles = np.sort(rng.uniform(0, 2 * np.pi, k))
        radii = r * rng.uniform(0.5, 1.0, k)
        pts = np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])
        polys.append(Polygon(pts))
    return np.array(polys, dtype=object)


def make_grid_polygons(nx: int, ny: int, *, cell: float = 1.0, origin=(0.0, 0.0)) -> np.ndarray:
    """A tiling of nx*ny axis-aligned squares sharing edges (adjacency stress)."""
    ox, oy = origin
    polys = []
    for j in range(ny):
        for i in range(nx):
            x0, y0 = ox + i * cell, oy + j * cell
            polys.append(shapely.box(x0, y0, x0 + cell, y0 + cell))
    return np.array(polys, dtype=object)


def make_donut(cx: float, cy: float, r_out: float, r_in: float, k: int = 64) -> Polygon:
    """Annulus: polygon with a hole."""
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    shell = np.column_stack([cx + r_out * np.cos(ang), cy + r_out * np.sin(ang)])
    hole = np.column_stack([cx + r_in * np.cos(ang), cy + r_in * np.sin(ang)])
    return Polygon(shell, holes=[hole])


def make_multipolygon(rng: np.random.Generator, *, parts: int = 3, extent=(-1, -1, 1, 1)):
    """MultiPolygon with several disjoint convex parts."""
    polys = make_convex_polygons(parts, rng, extent=extent, radius_frac=0.02)
    return MultiPolygon(list(polys))


def near_boundary_mask(points_xy: np.ndarray, geoms, tol: float) -> np.ndarray:
    """True for points within ``tol`` of any polygon boundary.

    Used to exclude the documented fp32 tolerance band from golden
    comparisons. Re-centering is a pure translation, so a tolerance computed
    in the local frame applies unchanged to original coordinates.
    """
    arr = np.asarray(geoms, dtype=object)
    boundary = shapely.union_all(shapely.boundary(arr))
    pts = shapely.points(np.asarray(points_xy, dtype=np.float64))
    return shapely.distance(pts, boundary) < tol


def reference_assign(points_xy: np.ndarray, geoms) -> np.ndarray:
    """Golden first-match assignment via shapely STRtree ('intersects')."""
    arr = np.asarray(geoms, dtype=object)
    tree = shapely.STRtree(arr)
    pt_idx, poly_idx = tree.query(
        shapely.points(np.asarray(points_xy, dtype=np.float64)), predicate="intersects"
    )
    result = np.full(len(points_xy), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(result, pt_idx, poly_idx)
    result[result == np.iinfo(np.int64).max] = -1
    return result


@pytest.fixture(autouse=True)
def _clean_backend_env(monkeypatch):
    """Isolate every test from an ambient MLX_GEOSPATIAL_BACKEND setting."""
    monkeypatch.delenv("MLX_GEOSPATIAL_BACKEND", raising=False)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260828)
