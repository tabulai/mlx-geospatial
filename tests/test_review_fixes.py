"""Regression pins for the adversarial-review fixes.

Each test corresponds to a confirmed review finding; see git history for the
full diagnoses (kernel int64 indexing, bounded pair expansion, ingestion
validation, sjoin suffix uniqueness, index= consistency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Polygon

import mlx_spatial as ms

from conftest import make_convex_polygons


def test_point_in_polygon_index_only():
    polys = [shapely.box(0, 0, 1, 1)]
    index = ms.GridIndex(ms.PolygonSet(polys))
    got = ms.point_in_polygon([[0.5, 0.5], [5.0, 5.0]], index=index)
    np.testing.assert_array_equal(got, [0, -1])


def test_point_in_polygon_index_mismatch_raises():
    set_a = ms.PolygonSet([shapely.box(0, 0, 1, 1)])
    set_b = ms.PolygonSet([shapely.box(100, 100, 101, 101)])
    index_b = ms.GridIndex(set_b)
    with pytest.raises(ValueError, match="different PolygonSet"):
        ms.point_in_polygon([[0.5, 0.5]], set_a, index=index_b)
    # the exact wrapped set is fine
    got = ms.point_in_polygon([[100.5, 100.5]], set_b, index=index_b)
    np.testing.assert_array_equal(got, [0])


def test_point_in_polygon_requires_polygons_or_index():
    with pytest.raises(ValueError, match="requires polygons"):
        ms.point_in_polygon([[0.5, 0.5]])


def test_validate_rejects_bowtie():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    with pytest.raises(ValueError, match="even-odd"):
        ms.PolygonSet([bowtie], validate=True)
    # without validate it ingests (documented even-odd semantics)
    assert ms.PolygonSet([bowtie]).n_polygons == 1
    # valid input passes validation
    assert ms.PolygonSet([shapely.box(0, 0, 1, 1)], validate=True).n_polygons == 1


def test_nonfinite_vertex_rejected():
    for bad in (np.nan, np.inf, -np.inf):
        poly = Polygon([(0, 0), (1, 0), (1, bad), (0, 1)])
        with pytest.raises(ValueError, match="finite"):
            ms.PolygonSet([poly, shapely.box(2, 2, 3, 3)])


def test_max_pairs_per_batch_validation(rng):
    polys = make_convex_polygons(10, rng)
    index = ms.GridIndex(ms.PolygonSet(polys))
    pts = np.zeros((4, 2))
    for bad in (0, -1, 2**31):
        with pytest.raises(ValueError, match="max_pairs_per_batch"):
            index.query(pts, max_pairs_per_batch=bad)


def test_sliced_expansion_matches_unbatched(rng):
    """The point-range-sliced expansion must equal a single-batch run exactly."""
    polys = make_convex_polygons(60, rng)
    pts = np.column_stack([rng.uniform(-1.1, 1.1, 4000), rng.uniform(-1.1, 1.1, 4000)])
    index = ms.GridIndex(ms.PolygonSet(polys))
    big = index.query(pts, backend="gpu")
    for batch in (1, 7, 113, 1000):
        got = index.query(pts, backend="gpu", max_pairs_per_batch=batch)
        np.testing.assert_array_equal(got, big)
    a_pt, a_pg = index.query(pts, backend="gpu", all_matches=True, max_pairs_per_batch=97)
    b_pt, b_pg = index.query(pts, backend="gpu", all_matches=True)
    np.testing.assert_array_equal(a_pt, b_pt)
    np.testing.assert_array_equal(a_pg, b_pg)


def test_sjoin_double_suffix_collision_keeps_both():
    left = pd.DataFrame(
        {"geometry": [shapely.Point(0.5, 0.5)], "zone": ["left-val"]}
    )
    right = pd.DataFrame(
        {
            "geometry": [shapely.box(0, 0, 1, 1)],
            "zone": ["right-a"],
            "zone_right": ["right-b"],
        }
    )
    out = ms.sjoin(left, right, how="inner")
    assert out.loc[0, "zone"] == "left-val"
    assert out.loc[0, "zone_right"] == "right-a"
    assert out.loc[0, "zone_right_right"] == "right-b"


def test_polygonset_snapshot_not_aliased():
    """Mutating the source array after ingestion must not change results."""
    arr = np.array([shapely.box(0, 0, 1, 1)], dtype=object)
    polyset = ms.PolygonSet(arr)
    arr[0] = shapely.box(100, 100, 101, 101)  # attacker mutates source
    got = ms.point_in_polygon([[0.5, 0.5]], polyset, backend="cpu")
    np.testing.assert_array_equal(got, [0])


def test_duplicate_consecutive_suffix_unique_names():
    """Suffixing never silently overwrites an already-emitted column."""
    left = pd.DataFrame({"geometry": [shapely.Point(0.5, 0.5)]})
    right = pd.DataFrame(
        {"geometry": [shapely.box(0, 0, 1, 1)], "a": [1], "b": [2]}
    )
    out = ms.sjoin(left, right, how="left")
    assert list(out.columns) == ["geometry", "index_right", "a", "b"]
