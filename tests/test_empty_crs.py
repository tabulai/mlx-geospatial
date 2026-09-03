"""CRS handling for the zero-polygon point-in-polygon fast path."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest
import shapely

import mlx_spatial as ms


gpd = pytest.importorskip("geopandas")


def _point_inputs(crs):
    series = gpd.GeoSeries([shapely.Point(0.0, 0.0)], crs=crs)
    frame = gpd.GeoDataFrame({"value": [1]}, geometry=series)
    return series, frame


def _empty_polygon_inputs(crs):
    series = gpd.GeoSeries([], crs=crs)
    frame = gpd.GeoDataFrame({"value": []}, geometry=series)
    return series, frame


@pytest.mark.parametrize("point_kind", [0, 1], ids=["geoseries", "geodataframe"])
@pytest.mark.parametrize("polygon_kind", [0, 1], ids=["geoseries", "geodataframe"])
def test_empty_polygons_reject_crs_mismatch(point_kind, polygon_kind):
    points = _point_inputs("EPSG:4326")[point_kind]
    polygons = _empty_polygon_inputs("EPSG:3857")[polygon_kind]

    with pytest.raises(ValueError, match="CRS mismatch"):
        ms.point_in_polygon(points, polygons)


@pytest.mark.parametrize("all_matches", [False, True])
@pytest.mark.parametrize("polygon_kind", [0, 1], ids=["geoseries", "geodataframe"])
@pytest.mark.parametrize(
    ("point_crs", "polygon_crs"),
    [
        ("EPSG:4326", "EPSG:4326"),
        ("EPSG:4326", None),
        (None, "EPSG:4326"),
        (None, None),
    ],
)
def test_empty_polygons_allow_matching_or_missing_crs(
    point_crs, polygon_crs, polygon_kind, all_matches
):
    points = _point_inputs(point_crs)[0]
    polygons = _empty_polygon_inputs(polygon_crs)[polygon_kind]

    result = ms.point_in_polygon(points, polygons, all_matches=all_matches)
    if all_matches:
        point_idx, polygon_idx = result
        assert point_idx.dtype == polygon_idx.dtype == np.int64
        assert len(point_idx) == len(polygon_idx) == 0
    else:
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, [-1])


def test_empty_polygons_preserve_device_output_and_localpoints_crs():
    empty = gpd.GeoSeries([], crs="EPSG:4326")

    raw = mx.array([[0.0, 0.0]], dtype=mx.float64)
    raw_result = ms.point_in_polygon(raw, empty)
    assert isinstance(raw_result, mx.array)
    assert raw_result.dtype == mx.int32
    np.testing.assert_array_equal(raw_result, [-1])

    source = ms.PolygonSet(
        gpd.GeoSeries([shapely.box(-1.0, -1.0, 1.0, 1.0)], crs="EPSG:4326")
    )
    local = source.to_local(gpd.GeoSeries([shapely.Point(0.0, 0.0)], crs="EPSG:4326"))
    point_idx, polygon_idx = ms.point_in_polygon(local, empty, all_matches=True)
    assert isinstance(point_idx, mx.array) and isinstance(polygon_idx, mx.array)
    assert point_idx.dtype == polygon_idx.dtype == mx.int64
    assert point_idx.size == polygon_idx.size == 0

    other = ms.PolygonSet(
        gpd.GeoSeries([shapely.box(-1.0, -1.0, 1.0, 1.0)], crs="EPSG:3857")
    ).to_local(gpd.GeoSeries([shapely.Point(0.0, 0.0)], crs="EPSG:3857"))
    with pytest.raises(ValueError, match="CRS mismatch"):
        ms.point_in_polygon(other, empty)
