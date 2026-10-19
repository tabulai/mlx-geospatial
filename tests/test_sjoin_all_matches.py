"""Full joined outputs, including overlaps and pandas extension dtypes."""

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import shapely

import mlx_geospatial as ms
from mlx_geospatial._dispatch import metal_available


BACKENDS = [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.skipif(not metal_available(), reason="Metal unavailable")),
]


def frames():
    left = gpd.GeoDataFrame(
        {"pid": np.arange(7), "point_geom": shapely.points(
            [[0.5, 0.5], [2.5, 0.5], [4.5, 0.5], [7, 7], [0, 0], [1.5, 1.5], [2, 0.5]]
        )},
        geometry="point_geom", crs="EPSG:4326", index=pd.Index([10, 12, 14, 16, 18, 20, 22]),
    )
    right = gpd.GeoDataFrame(
        {
            "zid": np.arange(2),
            "nullable": pd.array([7, None], dtype="Int64"),
            "string": pd.array(["west", "east"], dtype="string"),
            "category": pd.Categorical(["a", "b"], categories=["a", "b", "c"]),
            "time": pd.date_range("2026-01-01", periods=2, tz="UTC"),
            "zone_geom": [
                shapely.Polygon(
                    [(0, 0), (3, 0), (3, 3), (0, 3)],
                    holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
                ),
                shapely.box(2, 0, 5, 2),
            ],
        },
        geometry="zone_geom", crs="EPSG:4326", index=pd.Index([103, 101]),
    )
    return left, right


def canonical(frame):
    return frame.sort_values(["pid", "zid"], kind="stable")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("predicate", ["within", "intersects"])
@pytest.mark.parametrize("how", ["inner", "left"])
def test_all_matches_complete_geodataframe_matches_geopandas(backend, predicate, how):
    left, right = frames()
    ours = ms.sjoin(
        left, right, how=how, predicate=predicate, all_matches=True,
        exact_boundary=True, backend=backend,
    )
    expected = gpd.sjoin(left, right, how=how, predicate=predicate)
    assert_geodataframe_equal(canonical(ours), canonical(expected))
    assert_frame_equal(
        pd.DataFrame(canonical(ours).drop(columns="point_geom")),
        pd.DataFrame(canonical(expected).drop(columns="point_geom")),
        check_exact=True,
    )
    assert ours.geometry.name == "point_geom"
    assert ours.crs == left.crs
    assert ours.pid.tolist() == sorted(ours.pid.tolist())
    assert ours.loc[ours.pid == 1, "zid"].tolist() == [0, 1]
    if how == "left":
        assert len(ours[ours.pid == 3]) == 1
        assert ours.loc[ours.pid == 3, "index_right"].isna().all()
    for column in ("nullable", "string", "category", "time"):
        assert ours[column].dtype == right[column].dtype


@pytest.mark.parametrize("how", ["inner", "left"])
def test_duplicate_labels_do_not_add_matches(how):
    left = gpd.GeoDataFrame(
        {"pid": [0, 1, 2]}, geometry=shapely.points([[1.5, 0.5], [0.5, 0.5], [9, 9]]),
        index=pd.Index([7, 7, 9], name="sample"),
    )
    right = gpd.GeoDataFrame(
        {"zid": [0, 1]}, geometry=[shapely.box(0, 0, 2, 2), shapely.box(1, 0, 3, 2)],
        index=pd.Index([5, 5]),
    )
    ours = ms.sjoin(left, right, how=how, all_matches=True, backend="cpu")
    expected = gpd.sjoin(left, right, how=how)
    assert_geodataframe_equal(canonical(ours), canonical(expected))
    assert ours.pid.tolist() == ([0, 0, 1] if how == "inner" else [0, 0, 1, 2])
    assert ours.zid.dropna().tolist() == [0, 1, 0]


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("empty", ["left", "right", "both"])
def test_empty_frames_preserve_schema_and_dtypes(how, empty):
    left, right = frames()
    if empty in ("left", "both"):
        left = left.iloc[:0]
    if empty in ("right", "both"):
        right = right.iloc[:0]
    ours = ms.sjoin(left, right, how=how, all_matches=True, backend="cpu")
    expected = gpd.sjoin(left, right, how=how)
    # Empty-index class differs across GeoPandas versions; labels and dtype
    # are the contract here, as for the existing first-match API.
    assert_geodataframe_equal(ours, expected, check_index_type=False)
    for column in ("nullable", "string", "category", "time"):
        assert ours[column].dtype == right[column].dtype


@pytest.mark.parametrize("how", ["inner", "left"])
def test_default_first_match_and_right_extension_dtypes(how):
    left, right = frames()
    default = ms.sjoin(left, right, how=how, backend="cpu")
    explicit = ms.sjoin(left, right, how=how, all_matches=False, backend="cpu")
    assert_geodataframe_equal(default, explicit)
    assert default.loc[default.pid == 1, "zid"].tolist() == [0]
    for column in ("nullable", "string", "category", "time"):
        assert default[column].dtype == right[column].dtype


def test_plain_dataframe_collisions_and_inputs_unchanged():
    left = pd.DataFrame({
        "geometry": shapely.points([[1.5, 0.5], [9, 9]]),
        "kind": ["point", "outside"], "kind_right": ["keep", "also keep"],
    })
    right = pd.DataFrame({
        "geometry": [shapely.box(0, 0, 2, 2), shapely.box(1, 0, 3, 2)],
        "kind": pd.array(["west", "east"], dtype="string"),
        "kind_right": pd.array([1, 2], dtype="Int64"),
    })
    original_left, original_right = left.copy(), right.copy()
    out = ms.sjoin(left, right, how="left", all_matches=True, backend="cpu")
    assert type(out) is pd.DataFrame
    assert list(out.columns) == [
        "geometry", "kind", "kind_right", "index_right", "kind_right_right",
        "kind_right_right_right",
    ]
    assert out.kind.tolist() == ["point", "point", "outside"]
    assert out.kind_right_right.iloc[:2].tolist() == ["west", "east"]
    assert out.kind_right_right_right.iloc[:2].tolist() == [1, 2]
    assert out.kind_right_right.iloc[-1] is pd.NA
    assert_frame_equal(left, original_left)
    assert_frame_equal(right, original_right)


def test_missing_points_retained_once_in_left_join():
    left = gpd.GeoDataFrame(
        {"pid": [0, 1, 2]}, geometry=[shapely.Point(0.5, 0.5), None, shapely.Point()],
    )
    right = gpd.GeoDataFrame({"zid": [0, 1]}, geometry=[shapely.box(0, 0, 1, 1)] * 2)
    ours = ms.sjoin(left, right, how="left", all_matches=True, exact_boundary=True, backend="cpu")
    expected = gpd.sjoin(left, right, how="left")
    assert_geodataframe_equal(canonical(ours), canonical(expected))
    assert ours.pid.tolist() == [0, 0, 1, 2]


@pytest.mark.parametrize("all_matches", [False, True])
def test_multiindex_right_labels_kept_as_tuples(all_matches):
    left = gpd.GeoDataFrame(geometry=[shapely.Point(0.5, 0.5)])
    right = gpd.GeoDataFrame(
        {"zid": [0]}, geometry=[shapely.box(0, 0, 1, 1)],
        index=pd.MultiIndex.from_tuples([("zone", 7)]),
    )
    out = ms.sjoin(left, right, all_matches=all_matches, backend="cpu")
    assert out.index_right.tolist() == [("zone", 7)]
