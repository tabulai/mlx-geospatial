"""Tests for bulk sjoin DataFrame assembly and reusable PreparedJoin."""

from __future__ import annotations

import datetime as dt
import types
import warnings

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
    pytest.param(
        "gpu",
        marks=pytest.mark.skipif(not metal_available(), reason="Metal unavailable"),
    ),
]


def _rich_frames(*, crs="EPSG:4326"):
    left = gpd.GeoDataFrame(
        {
            "pid": np.arange(8, dtype=np.int64),
            "val": np.linspace(0.5, 7.5, 8, dtype=np.float64),
            "flag": np.array([True, False, True, False, True, True, False, True]),
            "nullable_i": pd.array([10, None, 30, 40, None, 60, 70, 80], dtype="Int64"),
            "nullable_f": pd.array([1.5, 2.5, None, 4.5, 5.5, None, 7.5, 8.5], dtype="Float64"),
            "nullable_b": pd.array([True, None, False, True, False, None, True, False], dtype="boolean"),
            "label": pd.array([f"p{i}" for i in range(8)], dtype="string"),
            "cat": pd.Categorical(
                ["a", "b", "a", "b", "a", "b", "a", "b"], categories=["a", "b", "unused"]
            ),
            "ts": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
            "point_geom": shapely.points(
                [
                    [0.5, 0.5],  # poly 0 only
                    [2.5, 0.5],  # poly 0 and poly 1
                    [4.5, 0.5],  # poly 1 only
                    [9.0, 9.0],  # unmatched
                    [0.0, 0.0],  # boundary of poly 0
                    [1.5, 1.5],  # inside hole of poly 0
                    [2.0, 0.5],  # boundary of poly 1, interior of poly 0
                    [2.5, 1.5],  # poly 0 and poly 1
                ]
            ),
        },
        geometry="point_geom",
        crs=crs,
        index=pd.Index([10, 12, 14, 16, 18, 20, 22, 24], name="left_id"),
    )
    right = gpd.GeoDataFrame(
        {
            "zid": np.arange(2, dtype=np.int64),
            "val": np.array([100.0, 200.0], dtype=np.float64),  # collides with left
            "r_bool": np.array([True, False]),
            "r_nullable": pd.array([7, None], dtype="Int64"),
            "r_nfloat": pd.array([None, 3.25], dtype="Float64"),
            "r_nbool": pd.array([True, None], dtype="boolean"),
            "r_string": pd.array(["west", "east"], dtype="string"),
            "r_cat": pd.Categorical(["x", "y"], categories=["x", "y", "z"]),
            "r_time": pd.date_range("2026-06-01", periods=2, freq="D", tz="UTC"),
            "r_extra_geom": gpd.GeoSeries(
                [shapely.Point(1.0, 1.0), shapely.Point(3.0, 1.0)], crs=crs
            ),
            "zone_geom": [
                shapely.Polygon(
                    [(0, 0), (3, 0), (3, 3), (0, 3)],
                    holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
                ),
                shapely.box(2, 0, 5, 2),
            ],
        },
        geometry="zone_geom",
        crs=crs,
        index=pd.Index([103, 101], name="right_id"),
    )
    return left, right


def _canonical(frame):
    return frame.sort_values(["pid", "zid"], kind="stable")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("all_matches", [False, True])
@pytest.mark.parametrize("predicate", ["within", "intersects"])
def test_bulk_assembly_and_prepared_join_match_raw_and_geopandas(
    backend, how, all_matches, predicate
):
    left, right = _rich_frames()
    prepared = ms.PreparedJoin(right)
    assert len(prepared) == len(right)
    assert prepared.crs == right.crs
    assert "zone_geom" not in prepared.columns
    assert "zid" in prepared.columns
    assert "PreparedJoin(" in repr(prepared)

    raw = ms.sjoin(
        left,
        right,
        how=how,
        predicate=predicate,
        all_matches=all_matches,
        exact_boundary=True,
        backend=backend,
    )
    via_sjoin = ms.sjoin(
        left,
        prepared,
        how=how,
        predicate=predicate,
        all_matches=all_matches,
        exact_boundary=True,
        backend=backend,
    )
    via_method = prepared.sjoin(
        left,
        how=how,
        predicate=predicate,
        all_matches=all_matches,
        exact_boundary=True,
        backend=backend,
    )

    assert_geodataframe_equal(raw, via_sjoin, check_like=False)
    assert_geodataframe_equal(raw, via_method, check_like=False)
    assert raw.geometry.name == "point_geom"
    assert raw.crs == left.crs
    assert raw.index.name == "left_id"
    for col in ("r_nullable", "r_nfloat", "r_nbool", "r_string", "r_cat", "r_time"):
        assert raw[col].dtype == right[col].dtype
    assert raw["r_extra_geom"].dtype == right["r_extra_geom"].dtype
    assert raw["r_extra_geom"].crs == right["r_extra_geom"].crs

    if all_matches:
        expected = gpd.sjoin(left, right, how=how, predicate=predicate).rename(
            columns={"val_left": "val", "right_id": "index_right"}
        )
        assert_geodataframe_equal(_canonical(raw), _canonical(expected), check_like=False)


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("all_matches", [False, True])
def test_mutation_isolation_for_raw_and_prepared_sjoin(how, all_matches):
    left, right = _rich_frames()
    left_orig = left.copy(deep=True)
    right_orig = right.copy(deep=True)
    prepared = ms.PreparedJoin(right)

    # Mutate output frame: must never alias left or right.
    out = ms.sjoin(left, right, how=how, all_matches=all_matches, backend="cpu")
    out.iloc[0, out.columns.get_loc("pid")] = -999
    out.iloc[0, out.columns.get_loc("val")] = -123.0
    out.iloc[0, out.columns.get_loc("nullable_i")] = 9999
    assert_geodataframe_equal(left, left_orig)
    assert_geodataframe_equal(right, right_orig)

    # Mutate original right frame AFTER constructing PreparedJoin: snapshot
    # must remain unaffected.
    before = ms.sjoin(left, prepared, how=how, all_matches=all_matches, backend="cpu")
    right.iloc[0, right.columns.get_loc("zid")] = 777
    right.iloc[0, right.columns.get_loc("r_string")] = "MUTATED"
    right.index = pd.Index([999, 888])
    right["extra_after"] = [1, 2]
    after = ms.sjoin(left, prepared, how=how, all_matches=all_matches, backend="cpu")
    assert_geodataframe_equal(before, after)


def test_prepared_join_copies_index_backing_array():
    left, right = _rich_frames()
    labels = np.array([101, 202], dtype=np.int64)
    right.index = pd.Index(labels, copy=False)
    prepared = ms.PreparedJoin(right)
    expected = ms.sjoin(left, right, all_matches=True, backend="cpu")

    labels[:] = [303, 404]
    assert_geodataframe_equal(
        prepared.sjoin(left, all_matches=True, backend="cpu"), expected
    )


def test_prepared_join_preserves_plain_dataframe_crs_column():
    left, right = _rich_frames()
    right = pd.DataFrame(
        {"geometry": list(right.geometry), "crs": ["north", "south"]},
        index=right.index,
    )
    expected = ms.sjoin(left, right, all_matches=True, backend="cpu")
    result = ms.PreparedJoin(right).sjoin(left, all_matches=True, backend="cpu")
    assert_geodataframe_equal(result, expected)


def test_prepared_join_reuses_polyset_and_grid_index_across_batches(monkeypatch):
    left, right = _rich_frames()
    prepared = ms.PreparedJoin(right)

    def must_not_rebuild(*args, **kwargs):
        raise AssertionError("PreparedJoin rebuilt PolygonSet or GridIndex during sjoin")

    monkeypatch.setattr("mlx_geospatial.join.PolygonSet", must_not_rebuild)
    monkeypatch.setattr("mlx_geospatial.join.GridIndex", must_not_rebuild)

    batches = [left.iloc[:4], left.iloc[4:]]
    results = [
        ms.sjoin(b, prepared, how="inner", all_matches=True, exact_boundary=True, backend="cpu")
        for b in batches
    ]
    full = ms.sjoin(left, prepared, how="inner", all_matches=True, exact_boundary=True, backend="cpu")
    assert_geodataframe_equal(pd.concat(results), full)


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("all_matches", [False, True])
def test_left_join_with_all_points_matched_preserves_dtypes(how, all_matches):
    left = gpd.GeoDataFrame(
        {"pid": [0, 1]},
        geometry=shapely.points([[0.5, 0.5], [2.5, 0.5]]),
        crs="EPSG:4326",
        index=pd.RangeIndex(10, 14, 2, name="ri"),
    )
    right = gpd.GeoDataFrame(
        {"zid": [10, 20], "nullable": pd.array([1, 2], dtype="Int64")},
        geometry=[shapely.box(0, 0, 2, 2), shapely.box(2, 0, 4, 2)],
        crs="EPSG:4326",
        index=pd.Index([50, 60]),
    )
    prepared = ms.PreparedJoin(right)
    raw = ms.sjoin(left, right, how=how, all_matches=all_matches, backend="cpu")
    prep = ms.sjoin(left, prepared, how=how, all_matches=all_matches, backend="cpu")
    assert_geodataframe_equal(raw, prep)
    assert raw.index.tolist() == [10, 12]
    assert raw.index.name == "ri"
    assert raw["zid"].tolist() == [10, 20]
    assert raw["nullable"].dtype == right["nullable"].dtype


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("all_matches", [False, True])
def test_duplicate_and_multiindex_on_left_and_right(how, all_matches):
    left = gpd.GeoDataFrame(
        {"pid": [0, 1, 2]},
        geometry=shapely.points([[0.5, 0.5], [1.5, 0.5], [9.0, 9.0]]),
        index=pd.MultiIndex.from_tuples([("a", 1), ("a", 1), ("b", 2)], names=["g", "k"]),
        crs="EPSG:4326",
    )
    right = gpd.GeoDataFrame(
        {"zid": [0, 1]},
        geometry=[shapely.box(0, 0, 2, 2), shapely.box(1, 0, 3, 2)],
        index=pd.MultiIndex.from_tuples([("z", 10), ("z", 10)]),
        crs="EPSG:4326",
    )
    prepared = ms.PreparedJoin(right)
    raw = ms.sjoin(left, right, how=how, all_matches=all_matches, backend="cpu")
    prep = ms.sjoin(left, prepared, how=how, all_matches=all_matches, backend="cpu")
    assert_geodataframe_equal(raw, prep)
    assert raw.index.names == ["g", "k"]


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("all_matches", [False, True])
def test_prepared_join_empty_right_and_empty_left(how, all_matches):
    left, right = _rich_frames()
    empty_right = right.iloc[:0]
    prep_empty = ms.PreparedJoin(empty_right)
    assert len(prep_empty) == 0
    assert prep_empty.polygons is None and prep_empty.index is None
    assert prep_empty.crs == right.crs

    raw = ms.sjoin(left, empty_right, how=how, all_matches=all_matches, backend="cpu")
    prep = ms.sjoin(left, prep_empty, how=how, all_matches=all_matches, backend="cpu")
    assert_geodataframe_equal(raw, prep)

    # CRS mismatch still raises on empty PreparedJoin.
    with pytest.raises(ValueError, match="CRS mismatch"):
        ms.sjoin(left.to_crs("EPSG:3857"), prep_empty, how=how, all_matches=all_matches)

    # Empty left against non-empty PreparedJoin.
    prep_full = ms.PreparedJoin(right)
    raw_el = ms.sjoin(left.iloc[:0], right, how=how, all_matches=all_matches, backend="cpu")
    prep_el = ms.sjoin(left.iloc[:0], prep_full, how=how, all_matches=all_matches, backend="cpu")
    assert_geodataframe_equal(raw_el, prep_el)


def test_prepared_join_plain_dataframe_geoseries_and_errors():
    left_pd = pd.DataFrame(
        {"geometry": shapely.points([[0.5, 0.5], [9.0, 9.0]]), "pid": [1, 2]}
    )
    right_pd = pd.DataFrame(
        {"geometry": [shapely.box(0, 0, 2, 2)], "tag": ["ok"]}, index=["Z0"]
    )
    prep_pd = ms.PreparedJoin(right_pd)
    out = ms.sjoin(left_pd, prep_pd, how="left", all_matches=True, backend="cpu")
    expected = ms.sjoin(left_pd, right_pd, how="left", all_matches=True, backend="cpu")
    assert type(out) is pd.DataFrame
    assert_frame_equal(out, expected)

    # Bare geometry list / GeoSeries right input.
    prep_seq = ms.PreparedJoin([shapely.box(0, 0, 2, 2)])
    assert prep_seq.columns == ()
    out_seq = ms.sjoin(left_pd, prep_seq, how="inner", backend="cpu")
    assert out_seq["index_right"].tolist() == [0]

    # Passing a PreparedJoin to PreparedJoin raises TypeError.
    with pytest.raises(TypeError, match="already a PreparedJoin"):
        ms.PreparedJoin(prep_pd)

    # validate=True forwards to PolygonSet and rejects self-intersecting polygons.
    bowtie = shapely.Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    with pytest.raises(ValueError, match="invalid geometries"):
        ms.PreparedJoin([bowtie], validate=True)


def test_duplicate_left_columns_and_custom_dataframe_subclass_fallback():
    left_dup = pd.DataFrame(
        [[shapely.Point(0.5, 0.5), 1, 2], [shapely.Point(9.0, 9.0), 3, 4]],
        columns=["geometry", "x", "x"],
    )
    right = pd.DataFrame({"geometry": [shapely.box(0, 0, 1, 1)], "y": [10]})
    out = ms.sjoin(left_dup, right, how="inner", all_matches=True, backend="cpu")
    assert list(out.columns) == ["geometry", "x", "x", "index_right", "y"]
    assert len(out) == 1 and out["y"].iloc[0] == 10

    class MyDF(pd.DataFrame):
        @property
        def _constructor(self):
            return MyDF

    left_sub = MyDF({"geometry": shapely.points([[0.5, 0.5], [9.0, 9.0]]), "pid": [0, 1]})
    out_sub = ms.sjoin(left_sub, right, how="inner", all_matches=True, backend="cpu")
    assert isinstance(out_sub, MyDF)
    assert out_sub["y"].tolist() == [10]


@pytest.mark.parametrize("all_matches", [False, True])
def test_sjoin_preserves_hierarchical_columns(all_matches):
    columns = pd.MultiIndex.from_tuples(
        [("geometry", ""), ("value", "left")], names=["column", "side"]
    )
    left = pd.DataFrame([[shapely.Point(0.5, 0.5), 1]], columns=columns)
    right = pd.DataFrame({"geometry": [shapely.box(0, 0, 1, 1)], "right_value": [2]})
    expected = left.copy()
    expected["index_right"] = [0]
    expected["right_value"] = [2]

    result = ms.sjoin(left, right, all_matches=all_matches, backend="cpu")
    pd.testing.assert_frame_equal(result, expected)


def _metadata_frames():
    left = gpd.GeoDataFrame(
        {
            "pid": [0, 1, 2],
            "s": pd.Series(["a", "b", "c"], dtype=object),
            "when": pd.Series([dt.datetime(2026, 1, d) for d in (1, 2, 3)], dtype=object),
        },
        geometry=shapely.points([[0.5, 0.5], [1.5, 0.5], [9.0, 9.0]]),
        index=pd.Index(["p0", "p1", "p2"], dtype=object, name="key"),
        crs="EPSG:4326",
    )
    left.columns.name = "fields"
    left.attrs["source"] = {"run": [1, 2]}
    right = gpd.GeoDataFrame(
        {"zid": [10, 20]},
        geometry=[shapely.box(0, 0, 2, 1), shapely.box(1, 0, 3, 1)],
        crs="EPSG:4326",
    )
    return left, right


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("all_matches", [False, True])
def test_left_metadata_and_object_dtypes_survive_bulk_assembly(how, all_matches, prepared):
    left, right = _metadata_frames()
    rhs = ms.PreparedJoin(right) if prepared else right
    out = ms.sjoin(left, rhs, how=how, all_matches=all_matches, backend="cpu")
    # What base.iloc[...] / base.copy() carried over before the bulk path.
    assert out.attrs == left.attrs
    assert out.columns.name == "fields"
    # The DataFrame constructor re-infers bare object arrays (str on
    # pandas >= 3, datetime64 everywhere); the left frame's dtypes must hold.
    assert out.index.dtype == object and out.index.name == "key"
    assert out["s"].dtype == object
    assert out["when"].dtype == object
    expected_pos = [0, 1, 1, 2] if all_matches else [0, 1, 2]
    if how == "inner":
        expected_pos = [p for p in expected_pos if p != 2]
    assert out.index.tolist() == left.index[expected_pos].tolist()
    assert out["s"].tolist() == left["s"].iloc[expected_pos].tolist()


@pytest.mark.parametrize("how", ["inner", "left"])
def test_allows_duplicate_labels_flag_is_propagated_and_enforced(how):
    left, right = _metadata_frames()
    left = left.set_flags(allows_duplicate_labels=False)
    first = ms.sjoin(left, right, how=how, all_matches=False, backend="cpu")
    assert first.flags.allows_duplicate_labels is False
    # Point p1 lies in both boxes, so all-matches output repeats its label.
    with pytest.raises(pd.errors.DuplicateLabelError):
        ms.sjoin(left, right, how=how, all_matches=True, backend="cpu")


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("all_matches", [False, True])
def test_duplicate_left_columns_fallback_emits_no_warnings(how, all_matches):
    left_dup = pd.DataFrame(
        [[shapely.Point(0.5, 0.5), 1, 2], [shapely.Point(9.0, 9.0), 3, 4]],
        columns=["geometry", "x", "x"],
    )
    right = pd.DataFrame({"geometry": [shapely.box(0, 0, 1, 1)], "y": [10]})
    with warnings.catch_warnings():
        # pandas < 3 raised SettingWithCopyWarning on an iloc gather.
        warnings.simplefilter("error")
        out = ms.sjoin(left_dup, right, how=how, all_matches=all_matches, backend="cpu")
    assert list(out.columns) == ["geometry", "x", "x", "index_right", "y"]


def _pandas_major_minor():
    return tuple(int(p) for p in pd.__version__.split(".")[:2])


@pytest.mark.skipif(
    not ((2, 2) <= _pandas_major_minor() < (3, 0)),
    reason='copy_on_write="warn" exists only in pandas 2.2.x',
)
def test_first_match_left_join_does_not_alias_left_under_cow_warn_mode():
    left = gpd.GeoDataFrame(
        {"v": np.arange(3.0)},
        geometry=shapely.points([[0.5, 0.5], [1.5, 0.5], [9.0, 9.0]]),
    )
    right = gpd.GeoDataFrame({"z": [1]}, geometry=[shapely.box(0, 0, 2, 1)])
    with pd.option_context("mode.copy_on_write", "warn"):
        out = ms.sjoin(left, right, how="left", all_matches=False, backend="cpu")
    # "warn" mode still shares memory between Series, so the output must copy.
    assert not np.shares_memory(out["v"].to_numpy(), left["v"].to_numpy())


def test_numpy_extension_array_type_falls_back_to_pandas_array_before_2_1():
    from mlx_geospatial.join import _numpy_extension_array_type

    class PandasArray:  # stand-in for the pandas < 2.1 class name
        pass

    old_pd = types.SimpleNamespace(arrays=types.SimpleNamespace(PandasArray=PandasArray))
    assert _numpy_extension_array_type(old_pd) is PandasArray
    if hasattr(pd.arrays, "NumpyExtensionArray"):
        assert _numpy_extension_array_type(pd) is pd.arrays.NumpyExtensionArray


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("all_matches", [False, True])
def test_python_storage_string_dtype_survives_bulk_assembly(how, all_matches, prepared):
    # StringArray subclasses NumpyExtensionArray; the fast gather must not
    # strip it to a bare object array. Explicit python storage keeps this
    # covered when pyarrow (ArrowStringArray) is the default string backend.
    py_str = pd.StringDtype("python")
    left = gpd.GeoDataFrame(
        {"ls": pd.array(["a", "b", None], dtype=py_str)},
        geometry=shapely.points([[0.5, 0.5], [1.5, 0.5], [9.0, 9.0]]),
        index=pd.Index(pd.array(["p0", "p1", "p2"], dtype=py_str), name="key"),
        crs="EPSG:4326",
    )
    right = gpd.GeoDataFrame(
        {"rs": pd.array(["x", "y"], dtype=py_str)},
        geometry=[shapely.box(0, 0, 2, 1), shapely.box(1, 0, 3, 1)],
        index=pd.Index(pd.array(["A", "B"], dtype=py_str)),
        crs="EPSG:4326",
    )
    rhs = ms.PreparedJoin(right) if prepared else right
    out = ms.sjoin(left, rhs, how=how, all_matches=all_matches, backend="cpu")
    assert out.index.dtype == py_str
    assert out["ls"].dtype == py_str
    assert out["rs"].dtype == py_str
    if how == "inner" or all_matches:
        assert out["index_right"].dtype == py_str
    if how == "left":
        assert out["rs"].isna().iloc[-1]
