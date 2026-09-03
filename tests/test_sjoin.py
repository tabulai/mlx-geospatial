"""Deep tests for :func:`mlx_geospatial.sjoin` against ``geopandas.sjoin`` as golden.

Covers index-label handling (RangeIndex, shuffled integer, string), column
handling (collisions, geometry exclusion, dtype preservation), row-for-row
golden equivalence on non-overlapping polygons, min-index reduction on
overlapping polygons, plain-pandas round trips, CRS checks, left-join NaN
semantics, error paths, empty results, and the GPU-forced backend.

Golden comparisons exclude the documented fp32 boundary-tolerance band via
``near_boundary_mask`` (points within ``PolygonSet.boundary_tolerance`` of an
edge may classify either way by contract) and call ``geopandas.sjoin`` with
its default predicate (``intersects``), which is also ours; outside the band
``within`` would give the same golden.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

import geopandas as gpd

import mlx_geospatial as ms
from mlx_geospatial._dispatch import metal_available

from conftest import make_convex_polygons, near_boundary_mask


# ---------------------------------------------------------------------------
# helpers (local to this file by the rules)
# ---------------------------------------------------------------------------


def disjoint_boxes(nx: int, ny: int, *, side: float = 0.8) -> np.ndarray:
    """nx*ny axis-aligned boxes on a unit lattice with gaps: pairwise disjoint,
    so first-match == only-match."""
    polys = []
    for j in range(ny):
        for i in range(nx):
            polys.append(shapely.box(i, j, i + side, j + side))
    return np.array(polys, dtype=object)


def overlapping_boxes() -> np.ndarray:
    """Three boxes with pairwise and triple overlap regions."""
    return np.array(
        [
            shapely.box(0.0, 0.0, 2.0, 2.0),
            shapely.box(1.0, 1.0, 3.0, 3.0),
            shapely.box(0.5, 0.5, 2.5, 2.5),
        ],
        dtype=object,
    )


def scatter_points(rng: np.random.Generator, n: int, extent) -> np.ndarray:
    x0, y0, x1, y1 = extent
    return np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])


def ok_labels(left: pd.DataFrame, pts: np.ndarray, polys: np.ndarray):
    """Left index labels for points outside the documented tolerance band."""
    tol = ms.PolygonSet(polys).boundary_tolerance
    ok = ~near_boundary_mask(pts, polys, tol)
    return set(np.asarray(left.index)[ok].tolist())


def pair_counter(df: pd.DataFrame, labels=None) -> Counter:
    """Multiset of (left label, index_right label) rows, optionally filtered
    to left labels in ``labels``. NaN index_right rows are dropped."""
    matched = df["index_right"].notna()
    items = zip(df.index[matched].tolist(), df.loc[matched, "index_right"].tolist())
    if labels is not None:
        items = (p for p in items if p[0] in labels)
    return Counter(items)


# ---------------------------------------------------------------------------
# golden equivalence: non-overlapping polygons, row-for-row
# ---------------------------------------------------------------------------


def test_inner_rangeindex_row_for_row_golden(rng):
    polys = disjoint_boxes(5, 5)
    pts = scatter_points(rng, 1500, (-0.5, -0.5, 5.5, 5.5))
    left = gpd.GeoDataFrame({"pid": np.arange(1500)}, geometry=list(shapely.points(pts)))
    right = gpd.GeoDataFrame({"zid": np.arange(25)}, geometry=list(polys))

    ours = ms.sjoin(left, right, how="inner")
    theirs = gpd.sjoin(left, right, how="inner")
    ok = ok_labels(left, pts, polys)

    # disjoint polygons: first-match == only-match, so the joins must agree
    # row for row (as multisets of (left label, right label)).
    assert pair_counter(ours, ok) == pair_counter(theirs, ok)
    # at most one row per left point ever (first-match contract)
    assert not ours.index.duplicated().any()
    # left row order is preserved
    assert list(ours.index) == sorted(ours.index)
    # right attribute column joined and aligned
    assert (ours["zid"].to_numpy() == right["zid"].to_numpy()[ours["index_right"].to_numpy()]).all()
    # sanity: the join is not vacuous
    assert len(ours) > 200


def test_left_rangeindex_golden(rng):
    polys = disjoint_boxes(5, 5)
    pts = scatter_points(rng, 1200, (-0.5, -0.5, 5.5, 5.5))
    left = gpd.GeoDataFrame({"pid": np.arange(1200)}, geometry=list(shapely.points(pts)))
    right = gpd.GeoDataFrame({"zid": np.arange(25)}, geometry=list(polys))

    ours = ms.sjoin(left, right, how="left")
    theirs = gpd.sjoin(left, right, how="left")
    ok = ok_labels(left, pts, polys)

    assert len(ours) == len(left)
    assert list(ours.index) == list(left.index)
    assert pair_counter(ours, ok) == pair_counter(theirs, ok)
    # unmatched rows: NaN index_right and NaN right columns
    unmatched = ours["index_right"].isna()
    theirs_unmatched = set(theirs.index[theirs["index_right"].isna()]) & ok
    assert (set(ours.index[unmatched]) & ok) == theirs_unmatched
    assert ours.loc[unmatched, "zid"].isna().all()
    # matched rows carry the right column values of their polygon
    m = ~unmatched
    assert (
        ours.loc[m, "zid"].to_numpy()
        == right["zid"].to_numpy()[ours.loc[m, "index_right"].to_numpy().astype(int)]
    ).all()


# ---------------------------------------------------------------------------
# index-label handling
# ---------------------------------------------------------------------------


def test_shuffled_int_index_labels_carried(rng):
    polys = disjoint_boxes(4, 4)
    pts = scatter_points(rng, 800, (-0.5, -0.5, 4.5, 4.5))
    left_labels = rng.permutation(800) * 3 + 7  # unique, shuffled, non-range
    right_labels = rng.permutation(16) * 11 + 1000
    left = gpd.GeoDataFrame(
        {"pid": np.arange(800)},
        geometry=list(shapely.points(pts)),
        index=pd.Index(left_labels),
    )
    right = gpd.GeoDataFrame(
        {"zid": np.arange(16)}, geometry=list(polys), index=pd.Index(right_labels)
    )

    for how in ("inner", "left"):
        ours = ms.sjoin(left, right, how=how)
        theirs = gpd.sjoin(left, right, how=how)
        ok = ok_labels(left, pts, polys)
        assert pair_counter(ours, ok) == pair_counter(theirs, ok)
        # index_right carries the right frame's ORIGINAL labels
        matched = ours["index_right"].notna()
        assert set(ours.loc[matched, "index_right"]).issubset(set(right_labels.tolist()))
        # left labels are preserved as the output index
        assert set(ours.index).issubset(set(left_labels.tolist()))
        # containment is real: each matched point lies within its labeled polygon
        sample = ours[matched].head(50)
        geoms = right.geometry.loc[sample["index_right"]].to_numpy()
        pts_s = left.geometry.loc[sample.index].to_numpy()
        assert shapely.intersects(geoms, pts_s).all()
    # left join keeps every left row, in order
    assert list(ms.sjoin(left, right, how="left").index) == list(left.index)


def test_string_index_labels_on_right(rng):
    polys = disjoint_boxes(4, 4)
    pts = scatter_points(rng, 600, (-0.5, -0.5, 4.5, 4.5))
    labels = [f"zone_{i:02d}" for i in rng.permutation(16)]
    left = gpd.GeoDataFrame({"pid": np.arange(600)}, geometry=list(shapely.points(pts)))
    right = gpd.GeoDataFrame(
        {"zid": np.arange(16)}, geometry=list(polys), index=pd.Index(labels)
    )

    ours = ms.sjoin(left, right, how="inner")
    theirs = gpd.sjoin(left, right, how="inner")
    ok = ok_labels(left, pts, polys)
    assert pair_counter(ours, ok) == pair_counter(theirs, ok)
    assert set(ours["index_right"]).issubset(set(labels))

    ours_l = ms.sjoin(left, right, how="left")
    assert len(ours_l) == len(left)
    matched = ours_l["index_right"].notna()
    assert set(ours_l.loc[matched, "index_right"]).issubset(set(labels))
    assert ours_l.loc[~matched, "zid"].isna().all()
    # matched string labels agree with the golden inner join
    assert pair_counter(ours_l, ok) == pair_counter(theirs, ok)


# ---------------------------------------------------------------------------
# column handling
# ---------------------------------------------------------------------------


def test_column_collision_and_geometry_exclusion(rng):
    polys = disjoint_boxes(3, 3)
    pts = scatter_points(rng, 400, (-0.2, -0.2, 3.2, 3.2))
    left = gpd.GeoDataFrame(
        {"name": [f"p{i}" for i in range(400)], "val": rng.normal(size=400)},
        geometry=list(shapely.points(pts)),
    )
    right = gpd.GeoDataFrame(
        {"name": [f"z{i}" for i in range(9)], "pop": np.arange(9) * 100},
        geometry=list(polys),
    )

    out = ms.sjoin(left, right, how="inner")
    assert len(out) > 0
    # right columns joined in; collision suffixed _right; left name kept
    assert {"name", "val", "index_right", "name_right", "pop"} <= set(out.columns)
    # exactly one geometry column: the left one
    assert list(out.columns).count("geometry") == 1
    # left values intact under collision
    assert (out["name"].to_numpy() == np.array([f"p{i}" for i in out.index])).all()
    # right values aligned to the matched polygon
    idx = out["index_right"].to_numpy()
    assert (out["name_right"].to_numpy() == right["name"].to_numpy()[idx]).all()
    assert (out["pop"].to_numpy() == right["pop"].to_numpy()[idx]).all()
    # output geometries are the left point geometries, exactly
    assert shapely.equals_exact(
        np.asarray(out.geometry.to_numpy(), dtype=object),
        np.asarray(left.geometry.loc[out.index].to_numpy(), dtype=object),
    ).all()


def test_right_geometry_excluded_nondefault_name(rng):
    polys = disjoint_boxes(2, 2)
    pts = np.array([[0.4, 0.4], [1.4, 0.4], [5.0, 5.0]])
    left = gpd.GeoDataFrame({"pid": [0, 1, 2]}, geometry=list(shapely.points(pts)))
    right = gpd.GeoDataFrame(
        {"zid": np.arange(4), "geom": list(polys)}, geometry="geom"
    )
    out = ms.sjoin(left, right, how="inner")
    assert "geom" not in out.columns
    assert "zid" in out.columns
    assert len(out) == 2

    # right with ONLY a geometry column: no extra columns appear
    right_bare = gpd.GeoDataFrame(geometry=list(polys))
    out_bare = ms.sjoin(left, right_bare, how="inner")
    assert set(out_bare.columns) == {"pid", "geometry", "index_right"}
    assert len(out_bare) == 2


def test_left_dtypes_preserved(rng):
    polys = disjoint_boxes(3, 3)
    pts = scatter_points(rng, 300, (-0.2, -0.2, 3.2, 3.2))
    left = gpd.GeoDataFrame(
        {
            "i": np.arange(300, dtype=np.int64),
            "f": rng.normal(size=300),
            "s": [f"row{i}" for i in range(300)],
            "t": pd.date_range("2026-01-01", periods=300, freq="h"),
        },
        geometry=list(shapely.points(pts)),
    )
    right = gpd.GeoDataFrame({"zid": np.arange(9, dtype=np.int64)}, geometry=list(polys))

    for how in ("inner", "left"):
        out = ms.sjoin(left, right, how=how)
        assert out["i"].dtype == np.int64
        assert out["f"].dtype == np.float64
        assert out["s"].dtype == left["s"].dtype
        assert out["t"].dtype == left["t"].dtype
        # values survive the join untouched
        sub = left.loc[out.index]
        assert (out["i"].to_numpy() == sub["i"].to_numpy()).all()
        np.testing.assert_array_equal(out["f"].to_numpy(), sub["f"].to_numpy())
        assert (out["t"].to_numpy() == sub["t"].to_numpy()).all()


# ---------------------------------------------------------------------------
# overlapping polygons: min-index reduction
# ---------------------------------------------------------------------------


def test_overlap_min_reduction_rangeindex(rng):
    polys = make_convex_polygons(60, rng, radius_frac=0.15)  # dense -> overlaps
    pts = scatter_points(rng, 2000, (-1.1, -1.1, 1.1, 1.1))
    left = gpd.GeoDataFrame({"pid": np.arange(2000)}, geometry=list(shapely.points(pts)))
    right = gpd.GeoDataFrame({"zid": np.arange(60)}, geometry=list(polys))

    ours = ms.sjoin(left, right, how="inner")
    theirs = gpd.sjoin(left, right, how="inner")
    ok = ok_labels(left, pts, polys)

    # sanity: overlaps actually occur (some point matches >= 2 polygons)
    assert theirs.index.duplicated().any()
    # RangeIndex: label == position, so min label == min polygon index
    theirs_min = theirs.groupby(level=0)["index_right"].min()
    ours_m = ours["index_right"]
    assert {k: v for k, v in theirs_min.items() if k in ok} == {
        k: v for k, v in ours_m.items() if k in ok
    }
    assert not ours.index.duplicated().any()


def test_overlap_shuffled_labels_use_positional_min(rng):
    polys = overlapping_boxes()
    labels = pd.Index([30, 10, 20])  # label order disagrees with position order
    pts = np.vstack(
        [
            scatter_points(rng, 400, (-0.5, -0.5, 3.5, 3.5)),
            [[0.7, 0.7]],  # in polys 0 and 2 only: pos-min 0 -> label 30;
            #                a label-min reduction would wrongly give 20
            [[1.5, 1.5]],  # in all three: pos-min 0 -> label 30
            [[2.7, 2.7]],  # in poly 1 only -> label 10
        ]
    )
    n = len(pts)
    left = gpd.GeoDataFrame({"pid": np.arange(n)}, geometry=list(shapely.points(pts)))
    right = gpd.GeoDataFrame({"zid": np.arange(3)}, geometry=list(polys), index=labels)

    ours = ms.sjoin(left, right, how="inner")
    theirs = gpd.sjoin(left, right, how="inner")
    ok = ok_labels(left, pts, polys)

    # golden expectation: label of the *minimum position* among all matches
    pos = right.index.get_indexer(theirs["index_right"])
    minpos = pd.Series(pos, index=theirs.index).groupby(level=0).min()
    expected = {li: labels[p] for li, p in minpos.items() if li in ok}
    got = {k: v for k, v in ours["index_right"].items() if k in ok}
    assert got == expected
    # the deterministic probes distinguish positional-min from label-min
    assert got[n - 3] == 30
    assert got[n - 2] == 30
    assert got[n - 1] == 10
    label_min = theirs.groupby(level=0)["index_right"].min().to_dict()
    assert label_min[n - 3] == 20  # proves the test can tell the two apart


# ---------------------------------------------------------------------------
# plain pandas round trip
# ---------------------------------------------------------------------------


def test_plain_pandas_dataframes_roundtrip(rng):
    polys = disjoint_boxes(4, 4)
    pts = scatter_points(rng, 700, (-0.5, -0.5, 4.5, 4.5))
    right_labels = pd.Index([f"z{i:02d}" for i in range(16)])

    left_pd = pd.DataFrame(
        {"geometry": list(shapely.points(pts)), "pid": np.arange(700)}
    )
    right_pd = pd.DataFrame(
        {"geometry": list(polys), "zname": [f"name_{i}" for i in range(16)]},
        index=right_labels,
    )
    left_g = gpd.GeoDataFrame(left_pd.drop(columns="geometry"), geometry=list(shapely.points(pts)))
    right_g = gpd.GeoDataFrame(right_pd.drop(columns="geometry"), geometry=list(polys), index=right_labels)

    ok = ok_labels(left_pd, pts, polys)
    golden = gpd.sjoin(left_g, right_g, how="inner")

    out = ms.sjoin(left_pd, right_pd, how="inner")
    assert isinstance(out, pd.DataFrame)
    assert pair_counter(out, ok) == pair_counter(golden, ok)
    # right geometry excluded, attribute joined and aligned
    assert list(out.columns).count("geometry") == 1
    assert "zname" in out.columns
    lab2name = dict(zip(right_labels, right_pd["zname"]))
    assert (out["zname"] == out["index_right"].map(lab2name)).all()
    # geometry column still holds the left shapely points
    assert shapely.intersects(
        np.asarray(out["geometry"].to_numpy(), dtype=object),
        np.asarray(right_pd["geometry"].loc[out["index_right"]].to_numpy(), dtype=object),
    ).all()

    # left join on plain pandas: full length, NaN for unmatched
    out_l = ms.sjoin(left_pd, right_pd, how="left")
    assert len(out_l) == len(left_pd)
    assert pair_counter(out_l, ok) == pair_counter(golden, ok)
    unmatched = out_l["index_right"].isna()
    assert out_l.loc[unmatched, "zname"].isna().all()


# ---------------------------------------------------------------------------
# CRS handling
# ---------------------------------------------------------------------------


def test_crs_mismatch_raises_matching_passes(rng):
    polys = disjoint_boxes(2, 2)
    pts = np.array([[0.4, 0.4], [1.4, 1.4], [9.0, 9.0]])
    left = gpd.GeoDataFrame({"pid": [0, 1, 2]}, geometry=list(shapely.points(pts)))
    right = gpd.GeoDataFrame({"zid": np.arange(4)}, geometry=list(polys))

    with pytest.raises(ValueError):
        ms.sjoin(left.set_crs("EPSG:4326"), right.set_crs("EPSG:3857"))

    # matching CRS passes
    out = ms.sjoin(left.set_crs("EPSG:4326"), right.set_crs("EPSG:4326"), how="inner")
    assert len(out) == 2
    # one side None passes, either way around
    assert len(ms.sjoin(left.set_crs("EPSG:4326"), right, how="inner")) == 2
    assert len(ms.sjoin(left, right.set_crs("EPSG:4326"), how="inner")) == 2
    # both None passes
    assert len(ms.sjoin(left, right, how="inner")) == 2


# ---------------------------------------------------------------------------
# errors and empty results
# ---------------------------------------------------------------------------


def test_bad_how_and_predicate_raise(rng):
    left = gpd.GeoDataFrame({"pid": [0]}, geometry=[shapely.Point(0.5, 0.5)])
    right = gpd.GeoDataFrame({"zid": [0]}, geometry=[shapely.box(0, 0, 1, 1)])
    for bad_how in ("outer", "right", "cross", "INNER"):
        with pytest.raises(ValueError):
            ms.sjoin(left, right, how=bad_how)
    for bad_pred in ("touches", "contains", "dwithin", ""):
        with pytest.raises(ValueError):
            ms.sjoin(left, right, predicate=bad_pred)
    # the two documented predicates are accepted
    assert len(ms.sjoin(left, right, predicate="within")) == 1
    assert len(ms.sjoin(left, right, predicate="intersects")) == 1


def test_empty_right_no_matches_and_preserved_schema(rng):
    left = gpd.GeoDataFrame(
        {"pid": [0, 1, 2]},
        geometry=list(shapely.points([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])),
    )
    right = gpd.GeoDataFrame({"zid": pd.Series(dtype=int)}, geometry=[])

    out_inner = ms.sjoin(left, right, how="inner")
    assert len(out_inner) == 0
    assert set(out_inner.columns) == {"pid", "geometry", "index_right", "zid"}

    out_left = ms.sjoin(left, right, how="left")
    assert len(out_left) == len(left)
    assert out_left["index_right"].isna().all()
    assert out_left["zid"].isna().all()


def test_left_with_existing_index_right_column_errors():
    left = gpd.GeoDataFrame(
        {
            "index_right": [10, 20],
            "value": ["a", "b"],
            "geometry": list(shapely.points([[0.5, 0.5], [1.5, 1.5]])),
        }
    )
    right = gpd.GeoDataFrame({"kind": ["zone"]}, geometry=[shapely.box(0, 0, 1, 1)])
    with pytest.raises(ValueError):
        ms.sjoin(left, right, how="inner")


def test_empty_inner_result_well_formed(rng):
    polys = disjoint_boxes(3, 3)
    pts = scatter_points(rng, 50, (100.0, 100.0, 101.0, 101.0))  # far away
    left = gpd.GeoDataFrame(
        {"pid": np.arange(50), "tag": [f"t{i}" for i in range(50)]},
        geometry=list(shapely.points(pts)),
    )
    right = gpd.GeoDataFrame({"zid": np.arange(9)}, geometry=list(polys))

    out = ms.sjoin(left, right, how="inner")
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0
    assert {"pid", "tag", "geometry", "index_right", "zid"} == set(out.columns)
    # left dtypes survive even in the empty frame
    assert out["pid"].dtype == np.int64

    # string right labels: still a well-formed empty frame
    right_s = gpd.GeoDataFrame(
        {"zid": np.arange(9)},
        geometry=list(polys),
        index=pd.Index([f"z{i}" for i in range(9)]),
    )
    out_s = ms.sjoin(left, right_s, how="inner")
    assert len(out_s) == 0
    assert {"pid", "tag", "geometry", "index_right", "zid"} == set(out_s.columns)

    # left join on the same data: all rows kept, everything right-side NaN
    out_l = ms.sjoin(left, right, how="left")
    assert len(out_l) == 50
    assert out_l["index_right"].isna().all()
    assert out_l["zid"].isna().all()


# ---------------------------------------------------------------------------
# GPU-forced backend through sjoin
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not metal_available(), reason="Metal not available")
def test_gpu_backend_golden(rng, monkeypatch):
    polys = np.concatenate([disjoint_boxes(4, 4), overlapping_boxes()])
    pts = scatter_points(rng, 1500, (-0.5, -0.5, 4.5, 4.5))
    left = gpd.GeoDataFrame({"pid": np.arange(1500)}, geometry=list(shapely.points(pts)))
    right = gpd.GeoDataFrame({"zid": np.arange(len(polys))}, geometry=list(polys))

    default = ms.sjoin(left, right, how="inner")  # auto -> CPU at this size
    monkeypatch.setenv("MLX_GEOSPATIAL_BACKEND", "gpu")
    ours = ms.sjoin(left, right, how="inner")
    theirs = gpd.sjoin(left, right, how="inner")
    ok = ok_labels(left, pts, polys)

    theirs_min = theirs.groupby(level=0)["index_right"].min()  # RangeIndex: label==pos
    assert {k: v for k, v in theirs_min.items() if k in ok} == {
        k: v for k, v in ours["index_right"].items() if k in ok
    }
    # GPU and CPU paths agree outside the tolerance band
    assert pair_counter(ours, ok) == pair_counter(default, ok)

    # left join through the GPU path
    ours_l = ms.sjoin(left, right, how="left")
    assert len(ours_l) == len(left)
    assert pair_counter(ours_l, ok) == pair_counter(ours, ok)


# ---------------------------------------------------------------------------
# multipolygons and holes through sjoin
# ---------------------------------------------------------------------------


def test_multipolygon_and_hole_right(rng):
    holed = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)]]
    )
    multi = MultiPolygon([shapely.box(6, 6, 7, 7), shapely.box(9, 9, 10, 10)])
    polys = np.array([holed, multi], dtype=object)
    pts = np.array(
        [
            [0.5, 0.5],  # in holed shell
            [2.0, 2.0],  # in the hole -> no match
            [6.5, 6.5],  # part 1 of multi
            [9.5, 9.5],  # part 2 of multi
            [8.0, 8.0],  # between the parts -> no match
        ]
    )
    left = gpd.GeoDataFrame({"pid": np.arange(5)}, geometry=list(shapely.points(pts)))
    right = gpd.GeoDataFrame(
        {"kind": ["holed", "multi"]},
        geometry=list(polys),
        index=pd.Index(["H", "M"]),
    )
    ours = ms.sjoin(left, right, how="inner")
    theirs = gpd.sjoin(left, right, how="inner")
    assert pair_counter(ours) == pair_counter(theirs)
    assert pair_counter(ours) == Counter({(0, "H"): 1, (2, "M"): 1, (3, "M"): 1})
    out_l = ms.sjoin(left, right, how="left")
    assert out_l["index_right"].isna().tolist() == [False, True, False, False, True]


# ---------------------------------------------------------------------------
# known defect: duplicate right index labels
# ---------------------------------------------------------------------------


def test_duplicate_right_labels_no_row_fanout():
    polys = [shapely.box(0, 0, 1, 1), shapely.box(2, 2, 3, 3)]
    pts = shapely.points(np.array([[0.5, 0.5], [2.5, 2.5]]))
    left = gpd.GeoDataFrame({"pid": [0, 1]}, geometry=list(pts))
    right = gpd.GeoDataFrame(
        {"zid": ["A", "B"]}, geometry=polys, index=pd.Index([5, 5])
    )
    ours = ms.sjoin(left, right, how="inner")
    theirs = gpd.sjoin(left, right, how="inner")
    # each point is in exactly one polygon: one row per point, and the right
    # attribute must come from the containing polygon
    assert len(ours) == len(theirs) == 2
    assert sorted(ours["zid"]) == ["A", "B"]
