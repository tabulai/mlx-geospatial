"""Deep tests for mlx_spatial._ingest (PolygonSet, as_point_array) and _precision.

Covers: every accepted input form, ragged-offset collapse for mixed
Polygon/MultiPolygon sets (validated against shapely.get_rings /
shapely.get_coordinates ground truth), the EdgeData edge list, empty-geometry
handling, origin/re-centering semantics, and validation errors.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

import mlx_spatial as ms
from mlx_spatial._ingest import EdgeData
from mlx_spatial._precision import (
    boundary_tolerance,
    compute_origin,
    float32_resolution,
    recenter,
)

from conftest import (
    make_convex_polygons,
    make_donut,
    make_multipolygon,
    near_boundary_mask,
)


# ---------------------------------------------------------------------------
# Ground-truth helpers (shapely float64 is the oracle)
# ---------------------------------------------------------------------------


def _rings_of_geom(geom):
    """All rings of one (multi)polygon in ragged-array traversal order:
    parts in order, each part exterior-then-interiors."""
    if geom.is_empty:
        return []
    parts = shapely.get_parts(geom) if geom.geom_type == "MultiPolygon" else [geom]
    rings = []
    for p in parts:
        if not p.is_empty:
            rings.extend(shapely.get_rings(p))
    return rings


def _traversal_rings(geoms):
    rings = []
    for g in np.asarray(geoms, dtype=object):
        rings.extend(_rings_of_geom(g))
    return rings


def _gt_counts(geoms):
    """(n_polygons, n_rings, n_vertices) from shapely."""
    arr = np.asarray(geoms, dtype=object)
    rings = _traversal_rings(arr)
    n_vertices = int(sum(len(shapely.get_coordinates(r)) for r in rings))
    return len(arr), len(rings), n_vertices


def _gt_per_geom(geoms):
    """Per-geometry (ring_count, vertex_count) lists."""
    ring_counts, vert_counts = [], []
    for g in np.asarray(geoms, dtype=object):
        rs = _rings_of_geom(g)
        ring_counts.append(len(rs))
        vert_counts.append(int(sum(len(shapely.get_coordinates(r)) for r in rs)))
    return ring_counts, vert_counts


def _expected_edges(geom, origin):
    """(E, 4) float32 segments of one geometry in the local frame — computed
    exactly the way ingestion does (float64 subtract, then float32 cast), so
    the comparison is bit-exact."""
    segs = []
    for r in _rings_of_geom(geom):
        local = recenter(shapely.get_coordinates(r), origin)  # float32
        segs.append(np.concatenate([local[:-1], local[1:]], axis=1))
    if not segs:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(segs, axis=0)


def _mixed_geoms():
    """Hand-built mix: polygon-with-hole, plain triangle, multipolygon with a
    holed part, plain multipolygon. Forces the ragged MULTIPOLYGON promotion."""
    sq_hole = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
    )
    tri = Polygon([(20, 0), (25, 0), (25, 5)])
    mp_holed = MultiPolygon(
        [
            Polygon(
                [(30, 30), (35, 30), (35, 35), (30, 35)],
                holes=[[(32, 31), (33, 31), (33, 32)]],
            ),
            Polygon([(40, 40), (45, 40), (45, 45)]),
        ]
    )
    mp_plain = MultiPolygon(
        [
            Polygon([(-10, -10), (-8, -10), (-8, -8)]),
            Polygon([(-6, -6), (-4, -6), (-4, -4), (-6, -4)]),
        ]
    )
    return np.array([sq_hole, tri, mp_holed, mp_plain], dtype=object)


def _assert_same_polyset(a: ms.PolygonSet, b: ms.PolygonSet):
    assert a.n_polygons == b.n_polygons
    assert a.n_rings == b.n_rings
    assert a.n_vertices == b.n_vertices
    np.testing.assert_array_equal(a.origin, b.origin)
    assert np.array_equal(np.array(a.coords), np.array(b.coords))
    assert np.array_equal(np.array(a.ring_offsets), np.array(b.ring_offsets))
    assert np.array_equal(
        np.array(a.geom_ring_offsets), np.array(b.geom_ring_offsets)
    )
    assert np.array_equal(a.bboxes, b.bboxes, equal_nan=True)


# ---------------------------------------------------------------------------
# PolygonSet construction: input forms
# ---------------------------------------------------------------------------


def test_input_forms_equivalent(rng):
    """list / object array / GeoSeries / GeoDataFrame all ingest identically."""
    gpd = pytest.importorskip("geopandas")
    polys = make_convex_polygons(12, rng)
    ref = ms.PolygonSet(polys)  # np object array

    forms = [
        list(polys),
        gpd.GeoSeries(list(polys)),
        gpd.GeoDataFrame({"name": [f"z{i}" for i in range(12)]}, geometry=list(polys)),
    ]
    for form in forms:
        _assert_same_polyset(ms.PolygonSet(form), ref)
    # from_shapely is the same constructor
    _assert_same_polyset(ms.from_shapely(list(polys)), ref)


def test_single_geometry_not_in_a_list():
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (2, 1), (2, 2)]])
    ps = ms.PolygonSet(poly)
    assert ps.n_polygons == 1
    assert ps.n_rings == 2
    assert ps.n_vertices == len(shapely.get_coordinates(poly))
    _assert_same_polyset(ps, ms.PolygonSet([poly]))

    mp = MultiPolygon([shapely.box(0, 0, 1, 1), shapely.box(3, 3, 4, 4)])
    ps_mp = ms.PolygonSet(mp)
    assert ps_mp.n_polygons == 1
    assert ps_mp.n_rings == 2
    assert ps_mp.n_vertices == 10


def test_mixed_promotion_offsets_collapse():
    """Mixed Polygon+MultiPolygon: shapely promotes to the multipolygon ragged
    type; the collapsed geom_ring_offsets must still delimit each input
    geometry's rings/vertices exactly."""
    arr = _mixed_geoms()
    ps = ms.PolygonSet(arr)

    n_polygons, n_rings, n_vertices = _gt_counts(arr)
    assert (ps.n_polygons, ps.n_rings, ps.n_vertices) == (n_polygons, n_rings, n_vertices)
    # hand-checked: rings per geom [2,1,3,2]; verts per geom [10,4,13,9]
    ring_counts, vert_counts = _gt_per_geom(arr)
    assert ring_counts == [2, 1, 3, 2]
    assert vert_counts == [10, 4, 13, 9]

    geom_ring = np.array(ps.geom_ring_offsets)
    ring_off = np.array(ps.ring_offsets)
    assert geom_ring.shape == (n_polygons + 1,)
    assert ring_off.shape == (n_rings + 1,)
    assert geom_ring[0] == 0 and geom_ring[-1] == n_rings
    assert ring_off[0] == 0 and ring_off[-1] == n_vertices
    np.testing.assert_array_equal(np.diff(geom_ring), ring_counts)
    # per-geometry vertex spans
    np.testing.assert_array_equal(np.diff(ring_off[geom_ring]), vert_counts)
    # per-ring vertex counts in traversal order
    ring_lens = [len(shapely.get_coordinates(r)) for r in _traversal_rings(arr)]
    np.testing.assert_array_equal(np.diff(ring_off), ring_lens)
    # coords are exactly the recentered shapely coordinate dump, in order
    expected = recenter(shapely.get_coordinates(arr), ps.origin)  # float32
    assert np.array_equal(np.array(ps.coords), expected)


def test_counts_polygon_path_with_holes(rng):
    """Pure-Polygon ragged path: donuts + random convex polygons."""
    polys = np.concatenate(
        [
            np.array(
                [make_donut(0.0, 0.0, 1.0, 0.5, k=16), make_donut(5.0, 5.0, 2.0, 1.0, k=12)],
                dtype=object,
            ),
            make_convex_polygons(20, rng),
        ]
    )
    ps = ms.PolygonSet(polys)
    assert (ps.n_polygons, ps.n_rings, ps.n_vertices) == _gt_counts(polys)
    assert ps.n_vertices == len(shapely.get_coordinates(polys))
    ring_counts, vert_counts = _gt_per_geom(polys)
    geom_ring = np.array(ps.geom_ring_offsets)
    ring_off = np.array(ps.ring_offsets)
    np.testing.assert_array_equal(np.diff(geom_ring), ring_counts)
    np.testing.assert_array_equal(np.diff(ring_off[geom_ring]), vert_counts)


def test_counts_multipolygon_path_with_holes(rng):
    """Pure-MultiPolygon ragged path, including a holed part."""
    holed_part = Polygon(
        [(10, 10), (20, 10), (20, 20), (10, 20)],
        holes=[[(12, 12), (14, 12), (14, 14), (12, 14)], [(16, 16), (18, 16), (18, 18)]],
    )
    geoms = np.array(
        [
            make_multipolygon(rng, parts=3),
            MultiPolygon([holed_part, shapely.box(30, 30, 31, 31)]),
            make_multipolygon(rng, parts=2),
        ],
        dtype=object,
    )
    ps = ms.PolygonSet(geoms)
    assert (ps.n_polygons, ps.n_rings, ps.n_vertices) == _gt_counts(geoms)
    ring_counts, vert_counts = _gt_per_geom(geoms)
    assert ring_counts[1] == 4  # 2 shells + 2 holes in geometry 1
    geom_ring = np.array(ps.geom_ring_offsets)
    ring_off = np.array(ps.ring_offsets)
    np.testing.assert_array_equal(np.diff(geom_ring), ring_counts)
    np.testing.assert_array_equal(np.diff(ring_off[geom_ring]), vert_counts)
    expected = recenter(shapely.get_coordinates(geoms), ps.origin)
    assert np.array_equal(np.array(ps.coords), expected)


# ---------------------------------------------------------------------------
# edges property
# ---------------------------------------------------------------------------


def _check_edges(ps: ms.PolygonSet, geoms):
    ed = ps.edges
    assert isinstance(ed, EdgeData)
    edges_np = np.array(ed.edges)
    rings = _traversal_rings(geoms)
    total_edges = int(sum(len(shapely.get_coordinates(r)) - 1 for r in rings))
    assert total_edges == ps.n_vertices - ps.n_rings
    assert edges_np.shape == (total_edges, 4)
    assert edges_np.dtype == np.float32

    offs = ed.geom_offsets_np
    assert offs.shape == (ps.n_polygons + 1,)
    assert offs[0] == 0 and offs[-1] == total_edges
    assert np.all(np.diff(offs) >= 0)
    assert np.array_equal(np.array(ed.geom_offsets_mx), offs)

    arr = np.asarray(geoms, dtype=object)
    for i, g in enumerate(arr):
        seg = edges_np[offs[i] : offs[i + 1]]
        exp = _expected_edges(g, ps.origin)
        assert np.array_equal(seg, exp), f"edge mismatch for geometry {i}"
        # each geometry contributes (vertices - rings) edges
        rs = _rings_of_geom(g)
        assert offs[i + 1] - offs[i] == sum(
            len(shapely.get_coordinates(r)) - 1 for r in rs
        )


def test_edges_mixed_set():
    arr = _mixed_geoms()
    _check_edges(ms.PolygonSet(arr), arr)


def test_edges_polygon_path(rng):
    polys = np.concatenate(
        [make_convex_polygons(15, rng), np.array([make_donut(0, 0, 0.5, 0.2, k=10)], dtype=object)]
    )
    ps = ms.PolygonSet(polys)
    _check_edges(ps, polys)
    # cached: same object on second access
    assert ps.edges is ps.edges


# ---------------------------------------------------------------------------
# Empty polygons
# ---------------------------------------------------------------------------


def test_empty_polygon_polygon_path():
    box = shapely.box(0, 0, 2, 2)
    ps = ms.PolygonSet([Polygon(), box])  # stays on the pure-Polygon path
    assert ps.n_polygons == 2
    assert ps.n_rings == 1
    assert ps.n_vertices == 5
    assert np.isnan(ps.bboxes[0]).all()
    np.testing.assert_array_equal(ps.bboxes[1], [0, 0, 2, 2])
    # empty row replaced by an impossible box
    np.testing.assert_array_equal(np.array(ps.bboxes_local)[0], [1.0, 1.0, -1.0, -1.0])
    # edges: geometry 0 spans zero edges
    offs = ps.edges.geom_offsets_np
    assert offs[0] == offs[1] == 0 and offs[2] == 4

    pts = np.array([[1.0, 1.0], [0.5, 1.5], [5.0, 5.0]])
    for backend in ("gpu", "cpu"):
        got = np.asarray(ms.point_in_polygon(pts, ps, backend=backend))
        # empty polygon is index 0 and must never win the min-index reduction
        np.testing.assert_array_equal(got, [1, 1, -1])
    cm = np.array(ms.contains_matrix(pts, ps))
    assert not cm[:, 0].any()
    np.testing.assert_array_equal(cm[:, 1], [True, True, False])


def test_empty_polygon_promoted_path():
    """Empty polygon inside a mixed set (promoted to the multipolygon ragged
    type: the empty becomes a zero-ring geometry after collapse)."""
    box = shapely.box(0, 0, 2, 2)
    mp = MultiPolygon([shapely.box(10, 10, 12, 12), shapely.box(20, 20, 21, 21)])
    geoms = np.array([Polygon(), box, mp], dtype=object)
    ps = ms.PolygonSet(geoms)
    assert (ps.n_polygons, ps.n_rings, ps.n_vertices) == (3, 3, 15)
    ring_counts, vert_counts = _gt_per_geom(geoms)
    assert ring_counts == [0, 1, 2]
    geom_ring = np.array(ps.geom_ring_offsets)
    np.testing.assert_array_equal(np.diff(geom_ring), ring_counts)
    np.testing.assert_array_equal(
        np.diff(np.array(ps.ring_offsets)[geom_ring]), vert_counts
    )
    assert np.isnan(ps.bboxes[0]).all()
    _check_edges(ps, geoms)

    pts = np.array([[1.0, 1.0], [11.0, 11.0], [20.5, 20.5], [-9.0, -9.0]])
    for backend in ("gpu", "cpu"):
        got = np.asarray(ms.point_in_polygon(pts, ps, backend=backend))
        np.testing.assert_array_equal(got, [1, 2, 2, -1])
    cm = np.array(ms.contains_matrix(pts, ps))
    assert not cm[:, 0].any()


def test_all_empty_set_ingests():
    ps = ms.PolygonSet([Polygon(), Polygon()])
    assert (ps.n_polygons, ps.n_rings, ps.n_vertices) == (2, 0, 0)
    assert np.isnan(ps.bboxes).all()
    np.testing.assert_array_equal(ps.origin, [0.0, 0.0])
    assert ps.boundary_tolerance > 0.0
    ed = ps.edges
    assert np.array(ed.edges).shape == (0, 4)
    np.testing.assert_array_equal(ed.geom_offsets_np, [0, 0, 0])
    cm = np.array(ms.contains_matrix(np.array([[0.0, 0.0], [1.0, 1.0]]), ps))
    assert cm.shape == (2, 2) and not cm.any()


# ---------------------------------------------------------------------------
# Origin / precision (_precision.py)
# ---------------------------------------------------------------------------


def test_default_origin_is_vertex_bbox_center(rng):
    polys = make_convex_polygons(10, rng, extent=(-74.3, 40.5, -73.7, 40.9))
    ps = ms.PolygonSet(polys)
    coords = shapely.get_coordinates(polys)
    expected = (coords.min(axis=0) + coords.max(axis=0)) / 2.0
    np.testing.assert_allclose(ps.origin, expected, rtol=0, atol=0)
    assert ps.origin.dtype == np.float64
    # max_abs_coord is the local-frame half extent
    local = recenter(coords, ps.origin)
    assert ps.max_abs_coord == float(np.abs(local).max())
    assert ps.boundary_tolerance == boundary_tolerance(ps.max_abs_coord)


def test_origin_override(rng):
    polys = make_convex_polygons(10, rng)
    ps_default = ms.PolygonSet(polys)
    override = np.array([0.25, -0.5])
    ps = ms.PolygonSet(polys, origin=override)
    np.testing.assert_array_equal(ps.origin, override)
    expected = recenter(shapely.get_coordinates(polys), override)
    assert np.array_equal(np.array(ps.coords), expected)
    # original-frame bboxes are unaffected by the origin choice
    np.testing.assert_array_equal(ps.bboxes, ps_default.bboxes)
    # containment is origin-invariant outside the tolerance band
    pts = np.column_stack([rng.uniform(-1.1, 1.1, 400), rng.uniform(-1.1, 1.1, 400)])
    tol = max(ps.boundary_tolerance, ps_default.boundary_tolerance)
    ok = ~near_boundary_mask(pts, polys, tol)
    a = np.array(ms.contains_matrix(pts, ps))
    b = np.array(ms.contains_matrix(pts, ps_default))
    assert np.array_equal(a[ok], b[ok])
    # from_shapely forwards the keyword
    np.testing.assert_array_equal(
        ms.from_shapely(polys, origin=(1.0, 2.0)).origin, [1.0, 2.0]
    )


def test_recenter_roundtrip_float64(rng):
    pts = np.column_stack([rng.uniform(-74.3, -73.7, 500), rng.uniform(40.5, 40.9, 500)])
    origin = compute_origin(pts)
    local = recenter(pts, origin, dtype=np.float64)
    assert local.dtype == np.float64
    assert local.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(local + origin, pts, rtol=0, atol=1e-12)
    # translation only: pairwise deltas survive exactly in float64
    np.testing.assert_allclose(
        local[1:] - local[:-1], pts[1:] - pts[:-1], rtol=0, atol=1e-12
    )
    # default cast is float32
    assert recenter(pts, origin).dtype == np.float32


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_compute_origin_ignores_nan():
    arr = np.array([[np.nan, 5.0], [2.0, np.nan], [4.0, 7.0], [0.0, 3.0]])
    np.testing.assert_array_equal(compute_origin(arr), [2.0, 5.0])
    # all-NaN and empty inputs fall back to the zero origin
    np.testing.assert_array_equal(compute_origin(np.full((3, 2), np.nan)), [0.0, 0.0])
    np.testing.assert_array_equal(compute_origin(np.empty((0, 2))), [0.0, 0.0])
    np.testing.assert_array_equal(compute_origin(), [0.0, 0.0])


def test_compute_origin_multiple_arrays_and_shapes():
    a = np.array([[0.0, 0.0], [2.0, 2.0]])
    b = np.array([[[10.0, 6.0], [4.0, -2.0]]])  # (..., 2) is flattened
    np.testing.assert_array_equal(compute_origin(a, b), [5.0, 2.0])
    np.testing.assert_array_equal(
        compute_origin(a, b), compute_origin(np.concatenate([a, b.reshape(-1, 2)]))
    )


def test_compute_origin_midpoint_does_not_overflow():
    coords = np.array(
        [
            [1.6e308, -1.7e308],
            [1.7e308, -1.6e308],
        ]
    )
    origin = compute_origin(coords)
    assert np.isfinite(origin).all()
    np.testing.assert_allclose(origin, [1.65e308, -1.65e308], rtol=1e-15)

    # Opposite-sign extrema need a different overflow-safe midpoint form.
    max_f64 = np.finfo(np.float64).max
    np.testing.assert_array_equal(
        compute_origin(np.array([[-max_f64, -max_f64], [max_f64, max_f64]])),
        [0.0, 0.0],
    )


def test_float32_resolution_properties():
    assert float32_resolution(0.0) > 0.0
    assert float32_resolution(np.inf) > 0.0
    assert float32_resolution(-100.0) == float32_resolution(100.0)
    assert float32_resolution(1.0) == float(np.spacing(np.float32(1.0)))
    # monotone growth with magnitude
    mags = [1e-3, 1.0, 100.0, 1e4]
    res = [float32_resolution(m) for m in mags]
    assert all(r2 > r1 for r1, r2 in zip(res, res[1:]))
    assert boundary_tolerance(100.0) == 8.0 * float32_resolution(100.0)
    assert boundary_tolerance(0.0) > 0.0


def test_boundary_tolerance_scales_with_extent_not_translation():
    small = ms.PolygonSet([shapely.box(0, 0, 1, 1), shapely.box(2, 2, 3, 3)])
    big = ms.PolygonSet(
        [shapely.box(0, 0, 1000, 1000), shapely.box(2000, 2000, 3000, 3000)]
    )
    assert small.boundary_tolerance > 0.0
    assert big.boundary_tolerance > 100.0 * small.boundary_tolerance
    # a pure translation must NOT inflate the tolerance (re-centering at work)
    shifted = ms.PolygonSet(
        [shapely.box(10000, 10000, 10001, 10001), shapely.box(10002, 10002, 10003, 10003)]
    )
    assert shifted.max_abs_coord == small.max_abs_coord == 1.5
    assert shifted.boundary_tolerance == small.boundary_tolerance


def test_bboxes_local_widened_conservatively(rng):
    polys = make_convex_polygons(30, rng, extent=(-74.3, 40.5, -73.7, 40.9))
    ps = ms.PolygonSet(polys)
    local_true = ps.bboxes.copy()
    local_true[:, [0, 2]] -= ps.origin[0]
    local_true[:, [1, 3]] -= ps.origin[1]
    bl = np.array(ps.bboxes_local, dtype=np.float64)
    assert (bl[:, 0] < local_true[:, 0]).all()
    assert (bl[:, 1] < local_true[:, 1]).all()
    assert (bl[:, 2] > local_true[:, 2]).all()
    assert (bl[:, 3] > local_true[:, 3]).all()
    # ...but only by a few ulps, not sloppily
    pad = 16.0 * float32_resolution(ps.max_abs_coord)
    assert (np.abs(bl - local_true) < pad).all()


def test_polygonset_dtypes_and_shapes(rng):
    polys = make_convex_polygons(7, rng)
    ps = ms.PolygonSet(polys)
    assert ps.coords.dtype == mx.float32
    assert ps.coords.shape == (ps.n_vertices, 2)
    assert ps.ring_offsets.dtype == mx.int32
    assert ps.geom_ring_offsets.dtype == mx.int32
    assert ps.bboxes_local.dtype == mx.float32
    assert ps.bboxes_local.shape == (ps.n_polygons, 4)
    assert ps.bboxes.dtype == np.float64
    assert ps.bboxes.shape == (ps.n_polygons, 4)
    np.testing.assert_array_equal(ps.bboxes, shapely.bounds(polys))
    assert len(ps) == 7

    pts = np.array([[0.1, 0.2], [-0.3, 0.4]])
    pts64, pts_local = ps.recenter_points(pts)
    assert pts64.dtype == np.float64
    np.testing.assert_array_equal(pts64, pts)
    assert pts_local.dtype == mx.float32
    assert np.array_equal(np.array(pts_local), recenter(pts, ps.origin))


# ---------------------------------------------------------------------------
# as_point_array
# ---------------------------------------------------------------------------


def test_as_point_array_array_like_forms():
    expected = np.array([[1.5, 2.5], [3.0, 4.0]])
    forms = [
        np.array([[1.5, 2.5], [3.0, 4.0]]),
        np.asfortranarray([[1.5, 2.5], [3.0, 4.0]]),
        [[1.5, 2.5], [3.0, 4.0]],
        [(1.5, 2.5), (3.0, 4.0)],
        mx.array(np.array([[1.5, 2.5], [3.0, 4.0]], dtype=np.float32)),
    ]
    for form in forms:
        out = ms.as_point_array(form)
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(out, expected)
    # integer input upcasts
    np.testing.assert_array_equal(
        ms.as_point_array(np.array([[1, 2], [3, 4]])), [[1.0, 2.0], [3.0, 4.0]]
    )
    # a single (2,) point in every flavor becomes (1, 2)
    single = np.array([[1.5, 2.5]])
    for form in [np.array([1.5, 2.5]), [1.5, 2.5], (1.5, 2.5), mx.array([1.5, 2.5])]:
        np.testing.assert_array_equal(ms.as_point_array(form), single)
    # empty (0, 2) input is preserved
    assert ms.as_point_array(np.zeros((0, 2))).shape == (0, 2)


def test_as_point_array_shapely_forms():
    expected = np.array([[1.5, 2.5], [3.0, 4.0]])
    pt_objs = shapely.points(expected)
    np.testing.assert_array_equal(ms.as_point_array(np.array(pt_objs, dtype=object)), expected)
    np.testing.assert_array_equal(ms.as_point_array(list(pt_objs)), expected)
    # a bare shapely Point (not in a sequence)
    np.testing.assert_array_equal(ms.as_point_array(Point(7.0, 8.0)), [[7.0, 8.0]])


def test_as_point_array_geopandas_and_pandas():
    gpd = pytest.importorskip("geopandas")
    import pandas as pd

    expected = np.array([[1.5, 2.5], [3.0, 4.0]])
    pt_objs = list(shapely.points(expected))
    np.testing.assert_array_equal(ms.as_point_array(gpd.GeoSeries(pt_objs)), expected)
    gdf = gpd.GeoDataFrame({"a": [1, 2]}, geometry=pt_objs)
    np.testing.assert_array_equal(ms.as_point_array(gdf), expected)
    # plain pandas DataFrame with a geometry column of shapely objects
    df = pd.DataFrame({"geometry": pt_objs, "b": [9, 9]})
    np.testing.assert_array_equal(ms.as_point_array(df), expected)


def test_as_point_array_errors():
    with pytest.raises(ValueError, match=r"expected points of shape \(N, 2\)"):
        ms.as_point_array(np.zeros((4, 3)))
    with pytest.raises(ValueError, match=r"expected points of shape \(N, 2\)"):
        ms.as_point_array(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match=r"expected points of shape \(N, 2\)"):
        ms.as_point_array(5.0)
    with pytest.raises(ValueError, match="only Point geometries"):
        ms.as_point_array(
            np.array([Point(0, 0), LineString([(0, 0), (1, 1)])], dtype=object)
        )
    with pytest.raises(ValueError, match="only Point geometries"):
        ms.as_point_array(np.array([shapely.box(0, 0, 1, 1)], dtype=object))
    gpd = pytest.importorskip("geopandas")
    with pytest.raises(ValueError, match="only Point geometries"):
        ms.as_point_array(gpd.GeoSeries([shapely.box(0, 0, 1, 1)]))


def test_as_point_array_empty_point_alignment():
    """Empty Points become NaN rows, preserving row alignment — silently
    returning fewer rows than input geometries would shift every later point
    onto the wrong row downstream."""
    pts = [Point(1.0, 2.0), Point(), Point(3.0, 4.0)]
    out = ms.as_point_array(pts)
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out[0], [1.0, 2.0])
    assert np.isnan(out[1]).all()
    np.testing.assert_array_equal(out[2], [3.0, 4.0])
    # End-to-end: the empty point matches nothing; neighbors stay aligned.
    box = shapely.box(0.0, 0.0, 10.0, 10.0)
    for backend in ("cpu", "gpu"):
        got = ms.point_in_polygon(pts, [box], backend=backend)
        np.testing.assert_array_equal(got, [0, -1, 0])


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_polygonset_validation_errors():
    with pytest.raises(ValueError, match="at least one geometry"):
        ms.PolygonSet([])
    with pytest.raises(ValueError, match="at least one geometry"):
        ms.PolygonSet(np.array([], dtype=object))
    with pytest.raises(ValueError, match="Polygon and MultiPolygon"):
        ms.PolygonSet([Point(0, 0), Point(1, 1)])
    with pytest.raises(ValueError, match="Polygon and MultiPolygon"):
        ms.PolygonSet([LineString([(0, 0), (1, 1)])])
    # one bad geometry poisons the whole set — mixed valid+invalid still raises
    with pytest.raises(ValueError, match="Polygon and MultiPolygon"):
        ms.PolygonSet([shapely.box(0, 0, 1, 1), LineString([(0, 0), (1, 1)])])
    with pytest.raises(ValueError, match="1-D sequence"):
        ms.PolygonSet(
            np.array(
                [[shapely.box(0, 0, 1, 1)], [shapely.box(1, 1, 2, 2)]], dtype=object
            )
        )
