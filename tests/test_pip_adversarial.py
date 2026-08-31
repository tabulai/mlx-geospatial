"""Adversarial geometry for the crossing-number point-in-polygon kernels.

Targets the classic ray-casting failure modes (rays through vertices, points
sharing y with vertices, horizontal edges, slivers, degenerate rings, holes
touching shells, sub-tolerance geometry) and cross-validates the pair,
direct-dense, first-match, and compact all-match Metal kernels against the
retained composed-MLX oracle exactly, and against shapely float64 outside the
documented ``boundary_tolerance`` band.

Everything here is deterministic; coordinates in the exact-y tests are chosen
to be exactly representable in float32 after re-centering.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon

import mlx_spatial as ms
from mlx_spatial._dispatch import metal_available
from mlx_spatial._pip import _pip_dense_composed, pip_pairs
from conftest import make_grid_polygons, near_boundary_mask, reference_assign

pytestmark = pytest.mark.skipif(
    not metal_available(), reason="GPU point-in-polygon tests require Metal"
)


# ---------------------------------------------------------------------------
# helpers (local to this file per suite rules)
# ---------------------------------------------------------------------------


def _pair_matrix(pts: np.ndarray, polyset: ms.PolygonSet) -> np.ndarray:
    """Exhaustive (N, M) containment flags through the Metal pair kernel."""
    n, m = len(pts), polyset.n_polygons
    pair_pt = np.repeat(np.arange(n, dtype=np.int32), m)
    pair_pg = np.tile(np.arange(m, dtype=np.int32), n)
    _, pts_local = polyset.recenter_points(pts)
    flags = pip_pairs(pts_local, polyset, mx.array(pair_pt), mx.array(pair_pg))
    return np.array(flags).reshape(n, m)


def _dense_matrix(pts: np.ndarray, polyset: ms.PolygonSet) -> np.ndarray:
    return np.array(ms.contains_matrix(pts, polyset)).astype(np.uint8)


def _composed_matrix(pts: np.ndarray, polyset: ms.PolygonSet) -> np.ndarray:
    _, pts_local = polyset.recenter_points(pts)
    return np.array(_pip_dense_composed(pts_local, polyset)).astype(np.uint8)


def _paths_agree(pts, polyset: ms.PolygonSet) -> np.ndarray:
    """Assert pair, direct-dense, and composed paths agree exactly."""
    pts = np.asarray(pts, dtype=np.float64)
    dense = _dense_matrix(pts, polyset)
    composed = _composed_matrix(pts, polyset)
    pairs = _pair_matrix(pts, polyset)
    np.testing.assert_array_equal(
        dense, composed, err_msg="direct pip_dense and composed oracle disagree"
    )
    np.testing.assert_array_equal(
        dense, pairs, err_msg="pip_dense and pip_pairs disagree"
    )
    return dense


def _first_match(matrix: np.ndarray) -> np.ndarray:
    """Lowest containing polygon index per row, -1 for none."""
    hit = matrix.astype(bool)
    any_hit = hit.any(axis=1)
    return np.where(any_hit, hit.argmax(axis=1), -1).astype(np.int64)


def _join_matches_matrix(pts, polyset: ms.PolygonSet, matrix: np.ndarray) -> np.ndarray:
    """Grid first/all-match outputs must equal the exact dense matrix."""
    pts = np.asarray(pts, float)
    got = np.asarray(
        ms.point_in_polygon(pts, polyset, backend="gpu"),
        dtype=np.int64,
    )
    np.testing.assert_array_equal(
        got, _first_match(matrix), err_msg="grid join != dense first-match"
    )
    point_idx, poly_idx = ms.point_in_polygon(
        pts, polyset, all_matches=True, backend="gpu"
    )
    expected_point, expected_poly = np.nonzero(matrix)
    np.testing.assert_array_equal(point_idx, expected_point.astype(np.int64))
    np.testing.assert_array_equal(poly_idx, expected_poly.astype(np.int64))
    return got


def _assert_golden(pts, polys, polyset, *, expect_no_band: bool = True):
    """GPU first-match vs shapely STRtree, excluding the tolerance band."""
    pts = np.asarray(pts, dtype=np.float64)
    got = np.asarray(ms.point_in_polygon(pts, polyset, backend="gpu"), dtype=np.int64)
    ref = reference_assign(pts, np.asarray(polys, dtype=object))
    ok = ~near_boundary_mask(pts, np.asarray(polys, dtype=object), polyset.boundary_tolerance)
    if expect_no_band:
        assert ok.all(), "test setup error: some points fall in the tolerance band"
    bad = np.flatnonzero((got != ref) & ok)
    assert bad.size == 0, (
        f"mismatch vs shapely outside band at points {pts[bad][:5].tolist()}: "
        f"got {got[bad][:5].tolist()}, ref {ref[bad][:5].tolist()}"
    )
    return got, ref


# ---------------------------------------------------------------------------
# 1. rays through vertices / points sharing y with vertex y-coordinates
# ---------------------------------------------------------------------------


def test_ray_through_vertex_y_inside_points():
    """Interior points whose y exactly equals vertex y; rays hit vertices dead-on.

    origin=(0,0) is forced so every coordinate (halves and integers <= 16) is
    exactly representable in float32 -- the y == vertex-y collisions are exact
    on the GPU, not approximate.
    """
    hexagon = Polygon([(0, 0), (4, 0), (6, 2), (4, 4), (0, 4), (-2, 2)])
    diamond1 = Polygon([(9, 1), (10, 0), (11, 1), (10, 2)])
    diamond2 = Polygon([(9, 5), (10, 4), (11, 5), (10, 6)])
    polys = np.array([hexagon, diamond1, diamond2], dtype=object)
    polyset = ms.PolygonSet(polys, origin=(0.0, 0.0))

    pts = np.array(
        [
            # inside, y shared with vertices, rays pass exactly through vertices
            [0.5, 2.0],   # hexagon; ray exits through vertex (6, 2)
            [2.0, 2.0],   # hexagon, same y as (-2,2) and (6,2)
            [-1.0, 2.0],  # hexagon, just right of vertex (-2, 2)
            [5.0, 2.0],   # hexagon, close to vertex (6, 2) but 0.7 from edges
            [10.0, 1.0],  # diamond1 center; ray through vertex (11, 1)
            [10.0, 5.0],  # diamond2 center; ray through vertex (11, 5)
            # outside, rays through vertices of polygons to the right
            [-5.0, 2.0],  # ray through (-2,2) and (6,2)
            [-5.0, 1.0],  # ray through diamond1 vertices (9,1) and (11,1)
            [7.0, 2.0],   # right of hexagon at vertex y
            [12.0, 1.0],  # right of diamond1 at vertex y
        ]
    )
    expected = np.array([0, 0, 0, 0, 1, 2, -1, -1, -1, -1], dtype=np.int64)

    got, ref = _assert_golden(pts, polys, polyset)
    # sanity: shapely itself agrees with the hand-computed expectations
    np.testing.assert_array_equal(ref, expected)
    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)

    # bbox corners / on-vertex points of diamond1: bbox corners are outside the
    # diamond but inside its (widened) bbox, so the kernel's bbox pre-test must
    # not claim them; on-vertex points are in-band so only cross-path agreement
    # and join consistency are asserted.
    extra = np.array(
        [
            [9.0, 0.0], [11.0, 2.0], [9.0, 2.0], [11.0, 0.0],  # diamond1 bbox corners
            [9.0, 0.5],                                        # on the bbox edge
            [9.0, 1.0], [10.0, 0.0], [10.0, 2.0], [11.0, 1.0], # exact vertices
        ]
    )
    extra_matrix = _paths_agree(extra, polyset)
    _join_matches_matrix(extra, polyset, extra_matrix)
    # the four bbox corners are ~0.7 from the diamond: definitely outside
    assert not extra_matrix[:5].any()


# ---------------------------------------------------------------------------
# 2. horizontal edges: staircase + rectangles, rays collinear with edges
# ---------------------------------------------------------------------------


def test_horizontal_edges_staircase_and_rectangles():
    staircase = Polygon(
        [
            (0, 0), (6, 0), (6, 1), (5, 1), (5, 2), (4, 2), (4, 3),
            (3, 3), (3, 4), (2, 4), (2, 5), (1, 5), (1, 6), (0, 6),
        ]
    )
    box1 = shapely.box(8, 0, 11, 2)
    box2 = shapely.box(13, 0, 16, 2)
    polys = np.array([staircase, box1, box2], dtype=object)
    polyset = ms.PolygonSet(polys)

    pts = np.array(
        [
            # interior points at exact step-edge y-levels (>= 0.5 from any edge)
            [0.5, 1.0], [2.5, 1.0], [4.5, 1.0],
            [0.5, 2.0], [3.5, 2.0],
            [1.5, 3.0],
            [0.5, 4.0], [1.5, 4.0],
            [0.5, 5.0],
            # outside at exact step ys (rightward rays hit vertical-edge endpoints)
            [6.5, 1.0], [5.5, 3.0], [-1.0, 3.0], [7.0, 0.5],
            # rays exactly collinear with the rectangles' horizontal edges
            [7.0, 0.0], [12.0, 0.0], [17.0, 0.0], [12.0, 2.0], [7.0, 2.0],
            # rectangle interiors
            [9.5, 1.0], [14.5, 1.0],
        ]
    )
    expected = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2],
        dtype=np.int64,
    )
    got, ref = _assert_golden(pts, polys, polyset)
    np.testing.assert_array_equal(ref, expected)
    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)


def test_horizontal_edges_comb():
    """Comb polygon: many horizontal edges and many vertices sharing y=1 and y=3."""
    comb = Polygon(
        [
            (0, 0), (10, 0), (10, 3), (9, 3), (9, 1), (8, 1), (8, 3),
            (7, 3), (7, 1), (6, 1), (6, 3), (5, 3), (5, 1), (4, 1),
            (4, 3), (3, 3), (3, 1), (2, 1), (2, 3), (1, 3), (1, 1), (0, 1),
        ]
    )
    polys = np.array([comb], dtype=object)
    polyset = ms.PolygonSet(polys)

    tooth_centers = [[x, 1.0] for x in (1.5, 3.5, 5.5, 7.5, 9.5)]  # inside, y==1 exactly
    gap_points = [[x, 2.0] for x in (2.5, 4.5, 6.5, 8.5)]          # outside, between teeth
    bar_points = [[2.5, 0.5], [8.5, 0.5]]                          # inside the base bar
    trap_points = [
        [-1.0, 1.0],   # outside; ray at y=1 crosses 10 tooth-wall endpoints
        [-1.0, 3.0],   # outside; ray collinear with all tooth-top edges
        [11.0, 1.0],   # outside to the right
        [0.5, 2.0],    # outside; above the leftmost gap
    ]
    pts = np.array(tooth_centers + gap_points + bar_points + trap_points)
    expected = np.array([0] * 5 + [-1] * 4 + [0] * 2 + [-1] * 4, dtype=np.int64)

    got, ref = _assert_golden(pts, polys, polyset)
    np.testing.assert_array_equal(ref, expected)
    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)

    # chunk_bytes remains behaviorally compatible on the direct Metal path;
    # forced composed-fallback chunking is covered in test_dispatch_api.
    chunked = np.array(ms.contains_matrix(pts, polyset, chunk_bytes=1)).astype(np.uint8)
    np.testing.assert_array_equal(chunked, matrix)


def test_half_open_shared_edges_exact():
    """Points exactly on shared edges of a 3x3 integer grid of squares.

    With origin at the bbox center all coordinates are exact in fp32, so the
    crossing arithmetic is exact and the documented half-open rule is
    deterministic: every on-edge point (including 4-square corners) must land
    in exactly one square.
    """
    polys = make_grid_polygons(3, 3, cell=1.0)
    polyset = ms.PolygonSet(polys)

    on_edge = []
    for e in (1.0, 2.0):  # interior grid lines
        for f in (0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75):
            on_edge.append([e, f])  # on a vertical shared edge
            on_edge.append([f, e])  # on a horizontal shared edge
    corners = [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]  # 4-square corners
    pts = np.array(on_edge + corners)

    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)
    memberships = matrix.sum(axis=1)
    np.testing.assert_array_equal(
        memberships,
        np.ones(len(pts), dtype=matrix.dtype),
        err_msg="half-open rule must put an exact on-edge point in exactly one square",
    )


# ---------------------------------------------------------------------------
# 3. slivers
# ---------------------------------------------------------------------------


def test_sliver_triangle_3x_tolerance_offsets():
    """Tilted thin triangle (aspect ~1e4): points at +/-3*tol along the normal.

    The triangle is thin enough to stress fp32 crossing arithmetic but thick
    enough (1e-4 >> tol ~ 2.4e-7) that just-inside/just-outside points at
    3x boundary_tolerance are well-defined; shapely float64 is the oracle.
    """
    A = np.array([0.1, 0.1])
    B = np.array([0.9, 0.7])
    u = B - A
    L = float(np.hypot(*u))
    n_hat = np.array([-u[1], u[0]]) / L
    h = 1e-4  # apex height; aspect ratio = L / h = 1e4
    apex = (A + B) / 2.0 + h * n_hat
    tri = Polygon([tuple(A), tuple(B), tuple(apex)])
    polys = np.array([tri], dtype=object)
    polyset = ms.PolygonSet(polys)
    tol = polyset.boundary_tolerance
    assert 6.0 * tol < 2.0 * h * 0.1, "setup: sliver must be thicker than 6*tol"

    ts = np.linspace(0.1, 0.9, 33)
    ts = ts[np.abs(ts - 0.5) > 0.06]  # keep clear of the apex vertex
    base = A[None, :] + ts[:, None] * u[None, :]
    # along n_hat, the interior at base-parameter t spans (0, 2*h*min(t, 1-t))
    thick = 2.0 * h * np.minimum(ts, 1.0 - ts)

    groups, exp = [], []
    groups.append(base + 3.0 * tol * n_hat[None, :])
    exp.append(np.ones(len(ts)))
    groups.append(base + (thick - 3.0 * tol)[:, None] * n_hat)
    exp.append(np.ones(len(ts)))
    groups.append(base - 3.0 * tol * n_hat[None, :])
    exp.append(np.zeros(len(ts)))
    groups.append(base + (thick + 3.0 * tol)[:, None] * n_hat)
    exp.append(np.zeros(len(ts)))
    pts = np.concatenate(groups, axis=0)
    expected = np.concatenate(exp).astype(bool)

    # setup sanity: float64 shapely agrees with the construction, and no point
    # falls inside the tolerance band
    assert np.array_equal(shapely.contains_xy(tri, pts[:, 0], pts[:, 1]), expected)
    assert not near_boundary_mask(pts, polys, tol).any()

    matrix = _paths_agree(pts, polyset)
    np.testing.assert_array_equal(matrix[:, 0].astype(bool), expected)
    got = _join_matches_matrix(pts, polyset, matrix)
    np.testing.assert_array_equal(got, np.where(expected, 0, -1))


def test_extreme_sliver_aspect_1e6():
    """Aspect-ratio-1e6 sliver: thickness of the order of the fp32 band.

    tol ~ 8 ulp of the half-extent ~ 4.8e-7 * L while thickness = 1e-6 * L, so
    thickness ~ 2 * tol at any scale: midline points sit ~1 * tol from both
    edges -- too close to the band to golden-test. Assert no crash,
    dense == pairs exactly, and that clearly-far points are outside.
    """
    A = np.array([0.0, 0.0])
    B = np.array([0.8, 0.6])  # |AB| = 1.0
    n_hat = np.array([-0.6, 0.8])
    h = 1e-6  # aspect ratio 1e6
    apex = (A + B) / 2.0 + h * n_hat
    tri = Polygon([tuple(A), tuple(B), tuple(apex)])
    polys = np.array([tri], dtype=object)
    polyset = ms.PolygonSet(polys)
    # thickness within a small multiple of the band: band-scale geometry
    assert h < 10.0 * polyset.boundary_tolerance

    ts = np.linspace(0.05, 0.95, 19)
    midline = A[None, :] + ts[:, None] * (B - A)[None, :] + (h / 2.0) * n_hat[None, :]
    far = np.array(
        [[0.4, 0.3 + 0.1], [0.4, 0.3 - 0.1], [-0.5, -0.5], [1.5, 1.2], [0.0, 0.6]]
    )
    pts = np.concatenate([midline, far], axis=0)

    matrix = _paths_agree(pts, polyset)  # exact agreement, no crash
    assert not matrix[len(ts):, 0].any(), "far points must be outside the sliver"
    got = _join_matches_matrix(pts, polyset, matrix)
    assert set(np.unique(got)) <= {-1, 0}


# ---------------------------------------------------------------------------
# 4. degenerate-ish rings shapely accepts
# ---------------------------------------------------------------------------


def test_duplicate_and_collinear_vertices():
    p_dup = Polygon([(0, 0), (0, 0), (2, 0), (2, 0), (2, 2), (2, 2), (0, 2)])
    p_col = Polygon([(5, 0), (6, 0), (7, 0), (7, 1), (7, 2), (5, 2), (5, 1)])
    polys = np.array([p_dup, p_col], dtype=object)
    polyset = ms.PolygonSet(polys, origin=(0.0, 0.0))

    pts = np.array(
        [
            [1.0, 1.0],   # inside p_dup
            [1.0, 0.5],   # inside p_dup below the duplicate-vertex corner
            [6.0, 1.0],   # inside p_col; y==1 shared with collinear vertices (7,1),(5,1)
            [6.0, 0.5],   # inside p_col
            [4.0, 1.0],   # between the two, ray passes through (5,1) and (7,1)
            [3.0, 0.0],   # ray collinear with both bottom edges (through 4 vertices)
            [8.0, 1.0],   # right of p_col at collinear-vertex y
            [-1.0, 2.0],  # ray collinear with p_dup top edge and through (5,2),(7,2)
        ]
    )
    expected = np.array([0, 0, 1, 1, -1, -1, -1, -1], dtype=np.int64)
    got, ref = _assert_golden(pts, polys, polyset)
    np.testing.assert_array_equal(ref, expected)
    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)


def test_empty_polygons_in_set():
    """Empty polygons (NaN bounds) mixed with a real one: rejected everywhere."""
    polys = np.array([Polygon(), shapely.box(0, 0, 2, 2), Polygon()], dtype=object)
    polyset = ms.PolygonSet(polys)
    pts = np.array(
        [
            [1.0, 1.0],
            [3.0, 3.0],
            [0.5, 1.5],
            [-1.0, 1.0],
            [np.nan, np.nan],
            [np.nan, 1.0],
            [1.0, np.nan],
        ]
    )

    matrix = _paths_agree(pts, polyset)
    assert not matrix[:, 0].any() and not matrix[:, 2].any(), (
        "empty polygons must contain nothing"
    )
    got = _join_matches_matrix(pts, polyset, matrix)
    np.testing.assert_array_equal(got, [1, -1, 1, -1, -1, -1, -1])


def test_lattice_sweep_shared_edges_grid_join():
    """Exhaustive quarter-lattice sweep over a 6x6 tiling of unit squares.

    Every point on [-0.5, 6.5]^2 at 0.25 spacing -- including all shared
    edges, all shared 4-square corners, and the tiling's outer boundary.
    Asserts: dense == pairs exactly; grid join == dense first-match (cell
    boundary handling); half-open rule => no point in more than one square;
    strict-interior points in exactly one; and full golden agreement with
    shapely for all off-lattice points.
    """
    polys = make_grid_polygons(6, 6, cell=1.0)
    polyset = ms.PolygonSet(polys)
    g = np.arange(-0.5, 6.75, 0.25)
    pts = np.array([[x, y] for x in g for y in g])

    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)

    counts = matrix.sum(axis=1)
    assert counts.max() <= 1, "half-open rule violated: point in 2+ adjacent squares"
    strict = (
        (pts[:, 0] % 1 != 0) & (pts[:, 1] % 1 != 0)
        & (pts[:, 0] > 0) & (pts[:, 0] < 6) & (pts[:, 1] > 0) & (pts[:, 1] < 6)
    )
    assert (counts[strict] == 1).all(), "strictly interior point not in its square"

    ref = np.array(
        [shapely.intersects_xy(gm, pts[:, 0], pts[:, 1]) for gm in polys]
    ).T.astype(np.uint8)
    off_lattice = (pts[:, 0] % 1 != 0) & (pts[:, 1] % 1 != 0)
    np.testing.assert_array_equal(matrix[off_lattice], ref[off_lattice])


def test_bowtie_self_intersecting():
    """Invalid bowtie: GEOS semantics are undefined, so no shapely oracle.

    Assert only that (a) nothing crashes, (b) the two GPU paths agree exactly,
    (c) the grid join agrees with the dense first-match, and (d) the even-odd
    rule itself: lobe centers are inside (odd crossings), the regions above and
    below the self-intersection are outside (even crossings) -- a mathematical
    property of the crossing-number algorithm on this exact-fp32 geometry.
    """
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    polys = np.array([bowtie], dtype=object)
    polyset = ms.PolygonSet(polys, origin=(0.0, 0.0))

    xs = np.linspace(-0.5, 2.5, 21)
    grid = np.array([[x, y] for x in xs for y in xs])
    key_pts = np.array(
        [
            [0.5, 1.0],   # left lobe center  -> inside under even-odd
            [1.75, 1.0],  # right lobe center -> inside under even-odd
            [1.0, 1.75],  # above the crossing -> outside
            [1.0, 0.25],  # below the crossing -> outside
        ]
    )
    pts = np.concatenate([grid, key_pts], axis=0)

    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)
    np.testing.assert_array_equal(matrix[-4:, 0], [1, 1, 0, 0])


# ---------------------------------------------------------------------------
# 5. holes
# ---------------------------------------------------------------------------


def test_hole_touching_shell_at_vertex():
    """Valid polygon whose hole touches the shell at exactly one point (5, 0)."""
    poly = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(5, 0), (8, 3), (5, 6), (2, 3)]],
    )
    assert poly.is_valid  # touching at a point is OGC-valid
    polys = np.array([poly], dtype=object)
    polyset = ms.PolygonSet(polys)

    pts = np.array(
        [
            [1.0, 1.0],   # shell interior, left of hole
            [9.0, 1.0],   # shell interior, right of hole
            [5.0, 8.0],   # shell interior, above hole
            [4.0, 0.5],   # in the wedge between hole and shell bottom edge
            [6.0, 0.5],   # same, right side
            [5.0, 0.5],   # inside hole, just above the touch point -> outside
            [5.0, 3.0],   # hole center -> outside
            [5.0, 5.5],   # inside hole near top vertex -> outside
            [5.0, -0.5],  # outside shell below the touch point
            [1.0, 3.0],   # shell interior at the hole's widest y
        ]
    )
    expected = np.array([0, 0, 0, 0, 0, -1, -1, -1, -1, 0], dtype=np.int64)
    got, ref = _assert_golden(pts, polys, polyset)
    np.testing.assert_array_equal(ref, expected)
    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)


def test_hole_fills_shell_thin_ring():
    """Hole 1e-3 inside the shell: only a thin ring of interior remains."""
    d = 1e-3
    shell = shapely.box(0.0, 0.0, 10.0, 10.0)
    poly = Polygon(
        list(shell.exterior.coords)[:-1],
        holes=[[(d, d), (10 - d, d), (10 - d, 10 - d), (d, 10 - d)]],
    )
    polys = np.array([poly], dtype=object)
    polyset = ms.PolygonSet(polys)
    tol = polyset.boundary_tolerance
    assert d > 20.0 * tol, "setup: ring must be much wider than the band"

    mid = d / 2.0
    pts = np.array(
        [
            [5.0, mid],        # bottom band midline -> inside
            [5.0, 10.0 - mid], # top band midline -> inside
            [mid, 5.0],        # left band midline -> inside
            [10.0 - mid, 5.0], # right band midline -> inside
            [mid, mid],        # corner of the band -> inside
            [5.0, 5.0],        # deep in the hole -> outside
            [5.0, 2.0 * d],    # just inside the hole (d away from its edge) -> outside
            [2.0 * d, 2.0 * d],
            [5.0, -0.5],       # outside the shell
            [-mid, 5.0],
        ]
    )
    expected = np.array([0, 0, 0, 0, 0, -1, -1, -1, -1, -1], dtype=np.int64)
    got, ref = _assert_golden(pts, polys, polyset)
    np.testing.assert_array_equal(ref, expected)
    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)


# ---------------------------------------------------------------------------
# 6. multipolygon parts sharing a vertex; very high vertex count
# ---------------------------------------------------------------------------


def test_multipolygon_parts_share_vertex():
    mp = MultiPolygon(
        [
            Polygon([(0, 0), (2, 0), (1, 1)]),
            Polygon([(0, 2), (2, 2), (1, 1)]),
        ]
    )
    assert mp.is_valid
    tri = Polygon([(4, 0), (6, 0), (5, 2)])
    polys = np.array([mp, tri], dtype=object)
    polyset = ms.PolygonSet(polys, origin=(0.0, 0.0))

    pts = np.array(
        [
            [1.0, 0.5],   # part 1 interior
            [1.0, 1.5],   # part 2 interior
            [1.0, 0.9],   # part 1, close under the shared vertex
            [1.0, 1.1],   # part 2, close above the shared vertex
            [0.5, 1.0],   # outside; ray passes exactly through shared vertex (1,1)
            [-1.0, 1.0],  # outside; same ray from further left
            [5.0, 0.5],   # inside the separate triangle
            [3.0, 1.0],   # between mp and tri
        ]
    )
    expected = np.array([0, 0, 0, 0, -1, -1, 1, -1], dtype=np.int64)
    got, ref = _assert_golden(pts, polys, polyset)
    np.testing.assert_array_equal(ref, expected)
    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)


def test_high_vertex_count_circle(rng):
    """10k-vertex circle: correctness outside the band, no crash, paths agree."""
    k = 10_000
    ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    circle = Polygon(np.column_stack([np.cos(ang), np.sin(ang)]))
    polys = np.array([circle], dtype=object)
    polyset = ms.PolygonSet(polys)

    # random points around the boundary + points whose y equals vertex ys
    r = rng.uniform(0.5, 1.5, 250)
    theta = rng.uniform(0.0, 2.0 * np.pi, 250)
    random_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    vy = np.sin(ang[7:6000:97])  # a sample of exact vertex y-coordinates
    vy = vy[np.abs(vy) < 0.8]
    vertex_y_pts = np.column_stack([np.zeros(len(vy)), vy])       # inside, ray thru vertex
    left_pts = np.column_stack([np.full(len(vy), -1.5), vy])      # outside, ray thru 2 vertices
    pts = np.concatenate([random_pts, vertex_y_pts, left_pts], axis=0)

    got, ref = _assert_golden(pts, polys, polyset, expect_no_band=False)
    n_rand = len(random_pts)
    np.testing.assert_array_equal(got[n_rand:], ref[n_rand:])  # constructed pts: far from edges
    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)


# ---------------------------------------------------------------------------
# 7. tiny polygon below the tolerance band
# ---------------------------------------------------------------------------


def test_tiny_polygon_below_tolerance_band():
    """A 1e-9-degree triangle at lon 139 next to a normal polygon.

    The dataset's re-centering origin is dominated by the normal polygon, so
    the tiny triangle is smaller than the fp32 tolerance band at its local
    coordinate magnitude: every interior point sits inside the band and its
    classification is documented-undefined. Assert no crash, exact
    dense == pairs agreement, correct handling of the normal polygon, and that
    far points never match the tiny triangle.
    """
    e = 1e-9
    tiny = Polygon([(139.0, 35.0), (139.0 + e, 35.0), (139.0, 35.0 + e)])
    normal = shapely.box(139.1, 35.1, 139.2, 35.2)
    polys = np.array([tiny, normal], dtype=object)
    polyset = ms.PolygonSet(polys)
    assert polyset.boundary_tolerance > 2.0 * e  # tiny is smaller than the band

    centroid = [139.0 + e / 3.0, 35.0 + e / 3.0]
    pts = np.array(
        [
            centroid,                # anything goes for polygon 0; never polygon 1
            [139.0, 35.0],           # the tiny triangle's own vertex
            [139.15, 35.15],         # inside the normal polygon
            [139.05, 35.05],         # in the gap between the two
            [138.0, 34.0],           # far away
            [139.0 + 1e-6, 35.0],    # near tiny but well outside it (1000x its size)
        ]
    )

    matrix = _paths_agree(pts, polyset)  # exact agreement, no crash
    got = _join_matches_matrix(pts, polyset, matrix)

    assert got[0] in (-1, 0)  # in-band: either way, but never the normal polygon
    assert got[1] in (-1, 0)
    assert got[2] == 1        # normal polygon unaffected
    assert got[3] == -1
    assert got[4] == -1
    assert matrix[2:, 0].sum() == 0  # far points never inside the tiny polygon


# ---------------------------------------------------------------------------
# 8. negative coordinates / polygons straddling the axes
# ---------------------------------------------------------------------------


def test_negative_coordinates_and_axis_straddle():
    b_neg = shapely.box(-3, -3, -1, -1)     # entirely in the negative quadrant
    b_str = shapely.box(-1, -1, 1, 1)       # straddles x=0 and y=0; touches b_neg at (-1,-1)
    tri = Polygon([(-5, 1), (-3, 1), (-4, 3)])  # horizontal bottom edge at y=1
    polys = np.array([b_neg, b_str, tri], dtype=object)
    polyset = ms.PolygonSet(polys)

    pts = np.array(
        [
            [-2.0, -2.0],   # b_neg interior
            [-2.5, -1.5],   # b_neg interior
            [0.0, 0.0],     # exact origin, b_str interior
            [0.0, 0.5],     # on the y-axis
            [-0.5, 0.0],    # on the x-axis
            [0.5, -0.5],    # fourth quadrant
            [-4.0, 1.5],    # tri interior
            [-4.0, 2.0],    # tri interior higher up
            [-6.0, 1.0],    # outside; ray collinear with tri's bottom edge
            [-2.0, 0.5],    # outside, left of b_str
            [-4.0, -2.0],   # outside, left of b_neg
            [2.0, 2.0],     # outside everything
            [0.0, -2.0],    # below b_str on the y-axis
        ]
    )
    expected = np.array([0, 0, 1, 1, 1, 1, 2, 2, -1, -1, -1, -1, -1], dtype=np.int64)
    got, ref = _assert_golden(pts, polys, polyset)
    np.testing.assert_array_equal(ref, expected)
    matrix = _paths_agree(pts, polyset)
    _join_matches_matrix(pts, polyset, matrix)

    # exact-on-boundary points (including the shared corner (-1,-1) of the two
    # boxes): in-band, so no golden claim -- but the two GPU paths and the
    # grid join must stay mutually consistent, and the shared corner must land
    # in at most one box under the half-open rule.
    onb = np.array(
        [
            [-1.0, -1.0],  # corner shared by b_neg and b_str
            [-1.0, 0.0],   # on b_str's left edge (the y-axis crossing region)
            [0.0, 1.0],    # on b_str's top edge
            [1.0, 0.0],    # on b_str's right edge
            [0.0, -1.0],   # on b_str's bottom edge
            [-2.0, -1.0],  # on b_neg's top edge
            [-4.0, 1.0],   # on tri's bottom edge
            [-3.0, -3.0],  # b_neg's bottom-left corner
        ]
    )
    onb_matrix = _paths_agree(onb, polyset)
    _join_matches_matrix(onb, polyset, onb_matrix)
    assert onb_matrix[0, :2].sum() <= 1, "shared corner claimed by both boxes"
