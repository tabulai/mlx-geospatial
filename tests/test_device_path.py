"""Device-resident query path: ``LocalPoints`` / ``mx.array`` inputs.

Pins the array-native contract: MLX inputs stay in unified memory, cell
assignment runs on the GPU, results come back as MLX arrays and are bit for
bit the same as the host path; host inputs keep returning NumPy. Also covers
the ``PrecisionWarning`` for float32 device inputs, the ``LocalPoints`` safety
rails, and the padded-cell-range invariant that makes fp32 cell assignment
safe.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import mlx.core as mx
import pytest
import shapely

import mlx_spatial as ms
import mlx_spatial._pip as pip_impl
from mlx_spatial import GridIndex, LocalPoints, PolygonSet, PrecisionWarning
from mlx_spatial._dispatch import metal_available
from mlx_spatial._precision import float32_resolution, recenter
from conftest import (
    make_convex_polygons,
    make_grid_polygons,
    near_boundary_mask,
    reference_assign,
)

needs_metal = pytest.mark.skipif(not metal_available(), reason="Metal GPU required")


@pytest.fixture(scope="module")
def workload():
    rng = np.random.default_rng(20260903)
    polys = make_convex_polygons(120, rng)
    pts = rng.uniform(-1.1, 1.1, size=(5000, 2))
    polyset = PolygonSet(polys)
    index = GridIndex(polyset)
    return polys, pts, polyset, index


def _pairs(result):
    return np.asarray(result[0]), np.asarray(result[1])


# ---------------------------------------------------------------------------
# container rule
# ---------------------------------------------------------------------------


@needs_metal
def test_output_container_follows_input(workload):
    polys, pts, polyset, index = workload
    host = index.query(pts, backend="gpu")
    assert isinstance(host, np.ndarray) and host.dtype == np.int32
    assert isinstance(index.query(pts.tolist(), backend="gpu"), np.ndarray)
    assert isinstance(index.query(pts, backend="cpu"), np.ndarray)

    device = index.query(polyset.to_local(pts))
    assert isinstance(device, mx.array) and device.dtype == mx.int32
    assert isinstance(index.query(mx.array(pts, dtype=mx.float64)), mx.array)
    assert isinstance(index.query(polyset.to_local(pts), backend="cpu"), mx.array)

    h_pt, h_pg = index.query(pts, backend="gpu", all_matches=True)
    d_pt, d_pg = index.query(polyset.to_local(pts), all_matches=True)
    assert isinstance(h_pt, np.ndarray) and h_pt.dtype == h_pg.dtype == np.int64
    assert isinstance(d_pt, mx.array) and d_pt.dtype == d_pg.dtype == mx.int64

    cm_host = ms.contains_matrix(pts[:50], polyset)
    cm_device = ms.contains_matrix(polyset.to_local(pts[:50]), polyset)
    assert isinstance(cm_host, np.ndarray) and cm_host.dtype == np.bool_
    assert isinstance(cm_device, mx.array) and cm_device.dtype == mx.bool_
    np.testing.assert_array_equal(cm_device, cm_host)


@needs_metal
def test_sjoin_still_returns_pandas(workload):
    gpd = pytest.importorskip("geopandas")
    polys, pts, polyset, index = workload
    left = gpd.GeoDataFrame(geometry=list(shapely.points(pts[:500])))
    right = gpd.GeoDataFrame({"zone": np.arange(len(polys))}, geometry=list(polys))
    out = ms.sjoin(left, right, how="left")
    assert isinstance(out, gpd.GeoDataFrame)
    assert out["zone"].dtype != object


# ---------------------------------------------------------------------------
# bit-identical device / host parity across every GPU route
# ---------------------------------------------------------------------------


@needs_metal
@pytest.mark.parametrize("max_pairs_per_batch", [16_000_000, 997, 3])
def test_device_path_matches_host_path_bitwise(workload, max_pairs_per_batch):
    """Single launch, whole-point batches, explicit pairs, batched all-matches."""
    polys, pts, polyset, index = workload
    if max_pairs_per_batch < 16:
        pts = pts[:600]  # the explicit-pair route is host-planned; keep it quick
    local = polyset.to_local(pts)
    kw = {"max_pairs_per_batch": max_pairs_per_batch}

    host = index.query(pts, backend="gpu", **kw)
    device = index.query(local, **kw)
    np.testing.assert_array_equal(device, host)

    h_pt, h_pg = index.query(pts, backend="gpu", all_matches=True, **kw)
    d_pt, d_pg = _pairs(index.query(local, all_matches=True, **kw))
    np.testing.assert_array_equal(d_pt, h_pt)
    np.testing.assert_array_equal(d_pg, h_pg)

    ok = ~near_boundary_mask(pts, polys, polyset.boundary_tolerance)
    ref = reference_assign(pts, polys)
    assert ok.sum() > 0.95 * len(pts)
    np.testing.assert_array_equal(np.asarray(device)[ok], ref[ok])


@needs_metal
def test_mx_float64_input_is_exact_and_silent(workload):
    polys, pts, polyset, index = workload
    p64 = mx.array(pts, dtype=mx.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        local = polyset.to_local(p64)
        out = index.query(p64)
        index.query(polyset.to_local(pts))
    np.testing.assert_array_equal(np.asarray(local.coords), recenter(pts, polyset.origin))
    np.testing.assert_array_equal(out, index.query(pts, backend="gpu"))
    # Integer coordinates are exact as well.
    ints = np.array([[0, 0], [1, 1]], dtype=np.int64)
    np.testing.assert_array_equal(
        index.query(mx.array(ints)), index.query(ints.astype(np.float64), backend="gpu")
    )


@needs_metal
def test_mx_float32_input_warns_and_stays_in_band(workload):
    polys, pts, polyset, index = workload
    p32 = mx.array(pts.astype(np.float32))
    with pytest.warns(PrecisionWarning, match="float64"):
        out = index.query(p32)
    with pytest.warns(PrecisionWarning):
        local = polyset.to_local(p32)
    np.testing.assert_array_equal(index.query(local), out)
    # Here the raw fp32 rounding (~6e-8 at |x| ~ 1) is below the tolerance
    # band, so the documented contract still holds outside the band.
    ok = ~near_boundary_mask(pts, polys, polyset.boundary_tolerance)
    ref = reference_assign(pts, polys)
    np.testing.assert_array_equal(np.asarray(out)[ok], ref[ok])
    # hi/lo device re-centering is within one ulp of float64 re-centering
    # of the same fp32 values.
    exact = (pts.astype(np.float32).astype(np.float64) - polyset.origin).astype(np.float32)
    got = np.asarray(local.coords)
    assert np.all(np.abs(got - exact) <= np.spacing(np.abs(exact)))


# ---------------------------------------------------------------------------
# LocalPoints safety rails and ergonomics
# ---------------------------------------------------------------------------


def test_localpoints_api(workload):
    polys, pts, polyset, index = workload
    local = polyset.to_local(pts)
    assert isinstance(local, LocalPoints)
    assert len(local) == len(pts) and local.shape == (len(pts), 2)
    assert local.coords.dtype == mx.float32
    assert not local.origin.flags.writeable
    assert "LocalPoints(n=5000" in repr(local)

    head = local[:10]
    assert isinstance(head, LocalPoints) and len(head) == 10
    np.testing.assert_array_equal(head.coords, local.coords[:10])
    picked = local[np.array([5, 1, 7])]
    np.testing.assert_array_equal(picked.coords, mx.take(local.coords, mx.array([5, 1, 7]), axis=0))
    picked_mx = local[mx.array([2, 3], dtype=mx.int64)]
    assert len(picked_mx) == 2
    with pytest.raises(TypeError):
        local[3]
    with pytest.raises(TypeError):
        local[np.ones(len(pts), dtype=bool)]
    with pytest.raises(TypeError):
        iter(local)
    with pytest.raises(AttributeError):
        local.coords = local.coords

    restored = pickle.loads(pickle.dumps(head))
    assert isinstance(restored, LocalPoints) and len(restored) == 10
    assert not restored.origin.flags.writeable
    np.testing.assert_array_equal(restored.coords, head.coords)
    np.testing.assert_array_equal(restored.origin, head.origin)

    with pytest.raises(ValueError):
        LocalPoints(mx.zeros((3, 2), dtype=mx.float64), polyset.origin)
    with pytest.raises(ValueError):
        LocalPoints(mx.zeros((3,), dtype=mx.float32), polyset.origin)
    with pytest.raises(TypeError):
        LocalPoints(np.zeros((3, 2), dtype=np.float32), polyset.origin)


def test_localpoints_rejected_outside_the_join(workload):
    polys, pts, polyset, index = workload
    local = polyset.to_local(pts[:10])
    with pytest.raises(TypeError, match="LocalPoints"):
        polyset.to_local(local)
    with pytest.raises(TypeError, match="lon/lat"):
        ms.haversine(local, local)
    with pytest.raises(TypeError, match="lon/lat"):
        ms.haversine_knn(local, pts[:10], k=1)
    with pytest.raises(TypeError):
        ms.as_point_array(local)
    with pytest.raises(TypeError):
        PolygonSet(local)
    with pytest.raises(TypeError):
        PolygonSet(local.coords)


@needs_metal
def test_origin_mismatch_raises_and_shared_origin_works(workload):
    polys, pts, polyset, index = workload
    local = polyset.to_local(pts)
    other = PolygonSet(polys, origin=(0.0, 0.0))
    assert not np.array_equal(other.origin, polyset.origin)
    with pytest.raises(ValueError, match="origin"):
        GridIndex(other).query(local)
    with pytest.raises(ValueError, match="origin"):
        ms.contains_matrix(local, other)
    # The documented multi-layer pattern: build the second layer in the first
    # layer's frame, and one LocalPoints serves both.
    shared = PolygonSet(polys[:40], origin=polyset.origin)
    np.testing.assert_array_equal(
        GridIndex(shared).query(local), GridIndex(shared).query(pts, backend="gpu")
    )


def test_crs_mismatch_is_rejected_like_sjoin(workload):
    gpd = pytest.importorskip("geopandas")
    polys, pts, polyset, index = workload
    zones = gpd.GeoSeries(list(polys), crs="EPSG:4326")
    same = gpd.GeoSeries(list(shapely.points(pts[:20])), crs="EPSG:4326")
    other = gpd.GeoSeries(list(shapely.points(pts[:20])), crs="EPSG:3857")
    polyset_crs = PolygonSet(zones)
    assert polyset_crs.crs == zones.crs
    assert polyset_crs.to_local(same).crs == zones.crs
    with pytest.raises(ValueError, match="CRS mismatch"):
        polyset_crs.to_local(other)
    with pytest.raises(ValueError, match="CRS mismatch"):
        ms.point_in_polygon(other, polyset_crs)
    # A CRS-less side never raises.
    ms.point_in_polygon(pts[:20], polyset_crs)
    polyset.to_local(other)


# ---------------------------------------------------------------------------
# device inputs never touch the host; backend matrix; laziness
# ---------------------------------------------------------------------------


@needs_metal
def test_device_input_never_round_trips_through_the_host(workload, monkeypatch):
    polys, pts, polyset, index = workload
    local = polyset.to_local(pts)
    expected = np.asarray(index.query(pts, backend="gpu"))

    def forbidden(*args, **kwargs):
        raise AssertionError("device-resident query touched the host")

    class NoHostNumpy:
        """numpy proxy for the join module: any materialization is an error."""

        def __getattr__(self, name):
            if name in ("asarray", "array", "ascontiguousarray"):
                return forbidden
            return getattr(np, name)

    monkeypatch.setattr("mlx_spatial._ingest.as_point_array", forbidden)
    monkeypatch.setattr("mlx_spatial.join.as_point_array", forbidden)
    monkeypatch.setattr(GridIndex, "_candidates", forbidden)
    monkeypatch.setattr(GridIndex, "_query_cpu", forbidden)
    monkeypatch.setattr(PolygonSet, "_host_coords", forbidden)
    monkeypatch.setattr("mlx_spatial.join.np", NoHostNumpy())
    out = index.query(local)
    assert isinstance(out, mx.array)
    np.testing.assert_array_equal(out, expected)
    pt, pg = index.query(local, all_matches=True)
    assert isinstance(pt, mx.array) and isinstance(pg, mx.array)
    assert isinstance(ms.point_in_polygon(local, polyset), mx.array)


@needs_metal
def test_settled_single_launch_query_skips_the_device_round_trip(workload, monkeypatch):
    """Device input, default cap: cells are assigned lazily, no sync before the kernel."""
    polys, pts, polyset, index = workload
    local = polyset.to_local(pts)
    expected = np.asarray(index.query(pts, backend="gpu"))

    def no_sync(*args, **kwargs):
        raise AssertionError("single-launch first match synchronized before the kernel")

    monkeypatch.setattr(GridIndex, "_device_candidates", no_sync)
    out = index.query(local)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(index.query(pts, backend="gpu"), expected)
    with pytest.raises(AssertionError):  # all_matches still needs the exact plan
        index.query(local, all_matches=True)


@needs_metal
def test_precision_warning_is_attributed_to_the_caller(workload):
    polys, pts, polyset, index = workload
    p32 = mx.array(pts[:100].astype(np.float32))
    for call in (
        lambda: ms.point_in_polygon(p32, polyset),
        lambda: index.query(p32),
        lambda: polyset.to_local(p32),
        lambda: ms.contains_matrix(p32, polyset),
        lambda: polyset.recenter_points(p32),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        precision = [w for w in caught if issubclass(w.category, PrecisionWarning)]
        assert len(precision) == 1
        assert precision[0].filename == __file__, precision[0].filename


def test_localpoints_gather_bounds_and_keys(workload):
    polys, pts, polyset, index = workload
    local = polyset.to_local(pts[:20])
    with pytest.raises(IndexError):
        local[np.array([25])]
    with pytest.raises(IndexError):
        local[mx.array([0, 10**9])]
    with pytest.raises(IndexError):
        local[np.array([-21])]
    with pytest.raises(TypeError, match="coords"):
        local[:, 0]
    assert len(local[[]]) == 0
    wrapped = local[np.array([-1, -20])]  # negative indices wrap like numpy
    np.testing.assert_array_equal(wrapped.coords, local.coords[[19, 0]])
    with pytest.raises(ValueError, match="shape"):
        ms.point_in_polygon(mx.zeros((4, 3)), [])
    with pytest.raises(ValueError, match="shape"):
        ms.point_in_polygon(mx.zeros((5,)), [])


@needs_metal
def test_device_input_backend_matrix(workload, monkeypatch):
    polys, pts, polyset, index = workload
    small = polyset.to_local(pts[:64])  # far below the host CPU/GPU crossover
    calls = []
    original = pip_impl.pip_first_match

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr("mlx_spatial.join.pip_first_match", spy)
    monkeypatch.delenv("MLX_SPATIAL_BACKEND", raising=False)
    auto = index.query(small)
    assert calls, "device input with backend=auto must run on the GPU"
    calls.clear()
    cpu = index.query(small, backend="cpu")
    assert not calls and isinstance(cpu, mx.array)
    np.testing.assert_array_equal(cpu, auto)
    monkeypatch.setenv("MLX_SPATIAL_BACKEND", "cpu")
    index.query(small)
    assert not calls, "MLX_SPATIAL_BACKEND=cpu must be honored for device inputs"


@needs_metal
def test_device_input_under_cpu_default_device(workload):
    polys, pts, polyset, index = workload
    local = polyset.to_local(pts[:2000])
    expected = np.asarray(index.query(pts[:2000], backend="gpu"))
    old = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        np.testing.assert_array_equal(index.query(local, backend="gpu"), expected)
        np.testing.assert_array_equal(index.query(local), expected)
    finally:
        mx.set_default_device(old)


@needs_metal
def test_lazy_device_result_is_consumable_without_eval(workload):
    polys, pts, polyset, index = workload
    local = polyset.to_local(pts[:1000])
    labels = index.query(local)
    # Downstream MLX ops fuse onto the lazy result: the -1 label maps to a
    # spare embedding row, the pattern the README example uses.
    table = mx.arange(len(polys) + 1, dtype=mx.int32)
    rows = mx.take(table, mx.where(labels < 0, len(polys), labels))
    expected = np.asarray(index.query(pts[:1000], backend="gpu"))
    np.testing.assert_array_equal(
        rows, np.where(expected < 0, len(polys), expected)
    )
    assert labels.tolist() == expected.tolist()


@needs_metal
def test_empty_nan_and_zero_polygon_device_results(workload):
    polys, pts, polyset, index = workload
    empty = polyset.to_local(np.empty((0, 2)))
    assert len(empty) == 0
    out = index.query(empty)
    assert isinstance(out, mx.array) and out.shape == (0,)
    pt, pg = index.query(empty, all_matches=True)
    assert pt.shape == (0,) and pg.shape == (0,)

    weird = polyset.to_local(np.array([[np.nan, 0.0], [np.inf, 0.0], [0.0, -np.inf]]))
    assert index.query(weird).tolist() == [-1, -1, -1]
    assert index.query(weird, all_matches=True)[0].shape == (0,)

    none = ms.point_in_polygon(polyset.to_local(pts[:7]), [])
    assert isinstance(none, mx.array) and none.tolist() == [-1] * 7
    assert isinstance(ms.point_in_polygon(pts[:7], []), np.ndarray)


# ---------------------------------------------------------------------------
# fp32 cell assignment: padded ranges make the cell choice irrelevant
# ---------------------------------------------------------------------------


@needs_metal
def test_padded_ranges_cover_cell_line_points_on_a_tiling():
    """Points within a few ulps of cell lines get one answer on every route.

    An 8x8 unit tiling puts every bbox edge exactly on a cell line (every
    fourth line with the default quarter-median cells), the worst case for
    fp32 assignment. Padding lists each tile on both sides of those lines, so
    host float64 planning and device fp32 assignment agree bit for bit and
    match shapely outside the band; a point in a tile's central cells sees
    one candidate, edge cells two, corner cells four.
    """
    polys = make_grid_polygons(8, 8)
    polyset = PolygonSet(polys)
    index = GridIndex(polyset)
    assert index.cell_size == 0.25 and (index.nx, index.ny) == (32, 32)
    ulp = float32_resolution(polyset.max_abs_coord)
    lines = np.arange(0.0, 9.0)
    ks = np.array([-32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32])
    xs = (lines[:, None] + ks[None, :] * ulp).ravel()
    rng = np.random.default_rng(3)
    pts = np.concatenate(
        [
            np.column_stack([xs, rng.uniform(0.2, 0.8, len(xs)) + rng.integers(0, 8, len(xs))]),
            np.column_stack([rng.uniform(0.2, 0.8, len(xs)) + rng.integers(0, 8, len(xs)), xs]),
            np.array([[0.0, 0.0], [8.0, 8.0], [8.0, 0.0], [0.0, 8.0], [4.0, 4.0]]),
            np.array([[4.5, 4.5], [0.5, 7.5]]),
        ]
    )
    counts, _, _, _ = index._candidates(np.array([[4.5, 4.5], [4.0, 4.5]]))
    assert counts[0] == 1  # interior point: only its own tile
    assert counts[1] == 2  # on a shared edge: padding lists both tiles

    host = index.query(pts, backend="gpu")
    device = index.query(polyset.to_local(pts))
    np.testing.assert_array_equal(device, host)
    h_pt, h_pg = index.query(pts, backend="gpu", all_matches=True)
    d_pt, d_pg = _pairs(index.query(polyset.to_local(pts), all_matches=True))
    np.testing.assert_array_equal(d_pt, h_pt)
    np.testing.assert_array_equal(d_pg, h_pg)

    ok = ~near_boundary_mask(pts, polys, polyset.boundary_tolerance)
    ref = reference_assign(pts, polys)
    assert ok.sum() > 0  # the +-16/32 ulp points are outside the band
    np.testing.assert_array_equal(np.asarray(device)[ok], ref[ok])

    # Device cells differ from float64 cells only at cell lines, by one cell.
    cell_d = np.asarray(index._device_candidates(polyset.to_local(pts).coords)[0])
    _, _, _, cell_h = index._candidates(pts)
    both = (cell_d >= 0) & (cell_h >= 0)
    assert np.all(np.abs(cell_d[both] % index.nx - cell_h[both] % index.nx) <= 1)
    assert np.all(np.abs(cell_d[both] // index.nx - cell_h[both] // index.nx) <= 1)


def test_padded_csr_is_a_superset_of_unpadded_ranges(workload):
    polys, pts, polyset, index = workload
    padded = set()
    for c in range(index.n_cells):
        lo, hi = index._cell_offsets[c], index._cell_offsets[c + 1]
        padded.update((c, int(p)) for p in index._cell_poly[lo:hi])
    bboxes = polyset.bboxes
    unpadded = set()
    for poly, (x0, y0, x1, y1) in enumerate(bboxes):
        if not np.isfinite([x0, y0, x1, y1]).all():
            continue
        ix0 = int(np.clip(np.floor((x0 - index.x0) / index.cell_size), 0, index.nx - 1))
        ix1 = int(np.clip(np.floor((x1 - index.x0) / index.cell_size), 0, index.nx - 1))
        iy0 = int(np.clip(np.floor((y0 - index.y0) / index.cell_size), 0, index.ny - 1))
        iy1 = int(np.clip(np.floor((y1 - index.y0) / index.cell_size), 0, index.ny - 1))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                unpadded.add((iy * index.nx + ix, poly))
    assert unpadded <= padded
    # Extra entries only come from bboxes within the pad of a cell line.
    pad = 2.0 * polyset.boundary_tolerance
    for cell, poly in padded - unpadded:
        x0, y0, x1, y1 = bboxes[poly]
        cx0 = index.x0 + (cell % index.nx) * index.cell_size
        cy0 = index.y0 + (cell // index.nx) * index.cell_size
        near_x = min(abs(x0 - (cx0 + index.cell_size)), abs(x1 - cx0)) <= pad
        near_y = min(abs(y0 - (cy0 + index.cell_size)), abs(y1 - cy0)) <= pad
        assert near_x or near_y


# ---------------------------------------------------------------------------
# haversine and recenter_points compatibility
# ---------------------------------------------------------------------------


def test_haversine_accepts_mx_float64_bitwise(workload):
    polys, pts, polyset, index = workload
    a, b = pts[:500] * 60.0, pts[500:1000] * 60.0
    a64, b64 = mx.array(a, dtype=mx.float64), mx.array(b, dtype=mx.float64)
    np.testing.assert_array_equal(ms.haversine(a64, b64), ms.haversine(a, b))
    np.testing.assert_array_equal(
        ms.haversine(a64, b64, degrees=False), ms.haversine(a, b, degrees=False)
    )
    d64, i64 = ms.haversine_knn(a64, b64, k=3)
    d, i = ms.haversine_knn(a, b, k=3)
    np.testing.assert_array_equal(i64, i)
    np.testing.assert_array_equal(d64, d)


def test_recenter_points_contract_is_unchanged(workload):
    polys, pts, polyset, index = workload
    pts64, local = polyset.recenter_points(pts[:100])
    assert pts64.dtype == np.float64 and local.dtype == mx.float32
    np.testing.assert_array_equal(pts64, pts[:100])
    np.testing.assert_array_equal(local, recenter(pts[:100], polyset.origin))
    back64, back_local = polyset.recenter_points(polyset.to_local(pts[:100]))
    np.testing.assert_array_equal(back_local, local)
    assert np.allclose(back64, pts[:100], atol=4 * polyset.boundary_tolerance)
