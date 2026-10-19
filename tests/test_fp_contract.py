"""Guards for the fp32 operation-order contract of the Metal kernels.

Every predicate kernel evaluates the x-intersection
``x1 + (py - y1) / (y2 - y1) * (x2 - x1)`` with each operation rounded on its
own. Metal compiles with fast-math, which fuses the multiply-add into an FMA
unless ``#pragma clang fp contract(off)`` is in scope. The pragma is lexically
scoped: a header's file-scope pragma governs the kernel body appended after
it, but a pragma inside a kernel body does not reach a header's template
function. Hence every string that carries the arithmetic must carry its own
pragma, before the arithmetic.

Two guards:

- static (no Metal needed): every ``metal_kernel`` call in the module is
  mapped to the header/source strings it compiles (comments stripped), and the
  one string that evaluates the x-intersection must have a contract pragma
  before it whose last setting is ``off``; the index-scatter kernel is the
  only one without arithmetic;
- behavioural (Metal): probe points constructed at the fp32 x-intersection of
  an edge where fused and unfused arithmetic disagree by a rounding (usually
  one ulp, more near cancellation; ``px`` is set to the smaller candidate, so
  the crossing decision differs), plus exact intersection points, points at
  vertex heights and points on a horizontal chain, must give the same answer
  on all 14 predicate kernels as the NumPy fp32 fallback, the repo's
  bit-identity arbiter; the composed MLX oracle (the non-Metal fallback) is
  held to the same reference in its own test. When this file was written,
  stripping any load-bearing pragma flipped every FMA probe on every kernel
  that string protects; the test does not assert that, because it would pin
  compiler behaviour rather than the library's.

The kNN selectors in ``haversine.py`` carry the same contract for the
haversine ``h`` (``h_lat + qc * cos_lat * sdlon * sdlon``), which must round
like ``_hav_graph`` for selection to rank exactly as ``haversine_cross``. Their
static guard checks that every selector source turns contraction off before
the first sine, keeps the ``h`` expression tree, and uses explicit ``fma``
only in the squared-chord skip test, whose value never reaches a result.
Their behavioural guard is the bit-identity suite in tests/test_knn_fused.py.

Helpers are local to this file.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re

import numpy as np
import mlx.core as mx
import pytest
import shapely

import mlx_geospatial as ms
import mlx_geospatial._pip as pip
from mlx_geospatial._dispatch import metal_available
from mlx_geospatial._ingest import LocalPoints

# The x-intersection's division; tolerant to reformatting.
ARITH = re.compile(r"/\s*\(\s*y2\s*-\s*y1\s*\)")
CONTRACT = re.compile(r"^\s*#pragma\s+clang\s+fp\s+contract\((\w+)\)\s*$", re.M)
# Scatters already-computed match flags into the output; no fp arithmetic.
EXEMPT = {"mlx_geospatial_pip_all_matches_pair_write"}

requires_metal = pytest.mark.skipif(not metal_available(), reason="Metal not available")


# ---------------------------------------------------------------------------
# static guard
# ---------------------------------------------------------------------------


def _code(text: str) -> str:
    """Metal source without comments (a commented-out pragma is no pragma)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _protected(code: str) -> bool:
    """The last contract pragma before the first x-intersection says ``off``."""
    first = ARITH.search(code)
    assert first is not None
    modes = [m.group(1) for m in CONTRACT.finditer(code) if m.start() < first.start()]
    return bool(modes) and modes[-1] == "off"


def _module_strings(module=pip) -> dict[str, str]:
    return {n: v for n, v in vars(module).items() if isinstance(v, str) and not n.startswith("__")}


def _kernel_calls(module=pip) -> list[tuple[str, str | None, str]]:
    """``(kernel name, header text or None, source text)`` per metal_kernel call.

    Read from the module's source with ``ast`` so runtime wrappers around the
    getters (route spies) cannot hide a call; string expressions may be module
    constants, literals or concatenations of those.
    """
    consts = _module_strings(module)

    def resolve(node):
        if isinstance(node, ast.Name):
            return consts[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return resolve(node.left) + resolve(node.right)
        raise AssertionError(f"cannot resolve a kernel string from {ast.dump(node)[:80]}")

    calls = []
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "metal_kernel":
            kw = {k.arg: k.value for k in node.keywords}
            header = resolve(kw["header"]) if "header" in kw else None
            calls.append((kw["name"].value, header, resolve(kw["source"])))
    return calls


def test_every_predicate_kernel_string_carries_its_own_pragma():
    calls = _kernel_calls()
    assert len(calls) == 15, [c[0] for c in calls]
    predicate_kernels = 0
    for name, header, source in calls:
        carriers = [t for t in (header, source) if t is not None and ARITH.search(_code(t))]
        if name in EXEMPT:
            assert carriers == [], f"{name} evaluates the predicate but is exempt"
            continue
        assert len(carriers) == 1, f"{name}: x-intersection in {len(carriers)} strings, expected one"
        code = _code(carriers[0])
        assert _protected(code), (
            f"{name} evaluates the x-intersection without a live, preceding "
            "'#pragma clang fp contract(off)': Metal would fuse it into an FMA"
        )
        if carriers[0] is header:
            # A header pragma protects only what follows it in the header.
            assert CONTRACT.search(code).start() < code.index("METAL_FUNC"), name
        predicate_kernels += 1
    assert predicate_kernels == 14


def test_pragma_and_arithmetic_pair_up_in_every_module_string():
    """Every string with the arithmetic is protected; no stray pragmas.

    A pragma in a string without arithmetic would suggest a kernel body can
    protect its header, which it cannot (the header's own pragma does that).
    """
    strings = {n: _code(v) for n, v in _module_strings().items()}
    carriers = sorted(n for n, code in strings.items() if ARITH.search(code))
    assert len(carriers) >= 6, carriers  # three scalar sources, three headers today
    for name in carriers:
        assert _protected(strings[name]), name
    with_pragma = sorted(n for n, code in strings.items() if CONTRACT.search(code))
    assert with_pragma == carriers


# ---------------------------------------------------------------------------
# static guard: haversine kNN selectors
# ---------------------------------------------------------------------------

hv = importlib.import_module("mlx_geospatial.haversine")

KNN_SELECTORS = {
    "mlx_geospatial_knn_select",
    "mlx_geospatial_knn_simdgroup_select",
    "mlx_geospatial_knn_chord_select",
    "mlx_geospatial_knn_chord_simdgroup_select",
}
HAV_ARITH = re.compile(r"metal::precise::sin\(")
FMA_CALL = re.compile(r"\bfma\s*\(")
CHORD_DEF = re.compile(r"^float\s+(chord\d+)\s*=\s*metal::fma\(")
# The _hav_graph tree: sin^2 terms, ((qc * cos_lat) * sdlon) * sdlon, sum.
H_TREE = (
    re.compile(r"float h_lat = sdlat \* sdlat;"),
    re.compile(r"float h_lon = qc \* [\w\[\]\s*()+]+ \* sdlon \* sdlon;"),
    re.compile(r"float h = h_lat \+ h_lon;"),
)


def _statements(code: str) -> list[str]:
    return [" ".join(s.split()) for s in code.split(";")]


def test_every_knn_selector_turns_contraction_off_before_h():
    calls = _kernel_calls(hv)
    assert {name for name, _, _ in calls} == KNN_SELECTORS, [c[0] for c in calls]
    assert len(calls) == len(KNN_SELECTORS)
    for name, header, source in calls:
        assert header is None, name
        code = _code(source)
        first = HAV_ARITH.search(code)
        assert first is not None, name
        modes = [m for m in CONTRACT.finditer(code)]
        # One pragma, before the first sine, and nothing re-enables fusion.
        assert [m.group(1) for m in modes] == ["off"], name
        assert modes[0].start() < first.start(), name
        n_h = len(HAV_ARITH.findall(code)) // 2
        assert n_h >= 1, name
        for pattern in H_TREE:
            assert len(pattern.findall(code)) == n_h, (name, pattern.pattern)


def test_explicit_fma_only_feeds_the_chord_skip_test():
    chord_selectors = 0
    for name, _, source in _kernel_calls(hv):
        code = _code(source)
        fma_statements = [s for s in _statements(code) if FMA_CALL.search(s)]
        chords = set()
        for statement in fma_statements:
            match = CHORD_DEF.match(statement)
            assert match, f"{name}: fma outside a chord definition: {statement}"
            chords.add(match.group(1))
        for chord in chords:
            for use in re.finditer(rf"\b{chord}\b", code):
                rest = code[use.end():]
                assert re.match(r"\s*(=\s*metal::fma\(|<=\s*thr\b)", rest), (
                    f"{name}: {chord} (fma-contracted) used outside the skip test"
                )
        chord_selectors += bool(chords)
    assert chord_selectors == 2


# ---------------------------------------------------------------------------
# behavioural guard
# ---------------------------------------------------------------------------

F32 = np.float32


def _kernel_getters() -> list[str]:
    return sorted(n for n in dir(pip) if n.startswith("_get_") and n.endswith("_kernel"))


def _spy_getters(monkeypatch) -> dict[str, int]:
    """Count launches per kernel getter (one call per launch)."""
    counts = dict.fromkeys(_kernel_getters(), 0)

    def wrap(name, original):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapped

    for name in counts:
        monkeypatch.setattr(pip, name, wrap(name, getattr(pip, name)))
    return counts


def _one_cell_csr(n_points: int, n_polys: int):
    """Grid CSR with one cell holding every polygon (ascending)."""
    return (
        mx.array(np.zeros(n_points, dtype=np.int32)),
        mx.array(np.array([0, n_polys], dtype=np.int64)),
        mx.array(np.arange(n_polys, dtype=np.int32)),
    )


def _triangle_layer() -> np.ndarray:
    """100 non-overlapping triangles near the origin (|x| <= 1.5)."""
    rng = np.random.default_rng(1234)
    polys = []
    for i in range(10):
        for j in range(10):
            c = np.array([-1.35 + 0.3 * i, -1.35 + 0.3 * j])
            tri = c + rng.uniform(-0.13, 0.13, (3, 2))
            polys.append(shapely.Polygon(tri))
    return np.array(polys, dtype=object)


def _chord_ring(center, radius: float, n_dense: int = 1090, n_spread: int = 10) -> shapely.Polygon:
    """Convex >= 1024-edge ring: a dense arc plus a few long chords.

    The chords are the probed edges; their x extent (~0.6 radius) against a
    small |x1| keeps the fused/unfused disagreement rate high.
    """
    dense = np.linspace(0.0, 0.1, n_dense, endpoint=False)
    spread = np.linspace(0.1, 2.0 * np.pi, n_spread, endpoint=False)
    ang = np.concatenate([dense, spread])
    return shapely.Polygon(
        np.column_stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)])
    )


def _staircase_ring(center, radius: float, top: float, k: int = 1100) -> shapely.Polygon:
    """>= 1024-edge circle whose top is clamped to an exact horizontal chain."""
    ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    pts = np.column_stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)])
    pts[:, 1] = np.minimum(pts[:, 1], top)
    return shapely.Polygon(pts)


def _long_ring_layer() -> np.ndarray:
    """24 chord rings on a grid plus one staircase ring, none overlapping."""
    polys = [_staircase_ring((0.0, 0.0), 1.0, top=0.5)]
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                continue
            polys.append(_chord_ring((2.2 * (i - 2), 2.2 * (j - 2)), 1.0))
    return np.array(polys, dtype=object)


def _bboxes_disjoint(polys) -> bool:
    b = shapely.bounds(polys)
    m = len(b)
    for i in range(m):
        o = (b[i, 0] < b[:, 2]) & (b[:, 0] < b[i, 2]) & (b[i, 1] < b[:, 3]) & (b[:, 1] < b[i, 3])
        o[i] = False
        if o.any():
            return False
    return True


def _fma_probes(polyset: ms.PolygonSet, edge_mask: np.ndarray, rng, samples: int):
    """Probe points where fused and unfused x-intersections disagree.

    For each selected fp32 local edge and ``samples`` heights ``py`` strictly
    straddled by the edge: ``xu`` is the unfused intersection (divide, then
    multiply, then add, each rounded), ``xf`` the fused one (exact product,
    one rounding; float64 holds a 24x24-bit product exactly, and f32-tie sums
    are dropped so the emulation is exact). Where they differ the point
    ``(min(xu, xf), py)`` is inside by one rule and outside by the other.
    Returns local fp32 points, the exact-intersection points ``(xu, py)`` and
    the target polygon of each.
    """
    edges = polyset._edges_np()[edge_mask]
    offsets = polyset._geom_edge_offsets
    edge_poly = (np.searchsorted(offsets, np.flatnonzero(edge_mask), side="right") - 1)
    e = np.repeat(edges, samples, axis=0)
    poly = np.repeat(edge_poly, samples)
    x1, y1, x2, y2 = e[:, 0], e[:, 1], e[:, 2], e[:, 3]
    t = rng.uniform(0.05, 0.95, len(e)).astype(F32)
    py = (y1 + t * (y2 - y1)).astype(F32)
    straddle = ((y1 > py) != (y2 > py)) & (y2 != y1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = ((py - y1) / (y2 - y1)).astype(F32)
    dx = (x2 - x1).astype(F32)
    xu = (x1 + (q * dx).astype(F32)).astype(F32)
    s64 = q.astype(np.float64) * dx.astype(np.float64) + x1.astype(np.float64)
    tie = (s64.view(np.uint64) & np.uint64((1 << 29) - 1)) == np.uint64(1 << 28)
    xf = s64.astype(F32)
    keep = straddle & ~tie & (xu != xf)
    exact = straddle & ~tie
    fma_pts = np.column_stack([np.minimum(xu, xf)[keep], py[keep]]).astype(F32)
    exact_pts = np.column_stack([xu[exact], py[exact]]).astype(F32)
    return fma_pts, poly[keep], exact_pts, poly[exact]


def _probe_set(polys, chord_edges: bool, seed: int):
    """Local fp32 probes for a layer: FMA-sensitive, exact-xint, chain, interior."""
    rng = np.random.default_rng(seed)
    polyset = ms.PolygonSet(polys, origin=(0.0, 0.0))
    edges = polyset._edges_np()
    if chord_edges:
        mask = np.abs(edges[:, 2] - edges[:, 0]) > 0.2  # the long chords only
    else:
        mask = np.ones(len(edges), dtype=bool)
    fma_pts, fma_poly, exact_pts, exact_poly = _fma_probes(polyset, mask, rng, samples=40)
    parts = [fma_pts, exact_pts[:400]]
    counts = {"fma": len(fma_pts), "exact": min(400, len(exact_pts))}
    # points at exact vertex heights inside the owning polygon's bbox: the
    # half-open rule in y decides which incident edge counts the crossing
    pick = rng.integers(0, len(edges), 100)
    owner = np.searchsorted(polyset._geom_edge_offsets, pick, side="right") - 1
    bbl = polyset._bboxes_local_np()[owner]
    vertex_y = np.column_stack([rng.uniform(bbl[:, 0], bbl[:, 2]), edges[pick, 1]]).astype(F32)
    parts.append(vertex_y)
    counts["vertex_y"] = 100
    if chord_edges:
        # points on the staircase ring's horizontal chain (polygon 0 at the
        # origin, top clamped to y = 0.5 exactly)
        chain = np.column_stack([rng.uniform(-0.8, 0.8, 200), np.full(200, 0.5)]).astype(F32)
        parts.append(chain)
        counts["chain"] = 200
    # interior/exterior points inside every bbox so each polygon gets hits
    bb = shapely.bounds(polys)
    inside = np.concatenate(
        [rng.uniform(bb[i, :2], bb[i, 2:], size=(8, 2)) for i in range(len(polys))]
    ).astype(F32)
    parts.append(inside)
    pts = np.concatenate(parts)
    return polyset, pts, counts


def _numpy_matrix(polyset: ms.PolygonSet, pts32: np.ndarray) -> np.ndarray:
    """(N, M) bool containment from the NumPy fp32 fallback (the arbiter)."""
    index = ms.GridIndex(polyset, cell_size=1e6)
    local = LocalPoints(mx.array(pts32), polyset.origin)
    pt, pg = index.query(local, backend="cpu", all_matches=True)
    matrix = np.zeros((len(pts32), polyset.n_polygons), dtype=bool)
    matrix[np.asarray(pt), np.asarray(pg)] = True
    return matrix


def _first_match(matrix: np.ndarray) -> np.ndarray:
    hit = matrix.any(axis=1)
    return np.where(hit, matrix.argmax(axis=1), -1).astype(np.int32)


def _layer_and_probes(layer: str):
    if layer == "triangles":
        polys, chord = _triangle_layer(), False
    else:
        polys, chord = _long_ring_layer(), True
        assert all(p.is_valid for p in polys)
    assert _bboxes_disjoint(polys)  # flips stay visible to first match
    polyset, pts, counts = _probe_set(polys, chord, seed=7)
    # non-vacuity: enough decisive probes of each kind, none bbox-rejected
    assert counts["fma"] >= 200 and counts["exact"] >= 100 and counts["vertex_y"] == 100, counts
    if chord:
        assert counts["chain"] == 200 and polyset.max_edges_per_ring >= 1024
    bb = polyset._bboxes_local_np()
    in_some_bbox = (
        (pts[:, None, 0] >= bb[None, :, 0]) & (pts[:, None, 1] >= bb[None, :, 1])
        & (pts[:, None, 0] <= bb[None, :, 2]) & (pts[:, None, 1] <= bb[None, :, 3])
    ).any(axis=1)
    assert in_some_bbox.all()
    expected = _numpy_matrix(polyset, pts)
    hits = expected.sum()
    assert 0.2 * len(pts) <= hits <= 0.8 * len(pts), hits
    return polyset, pts, expected


@requires_metal
@pytest.mark.parametrize("layer", ["triangles", "long rings"])
def test_composed_oracle_matches_the_fallback_on_fma_sensitive_probes(layer):
    """The mx.compile'd crossing graph (the non-Metal pip_dense fallback and
    the tests' oracle) rounds each operation on its own, like NumPy."""
    polyset, pts, expected = _layer_and_probes(layer)
    composed = np.array(pip._pip_dense_composed(mx.array(pts), polyset)).astype(bool)
    np.testing.assert_array_equal(
        composed, expected,
        err_msg="composed MLX oracle (mx.compile'd _crossed_graph, the non-Metal fallback) != "
        "NumPy fp32 fallback: MLX fused or reordered the x-intersection; the Metal kernels "
        "are checked separately",
    )


@requires_metal
@pytest.mark.parametrize("layer", ["triangles", "long rings"])
def test_every_predicate_kernel_matches_the_fallback_on_fma_sensitive_probes(layer, monkeypatch):
    """All 14 predicate kernels agree with NumPy fp32 where FMA would not.

    Routing is forced through the strategy selectors so one probe set reaches
    every kernel; natural routing is covered by tests/test_kernel_routes.py.
    """
    polyset, pts, expected = _layer_and_probes(layer)
    n, m = len(pts), polyset.n_polygons
    local = mx.array(pts)

    counts_seen = _spy_getters(monkeypatch)
    pair_pt = mx.array(np.repeat(np.arange(n, dtype=np.int32), m))
    pair_pg = mx.array(np.tile(np.arange(m, dtype=np.int32), n))
    point_cell, cell_offsets, cell_poly = _one_cell_csr(n, m)
    # first match also sees out-of-grid points (cell -1), which the public
    # route passes through to the kernels; they must come back as -1
    outside = np.arange(0, n, 9)
    first_cell = np.zeros(n, dtype=np.int32)
    first_cell[outside] = -1
    point_id = mx.array(np.arange(n, dtype=np.int32))
    candidate_offsets = mx.array(np.arange(0, (n + 1) * m, m, dtype=np.int64))
    expected_first = _first_match(expected)
    expected_first[outside] = -1
    expected_pt, expected_pg = np.nonzero(expected)

    for strategy in ("scalar", "simdgroup", "threadgroup"):
        monkeypatch.setattr(pip, "_pip_long_polygon_strategy", lambda *a, s=strategy, **k: s)
        got = np.array(pip.pip_pairs(local, polyset, pair_pt, pair_pg)).reshape(n, m).astype(bool)
        np.testing.assert_array_equal(got, expected, err_msg=f"pip_pairs/{strategy}")
        got = np.array(pip.pip_dense(local, polyset))
        np.testing.assert_array_equal(got, expected, err_msg=f"pip_dense/{strategy}")
        got = np.array(
            pip.pip_first_match(local, polyset, mx.array(first_cell), cell_offsets, cell_poly)
        )
        np.testing.assert_array_equal(got, expected_first, err_msg=f"pip_first_match/{strategy}")
        for compaction in ("point", "pair"):
            if compaction == "point" and strategy != "scalar":
                continue  # the point-centric compactor has no cooperative variant
            monkeypatch.setattr(pip, "_all_matches_strategy", lambda *a, c=compaction, **k: c)
            pt, pg = pip.pip_all_matches(
                local, polyset, point_id, point_cell, candidate_offsets, cell_offsets, cell_poly,
                pair_begin=0, pair_end=n * m, max_point_pairs=m,
            )
            np.testing.assert_array_equal(pt, expected_pt, err_msg=f"all_matches/{compaction}/{strategy}")
            np.testing.assert_array_equal(pg, expected_pg, err_msg=f"all_matches/{compaction}/{strategy}")
    # one launch per kernel, except the pair write kernel shared by the three
    # pair-parallel count kernels
    expected_counts = {
        g: (3 if g == "_get_all_matches_pair_write_kernel" else 1) for g in counts_seen
    }
    assert counts_seen == expected_counts, counts_seen
