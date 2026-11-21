import itertools
import json
from statistics import median
from time import perf_counter

import bench
import mlx.core as mx
import mlx_geospatial as ms
import numpy as np
import shapely
from mlx_geospatial._edge_index import get_edge_index

def workloads():
    rng = np.random.default_rng(0)
    zones = bench.make_zones(260, rng)
    points = bench.make_points(1_000_000, rng)
    for vertices in (2048, 4096):
        yield f"260 x {vertices}", points, bench.densify_zones(zones, vertices)
    yield "16 x 2048", points, bench.densify_zones(zones[:16], 2048)
    yield "1 x 150k coastline", points, bench.make_coastline(150_000, rng)
    # Many candidates in one cell, but most traversals fail inside the hole.
    circle = np.linspace(0.0, 2 * np.pi, 2049)
    ring = np.column_stack((np.cos(circle), np.sin(circle)))
    shape = shapely.Polygon(ring, holes=[ring * 0.8])
    overlap = np.repeat(np.array([shape], dtype=object), 32)
    yield "32 overlapping detailed rings", rng.uniform(-0.7, 0.7, (262_144, 2)), overlap
    sparse = points.copy()
    sparse[1::2] += 20.0
    sparse[2::4] += 30.0
    yield "260 x 2048, 75% out of bounds", sparse, bench.densify_zones(zones, 2048)

def compare(ref, result):
    if isinstance(result, tuple):
        for a, b in zip(ref, result, strict=True):
            np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_array_equal(ref, result)

def run(route, index, points, mode, edge_index):
    p = index.polygons
    p._y_edge_index = None if mode == "traverse" else False if mode == "build" else edge_index
    p._y_edge_index_work = 0
    if route.startswith("exact"):
        # Time the cost of each cold exact-boundary route on reusable input
        # coordinates, including the first host-side refinement cache build.
        p._edge_box_strtree = None
        p._host_edge_grid_cache = False
        p._edges_f64_cache = None
        p._strtree = None
    start = perf_counter()
    if mode == "build":
        assert get_edge_index(p) is not None
    out = index.query(points, backend="gpu", all_matches=route.endswith("all"),
                      exact_boundary=route.startswith("exact"), predicate="within")
    mx.synchronize()
    return (perf_counter() - start) * 1e3, out

def main():
    print(json.dumps(dict(machine=bench.machine())), flush=True)
    for label, points, zones in workloads():
        polygon_set = ms.PolygonSet(zones)
        index = ms.GridIndex(polygon_set)
        edge_index = get_edge_index(polygon_set)
        if edge_index is None:
            print(json.dumps(dict(layer=label, index="rejected")), flush=True)
            continue
        sizes = (32_768, 65_536, 100_000, 262_144)
        if len(points) >= 1_000_000 and label in ("260 x 2048", "260 x 4096"):
            sizes += (524_288, 1_000_000)
        for n in sizes:
            pts = points[:n]
            local = polygon_set._local_from_f64(pts)
            cells, counts, total, _ = index._device_candidates(local)
            ids = np.asarray(cells)
            total_cell_edges = index._cell_edge_work[ids[ids >= 0]].sum(dtype=np.int64)
            n_active = int(np.count_nonzero(np.asarray(counts)))
            for route in ("first", "all", "exact-first", "exact-all"):
                if n > 262_144 and route.startswith("exact"):
                    continue
                if label.startswith("32 overlapping") and route.startswith("exact"):
                    continue
                _, reference = run(route, index, pts, "cached", edge_index)
                _, traversal = run(route, index, pts, "traverse", edge_index)
                compare(reference, traversal)
                modes = ("build", "traverse", "cached")
                samples = {k: [] for k in modes}
                for order in itertools.islice(itertools.cycle(itertools.permutations(modes)), 3):
                    for mode in order:
                        if route.startswith("exact"):
                            shapely.destroy_prepared(zones)
                        elapsed, result = run(route, index, pts, mode, edge_index)
                        samples[mode].append(elapsed)
                        del result
                print(json.dumps(dict(
                    layer=label, route=route, n=n, n_active=n_active, total_pairs=total,
                    n_polygons=polygon_set.n_polygons,
                    cell_edge_work=int(total_cell_edges),
                    total_edges=polygon_set._total_edges, max_ring=polygon_set.max_edges_per_ring,
                    max_cell_count=index._max_cell_count, average_edges_per_point=index._edges_per_point,
                    max_bin_edges=edge_index.max_bin_edges,
                    samples=samples, median_ms={k: median(x) for k, x in samples.items()},
                )), flush=True)

if __name__ == "__main__":
    main()
