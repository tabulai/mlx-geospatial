import importlib.util
import itertools
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import bench
import mlx_geospatial as tuned
import shapely
from calibrate import compare, workloads

def load_before(location):
    location = Path(location)
    spec = importlib.util.spec_from_file_location(
        "before_geospatial", location / "__init__.py",
        submodule_search_locations=[str(location)],
    )
    before = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = before
    spec.loader.exec_module(before)
    return before

def measure(route, index, points, zones):
    polygons = index.polygons
    polygons._y_edge_index = False
    polygons._y_edge_index_work = 0
    exact = route.startswith("exact")
    if exact:
        polygons._edge_box_strtree = None
        polygons._host_edge_grid_cache = False
        polygons._edges_f64_cache = None
        polygons._strtree = None
        shapely.destroy_prepared(zones)
    start = perf_counter()
    result = index.query(
        points, backend="gpu", all_matches=route.endswith("all"),
        exact_boundary=exact, predicate="within",
    )
    elapsed = (perf_counter() - start) * 1e3
    return elapsed, result, bool(polygons._y_edge_index)

def main():
    before = load_before(sys.argv[1])
    print(json.dumps({"machine": bench.machine()}), flush=True)
    for layer, points, zones in workloads():
        indices = [module.GridIndex(zones) for module in (before, tuned)]
        sizes = (32_768, 65_536, 100_000, 262_144)
        if layer in ("260 x 2048", "260 x 4096"):
            sizes += (524_288, 1_000_000)
        for n in sizes:
            for route in ("first", "all", "exact-first", "exact-all"):
                if route.startswith("exact") and (n > 262_144 or layer.startswith("32 overlapping")):
                    continue
                times = [[], []]
                builds = [[], []]
                for trial, order in enumerate(itertools.islice(itertools.cycle(((0, 1), (1, 0))), 7)):
                    outputs = [None, None]
                    for i in order:
                        t, outputs[i], built = measure(route, indices[i], points[:n], zones)
                        if trial:
                            times[i].append(t)
                            builds[i].append(built)
                    if trial in (0, 6):
                        compare(*outputs)
                print(json.dumps(dict(
                    layer=layer, route=route, n=n,
                    before_ms=median(times[0]), tuned_ms=median(times[1]),
                    before_build=builds[0], tuned_build=builds[1], equality=True,
                    samples_ms=dict(zip(("before", "tuned"), times, strict=True)),
                )), flush=True)

if __name__ == "__main__":
    main()
