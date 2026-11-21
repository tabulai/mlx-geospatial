"""Validate the archived runs and regenerate the benchmark tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
HOSTS = {"m1": "M1 Ultra", "m4": "M4 Max"}


def read(path):
    return json.loads(path.read_text())


def elapsed(seconds):
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 0.01:
        return f"{seconds * 1e3:.2f} ms"
    if seconds < 0.1:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    return f"{seconds:.2f} s"


def table(headers, rows):
    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in (headers, ["---"] * len(headers), *rows)
    )


def exact_comparison(manifest):
    runs = {host: read(ROOT / host / "e2e-results.json") for host in HOSTS}
    reference = runs["m4"]
    for run in runs.values():
        assert run["status"] == "completed"
        assert run["reference_fingerprints_verified"] == 25
        assert run["input_fingerprints"] == reference["input_fingerprints"]
        assert run["protocol"] == reference["protocol"]
        assert run["correctness_fixture"]["status"] == "passed"
        assert run["correctness_fixture"]["pair_sha256"] == reference["correctness_fixture"]["pair_sha256"]
        assert run["metadata"]["versions"] == reference["metadata"]["versions"]
        assert run["metadata"]["package_python_file_sha256"] == manifest["package_python_file_sha256"]
        assert len(run["cases"]) == 6

    joins = []
    for reference_case in reference["cases"]:
        i = reference_case["id"]
        row = {key: reference_case[key] for key in ("id", "label", "n", "m", "vertices", "output_rows")}
        row["detail"] = f"{row['vertices']:,} vertices"
        if row["m"] == 1:
            row["detail"] += " · coastline"
        for host, run in runs.items():
            case = run["cases"][i]
            assert case["status"] == "completed" and case["id"] == i
            assert case["output_rows"] == row["output_rows"]
            assert case["output_pair_sha256"] == reference_case["output_pair_sha256"]
            assert len(case["pairs"]) == 4
            assert all(pair["validation"]["status"] == "passed" for pair in case["pairs"])
            timings = row[host] = {}
            for method, summary in case["timings"].items():
                assert len(summary["samples_seconds"]) == 3
                assert median(summary["samples_seconds"]) == summary["median_seconds"]
                timings[f"{method}_seconds"] = summary["median_seconds"]
                timings[f"{method}_samples"] = summary["samples_seconds"]
            ratio = timings["geopandas_seconds"] / timings["mlx_seconds"]
            assert ratio == case["speedup_geopandas_over_mlx"]
            timings["speedup"] = ratio
        joins.append(row)
    return {
        "machines": manifest["hardware"],
        "methodology": {
            "date": manifest["date"],
            "repetitions": 3,
            "output_equality_checked": True,
            "exact_boundary": True,
            "point_extraction_workers": 8,
        },
        "joins": joins,
    }


def join_rows(reports):
    return [report["rows"] for report in reports if report["title"].startswith("join ")]


def matching_time(rows, prefix):
    values = [row["seconds"] for row in rows if row["name"].startswith(prefix)]
    assert len(values) == 1 and values[0] is not None, prefix
    return values[0]


def warm_pair_rows(reports):
    pairs = join_rows(reports)
    return [
        {
            "mlx": matching_time(rows, "mlx-geospatial point_in_polygon, all pairs"),
            "shapely": matching_time(rows, "shapely STRtree"),
            "geopandas": matching_time(rows, "geopandas sjoin"),
            "duckdb": matching_time(rows, "duckdb spatial join"),
        }
        for rows in pairs
    ]


def component_report(manifest, exact):
    blocks = [
        "# Component benchmarks", "",
        "Measurements on November 21, 2026 use the source revision in the "
        "[source manifest](source-manifest.json). Both hosts completed 12 component "
        "reports and 144 CPU/GPU/auto crossover timings. The raw results retain "
        "all trial samples and 30 matched input fingerprints.", "",
        "The component harness reports minimum wall times. Warm joins reuse the MLX "
        "index, and containment uses the float32 predicate with "
        "`exact_boundary=False`. The Shapely and GeoPandas rows build point geometry "
        "and their point index inside the timer, with prepared polygons on one CPU "
        "core. DuckDB queries preregistered inputs and returns a count. MLX returns "
        "pair arrays. These timing contracts describe distinct operations. The "
        "[benchmark guide](../../README.md) covers complete GeoDataFrame joins.", "",
        "Distance and nearest-neighbor CPU baselines use float64. The MLX paths "
        "use float32. BallTree build-plus-query and prebuilt-tree queries have "
        "separate rows.", "",
    ]
    cases = [f"{row['label']} · {row['detail']}" for row in exact["joins"]]
    data = {}
    for host, name in HOSTS.items():
        run = read(ROOT / host / "components/benchmark-results.json")
        assert run["status"] == "completed"
        assert run["metadata"]["package_python_file_sha256"] == manifest["package_python_file_sha256"]
        assert run["metadata"]["bench_sha256"] == manifest["harness_sha256"]["bench/bench.py"]
        assert run["metadata"]["runner_sha256"] == manifest["harness_sha256"]["bench/e2e/run_benchmarks.py"]
        assert len(run["reports"]) == 12 and len(run["input_fingerprints"]) == 30
        if host == "m1":
            fingerprints = run["input_fingerprints"]
        else:
            assert run["input_fingerprints"] == fingerprints
        for timing in run["timings"]:
            assert timing["status"] == "completed"
            assert timing["warmups_completed"] == timing["warmup_count"]
            assert len(timing["durations_seconds"]) == timing["repeats_requested"]
            assert min(timing["durations_seconds"]) == timing["best_seconds"]
        pairs = warm_pair_rows(run["reports"])
        data[host] = pairs
        blocks.extend([
            f"## {name}", "",
            f"[Raw samples]({host}/components/benchmark-results.json) · "
            f"[Console log]({host}/components/benchmark.log)", "",
            "### All matching pairs", "",
            table(["Workload", "MLX pair arrays", "Shapely", "GeoPandas", "DuckDB count"],
                  [[case, *(elapsed(times[k]) for k in ("mlx", "shapely", "geopandas", "duckdb"))]
                   for case, times in zip(cases, pairs, strict=True)]),
            "", "### First match per point", "",
        ])
        prefixes = (
            "mlx-geospatial point_in_polygon, first match (cold:",
            "mlx-geospatial point_in_polygon, first match (warm index)",
            "mlx-geospatial point_in_polygon, first match (warm index, device",
        )
        blocks.extend([
            table(["Workload", "Cold ingestion + query", "Warm NumPy", "Prepared LocalPoints"],
                  [[case, *(elapsed(matching_time(rows, prefix)) for prefix in prefixes)]
                   for case, rows in zip(cases, join_rows(run["reports"]), strict=True)]),
            "", "### Distance matrices, nearest neighbors and dense PIP", "",
            table(["Workload", "Method", "Elapsed"],
                  [[report["title"], row["name"], elapsed(row["seconds"])]
                   for report in run["reports"] if not report["title"].startswith("join ")
                   for row in report["rows"]]),
            "", "### CPU/GPU crossover", "",
        ])
        crossover = {}
        for timing in run["timings"]:
            ctx = timing["context"]
            if ctx.get("function") == "bench_crossover" and "pts" in timing["closure"]:
                key = (ctx["label"], ctx["n"], ctx["all_matches"])
                crossover.setdefault(key, []).append(timing["best_seconds"])
        assert len(crossover) == 48 and all(len(times) == 3 for times in crossover.values())
        blocks.extend([
            table(["Layer", "Points", "First CPU", "First GPU", "First auto", "All CPU", "All GPU", "All auto"],
                  [[layer, f"{n:,}", *(elapsed(t) for t in times),
                    *(elapsed(t) for t in crossover[layer, n, True])]
                   for (layer, n, all_matches), times in crossover.items() if not all_matches]),
            "",
        ])
    (ROOT / "COMPONENTS.md").write_text("\n".join(blocks))
    return data


def main():
    manifest = read(ROOT / "source-manifest.json")
    exact = exact_comparison(manifest)
    pairs = component_report(manifest, exact)
    (ROOT / "e2e-comparison.json").write_text(json.dumps(exact, indent=2) + "\n")
    with (ROOT / "e2e-comparison.csv").open("w", newline="") as f:
        output = csv.writer(f, lineterminator="\n")
        output.writerow(["points", "polygons", "vertices", "joined_rows", "host",
                         "mlx_seconds", "geopandas_seconds", "geopandas_over_mlx"])
        for row in exact["joins"]:
            for host in HOSTS:
                times = row[host]
                output.writerow([row["n"], row["m"], row["vertices"], row["output_rows"],
                                 HOSTS[host], times["mlx_seconds"], times["geopandas_seconds"],
                                 times["speedup"]])
    (ROOT / "warm-pip-comparison.json").write_text(json.dumps(pairs, indent=2) + "\n")
    print("Validated six exact joins, 12 component reports and 144 crossover timings per host.")


if __name__ == "__main__":
    main()
