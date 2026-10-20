#!/usr/bin/env python3
"""Validate completed current runs and derive tables without importing MLX.

Usage: python analyze_comparison.py [--hosts m1 m4]

The default discovers completed six-case runs in this result directory only.
An explicit host must be complete. Prior-session timings are read separately
for a labeled before/after comparison and never fill missing current results.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parent
E2E = ROOT.parents[1] / "e2e"
PRIOR = ROOT.parent / "2026-10-19-full-e2e"
HOSTS = ("m1", "m4")
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)
DEBUG_VARS = (
    "MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "METAL_DEVICE_WRAPPER_TYPE",
    "METAL_DEBUG_ERROR_MODE", "MLX_METAL_DEBUG", "MLX_GEOSPATIAL_BACKEND",
)
OUTPUT_SCHEMA = {
    "columns": ["point_id", "value", "geometry", "index_right", "zone_id",
                "zone_name", "zone_weight"],
    "dtypes": {"point_id": "int64", "value": "float64", "geometry": "geometry",
               "index_right": "int64", "zone_id": "int64", "zone_name": "string",
               "zone_weight": "float64"},
    "crs": "EPSG:4326", "index_dtype": "int64", "index_name": None,
}


def require(condition, message):
    """Keep evidence checks active even when Python is invoked with -O."""
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_digest(modules):
    return hashlib.sha256(
        json.dumps(modules, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def frozen_constants():
    """Read literal protocol/cases without running GPU or benchmark code."""
    parsed = ast.parse((E2E / "run_e2e.py").read_text())
    values = {}
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("PROTOCOL", "CASES"):
                    values[target.id] = ast.literal_eval(node.value)
    require(set(values) == {"PROTOCOL", "CASES"}, "Frozen harness constants missing")
    return values["PROTOCOL"], values["CASES"]


def positive(value, field):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0, f"Invalid timing: {field}")


def test_count(path):
    text = path.read_text()
    # Only accept a terminal successful pytest summary, not progress or a
    # passing count from an earlier run followed by failures/interruption.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    require(bool(lines), f"Empty pytest log: {path}")
    summary = lines[-1].strip("= ")
    match = re.fullmatch(r"(\d+) passed(?:, \d+ (?:skipped|warnings?|xfailed))* in .+", summary)
    require(match is not None, f"No successful final pytest summary: {path}: {summary}")
    count = int(match.group(1))
    require(count > 0, f"No passing tests: {path}")
    return count


def output_validation(value, field):
    require(value.get("status") == "passed", f"Output validation failed: {field}")
    for key, expected in OUTPUT_SCHEMA.items():
        require(value.get(key) == expected, f"Output schema mismatch: {field}.{key}")


def validate_run(host, run, build, protocol, cases, fingerprints, *, current):
    require(run.get("status") == "completed", f"{host}: incomplete benchmark")
    require(run.get("schema_version") == 1, f"{host}: unsupported result schema")
    require(run.get("protocol") == protocol, f"{host}: changed timing protocol")
    require([case["id"] for case in run["cases"]] == list(range(6)),
            f"{host}: expected every workload exactly once")
    require(run["input_fingerprints"] == fingerprints, f"{host}: input fingerprints changed")
    require(run["reference_fingerprints_verified"] == len(fingerprints) == 25,
            f"{host}: reference fingerprints not all verified")
    require(run["reference_json_sha256"] == sha256(E2E / "reference-benchmark-results.json"),
            f"{host}: different reference file")
    meta = run["metadata"]
    require(meta["package_python_file_sha256"] == build["python_modules"],
            f"{host}: installed source differs from build manifest")
    require(meta["package_python_manifest_sha256"] == build["package_python_manifest_sha256"],
            f"{host}: source manifest digest mismatch")
    for key, filename in (("e2e_runner_sha256", "run_e2e.py"),
                          ("bench_sha256", "bench.py"), ("runner_sha256", "run_benchmarks.py")):
        require(meta[key] == sha256(E2E / filename), f"{host}: changed harness: {filename}")
    require(meta["metal_available"] is True, f"{host}: Metal unavailable")
    require(meta["power"]["power_source"] == "AC Power", f"{host}: not on AC power")
    for name in THREAD_VARS:
        require(meta["environment"].get(name) == "1", f"{host}: {name} must equal 1")
    for name in DEBUG_VARS:
        require(meta["environment"].get(name) is None, f"{host}: {name} must be unset")
    fixture = run["correctness_fixture"]
    output_validation(fixture, f"{host}.fixture")
    output_validation(fixture["empty_output"], f"{host}.fixture.empty")
    require(fixture["rows"] == 8 and fixture["matched_unique_points"] == 7
            and fixture["multi_hit_points"] == 1 and fixture["empty_output"]["rows"] == 0,
            f"{host}: unexpected smoke fixture output")
    checks = 0
    fingerprint_offset = 12
    for expected, case in zip(cases, run["cases"]):
        field = f"{host}.case{expected['id']}"
        require(case["status"] == "completed", f"{field}: incomplete")
        for key, value in expected.items():
            require(case[key] == value, f"{field}: wrong case definition {key}")
        count = 3 if expected["vertices"] == 2048 else 2
        require(case["input_fingerprints"] == fingerprints[fingerprint_offset:fingerprint_offset + count],
                f"{field}: wrong case inputs")
        fingerprint_offset += count
        require(case["input_points"] == case["n"], f"{field}: point count mismatch")
        require(case["matched_unique_points"] + case["unmatched_points"] == case["n"],
                f"{field}: coverage does not sum to input")
        require(0 <= case["multi_hit_points"] <= case["matched_unique_points"] <= case["output_rows"],
                f"{field}: invalid output counts")
        require(re.fullmatch(r"[0-9a-f]{64}", case["output_pair_sha256"]) is not None,
                f"{field}: invalid pair hash")
        pairs = case["pairs"]
        require(len(pairs) == 4, f"{field}: expected one warmup and three trials")
        for trial, pair in enumerate(pairs):
            require(pair["trial"] == trial and pair["phase"] == ("warmup" if trial == 0 else "measured"),
                    f"{field}: incorrect trial sequence")
            order = ["mlx", "geopandas"] if (case["id"] + trial) % 2 == 0 else ["geopandas", "mlx"]
            require(pair["order"] == order, f"{field}: method order did not alternate")
            require(set(pair["seconds"]) == {"mlx", "geopandas"}, f"{field}: missing method")
            for method, seconds in pair["seconds"].items():
                positive(seconds, f"{field}.{trial}.{method}")
            validation = pair["validation"]
            output_validation(validation, f"{field}.{trial}")
            for output_key, case_key in (("rows", "output_rows"), ("pair_sha256", "output_pair_sha256"),
                                         ("input_points", "n"), ("matched_unique_points", "matched_unique_points"),
                                         ("multi_hit_points", "multi_hit_points"), ("unmatched_points", "unmatched_points")):
                require(validation[output_key] == case[case_key], f"{field}: changing output {output_key}")
            checks += 1
        require(set(case["timings"]) == {"mlx", "geopandas"}, f"{field}: wrong timing methods")
        for method, timing in case["timings"].items():
            samples = [pair["seconds"][method] for pair in pairs[1:]]
            expected_timing = {"samples_seconds": samples, "median_seconds": statistics.median(samples),
                               "minimum_seconds": min(samples), "maximum_seconds": max(samples),
                               "range_seconds": max(samples) - min(samples)}
            require(timing == expected_timing, f"{field}.{method}: summary disagrees with samples")
        ratio = case["timings"]["geopandas"]["median_seconds"] / case["timings"]["mlx"]["median_seconds"]
        require(case["speedup_geopandas_over_mlx"] == ratio, f"{field}: wrong speedup")
    require(fingerprint_offset == 25 and checks == 24, f"{host}: incomplete evidence")
    gpu = [gpu for gpu in meta["gpu_hardware"] if gpu.get("sppci_cores")]
    require(len(gpu) == 1, f"{host}: expected one GPU hardware record")
    machine = {"name": run["machine_label"], "cpu_cores": meta["cpu_count"],
               "gpu_cores": int(gpu[0]["sppci_cores"]), "macos": meta["macos"][0]}
    if current:
        machine["tests_passed"] = test_count(ROOT / host / "tests.log")
    return machine, checks


def duration(seconds):
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 0.01:
        return f"{seconds * 1000:.2f} ms"
    if seconds < 0.1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def markdown_tables(data, hosts):
    headings = ["Points × polygons", "Vertices"]
    for host in hosts:
        name = data["machines"][host]["name"]
        headings.extend([f"{name} MLX", f"{name} GeoPandas", "Speedup"])
    lines = ["# Current complete spatial joins", "", "| " + " | ".join(headings) + " |",
             "|---|" + "---:|" * (len(headings) - 1)]
    for row in data["joins"]:
        values = [row["label"], f"{row['vertices']:,}"]
        for host in hosts:
            values.extend([duration(row[host]["mlx_seconds"]), duration(row[host]["geopandas_seconds"]),
                           f"**{row[host]['speedup']:.2f}×**"])
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Each row uses newly measured medians; no earlier run fills a current timing.", "",
                  "# Change from the preceding measured build", "",
                  "Prior and current medians were measured in different sessions. These ratios describe",
                  "observed changes on the same host and inputs; they do not isolate the code changes",
                  "from background load or other session effects.", "",
                  "| Machine | Points × polygons | Vertices | Prior MLX | Current MLX | Prior/current MLX |",
                  "|---|---|---:|---:|---:|---:|"])
    for row in data["prior_comparison"]["joins"]:
        for host in hosts:
            value = row[host]
            lines.append(f"| {data['machines'][host]['name']} | {row['label']} | {row['vertices']:,} | "
                         f"{duration(value['prior_mlx_seconds'])} | {duration(value['current_mlx_seconds'])} | "
                         f"**{value['prior_over_current_mlx']:.2f}×** |")
    lines.extend(["", "A prior/current ratio above 1 means the current measured MLX run was faster.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", choices=HOSTS,
                        help="Require these hosts; default: completed current runs only")
    args = parser.parse_args()
    hosts = args.hosts
    if hosts is None:
        hosts = [host for host in HOSTS if (ROOT / host / "e2e-results.json").exists()
                 and read_json(ROOT / host / "e2e-results.json").get("status") == "completed"]
    require(bool(hosts), "No completed current runs found")
    require(len(hosts) == len(set(hosts)), "Duplicate host selection")
    hosts = [host for host in HOSTS if host in hosts]
    build = read_json(ROOT / "build-manifest.json")
    date.fromisoformat(build["measurement_date"])
    require(manifest_digest(build["python_modules"]) == build["package_python_manifest_sha256"],
            "Build manifest module digest is inconsistent")
    require(re.fullmatch(r"[0-9a-f]{64}", build["wheel_sha256"]) is not None, "Invalid wheel hash")
    revision = build.get("git_sha", build.get("source_git_sha", build.get("base_git_sha")))
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
            "Missing or invalid source Git revision")
    wheels = list((ROOT / "dist").glob("*.whl"))
    for wheel in wheels:
        require(sha256(wheel) == build["wheel_sha256"], f"Wheel hash mismatch: {wheel}")
    protocol, cases = frozen_constants()
    fingerprints = read_json(E2E / "reference-benchmark-results.json")["input_fingerprints"][:25]
    raw = {host: read_json(ROOT / host / "e2e-results.json") for host in hosts}
    machines, checks = {}, 0
    ref = raw[hosts[0]]
    for host, run in raw.items():
        machines[host], count = validate_run(host, run, build, protocol, cases, fingerprints, current=True)
        checks += count
        for key in ("versions", "geos_version"):
            require(run["metadata"][key] == ref["metadata"][key], f"{host}: runtime mismatch: {key}")
        require(run["metadata"]["python_version"].split()[0] == ref["metadata"]["python_version"].split()[0],
                f"{host}: Python version mismatch")
        require(run["correctness_fixture"] == ref["correctness_fixture"], f"{host}: different fixture output")
        for case, other in zip(run["cases"], ref["cases"]):
            for key in ("output_pair_sha256", "output_rows", "matched_unique_points", "multi_hit_points", "unmatched_points"):
                require(case[key] == other[key], f"{host}: cross-host output mismatch: case {case['id']} {key}")
    joins = []
    for i, case in enumerate(ref["cases"]):
        row = {key: case[key] for key in ("id", "label", "n", "m", "vertices", "output_rows",
                                        "matched_unique_points", "multi_hit_points", "unmatched_points")}
        row["detail"] = f"{case['vertices']:,} vertices"
        for host in hosts:
            timings = raw[host]["cases"][i]["timings"]
            ml, gp = timings["mlx"], timings["geopandas"]
            row[host] = {"mlx_seconds": ml["median_seconds"], "geopandas_seconds": gp["median_seconds"],
                         "speedup": gp["median_seconds"] / ml["median_seconds"],
                         "mlx_samples": ml["samples_seconds"], "geopandas_samples": gp["samples_seconds"],
                         "mlx_range": [ml["minimum_seconds"], ml["maximum_seconds"]],
                         "geopandas_range": [gp["minimum_seconds"], gp["maximum_seconds"]]}
        if set(hosts) == set(HOSTS):
            row["m1_over_m4_mlx"] = row["m1"]["mlx_seconds"] / row["m4"]["mlx_seconds"]
        joins.append(row)
    summary = {}
    for host in hosts:
        ratios = [row[host]["speedup"] for row in joins]
        summary[host] = {"min": min(ratios), "max": max(ratios), "median": statistics.median(ratios),
                         "faster_cases": sum(r > 1 for r in ratios), "case_count": len(ratios)}

    prior_build = read_json(PRIOR / "build-manifest.json")
    require(manifest_digest(prior_build["python_modules"]) == prior_build["package_python_manifest_sha256"],
            "Prior build manifest module digest is inconsistent")
    prior = {host: read_json(PRIOR / host / "e2e-results.json") for host in hosts}
    prior_rows = [{key: row[key] for key in ("id", "label", "n", "m", "vertices")} for row in joins]
    prior_sessions = {}
    for host, run in prior.items():
        machine, _ = validate_run(f"prior-{host}", run, prior_build, protocol, cases, fingerprints, current=False)
        for key in ("name", "cpu_cores", "gpu_cores"):
            require(machine[key] == machines[host][key], f"{host}: prior measurement used different {key}")
        meta = run["metadata"]
        for key in ("versions", "geos_version", "macos"):
            require(meta[key] == raw[host]["metadata"][key], f"{host}: prior/current runtime changed: {key}")
        require(meta["python_version"].split()[0] == raw[host]["metadata"]["python_version"].split()[0],
                f"{host}: prior/current Python changed")
        prior_sessions[host] = {"prior_started_at": run["started_at"], "prior_finished_at": run["finished_at"],
                                "current_started_at": raw[host]["started_at"], "current_finished_at": raw[host]["finished_at"]}
        for i, old in enumerate(run["cases"]):
            new = raw[host]["cases"][i]
            require(old["output_pair_sha256"] == new["output_pair_sha256"] and old["output_rows"] == new["output_rows"],
                    f"{host}: prior/current outputs changed for case {i}")
            old_s, new_s = old["timings"]["mlx"]["median_seconds"], new["timings"]["mlx"]["median_seconds"]
            prior_rows[i][host] = {"prior_mlx_seconds": old_s, "current_mlx_seconds": new_s,
                                  "prior_over_current_mlx": old_s / new_s,
                                  "prior_geopandas_seconds": old["timings"]["geopandas"]["median_seconds"],
                                  "current_geopandas_seconds": new["timings"]["geopandas"]["median_seconds"]}
    tests = "; ".join(f"{machines[h]['tests_passed']} passed on {machines[h]['name']}" for h in hosts)
    workers = build.get("point_extraction_workers", 8)
    require(isinstance(workers, int) and not isinstance(workers, bool) and 1 <= workers <= 8,
            "Invalid point extraction worker count")
    methodology = {
        "date": build["measurement_date"], "repetitions": 3, "warmups": 1,
        "paired_output_checks": checks, "output_equality_checked": True,
        "timing": "Median of three alternating paired measured trials after one untimed full warmup per method and workload.",
        "included": "Fresh PolygonSet/grid/edge indexes, threaded point extraction, GPU query and materialization, exact float64/GEOS boundary refinement, complete joined GeoDataFrames with attributes, and canonical sorting.",
        "excluded": "Ready input GeoDataFrame construction/copies, imports and runtime initialization, cold kernel compilation, I/O, correctness validation.",
        "predicate": "within", "join": "inner", "matching": "All matching point-polygon pairs.",
        "exact_boundary": True, "point_extraction_workers": workers,
        "mlx_adapter": "Unchanged all-pairs query plus GeoDataFrame attribute assembly; native ms.sjoin is validated separately, not timed here.",
        "cpu_baseline": "Native geopandas.sjoin(left, right, how=inner, predicate=within), without parallel partitioning.",
        "threads": f"CPU numerical-library thread limits are one. MLX Point coordinate extraction uses up to {workers} Python worker threads, independently of those limits; Metal uses the GPU.",
        "verification": f"{checks} paired full-output checks passed across {len(hosts)} current hosts; 25 reference input fingerprints and every output pair hash verified. All selected current hosts used the same source and runtime versions.",
        "tests": tests,
        "tests_passed": {host: machines[host]["tests_passed"] for host in hosts},
        "scope": "Six synthetic point-in-polygon joins from ready GeoDataFrames, with fresh indexes and full outputs. Three trials are descriptive measurements, not confidence intervals.",
        "prior_comparison": "Prior/current MLX ratios compare the same host, runtime, input fingerprints, output hashes, and timing protocol across different sessions; background conditions may differ. Prior timings never fill current results.",
    }
    data = {"machines": machines, "joins": joins, "summary": summary, "methodology": methodology,
            "wheel_sha256": build["wheel_sha256"], "source_git_sha": revision,
            "prior_comparison": {"result_directory": PRIOR.name, "wheel_sha256": prior_build["wheel_sha256"],
                                 "sessions": prior_sessions, "joins": prior_rows}}
    (ROOT / "comparison.json").write_text(json.dumps(data, indent=2) + "\n")
    with (ROOT / "comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["machine", "case", "vertices", "output_rows", "mlx_s", "geopandas_s", "geopandas_over_mlx",
                         "mlx_min_s", "mlx_max_s", "geopandas_min_s", "geopandas_max_s"])
        for row in joins:
            for host in hosts:
                value = row[host]
                writer.writerow([machines[host]["name"], row["label"], row["vertices"], row["output_rows"],
                                 value["mlx_seconds"], value["geopandas_seconds"], value["speedup"],
                                 *value["mlx_range"], *value["geopandas_range"]])
    (ROOT / "methodology.txt").write_text("\n\n".join(f"{key}: {value}" for key, value in methodology.items()) + "\n")
    (ROOT / "tables.md").write_text(markdown_tables(data, hosts))
    print(json.dumps({"hosts": hosts, "paired_output_checks": checks, "tests": tests, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
