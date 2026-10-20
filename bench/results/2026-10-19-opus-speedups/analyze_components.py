#!/usr/bin/env python3
"""Validate completed frozen component sweeps and regenerate their tables.

Run from any directory with Python 3.10+; no benchmark dependencies are needed.
Only completed m1/components and m4/components runs are published. Incomplete
or absent hosts are named explicitly, never filled with historical measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re


HOSTS = {"m1": "M1 Ultra", "m4": "M4 Max"}
LAYERS = ("48-vertex zones", "square tiling", "2048-vertex zones")
POINT_COUNTS = (100, 200, 500, 1000, 2000, 5000, 10000, 50000)
DEBUG_KEYS = (
    "MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION", "METAL_DEVICE_WRAPPER_TYPE",
    "METAL_DEBUG_ERROR_MODE", "MLX_METAL_DEBUG", "MLX_GEOSPATIAL_BACKEND",
)
THREAD_KEYS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
METHODS = {
    "matrix": (
        ("mlx", "mlx-geospatial haversine_cross", "ours"),
        ("numpy", "numpy float64", "np64"),
        ("sklearn", "sklearn haversine_distances", "sk"),
    ),
    "knn": (
        ("mlx", "mlx-geospatial haversine_knn", "ours"),
        ("balltree_build_query", "sklearn BallTree(haversine), build + query", "sk_build_and_query"),
        ("balltree_query", "sklearn BallTree(haversine), query only", "sk_query"),
    ),
    "dense": (
        ("direct", "mlx-geospatial dense PIP", "direct"),
        ("composed", "composed MLX dense PIP oracle", "composed"),
    ),
    "join": (
        ("cold_first", "mlx-geospatial point_in_polygon, first match (cold:", "ours_cold"),
        ("warm_first", "mlx-geospatial point_in_polygon, first match (warm index)", "ours_warm"),
        ("warm_all", "mlx-geospatial point_in_polygon, all pairs", "ours_all"),
        ("prepared_first", "mlx-geospatial point_in_polygon, first match (warm index, device-resident", "ours_device"),
        ("shapely", "shapely STRtree", "strtree"),
        ("geopandas", "geopandas sjoin", "gpd_join"),
        ("duckdb", "duckdb spatial", "q"),
    ),
}
EXPECTED_TITLES = (
    "haversine_cross 20k x 20k",
    "haversine_knn 1k queries, 1M refs, k=10",
    "haversine_knn 1M queries, 10k refs, k=10",
    "haversine_knn 200k queries, 200k refs, k=10",
    "haversine_knn 1M queries, 200k refs, k=10",
    "dense PIP 100,000 points x 32 polygons x 48 vertices",
    "join 100,000 points x 260 zones (48 vertices)",
    "join 1,000,000 points x 260 zones (48 vertices)",
    "join 10,000,000 points x 260 zones (48 vertices)",
    "join 1,000,000 points x 10,000 zones (48 vertices)",
    "join 1,000,000 points x 260 zones (2,048 vertices)",
    "join 1,000,000 points x one 150,000-vertex coastline polygon",
)
JOIN_LABELS = ("100k × 260; 48 vertices", "1M × 260; 48 vertices",
               "10M × 260; 48 vertices", "1M × 10k; 48 vertices",
               "1M × 260; 2,048 vertices", "1M × 1; 150,000 vertices")
KNN_LABELS = ("1k × 1M", "1M × 10k", "200k × 200k", "1M × 200k")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_digest(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def validate_samples(timing):
    samples = timing["durations_seconds"]
    require(timing["status"] == "completed", f"Timing {timing['sequence']} did not complete")
    require(len(samples) == timing["repeats_requested"] > 0, "Incomplete timing samples")
    require(timing["warmups_completed"] == timing["warmup_count"], "Incomplete warmups")
    require(all(isinstance(x, (float, int)) and math.isfinite(x) and x > 0 for x in samples),
            "Nonpositive or nonfinite timing")
    require(timing["best_seconds"] == min(samples), "Reported best is not the minimum sample")


def sample_summary(timing):
    validate_samples(timing)
    return {
        "seconds": timing["best_seconds"],
        "minimum_seconds": min(timing["durations_seconds"]),
        "maximum_seconds": max(timing["durations_seconds"]),
        "durations_seconds": timing["durations_seconds"],
        "warmup_count": timing["warmup_count"],
        "repeats": timing["repeats_requested"],
        "timing_sequence": timing["sequence"],
    }


def read_reports(data):
    require(tuple(r["title"] for r in data["reports"]) == EXPECTED_TITLES,
            "Expected all 12 frozen component reports in their original order")
    rows = []
    previous_end = 0
    for report_index, report in enumerate(data["reports"]):
        require(report["timing_sequence_start"] == previous_end, "Report timing ranges are not contiguous")
        previous_end = report["timing_sequence_end_exclusive"]
        timings = data["timings"][report["timing_sequence_start"]:previous_end]
        kind = ("matrix" if report_index == 0 else "knn" if report_index < 5 else
                "dense" if report_index == 5 else "join")
        for method, prefix, suffix in METHODS[kind]:
            matches = [r for r in report["rows"] if r["name"].startswith(prefix)]
            require(len(matches) == 1, f"Missing or duplicate method: {prefix}")
            source = matches[0]
            ts = [t for t in timings if t["callable_qualname"].endswith("." + suffix)]
            row = {"kind": kind, "workload": report["title"], "method": method,
                   "source_label": source["name"], "report_index": report_index}
            if source["seconds"] is None:
                require(method in {"sklearn", "balltree_build_query", "balltree_query", "geopandas", "duckdb"},
                        f"Required method is unavailable: {source['name']}")
                row.update(seconds=None, unavailable="Baseline unavailable or interrupted; see source label and raw log")
                if ts:
                    require(len(ts) == 1 and ts[0]["status"] == "failed", "Unavailable baseline has a completed timer")
                    row["timing_sequence"] = ts[0]["sequence"]
                    row["unavailable"] = ts[0].get("exception", row["unavailable"])
            else:
                require(len(ts) == 1, f"No unique timing for {source['name']}")
                row.update(sample_summary(ts[0]))
                require(row["seconds"] == source["seconds"], "Report and sample timings disagree")
                repeats, warmups = (2, 1) if method == "duckdb" else (3, 0) if method == "geopandas" else (3, 1)
                if method == "geopandas" and report_index == 8:
                    repeats = 1
                require((row["repeats"], row["warmup_count"]) == (repeats, warmups), "Unexpected timing protocol")
            rows.append(row)
    return rows


def log_metadata(path):
    """Auto choices and idle medians are only printed by the frozen harness."""
    result = {"auto_choices": {}, "idle_probe_microseconds": {}, "available": path.exists()}
    if not path.exists():
        return result
    layer = None
    for line in path.read_text().splitlines():
        match = re.match(r"### crossover: [\d,]+ (.+?) \(~", line)
        if match:
            layer = match[1]
        match = re.match(r"\s+([\d,]+)\s+(.+)\|(.+)$", line)
        if layer in LAYERS and match:
            n = int(match[1].replace(",", ""))
            for all_matches, part in ((False, match[2]), (True, match[3])):
                pick = re.search(r"\b(cpu|gpu)(!?)\s*$", part)
                if pick:
                    result["auto_choices"][(layer, n, all_matches)] = pick[1]
        if "GPU_LAUNCH_US" in line and " -> " in line:
            match = re.search(r"\((\d+) warm / (\d+) after the GPU idled 200 ms", line)
            if match:
                result["idle_probe_microseconds"]["first"] = int(match[2])
        if "GPU_ALL_MATCHES_LAUNCH_US" in line and " -> " in line:
            match = re.search(r"\((\d+) warm / (\d+) idle\)", line)
            if match:
                result["idle_probe_microseconds"]["all"] = int(match[2])
    return result


def crossover_rows(data, log):
    # These line identities are safe because the complete frozen script hash
    # is checked before parsing, rather than guessed from timing values.
    methods = {382: "cpu", 386: "gpu", 387: "auto"}
    rows, found = [], set()
    for timing in data["timings"]:
        context = timing["context"]
        if context.get("function") != "bench_crossover" or context["line"] == 368:
            continue
        method = methods.get(context["line"])
        require(method is not None, "Unknown crossover timer in frozen harness")
        key = (context["label"], context["n"], context["all_matches"])
        require(key + (method,) not in found, "Duplicate crossover timer")
        found.add(key + (method,))
        row = {"kind": "crossover", "workload": context["label"], "method": method,
               "n": context["n"], "all_matches": context["all_matches"],
               "edges_per_point": context["edges"], **sample_summary(timing)}
        require((row["warmup_count"], row["repeats"]) == (2, 7), "Unexpected crossover protocol")
        if method == "auto":
            row["auto_selected_backend"] = log["auto_choices"].get(key)
        rows.append(row)
    expected = {(layer, n, all_matches, method) for layer in LAYERS for n in POINT_COUNTS
                for all_matches in (False, True) for method in ("cpu", "gpu", "auto")}
    require(found == expected, "Incomplete crossover sweep")
    return rows


def validate_run(data, manifest, reference, harness_dir):
    require(data["schema_version"] == 1 and data["status"] == "completed", "Run is not completed schema 1")
    require(data["invocation"] == ["bench.py", "--all", "--seed", "0"], "Unexpected invocation")
    metadata = data["metadata"]
    require(metadata["package_python_file_sha256"] == manifest["python_modules"], "Installed source differs from build manifest")
    require(metadata["package_python_manifest_sha256"] == manifest_digest(manifest["python_modules"])
            == manifest["package_python_manifest_sha256"], "Package manifest digest mismatch")
    for file_name, field in (("bench.py", "bench_sha256"), ("run_benchmarks.py", "runner_sha256")):
        require(metadata[field] == manifest["harness_sha256"][file_name] == sha256(harness_dir / file_name),
                f"Frozen harness mismatch: {file_name}")
    require(data["input_fingerprints"] == reference["input_fingerprints"], "Inputs differ from the frozen seed-0 reference")
    require(len(data["input_fingerprints"]) == 30, "Expected 30 generated input fingerprints")
    require(metadata["metal_available"] and metadata["mlx_default_device"] == "Device(gpu, 0)", "Metal GPU not active")
    environment = metadata["environment"]
    require(all(environment.get(key) is None for key in DEBUG_KEYS), "Timing backend/debug override is set")
    require(all(environment.get(key) == "1" for key in THREAD_KEYS), "CPU thread limits differ from protocol")
    require(metadata["power"]["power_source"] == "AC Power", "Run did not record AC power")
    for sequence, timing in enumerate(data["timings"]):
        require(sequence == timing["sequence"], "Timing sequence gap")
        if timing["status"] == "completed":
            validate_samples(timing)
        else:
            require(timing["status"] == "failed" and timing["context"]["function"] == "_bench_duckdb_join",
                    "Failed nonoptional timing in completed sweep")


def prior_compatibility(data, reference):
    old, new = reference["metadata"], data["metadata"]
    reasons = []
    for field in ("bench_sha256", "runner_sha256", "architecture", "cpu_count", "macos", "gpu_hardware", "versions", "geos_version"):
        if old[field] != new[field]:
            reasons.append(f"{field} differs")
    if old["python_version"] != new["python_version"]:
        reasons.append("Python runtime build differs")
    for key in DEBUG_KEYS + THREAD_KEYS:
        if old["environment"].get(key) != new["environment"].get(key):
            reasons.append(f"{key} differs")
    if old["power"].get("power_source") != new["power"].get("power_source"):
        reasons.append("power source differs")
    return reasons


def validate_prior(data, reference, manifest, harness_dir, host):
    """Prior hosts must share the tracked reference's source and runtime versions.

    Python compiler/build strings may differ across hosts; before/after
    comparisons additionally require the exact same Python build on each host.
    """
    old_manifest = {**manifest,
                    "python_modules": reference["metadata"]["package_python_file_sha256"],
                    "package_python_manifest_sha256": reference["metadata"]["package_python_manifest_sha256"]}
    validate_run(data, old_manifest, reference, harness_dir)
    require(HOSTS[host] in data["metadata"]["machine_description"], f"Unexpected prior hardware for {host}")
    for field in ("versions", "geos_version", "architecture"):
        require(data["metadata"][field] == reference["metadata"][field], f"Prior reference {field} mismatch")
    require(data["metadata"]["python_version"].split()[0] == reference["metadata"]["python_version"].split()[0],
            "Prior reference Python version mismatch")
    return read_reports(data)


def fmt(seconds):
    if seconds is None:
        return "n/a"
    scale, unit = (1e6, "µs") if seconds < .001 else (1e3, "ms") if seconds < 1 else (1, "s")
    return f"{seconds * scale:.3g} {unit}"


def relative(seconds, base):
    if seconds is None or base is None:
        return fmt(seconds)
    return f"{fmt(seconds)} · {seconds / base:.2f}×"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows], ""]


def render(result, host_data, reference):
    manifest = result["build_manifest"]
    lines = ["# Component benchmark results — " + manifest["measurement_date"], "",
             f"Fresh frozen-harness sweeps of workspace revision `{manifest['git_sha'][:12]}`.",
             "These tables replace the current component figures; [historical measurements](../../HISTORICAL.md) remain unchanged.",
             "For complete exact joins from ready GeoDataFrames, use [the benchmark guide](../../README.md).", "",
             "## Evidence and coverage", ""]
    coverage = []
    for host, label in HOSTS.items():
        if host in host_data:
            data = host_data[host]
            coverage.append([label, "Completed", data["finished_at"],
                             f"[Raw samples]({host}/components/benchmark-results.json) · [log]({host}/components/benchmark.log)"])
        else:
            coverage.append([label, result["missing_hosts"][host], "—", "No current timings published"])
    lines += table(["Host", "Status", "Finished (UTC)", "Evidence"], coverage)
    lines += ["Each completed run passed checks for all 12 component reports, all 144 CPU/GPU/auto crossover timers,",
              "complete sample counts, report minima, installed module hashes, the [build manifest](build-manifest.json),",
              "and the frozen harness hashes. All 30 generated input fingerprints match the recorded seed-0 reference.",
              "Completed hosts also share package source, Python/dependency versions, input fingerprints, and timer protocols.",
              "The JSON and CSV preserve unrounded samples for the operation tables and crossover timings.",
              "Warm launch-probe samples appear in JSON only; idle-probe medians come from rounded console output,",
              "and the frozen runner does not save their individual samples.", "",
              "The component sweep checks direct dense-PIP output against the composed oracle. It does **not** independently",
              "validate every join, distance, or kNN output against its CPU baseline; the separate full test suite and exact",
              "end-to-end join validation supply correctness evidence. Approximate component pair counts can differ near",
              "boundaries from the exact join counts.", "", "## Timing contract", "",
              "- Reported elapsed times are **minimum wall times**, not medians: best of 3 after one warmup by default; DuckDB best of 2 after one warmup; GeoPandas best of 3 without warmup, except the 10M-point case uses one observation. Crossover uses best of 7 after two warmups and warms GPU clocks before each forced-GPU timer.",
              "- Warm joins reuse a `PolygonSet` and `GridIndex`. NumPy rows include recentering/upload and materialized NumPy results. Prepared `LocalPoints` rows exclude preparation/upload and evaluate the returned MLX array. Cold first-match rows include polygon ingestion and index construction; cold kernel compilation is excluded by warmup.",
              "- These joins use the float32 predicate without `exact_boundary=True`. CPU fallback crossover rows use the library's NumPy float32 predicate, not GeoPandas. Isolated queries after idle time may cost more than these warm bursts.",
              "- All-pairs output contracts differ: MLX returns two NumPy index arrays; Shapely builds points and their STRtree inside the timer; GeoPandas builds point geometries/frame/index and returns a joined frame with prebuilt polygons; DuckDB prebuilds/registers inputs and only returns `count(*)`. Ratios describe elapsed times under these distinct contracts, not equivalent end-to-end speedups.",
              "- Distance/nearest-neighbor MLX computations use float32 versus float64 CPU baselines. MLX host-input timers include conversion and evaluation. BallTree radians conversion is excluded from both CPU timers; the query-only column excludes tree construction and represents reuse of a prebuilt index.",
              "- CPU library thread limits are one; DuckDB uses its recorded default threads, shown per host. MLX uses the GPU. All datasets are synthetic; 2,048-vertex polygons resample 48-vertex boundaries, and the 150,000-vertex case is a single generated ring.", "",
              "For baseline cells, `time · R×` means **baseline elapsed time / MLX elapsed time**. A ratio below 1 means the baseline was faster. Missing baselines remain `n/a`.", ""]
    lines += ["## Machines and runtime", ""]
    lines += table(["Host", "CPU cores", "GPU cores", "macOS", "Python", "MLX", "DuckDB threads"], [
        [HOSTS[h], d["metadata"]["cpu_count"], d["metadata"]["gpu_hardware"][0].get("sppci_cores", "n/a"),
         d["metadata"]["macos"][0], d["metadata"]["python_version"].split()[0], d["metadata"]["versions"]["mlx"],
         d["metadata"]["duckdb_defaults"].get("threads", "n/a")] for h, d in host_data.items()])
    for host, data in host_data.items():
        entries = [r for r in result["measurements"] if r["host"] == host]
        by_report = {(r.get("report_index"), r["method"]): r for r in entries if "report_index" in r}

        def seconds(index, method):
            return by_report[(index, method)]["seconds"]

        lines += [f"## {HOSTS[host]}", "", "### All matching pair arrays", ""]
        lines += table(["Points × polygons; vertices", "MLX all pairs", "Shapely STRtree", "GeoPandas", "DuckDB count(*)"], [
            [label, fmt(seconds(i, "warm_all")), *[relative(seconds(i, method), seconds(i, "warm_all"))
                                                    for method in ("shapely", "geopandas", "duckdb")]]
            for i, label in enumerate(JOIN_LABELS, start=6)])
        lines += ["### First match per point", "", "Returns the lowest-index containing polygon or `-1`, not every matching pair.", ""]
        lines += table(["Points × polygons; vertices", "Warm NumPy → NumPy", "Prepared LocalPoints → MLX", "Cold ingestion + index + query"], [
            [label, *[fmt(seconds(i, method)) for method in ("warm_first", "prepared_first", "cold_first")]]
            for i, label in enumerate(JOIN_LABELS, start=6)])
        lines += ["### Haversine distance matrix", ""]
        lines += table(["Queries × references", "MLX float32", "NumPy float64", "scikit-learn float64"], [
            ["20k × 20k", fmt(seconds(0, "mlx")), relative(seconds(0, "numpy"), seconds(0, "mlx")),
             relative(seconds(0, "sklearn"), seconds(0, "mlx"))]])
        lines += ["### Haversine nearest neighbors, k=10", ""]
        lines += table(["Queries × references", "MLX", "BallTree build + query", "BallTree query only (prebuilt)"], [
            [label, fmt(seconds(i, "mlx")), relative(seconds(i, "balltree_build_query"), seconds(i, "mlx")),
             relative(seconds(i, "balltree_query"), seconds(i, "mlx"))] for i, label in enumerate(KNN_LABELS, start=1)])
        lines += ["### Dense point-in-polygon kernel", "", "Pre-ingested polygons and device-resident points; both outputs are evaluated and synchronized.", ""]
        lines += table(["Points × polygons × vertices", "Direct Metal", "Composed MLX oracle", "Composed / direct"], [
            ["100k × 32 × 48", fmt(seconds(5, "direct")), fmt(seconds(5, "composed")),
             f"{seconds(5, 'composed') / seconds(5, 'direct'):.2f}×"]])
        cross = {(r["workload"], r["n"], r["all_matches"], r["method"]): r for r in entries if r["kind"] == "crossover"}
        lines += ["### CPU/GPU crossover", "",
                  "All rows use the same warm index and host NumPy points. `auto` shows measured elapsed time and the backend printed by the harness; `n/a` means that log evidence is missing.", ""]
        for layer in LAYERS:
            lines += [f"#### {layer} ({256 if layer == 'square tiling' else 260} polygons)", ""]
            values = []
            for n in POINT_COUNTS:
                cells = [f"{n:,}"]
                for all_matches in (False, True):
                    for method in ("cpu", "gpu", "auto"):
                        row = cross[(layer, n, all_matches, method)]
                        cells.append(fmt(row["seconds"]) + (" / " + (row.get("auto_selected_backend") or "n/a") if method == "auto" else ""))
                values.append(cells)
            lines += table(["Points", "First CPU", "First GPU", "First auto / pick", "All CPU", "All GPU", "All auto / pick"], values)
        probes = result["hosts"][host]["launch_probes"]
        lines += ["Warm launch probes at 500 points use best of 7; idle probes are the median of 7 individual queries after 200 ms idle, rounded to whole microseconds by the original log. Idle raw samples are not stored by the frozen runner.", ""]
        lines += table(["Route", "Warm probe", "After 200 ms GPU idle"], [
            [mode, fmt(probes[mode]["seconds"]), "n/a" if probes[mode]["idle_microseconds"] is None
             else f"{probes[mode]['idle_microseconds']:,} µs"] for mode in ("first", "all")])
    lines += ["## Prior recorded sessions: kNN context", "",
              "The earlier per-host sessions below used the same older package source. Their complete module hashes, input fingerprints, harness hashes, and dependency/Python versions match the tracked M4 reference. Python compiler/build strings differ across the two hosts; each before/after comparison requires the exact same Python build on its own host.",
              "Comparisons also require matching hardware, macOS, dependencies, CPU thread limits, and Metal settings. Source hashes and dates for both sessions are retained in the JSON.",
              "These are **prior-session comparisons**, not an isolated causal measurement of one optimization: package changes, elapsed session time, GPU clocks, thermals, and background activity can all affect the result.", ""]
    lines += table(["Host", "Prior session started (UTC)", "Prior source manifest", "Prior raw evidence"], [
        [HOSTS[host], prior["started_at"], f"`{prior['package_python_manifest_sha256'][:12]}`",
         f"[Raw samples]({prior['source']})"] for host, prior in result["prior_references"].items()])
    for host, reasons in result["prior_session_comparison_omissions"].items():
        lines += [f"{HOSTS[host]}: no prior-session comparison ({'; '.join(reasons)}).", ""]
    if result["prior_session_knn"]:
        lines += table(["Host", "Queries × references", "Prior MLX", "Current MLX", "Prior / current"], [
            [HOSTS[r["host"]], r["shape"], fmt(r["prior_seconds"]), fmt(r["current_seconds"]), f"{r['prior_over_current']:.2f}×"]
            for r in result["prior_session_knn"]])
    lines += ["## Reproduce the tables", "", "From the repository root:", "", "```bash",
              "python3 bench/results/2026-10-19-opus-speedups/analyze_components.py", "```", "",
              "The generator reads completed `m1/components/benchmark-results.json` and `m4/components/benchmark-results.json`, checks provenance and timing samples, and writes this report plus [JSON](component-comparison.json) and [CSV](component-comparison.csv). It performs no timed work and never replaces a missing current measurement with a historical value.", "",
              "To collect a new sweep, install the desired wheel and pinned [runtime requirements](../../e2e/requirements.txt), use the environment settings in [the benchmark guide](../../README.md), and run the frozen entry point with a new output directory:", "", "```bash",
              "python -u bench/e2e/run_benchmarks.py --output /path/to/new/components --machine-label 'My Apple silicon Mac'",
              "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--harness-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    harness_dir = args.harness_dir or Path(__file__).resolve().parents[2] / "e2e"
    reference_path = args.reference or harness_dir / "reference-benchmark-results.json"
    manifest = read_json(root / "build-manifest.json")
    reference = read_json(reference_path)
    require(reference["status"] == "completed", "Reference session incomplete")
    require(reference["metadata"]["package_python_manifest_sha256"] ==
            manifest_digest(reference["metadata"]["package_python_file_sha256"]), "Reference source manifest is inconsistent")
    result = {"schema_version": 1, "statistic": "minimum", "build_manifest": manifest,
              "hosts": {}, "missing_hosts": {}, "measurements": [], "prior_session_knn": [],
              "prior_session_comparison_omissions": {}, "prior_references": {},
              "reference": {"path": os.path.relpath(reference_path, harness_dir.parent),
                            "sha256": sha256(reference_path), "started_at": reference["started_at"],
                            "package_python_manifest_sha256": reference["metadata"]["package_python_manifest_sha256"],
                            "package_python_file_sha256": reference["metadata"]["package_python_file_sha256"]}}
    prior_data, prior_rows = {}, {}
    for host in HOSTS:
        path = reference_path if host == "m4" else root / "prior" / "m1-component-results.json"
        if not path.exists():
            continue
        data = reference if host == "m4" else read_json(path)
        prior_rows[host] = validate_prior(data, reference, manifest, harness_dir, host)
        prior_data[host] = data
        result["prior_references"][host] = {
            "source": os.path.relpath(path, root), "sha256": sha256(path),
            "started_at": data["started_at"], "finished_at": data["finished_at"],
            "package_python_manifest_sha256": data["metadata"]["package_python_manifest_sha256"],
            "package_python_file_sha256": data["metadata"]["package_python_file_sha256"],
            "python_version": data["metadata"]["python_version"], "versions": data["metadata"]["versions"],
            "bench_sha256": data["metadata"]["bench_sha256"], "runner_sha256": data["metadata"]["runner_sha256"],
            "verified_fingerprints": len(data["input_fingerprints"]),
        }
    host_data = {}
    for host, label in HOSTS.items():
        path = root / host / "components" / "benchmark-results.json"
        if not path.exists():
            result["missing_hosts"][host] = "Missing current component run"
            continue
        data = read_json(path)
        if data["status"] != "completed":
            result["missing_hosts"][host] = f"Current run {data['status']}; excluded until complete"
            continue
        validate_run(data, manifest, reference, harness_dir)
        require(label in data["metadata"]["machine_description"], f"Unexpected hardware under {host}")
        log = log_metadata(path.with_name("benchmark.log"))
        measurements = read_reports(data) + crossover_rows(data, log)
        for row in measurements:
            row["host"] = host
        result["measurements"].extend(measurements)
        probes = {}
        for timing in data["timings"]:
            if timing["context"]["function"] == "_launch_probe":
                mode = "all" if timing["context"]["kw"].get("all_matches") else "first"
                probes[mode] = {**sample_summary(timing), "idle_microseconds": log["idle_probe_microseconds"].get(mode)}
        require(set(probes) == {"first", "all"}, "Missing launch probes")
        result["hosts"][host] = {"label": label, "source": str(path.relative_to(root)), "sha256": sha256(path),
                                 "started_at": data["started_at"], "finished_at": data["finished_at"],
                                 "metadata": data["metadata"], "launch_probes": probes,
                                 "benchmark_log_available": log["available"],
                                 "verified_fingerprints": len(data["input_fingerprints"])}
        if host_data:
            previous = next(iter(host_data.values()))
            for key in ("versions", "geos_version"):
                require(data["metadata"][key] == previous["metadata"][key], f"Cross-host {key} mismatch")
            require(data["metadata"]["python_version"].split()[0] == previous["metadata"]["python_version"].split()[0],
                    "Cross-host Python version mismatch")
        host_data[host] = data
        reasons = (prior_compatibility(data, prior_data[host]) if host in prior_data
                   else ["no archived prior component session for this host"])
        if reasons:
            result["prior_session_comparison_omissions"][host] = reasons
        else:
            for i, shape in enumerate(KNN_LABELS, start=1):
                current = next(r for r in measurements if r.get("report_index") == i and r["method"] == "mlx")
                prior = next(r for r in prior_rows[host] if r["report_index"] == i and r["method"] == "mlx")
                result["prior_session_knn"].append({"host": host, "shape": shape, "prior_seconds": prior["seconds"],
                                                    "prior_source": result["prior_references"][host]["source"],
                                                    "current_seconds": current["seconds"],
                                                    "prior_over_current": prior["seconds"] / current["seconds"],
                                                    "prior_samples_seconds": prior["durations_seconds"],
                                                    "current_samples_seconds": current["durations_seconds"]})
    for host, reason in result["missing_hosts"].items():
        result["prior_session_comparison_omissions"][host] = [reason.lower()]
    require(host_data, "No completed current component run; no report was written")
    (root / "COMPONENTS.md").write_text(render(result, host_data, reference))
    (root / "component-comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    fields = ("host", "kind", "workload", "method", "n", "all_matches", "seconds", "minimum_seconds", "maximum_seconds",
              "warmup_count", "repeats", "timing_sequence", "auto_selected_backend", "unavailable", "durations_seconds")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in result["measurements"]:
        writer.writerow({**row, "durations_seconds": json.dumps(row.get("durations_seconds", []), separators=(",", ":"))})
    (root / "component-comparison.csv").write_text(stream.getvalue())
    print(f"Validated {len(host_data)} completed hosts; wrote {len(result['measurements'])} measurements and COMPONENTS.md")
    for host, reason in result["missing_hosts"].items():
        print(f"{HOSTS[host]}: {reason}")


if __name__ == "__main__":
    main()
