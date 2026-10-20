#!/usr/bin/env python3
"""Instrument the adjacent, unmodified bench.py and run its --all --seed 0 sweep.

Usage: python -u run_benchmarks.py --output results/m1-ultra --machine-label 'M1 Ultra'
The original benchmark prints normally. Structured evidence is written to
benchmark-results.json after every report and on completion or failure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_value(fn):
    try:
        return fn()
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar(value):
    return value is None or isinstance(value, (bool, int, float, str))


def compact_value(value):
    if scalar(value):
        return value
    if isinstance(value, dict) and all(isinstance(k, str) and scalar(v) for k, v in value.items()):
        return dict(value)
    return None


def load_snapshot():
    return {
        "utc": utc_now(),
        "load_average_1_5_15_minutes": safe_value(lambda: list(os.getloadavg())),
        "vm_stat": safe_value(lambda: command_output(["vm_stat"])),
        "swap_usage": safe_value(lambda: read_sysctl("vm.swapusage")),
    }


def command_output(argv):
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=30, check=True
    ).stdout.strip()


def gpu_hardware():
    data = json.loads(command_output(["system_profiler", "SPDisplaysDataType", "-json"]))
    return [
        {
            key: value for key, value in device.items()
            if scalar(value) and (key == "_name" or any(
                token in key.lower() for token in ("model", "chip", "core", "metal", "vram")
            ))
        }
        for device in data.get("SPDisplaysDataType", [])
    ]


def power_metadata():
    battery = command_output(["pmset", "-g", "batt"])
    source = re.search(r"Now drawing from '([^']+)'", battery)
    settings = command_output(["pmset", "-g", "custom"])
    return {
        "power_source": source.group(1) if source else None,
        "battery_percent": re.findall(r"(\d+)%", battery),
        "power_mode_settings": [
            line.strip() for line in settings.splitlines()
            if any(token in line.lower() for token in ("power:", "powermode", "cpupower", "gpuswitch"))
        ],
    }


def read_sysctl(key):
    result = subprocess.run(
        ["sysctl", "-n", key], capture_output=True, text=True, timeout=10, check=True
    )
    return result.stdout.strip()


def environment_metadata(bench):
    versions = {}
    for distribution in (
        "mlx-geospatial", "mlx", "mlx-metal", "numpy", "shapely", "pandas",
        "geopandas", "duckdb", "pyarrow", "scikit-learn", "scipy", "threadpoolctl",
    ):
        versions[distribution] = safe_value(lambda name=distribution: importlib.metadata.version(name))

    package_root = Path(bench.ms.__file__).resolve().parent
    hashes = {
        str(path.relative_to(package_root)): file_hash(path)
        for path in sorted(package_root.rglob("*.py"))
    }
    package_hash = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    def duckdb_settings():
        import duckdb

        con = duckdb.connect()
        try:
            return {
                key: con.execute(f"SELECT current_setting('{key}')").fetchone()[0]
                for key in ("threads", "memory_limit")
            }
        finally:
            con.close()

    environment_keys = (
        "MLX_GEOSPATIAL_BACKEND", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        "PYTHONHASHSEED", "PYTHONUNBUFFERED", "PYTHONOPTIMIZE", "PYTHONDEVMODE",
        "PYTHONMALLOC", "PYTHONPATH", "PYTHONFAULTHANDLER", "MallocStackLogging",
        "MTL_DEBUG_LAYER", "METAL_DEVICE_WRAPPER_TYPE", "METAL_DEBUG_ERROR_MODE",
        "MTL_SHADER_VALIDATION", "MLX_METAL_DEBUG",
    )
    debug_keys = [name for name in os.environ if name.startswith(("MLX_", "MTL_", "METAL_"))]
    return {
        "machine_description": safe_value(bench.machine),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "macos": list(platform.mac_ver()),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu_hardware": safe_value(gpu_hardware),
        "power": safe_value(power_metadata),
        "sysctl": {
            key: safe_value(lambda key=key: read_sysctl(key))
            for key in (
                "machdep.cpu.brand_string", "hw.memsize", "hw.physicalcpu", "hw.logicalcpu",
                "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu",
            )
        },
        "python_version": sys.version,
        "python_executable": sys.executable,
        "python_flags": repr(sys.flags),
        "python_xoptions": dict(sys._xoptions),
        "working_directory": str(Path.cwd()),
        "sys_path": list(sys.path),
        "versions": versions,
        "geos_version": getattr(bench.shapely, "geos_version_string", None),
        "package_import_path": str(Path(bench.ms.__file__).resolve()),
        "package_python_file_sha256": hashes,
        "package_python_manifest_sha256": package_hash,
        "bench_sha256": file_hash(Path(bench.__file__)),
        "runner_sha256": file_hash(Path(__file__)),
        "metal_available": safe_value(bench.metal_available),
        "mlx_default_device": safe_value(lambda: str(bench.mx.default_device())),
        "metal_device_info": safe_value(lambda: dict(bench.mx.device_info())),
        "duckdb_defaults": safe_value(duckdb_settings),
        "environment": {key: os.environ.get(key) for key in sorted(set(environment_keys) | set(debug_keys))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine-label", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "benchmark-results.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing benchmark: {result_path}")

    start = time.perf_counter()
    state = {
        "schema_version": 1,
        "machine_label": args.machine_label,
        "status": "initializing",
        "started_at": utc_now(),
        "invocation": ["bench.py", "--all", "--seed", "0"],
        "load_start": load_snapshot(),
        "metadata": {},
        "input_fingerprints": [],
        "timings": [],
        "reports": [],
    }

    def save():
        state["updated_at"] = utc_now()
        state["elapsed_seconds"] = time.perf_counter() - start
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(temporary, result_path)

    save()
    previous_argv = sys.argv[:]
    try:
        bench_path = Path(__file__).resolve().with_name("bench.py")
        spec = importlib.util.spec_from_file_location("instrumented_bench", bench_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import adjacent benchmark: {bench_path}")
        bench = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bench
        spec.loader.exec_module(bench)
        state["metadata"] = environment_metadata(bench)
        state["status"] = "running"
        save()

        context_names = (
            "n", "m", "k", "vertices", "coastline", "label", "all_matches", "kw",
            "edges", "deadline_s",
        )

        def frame_context(frame):
            while frame is not None:
                if frame.f_globals.get("__name__") == bench.__name__:
                    return {
                        "function": frame.f_code.co_name,
                        "line": frame.f_lineno,
                        **{
                            name: compact_value(frame.f_locals[name])
                            for name in context_names
                            if name in frame.f_locals and (
                                scalar(frame.f_locals[name]) or isinstance(frame.f_locals[name], dict)
                            )
                        },
                    }
                frame = frame.f_back
            return {}

        def timed(fn, *, warmup=1, repeats=3):
            frame = inspect.currentframe()
            try:
                context = frame_context(frame.f_back)
            finally:
                del frame
            closure = {}
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
                value = cell.cell_contents
                if scalar(value) or isinstance(value, dict):
                    closure[name] = compact_value(value)
                elif isinstance(value, bench.np.ndarray):
                    closure[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            record = {
                "sequence": len(state["timings"]),
                "callable_qualname": fn.__qualname__,
                "callable_first_line": fn.__code__.co_firstlineno,
                "closure": closure,
                "context": context,
                "warmup_count": warmup,
                "repeats_requested": repeats,
                "warmups_completed": 0,
                "durations_seconds": [],
                "started_elapsed_seconds": time.perf_counter() - start,
                "status": "running",
            }
            state["timings"].append(record)
            try:
                for _ in range(warmup):
                    fn()
                    record["warmups_completed"] += 1
                best = float("inf")
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    fn()
                    duration = time.perf_counter() - t0
                    best = min(best, duration)
                    record["durations_seconds"].append(duration)
                record["best_seconds"] = best
                record["status"] = "completed"
                return best
            except BaseException as exc:
                record["status"] = "failed"
                record["exception"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["ended_elapsed_seconds"] = time.perf_counter() - start

        bench.timeit = timed
        original_report = bench.report
        previous_report_end = 0

        def report(title, rows):
            nonlocal previous_report_end
            original_report(title, rows)
            state["reports"].append({
                "title": title,
                "rows": [{"name": name, "seconds": seconds} for name, seconds in rows],
                "timing_sequence_start": previous_report_end,
                "timing_sequence_end_exclusive": len(state["timings"]),
                "elapsed_seconds": time.perf_counter() - start,
            })
            previous_report_end = len(state["timings"])
            save()

        bench.report = report

        def fingerprint_generator(name):
            original = getattr(bench, name)
            signature = inspect.signature(original)

            @functools.wraps(original)
            def generated(*fn_args, **fn_kwargs):
                result = original(*fn_args, **fn_kwargs)
                bound = signature.bind(*fn_args, **fn_kwargs)
                bound.apply_defaults()
                digest = hashlib.sha256()
                digest.update(str(result.dtype).encode() + b"\0")
                digest.update(json.dumps(list(result.shape)).encode() + b"\0")
                if result.dtype == object:
                    kind = "geometry_wkb_little_endian"
                    for wkb in bench.shapely.to_wkb(result, byte_order=1, include_srid=True):
                        digest.update(len(wkb).to_bytes(8, "little"))
                        digest.update(wkb)
                else:
                    kind = "numpy_contiguous_bytes"
                    digest.update(memoryview(bench.np.ascontiguousarray(result)).cast("B"))
                state["input_fingerprints"].append({
                    "sequence": len(state["input_fingerprints"]),
                    "generator": name,
                    "arguments": {key: value for key, value in bound.arguments.items() if scalar(value)},
                    "kind": kind,
                    "shape": list(result.shape),
                    "dtype": str(result.dtype),
                    "sha256": digest.hexdigest(),
                })
                return result

            setattr(bench, name, generated)

        for name in ("make_points", "make_zones", "make_tiling", "densify_zones", "make_coastline"):
            fingerprint_generator(name)

        print(f"Instrumented run: {args.machine_label}; structured output: {result_path}", flush=True)
        sys.argv = [str(bench_path), "--all", "--seed", "0"]
        bench.main()
        state["status"] = "completed"
    except BaseException as exc:
        state["status"] = "failed"
        state["exception"] = {
            "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()
        }
        raise
    finally:
        sys.argv = previous_argv
        state["finished_at"] = utc_now()
        state["load_end"] = load_snapshot()
        save()


if __name__ == "__main__":
    main()
