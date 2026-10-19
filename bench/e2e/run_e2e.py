#!/usr/bin/env python3
"""Matched, fresh-index, exact-predicate spatial joins producing GeoDataFrames.

Requires adjacent, unmodified bench.py and run_benchmarks.py from the preceding
benchmark. Run from an environment containing the same installed distribution.
No input construction, validation, metadata collection, or disk I/O is timed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import geopandas as gpd
from geopandas.array import GeometryArray
from geopandas.testing import assert_geodataframe_equal
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import shapely

import bench
import run_benchmarks as evidence


CASES = [
    {"id": 0, "label": "100k × 260", "n": 100_000, "m": 260, "vertices": 48},
    {"id": 1, "label": "1M × 260", "n": 1_000_000, "m": 260, "vertices": 48},
    {"id": 2, "label": "10M × 260", "n": 10_000_000, "m": 260, "vertices": 48},
    {"id": 3, "label": "1M × 10k", "n": 1_000_000, "m": 10_000, "vertices": 48},
    {"id": 4, "label": "1M × 260", "n": 1_000_000, "m": 260, "vertices": 2048},
    {"id": 5, "label": "1M × 1", "n": 1_000_000, "m": 1, "vertices": 150_000,
     "coastline": 150_000},
]

PROTOCOL = {
    "predicate": "within", "how": "inner", "matching": "all matching pairs",
    "crs": "EPSG:4326", "seed": 0,
    "headline_statistic": "median of three paired alternating trials",
    "warmups_per_method_per_case": 1, "trials_per_method": 3,
    "input_start": "Already constructed matching GeoDataFrames, without cached indexes or prepared polygons",
    "included": ["MLX PolygonSet construction and GPU ingestion", "all spatial index construction",
                 "point-coordinate extraction for MLX", "GPU query and host result materialization",
                 "exact-boundary GEOS refinement", "complete joined GeoDataFrame materialization",
                 "all left and right nongeometry attributes", "canonical row sorting"],
    "excluded": ["input GeoDataFrame construction and fresh copies", "imports/runtime initialization",
                 "correctness validation", "metadata collection", "disk I/O"],
    "geopandas": "geopandas.sjoin(left, right, how='inner', predicate='within') with native default indexing",
    "mlx": "Fresh PolygonSet(right), GridIndex; query(left.geometry, all_matches=True, predicate='within', exact_boundary=True, backend='gpu'); attribute assembly",
    "mlx_native_sjoin_used": False,
    "sorting": "Both outputs sorted by point_id, then index_right inside each timer; original left index preserved",
    "input_copy": "Independent GeometryArrays and attribute arrays; immutable Shapely geometry objects may be shared, but polygon prepared state is destroyed before every method",
    "validation": "Every warmup pair and trial pair: GeoDataFrame equality, exact attrs/index/columns/dtypes/CRS, zero-tolerance geometry coordinate equality",
    "cpu_threads": "GeoPandas/GEOS native spatial join; no parallel partitioning",
    "scope": "Six synthetic point-in-polygon workloads; valid polygons, unique unnamed indexes, noncolliding attribute names",
}


def canonical(frame):
    return frame.sort_values(["point_id", "index_right"], kind="mergesort")


def geopandas_join(left, right):
    return canonical(gpd.sjoin(left, right, how="inner", predicate="within"))


def mlx_join(left, right):
    polygon_set = bench.ms.PolygonSet(right)
    spatial_index = bench.ms.GridIndex(polygon_set)
    point_pos, polygon_pos = spatial_index.query(
        left.geometry, backend="gpu", all_matches=True,
        predicate="within", exact_boundary=True,
    )
    point_pos = np.asarray(point_pos, dtype=np.int64)
    polygon_pos = np.asarray(polygon_pos, dtype=np.int64)
    result = left.iloc[point_pos].copy()
    result["index_right"] = right.index.to_numpy()[polygon_pos]
    for column in right.columns:
        if column != right.geometry.name:
            result[column] = right[column].array.take(polygon_pos)
    if len(result) == 0:
        # GeoPandas 1.2 uses an empty RangeIndex for an empty inner join.
        result.index = pd.RangeIndex(0, name=left.index.name)
    result = canonical(result)
    bench.mx.synchronize()
    return result


METHODS = {"mlx": mlx_join, "geopandas": geopandas_join}


def frames_from_geometry(points, zones):
    n, m = len(points), len(zones)
    point_ids = np.arange(n, dtype=np.int64)
    zone_ids = np.arange(m, dtype=np.int64)
    # Non-positional index labels catch accidental row-position/index confusion.
    left = gpd.GeoDataFrame(
        {"point_id": point_ids, "value": (point_ids % 1009).astype(np.float64) / 1009.0},
        geometry=GeometryArray(np.asarray(points, dtype=object), crs="EPSG:4326"),
        index=pd.Index(point_ids * 2 + 101, name=None), crs="EPSG:4326",
    )
    right = gpd.GeoDataFrame(
        {"zone_id": zone_ids,
         "zone_name": pd.array([f"zone_{i:05d}" for i in zone_ids], dtype="string"),
         "zone_weight": zone_ids.astype(np.float64) * 0.125 + 0.5},
        geometry=GeometryArray(np.asarray(zones, dtype=object), crs="EPSG:4326"),
        index=pd.Index(zone_ids * 3 + 1001, name=None), crs="EPSG:4326",
    )
    return left, right


def fresh_frame(source):
    attrs = {column: source[column].array.copy() for column in source.columns
             if column != source.geometry.name}
    geometry = GeometryArray(np.array(source.geometry.array, dtype=object, copy=True), crs=source.crs)
    result = gpd.GeoDataFrame(attrs, geometry=geometry, index=source.index.copy(), crs=source.crs)
    assert result.geometry.array is not source.geometry.array
    assert not result.has_sindex
    return result


def fresh_inputs(left, right):
    fresh_left, fresh_right = fresh_frame(left), fresh_frame(right)
    shapely.destroy_prepared(fresh_right.geometry.array)
    assert not shapely.is_prepared(fresh_right.geometry.array).any()
    assert not fresh_left.has_sindex and not fresh_right.has_sindex
    return fresh_left, fresh_right


def validate_outputs(mlx_result, geopandas_result):
    assert_geodataframe_equal(
        mlx_result, geopandas_result, check_dtype=True,
        check_index_type=True, check_column_type=True, check_frame_type=True,
        check_like=False, check_less_precise=False, check_geom_type=True,
        check_crs=True, normalize=False,
    )
    assert_index_equal(mlx_result.index, geopandas_result.index, exact=True)
    assert_index_equal(mlx_result.columns, geopandas_result.columns, exact=True)
    assert_frame_equal(
        pd.DataFrame(mlx_result.drop(columns=mlx_result.geometry.name)),
        pd.DataFrame(geopandas_result.drop(columns=geopandas_result.geometry.name)),
        check_dtype=True, check_exact=True, check_names=True,
    )
    assert np.all(shapely.equals_exact(mlx_result.geometry.array, geopandas_result.geometry.array, 0.0))
    assert mlx_result.geometry.name == geopandas_result.geometry.name
    assert mlx_result.crs == geopandas_result.crs
    point_ids = mlx_result["point_id"].to_numpy(dtype=np.int64)
    starts = np.r_[True, point_ids[1:] != point_ids[:-1]] if len(point_ids) else np.array([], dtype=bool)
    matched_points = int(np.count_nonzero(starts))
    multi_hit_points = int(np.count_nonzero(starts[:-1] & (point_ids[:-1] == point_ids[1:])))
    return {
        "status": "passed", "rows": len(mlx_result),
        "matched_unique_points": matched_points, "multi_hit_points": multi_hit_points,
        "columns": list(mlx_result.columns),
        "dtypes": {column: str(dtype) for column, dtype in mlx_result.dtypes.items()},
        "index_dtype": str(mlx_result.index.dtype), "index_name": mlx_result.index.name,
        "crs": str(mlx_result.crs),
    }


def pair_digest(result):
    digest = hashlib.sha256()
    for column in ("point_id", "index_right"):
        digest.update(column.encode() + b"\0")
        array = result[column].to_numpy(dtype=np.int64)
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def run_pair(left, right, order, *, case_id, phase, trial, save):
    outputs, samples = {}, {}
    for method in order:
        fresh_left, fresh_right = fresh_inputs(left, right)
        gc.collect()
        bench.mx.synchronize()
        print(f"case={case_id} phase={phase} trial={trial} method={method} start", flush=True)
        t0 = time.perf_counter()
        outputs[method] = METHODS[method](fresh_left, fresh_right)
        duration = time.perf_counter() - t0
        samples[method] = duration
        print(f"case={case_id} phase={phase} trial={trial} method={method} seconds={duration:.6f} rows={len(outputs[method]):,}", flush=True)
        # Release the point STRtree before the next method. Outputs retain only
        # the matching geometry references and materialized attributes.
        del fresh_left, fresh_right
        gc.collect()
    validation = validate_outputs(outputs["mlx"], outputs["geopandas"])
    validation["input_points"] = len(left)
    validation["unmatched_points"] = len(left) - validation["matched_unique_points"]
    validation["pair_sha256"] = pair_digest(outputs["mlx"])
    print(f"case={case_id} phase={phase} trial={trial} equality=PASS rows={validation['rows']:,}", flush=True)
    del outputs
    gc.collect()
    return {"phase": phase, "trial": trial, "order": order,
            "seconds": samples, "validation": validation}


def smoke_fixture():
    shell = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
    hole = [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]
    zones = np.array([shapely.Polygon(shell, [hole]), shapely.box(3, 0, 5, 3)], dtype=object)
    xy = np.array([
        [0.5, 0.5], [3.5, 0.5], [1.5, 1.5], [6, 6],
        [0, 0], [4, 4], [0, 2], [4, 2], [1, 1], [1.5, 1],
        [3, 1.5], [5, 1.5], [4, 3], [1e-10, 0.5], [-1e-10, 0.5],
        [1 + 1e-10, 1.5], [1 - 1e-10, 1.5], [2, 2], [2.5, 2.5],
    ], dtype=np.float64)
    left, right = frames_from_geometry(shapely.points(xy), zones)
    a, b = fresh_inputs(left, right)
    ml = mlx_join(a, b)
    a, b = fresh_inputs(left, right)
    gp = geopandas_join(a, b)
    result = validate_outputs(ml, gp)
    result["pair_sha256"] = pair_digest(ml)
    result["fixture"] = "Overlaps, hole, outside, exact edges/vertices, points ±1e-10 from edges"
    assert len(ml[ml.point_id == 1]) == 2
    assert not any(ml.point_id.isin([2, 3, 4, 5, 6, 8, 9, 11, 14, 15, 17]))
    empty_left, _ = frames_from_geometry(shapely.points([[99.0, 99.0]]), zones)
    a, b = fresh_inputs(empty_left, right)
    ml_empty = mlx_join(a, b)
    a, b = fresh_inputs(empty_left, right)
    gp_empty = geopandas_join(a, b)
    result["empty_output"] = validate_outputs(ml_empty, gp_empty)
    return result


class Replay:
    """Reproduce the original --all generator sequence and digest format."""
    def __init__(self, state, reference):
        self.rng = np.random.default_rng(0)
        self.state = state
        self.reference = reference

    def generate(self, name, *args, **kwargs):
        fn = getattr(bench, name)
        result = fn(*args, **kwargs)
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        digest = hashlib.sha256()
        digest.update(str(result.dtype).encode() + b"\0")
        digest.update(json.dumps(list(result.shape)).encode() + b"\0")
        if result.dtype == object:
            kind = "geometry_wkb_little_endian"
            for wkb in shapely.to_wkb(result, byte_order=1, include_srid=True):
                digest.update(len(wkb).to_bytes(8, "little"))
                digest.update(wkb)
        else:
            kind = "numpy_contiguous_bytes"
            digest.update(memoryview(np.ascontiguousarray(result)).cast("B"))
        sequence = len(self.state["input_fingerprints"])
        record = {
            "sequence": sequence, "generator": name,
            "arguments": {key: value for key, value in bound.arguments.items() if evidence.scalar(value)},
            "kind": kind, "shape": list(result.shape), "dtype": str(result.dtype),
            "sha256": digest.hexdigest(),
        }
        if self.reference is not None:
            expected = self.reference[sequence]
            assert record == expected, f"Input fingerprint mismatch at sequence {sequence}: {record} != {expected}"
        self.state["input_fingerprints"].append(record)
        return result

    def prefix(self):
        for n in (20_000, 20_000, 1_000, 1_000_000, 1_000_000, 10_000,
                  200_000, 200_000, 1_000_000, 200_000, 100_000):
            self.generate("make_points", n, self.rng)
        self.generate("make_zones", 32, self.rng, vertices=48)

    def case(self, case):
        points = self.generate("make_points", case["n"], self.rng)
        if case.get("coastline"):
            zones = self.generate("make_coastline", case["coastline"], self.rng)
        else:
            zones = self.generate("make_zones", case["m"], self.rng)
            if case["vertices"] != 48:
                zones = self.generate("densify_zones", zones, case["vertices"])
        return points, zones


def timing_summary(samples):
    return {"samples_seconds": samples, "median_seconds": statistics.median(samples),
            "minimum_seconds": min(samples), "maximum_seconds": max(samples),
            "range_seconds": max(samples) - min(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--machine-label", required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--cases", nargs="+", type=int, choices=range(6), default=list(range(6)))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "e2e-results.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    started = time.perf_counter()
    state = {
        "schema_version": 1, "machine_label": args.machine_label,
        "started_at": evidence.utc_now(), "status": "initializing",
        "invocation": sys.argv, "protocol": PROTOCOL, "cases": [],
        "input_fingerprints": [], "load_start": evidence.load_snapshot(),
    }
    def save():
        state["updated_at"] = evidence.utc_now()
        state["elapsed_seconds"] = time.perf_counter() - started
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(temp, path)
    save()
    try:
        state["metadata"] = evidence.environment_metadata(bench)
        state["metadata"]["e2e_runner_sha256"] = evidence.file_hash(__file__)
        print(f"Running exact-boundary correctness fixture: {args.machine_label}", flush=True)
        state["correctness_fixture"] = smoke_fixture()
        print("Correctness fixture PASS", flush=True)
        save()
        if args.smoke_only:
            state["status"] = "completed"
            return
        reference = None
        if args.reference_json:
            reference = json.loads(args.reference_json.read_text())["input_fingerprints"]
            state["reference_json_sha256"] = evidence.file_hash(args.reference_json)
        replay = Replay(state, reference)
        replay.prefix()
        state["status"] = "running"
        for definition in CASES:
            fingerprints_start = len(state["input_fingerprints"])
            points, zones = replay.case(definition)
            if definition["id"] not in args.cases:
                del points, zones
                continue
            case = dict(definition, status="constructing_inputs", pairs=[])
            case["input_fingerprints"] = state["input_fingerprints"][fingerprints_start:]
            case["load_start"] = evidence.load_snapshot()
            state["cases"].append(case)
            save()
            assert shapely.is_valid(zones).all(), "Invalid input geometry"
            left, right = frames_from_geometry(shapely.points(points), zones)
            del points, zones
            gc.collect()
            case["status"] = "running"
            for round_number in range(4):
                phase = "warmup" if round_number == 0 else "measured"
                # First measured method alternates across cases, then alternates
                # between successive trials. Warmup also gets fresh inputs.
                order = ["mlx", "geopandas"] if (definition["id"] + round_number) % 2 == 0 else ["geopandas", "mlx"]
                pair = run_pair(left, right, order, case_id=definition["id"],
                                phase=phase, trial=round_number, save=save)
                case["pairs"].append(pair)
                save()
            measured = [p for p in case["pairs"] if p["phase"] == "measured"]
            case["timings"] = {
                method: timing_summary([p["seconds"][method] for p in measured])
                for method in METHODS
            }
            digests = {p["validation"]["pair_sha256"] for p in case["pairs"]}
            assert len(digests) == 1, "Output pair set changed across trials"
            case["output_pair_sha256"] = digests.pop()
            case["output_rows"] = measured[0]["validation"]["rows"]
            case["input_points"] = len(left)
            for metric in ("matched_unique_points", "multi_hit_points", "unmatched_points"):
                case[metric] = measured[0]["validation"][metric]
            case["speedup_geopandas_over_mlx"] = (
                case["timings"]["geopandas"]["median_seconds"] / case["timings"]["mlx"]["median_seconds"]
            )
            case["status"] = "completed"
            del left, right
            gc.collect()
            case["load_end"] = evidence.load_snapshot()
            save()
            print(f"case={definition['id']} complete median_speedup={case['speedup_geopandas_over_mlx']:.4f}x", flush=True)
        state["reference_fingerprints_verified"] = len(state["input_fingerprints"]) if reference is not None else None
        state["status"] = "completed"
    except BaseException as exc:
        state["status"] = "failed"
        state["exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        state["finished_at"] = evidence.utc_now()
        state["load_end"] = evidence.load_snapshot()
        save()


if __name__ == "__main__":
    main()
