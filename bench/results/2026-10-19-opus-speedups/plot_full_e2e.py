#!/usr/bin/env python3
"""Render verified six-workload, complete-output MLX / GeoPandas benchmarks.

Usage: python plot_full_e2e.py comparison.json [--output-dir PATH]
       python plot_full_e2e.py comparison.json --validate-only

Requires matplotlib. Expected JSON:
  machines: {m4: {name,cpu_cores,gpu_cores,...}, m1: optional matching host}
  joins: six {id,label,detail,n,m,vertices,output_rows,
              m1:{mlx_seconds,geopandas_seconds,speedup,mlx_samples,geopandas_samples},
              m4:{same}}
  methodology: {date,repetitions,output_equality_checked,exact_boundary,
                point_extraction_workers,...}
  summary: optional source summaries (figures recompute summaries from timings)

All timings are median complete-join durations. The renderer validates supplied
sample medians and ratios. It does not read partially written raw benchmark
files or reuse previous measurements. Outputs PNG/SVG pairs for every measured
host (800 x 1120) and full-e2e-overview (800 or 1600 x 1120 for one or two hosts).
An unavailable host is omitted, never filled with previous measurements.

All charts share one zero-based GeoPandas/MLX ratio scale, with a 1x reference.
The full-output contract is: fresh indexes, exact boundaries, all matching
pairs, complete point-left GeoDataFrames, attributes and canonical sorting;
ready input GeoDataFrame creation is excluded equally.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
import statistics
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

INK, MUTED = "#25282B", "#898D91"
CARD, TRACK = "#F4F4F4", "#EDEFEF"
BLUE, GOLD, PINK = "#2E6BD4", "#F4CB42", "#EB6FAB"
HOST_ORDER = ("m1", "m4")
COLORS = {"m1": BLUE, "m4": PINK}


def positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a finite, positive measured value")
    return float(value)


def measured_hosts(data: dict[str, Any]) -> tuple[str, ...]:
    machines = data.get("machines")
    if not isinstance(machines, dict) or "m4" not in machines or set(machines) - set(HOST_ORDER):
        raise ValueError("machines must contain m4, optionally with m1")
    return tuple(host for host in HOST_ORDER if host in machines)


def validate(data: dict[str, Any]) -> None:
    hosts = measured_hosts(data)
    rows = data.get("joins", [])
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("joins must contain exactly six measured workloads")
    method = data["methodology"]
    date.fromisoformat(method["date"])
    repeats = method["repetitions"]
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("methodology.repetitions must be a positive integer")
    if method.get("output_equality_checked") is not True:
        raise ValueError("methodology.output_equality_checked must confirm full-output validation")
    if method.get("exact_boundary") is not True:
        raise ValueError("methodology.exact_boundary must confirm exact predicate refinement")
    workers = method.get("point_extraction_workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("methodology.point_extraction_workers must be a positive integer")
    for host in hosts:
        info = data["machines"][host]
        if not isinstance(info.get("name"), str) or not info["name"]:
            raise ValueError(f"machines.{host}.name is required")
        for key in ("cpu_cores", "gpu_cores"):
            if not isinstance(info.get(key), int) or isinstance(info[key], bool) or info[key] <= 0:
                raise ValueError(f"machines.{host}.{key} must be a positive integer")
    for i, row in enumerate(rows):
        for key in ("label", "detail"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"joins[{i}].{key} is required")
        count = row.get("output_rows")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"joins[{i}].output_rows must be a nonnegative integer")
        for host in hosts:
            for name in ("mlx", "geopandas"):
                measured = positive(row[host][f"{name}_seconds"], f"joins[{i}].{host}.{name}_seconds")
                samples = row[host].get(f"{name}_samples")
                if not isinstance(samples, list) or len(samples) != repeats:
                    raise ValueError(f"joins[{i}].{host}.{name}_samples needs {repeats} trials")
                for value in samples:
                    positive(value, f"joins[{i}].{host}.{name}_samples")
                if not math.isclose(statistics.median(samples), measured, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"joins[{i}].{host}.{name}_seconds is not its sample median")
            expected = row[host]["geopandas_seconds"] / row[host]["mlx_seconds"]
            if "speedup" in row[host] and not math.isclose(row[host]["speedup"], expected, rel_tol=1e-9):
                raise ValueError(f"joins[{i}].{host}.speedup disagrees with its measured times")


def ratio(value: float) -> str:
    if value < 100:
        return f"{value:.2f}×"
    return f"{value:,.0f}×"


def duration(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 0.01:
        return f"{seconds * 1000:.2f} ms"
    if seconds < 0.1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s" if seconds < 10 else f"{seconds:.1f} s"


def nice_max(value: float) -> float:
    magnitude = 10 ** math.floor(math.log10(value))
    for multiple in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12):
        candidate = multiple * magnitude
        if candidate >= value * 1.045:
            return candidate
    return 20 * magnitude


def text(fig: Any, x: float, y: float, label: str, size: float = 12,
         color: str = INK, ha: str = "left", weight: str = "normal") -> None:
    fig.text(x, y, label, fontsize=size, color=color, ha=ha,
             va="center", fontweight=weight, linespacing=1.3)


def panel(fig: Any, data: dict[str, Any], host: str, left: float,
          width: float, maximum: float) -> None:
    def x(relative: float) -> float:
        return left + width * relative
    info = data["machines"][host]
    name = info["name"].removeprefix("Apple ")
    rows = data["joins"]
    ratios = [row[host]["geopandas_seconds"] / row[host]["mlx_seconds"] for row in rows]
    text(fig, left, 0.923, f"{name} · MLX vs GeoPandas", 16.5, weight="bold")
    text(fig, left, 0.895, f"{info['cpu_cores']} CPU cores · {info['gpu_cores']} GPU cores", 11.1, MUTED)
    gap, card_width = 0.020, (1 - 0.040) / 3
    cards = (
        ("Compared output", "Full join", "Attributes + geometry"),
        ("Observed time ratio", f"{ratio(min(ratios))[:-1]}–{ratio(max(ratios))}", "GeoPandas / MLX"),
        ("Median time ratio", ratio(statistics.median(ratios)), "Across six workloads"),
    )
    for i, (heading, value, detail) in enumerate(cards):
        card_x, card_y, height = x(i * (card_width + gap)), 0.754, 0.114
        fig.add_artist(FancyBboxPatch(
            (card_x, card_y), width * card_width, height,
            transform=fig.transFigure, boxstyle="round,pad=0,rounding_size=0.018",
            facecolor=CARD, edgecolor="none"))
        text(fig, card_x + width * 0.015, card_y + height * 0.78, heading, 10.8, MUTED)
        text(fig, card_x + width * 0.015, card_y + height * 0.49, value, 18.0)
        text(fig, card_x + width * 0.015, card_y + height * 0.20, detail, 10.1, MUTED)
    text(fig, left, 0.719, "GeoPandas / MLX · above 1×: MLX faster · below 1×: GeoPandas faster", 10.2, MUTED)

    bar_x, bar_width, time_x = x(0.58), width * 0.32, x(0.300)
    parity = bar_x + bar_width / maximum
    text(fig, parity, 0.679, "1×", 9.8, MUTED, ha="center")
    centers = []
    for i, (row, value) in enumerate(zip(rows, ratios)):
        top = 0.652 - i * 0.0826
        center = top - 0.017
        centers.append(center)
        text(fig, left, top, row["label"], 12.6)
        text(fig, left, top - 0.023, row["detail"], 10.6, MUTED)
        text(fig, left, top - 0.046, f"{row['output_rows']:,} rows", 10.2, MUTED)
        text(fig, time_x, top - 0.002, f"MLX  {duration(row[host]['mlx_seconds'])}", 11.2)
        text(fig, time_x, top - 0.032, f"GeoPandas  {duration(row[host]['geopandas_seconds'])}", 10.6, MUTED)
        for bar_length, fill in ((bar_width, TRACK), (bar_width * value / maximum, COLORS[host])):
            fig.add_artist(Rectangle(
                (bar_x, center - 0.005), bar_length, 0.010,
                transform=fig.transFigure, facecolor=fill, edgecolor="none"))
        text(fig, x(1), center, ratio(value), 11.8, ha="right")
    fig.add_artist(Line2D(
        [parity, parity], [min(centers) - 0.020, max(centers) + 0.022],
        transform=fig.transFigure, color="#747A80", linewidth=1.0,
        linestyle=(0, (3, 3)), zorder=4))
    for tick in (0, maximum / 2, maximum):
        text(fig, bar_x + bar_width * tick / maximum, 0.159, f"{tick:g}×", 10.2,
             MUTED, ha="left" if tick == 0 else "right" if tick == maximum else "center")


def footer(data: dict[str, Any]) -> list[str]:
    method = data["methodology"]
    hosts = measured_hosts(data)
    host_names = " and ".join(data["machines"][host]["name"].removeprefix("Apple ") for host in hosts)
    os_note = "; macOS versions differ by host" if len({data["machines"][host].get("macos") for host in hosts}) > 1 else ""
    return [
        f"Optimized build · median of {method['repetitions']} trials · fresh indexes · exact_boundary=True",
        f"MLX point extraction: up to {method['point_extraction_workers']} workers · GeoPandas: native join, no partitioning.",
        "All pairs; complete point-left GeoDataFrames, attributes and canonical sorting.",
        "Ready input creation excluded equally; complete output materialization included.",
        f"Full output equality checked for every workload on {host_names}.",
        f"Measured {method['date']} · Shared zero-based ratio scale{os_note}.",
    ]


def save(fig: Any, outdir: Path, stem: str) -> None:
    for extension in ("png", "svg"):
        path = outdir / f"{stem}.{extension}"
        fig.savefig(path, dpi=100, facecolor="white", edgecolor="none")
        if path.suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(path)
    plt.close(fig)


def render(data: dict[str, Any], outdir: Path) -> None:
    hosts = measured_hosts(data)
    maximum = nice_max(max(1, *(row[host]["geopandas_seconds"] / row[host]["mlx_seconds"]
                               for row in data["joins"] for host in hosts)))
    for host in hosts:
        fig = plt.figure(figsize=(8, 11.2), facecolor="white")
        text(fig, 0.04, 0.964, "Complete spatial joins", 21, weight="bold")
        panel(fig, data, host, 0.04, 0.92, maximum)
        for i, line in enumerate(footer(data)):
            text(fig, 0.04, 0.123 - i * 0.0195, line, 9.1, MUTED)
        save(fig, outdir, f"full-e2e-{host}")
    fig = plt.figure(figsize=(8 * len(hosts), 11.2), facecolor="white")
    title = "Complete spatial joins: MLX vs GeoPandas" if len(hosts) == 2 else "Complete spatial joins"
    margin, panel_width = (0.03, 0.455) if len(hosts) == 2 else (0.04, 0.92)
    text(fig, margin, 0.964, title, 22 if len(hosts) == 2 else 21, weight="bold")
    for host, left in zip(hosts, (margin, 0.515)):
        panel(fig, data, host, left, panel_width, maximum)
    for i, line in enumerate(footer(data)):
        text(fig, margin, 0.123 - i * 0.0195, line, 10.1 if len(hosts) == 2 else 9.1, MUTED)
    save(fig, outdir, "full-e2e-overview")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    validate(data)
    if args.validate_only:
        print("Schema, measured sample medians, and supplied ratios validated; no figures rendered.")
        return
    outdir = args.output_dir or args.input.parent
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "svg.fonttype": "none", "savefig.bbox": None})
    render(data, outdir)


if __name__ == "__main__":
    main()
