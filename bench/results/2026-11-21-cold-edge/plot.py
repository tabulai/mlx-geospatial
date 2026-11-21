"""Render the verified exact-join, warm-pair and cold-routing charts."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from math import ceil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from summarize import HOSTS, ROOT, read

INK = "#25282B"
MUTED = "#545D65"
PAIRS = {"shapely": "#2E6BD4", "geopandas": "#AD497E", "duckdb": "#966600"}
BEFORE, TUNED = "#A34B28", "#235FBF"


def complete_joins(exact):
    # Reuse the previous release's full-join panels and common ratio scale.
    path = ROOT.parent / "2026-10-19-opus-speedups" / "plot_full_e2e.py"
    spec = spec_from_file_location("exact_renderer", path)
    renderer = module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.validate(exact)
    maximum = renderer.nice_max(max(case[h]["speedup"] for h in HOSTS for case in exact["joins"]))
    notes = [
        "Median of three paired trials after one full warmup.",
        "Fresh indexes, exact boundaries, complete GeoDataFrames and canonical sorting.",
        "MLX uses the GPU and up to eight Point extraction workers. GeoPandas uses its native join.",
        "Both methods start from ready GeoDataFrames. All 48 full-output checks passed.",
        f"Measured {exact['methodology']['date']} on the same six synthetic workloads on both Macs.",
    ]
    for hosts, stem in ((("m1",), "full-e2e-m1"), (("m4",), "full-e2e-m4"),
                        (tuple(HOSTS), "full-e2e-overview")):
        side_by_side = len(hosts) == 2
        fig = plt.figure(figsize=(8 * len(hosts), 11.2), facecolor="white")
        margin, width = (0.03, 0.455) if side_by_side else (0.04, 0.92)
        renderer.text(fig, margin, 0.964, "Complete spatial joins", 22, weight="bold")
        for host, left in zip(hosts, (margin, 0.515)):
            renderer.panel(fig, exact, host, left, width, maximum)
        for i, line in enumerate(notes):
            renderer.text(fig, margin, 0.124 - i * 0.0195, line, 9.0, MUTED)
        renderer.save(fig, ROOT, stem)


def save(fig, name):
    for ext in ("png", "svg"):
        path = ROOT / f"{name}.{ext}"
        fig.savefig(path, dpi=140, facecolor="white")
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def warm_pairs(exact, pairs):
    labels = [f"{row['label']}\n{row['detail'].split(' · ')[0]}" for row in exact["joins"]]
    maximum = max(r[key] / r["mlx"] for rows in pairs.values() for r in rows for key in PAIRS)
    limit = ceil((maximum + 100) / 500) * 500
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.4), sharex=True, facecolor="white")
    fig.subplots_adjust(left=0.105, right=0.973, bottom=0.174, top=0.79, wspace=0.36)
    fig.text(0.06, 0.947, "Warm-index point-in-polygon", fontsize=23, weight="bold", color=INK)
    fig.text(0.06, 0.897, "CPU baseline time ÷ MLX pair-array time  ·  Minimum wall times",
             fontsize=12, color=MUTED)
    for ax, (host, name) in zip(axes, HOSTS.items(), strict=True):
        ax.set_title(name, loc="left", weight="bold", fontsize=16, pad=12)
        for j, (key, color) in enumerate(PAIRS.items()):
            y = [i + (j - 1) * 0.235 for i in range(len(labels))]
            values = [r[key] / r["mlx"] for r in pairs[host]]
            ax.barh(y, values, height=0.185, color=color, label=key.capitalize(), zorder=3)
            for yloc, value in zip(y, values, strict=True):
                ax.text(value + limit * 0.012, yloc, f"{value:,.0f}×", va="center",
                        fontsize=8.8, color=INK)
        ax.set_yticks(range(len(labels)), labels, fontsize=10)
        ax.set_ylim(len(labels) - 0.55, -0.55)
        ax.set_xlim(0, limit)
        ax.set_xticks(range(0, limit + 1, 500))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}×"))
        ax.tick_params(axis="both", length=0, pad=7, labelcolor=INK)
        ax.grid(axis="x", color="#E5E8EB", linewidth=0.8)
        ax.spines[:].set_visible(False)
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["Shapely STRtree", "GeoPandas sjoin", "DuckDB count"],
               loc="lower center", bbox_to_anchor=(0.5, 0.084), frameon=False, ncol=3)
    fig.text(0.06, 0.049, "Shapely and GeoPandas construct point geometry and a point index. "
             "DuckDB counts matches over registered inputs.", fontsize=10, color=MUTED)
    fig.text(0.06, 0.025, "MLX uses a warm polygon index with exact_boundary=False. "
             "Measured November 21, 2026.", fontsize=10, color=MUTED)
    save(fig, "pip-m1-m4-side-by-side")


def cold_routing(comparison):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), sharey=True, facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.955, bottom=0.206, top=0.744, wspace=0.15)
    fig.text(0.057, 0.928, "Cold first-match point-in-polygon", fontsize=22, weight="bold", color=INK)
    fig.text(0.057, 0.854, "260 zones with 2,048 vertices per polygon  ·  Median query latency",
             fontsize=12, color=MUTED)
    styles = [("before", "Before cutoff tuning", BEFORE, "s"), ("tuned", "Tuned cutoff", TUNED, "o")]
    for ax, (host, name) in zip(axes, HOSTS.items(), strict=True):
        rows = comparison[host]["public_pip"]
        points = [r["points"] for r in rows]
        assert points[-1] == 65_536
        ax.set_title(name, loc="left", fontsize=15, weight="bold", pad=11)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(860, 105_000)
        ax.set_ylim(0.3, 55)
        ax.set_xticks([1_024, 4_096, 16_384, 65_536], ["1,024", "4,096", "16,384", "65,536"])
        ax.set_yticks([0.5, 1, 2, 5, 10, 20, 40], ["0.5", "1", "2", "5", "10", "20", "40"])
        ax.minorticks_off()
        ax.grid(axis="y", color="#E5E8EB", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#C8CFD5")
        for key, label, color, marker in styles:
            times = [r[f"{key}_cold_ms"] for r in rows]
            ax.plot(points, times, color=color, marker=marker, label=label, linewidth=1.9, markersize=4.5)
            ax.annotate(f"{times[-1]:.2f} ms", (points[-1], times[-1]), xytext=(6, 0),
                        textcoords="offset points", ha="left", va="center", color=color,
                        fontsize=9)
        ax.set_xlabel("Query points", color=INK)
    axes[0].set_ylabel("Milliseconds", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.058),
               ncol=2, frameon=False)
    fig.text(0.058, 0.036, "Each trial starts with a cold edge cache. The PolygonSet and grid already exist.",
             fontsize=10, color=MUTED)
    save(fig, "cold-edge-routing")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "mlx-geospatial-2026-11-21"})
    exact = read(ROOT / "e2e-comparison.json")
    complete_joins(exact)
    warm_pairs(exact, read(ROOT / "warm-pip-comparison.json"))
    cold_routing(read(ROOT / "comparison.json"))


if __name__ == "__main__":
    main()
