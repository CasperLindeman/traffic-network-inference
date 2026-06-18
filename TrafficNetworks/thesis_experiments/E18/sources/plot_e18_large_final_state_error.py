#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

import plot_e18_large_pruned_sections as sections
import plot_e18_large_smoke_snapshots as smoke_plot


HERE = Path(__file__).resolve().parent
GRAPH_OUTPUT_DIR = HERE.parent / "outputs" / "graph"
DEFAULT_ERROR_CSV = HERE / "inference_outputs_128x4_180s" / "final_state_error" / "final_state_cell_abs_error.csv"
DEFAULT_OUTPUT_DIR = HERE / "inference_outputs_128x4_180s" / "final_state_error"

smoke_plot.OUTDIR = GRAPH_OUTPUT_DIR
sections.OUTDIR = GRAPH_OUTPUT_DIR
sections.SIM_ROADS_CSV = GRAPH_OUTPUT_DIR / "e18_large_pruned_sim_roads.csv"
sections.SIM_NODES_CSV = GRAPH_OUTPUT_DIR / "e18_large_pruned_sim_nodes.csv"

ERROR_CMAP = LinearSegmentedColormap.from_list(
    "white_red_error",
    ["#ffffff", "#fff0e6", "#f7b799", "#dc5c4d", "#8b0000"],
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def load_errors(path: Path) -> dict[int, list[float]]:
    errors: dict[int, list[float]] = {}
    for row in read_csv(path):
        road_id = int(row["road_id"])
        errors.setdefault(road_id, []).append(float(row["abs_error"]))
    return errors


def draw_error_roads(ax, road_geoms, errors_by_road, extent, norm):
    base_segments = []
    error_segments = []
    error_values = []

    for road_id, geometry in sorted(road_geoms.items()):
        coords = geometry["coords"]
        if not sections.geometry_intersects_extent(coords, extent, margin=30.0):
            continue
        values = errors_by_road.get(road_id)
        if not values:
            base_segments.append(coords)
            continue
        cell_segs = smoke_plot.cell_segments(coords, len(values))
        if not cell_segs:
            base_segments.append(coords)
            continue
        error_segments.extend(cell_segs)
        error_values.extend(values)

    if base_segments:
        ax.add_collection(
            LineCollection(
                base_segments,
                colors="#d7dce3",
                linewidths=1.0,
                alpha=0.45,
                zorder=1,
            )
        )

    if error_segments:
        ax.add_collection(
            LineCollection(
                error_segments,
                cmap=ERROR_CMAP,
                norm=norm,
                array=error_values,
                linewidths=4.2,
                alpha=0.96,
                zorder=4,
                capstyle="round",
            )
        )


def draw_plain_nodes(ax, nodes, extent):
    sections.draw_nodes(
        ax,
        nodes,
        extent,
        labeled_node_labels=[],
        quiet=True,
    )


def plot_window(road_geoms, nodes, errors_by_road, window, output_path, norm):
    if "center_label" in window:
        extent = sections.node_extent(nodes, window["center_label"], window["radius_m"])
    else:
        extent = sections.labels_extent(nodes, window["labels"], window["pad_m"])

    fig, ax = plt.subplots(figsize=window["figsize"])
    draw_error_roads(ax, road_geoms, errors_by_road, extent, norm)
    draw_plain_nodes(ax, nodes, extent)
    sections.draw_target_nodes(ax, nodes, window["targets"], show_labels=False)
    sections.draw_sensor_locations(ax, road_geoms, window["sensors"], show_labels=False)

    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=ERROR_CMAP), ax=ax, shrink=0.72, pad=0.01)
    colorbar.set_label("Absolute density error", fontsize=19)
    colorbar.ax.tick_params(labelsize=16)

    handles = [
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#f03b8d",
               markeredgewidth=2.2, color="none", markersize=13.0, label="Inferred roundabout"),
        Line2D([0], [0], marker="o", markerfacecolor="#f03b8d", markeredgecolor="#222222",
               color="none", markersize=10.0, label="Inferred junction"),
        Line2D([0], [0], marker="o", markerfacecolor="#0072b2", markeredgecolor="white",
               color="none", markersize=10.5, label="Sensor"),
    ]
    legend = ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=16)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.96)

    x_min, x_max, y_min, y_max = extent
    pad = 30.0
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-csv", type=Path, default=DEFAULT_ERROR_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors_by_road = load_errors(args.error_csv)
    all_errors = [value for values in errors_by_road.values() for value in values]
    vmax = args.vmax if args.vmax is not None else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)

    arcs = smoke_plot.arc_geometries()
    road_geoms = smoke_plot.road_geometries(arcs, sections.SIM_ROADS_CSV)
    sections.enrich_road_metadata(road_geoms)
    nodes = smoke_plot.active_sim_nodes(sections.SIM_NODES_CSV, road_geoms)

    for window in sections.SELECTION_WINDOWS:
        suffix = window["name"].removeprefix("target_area_")
        output_name = f"e18_final_state_abs_error_{suffix}.png"
        plot_window(road_geoms, nodes, errors_by_road, window, args.output_dir / output_name, norm)


if __name__ == "__main__":
    main()
