from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


OUTDIR = Path(__file__).resolve().parent
OUTPUT_DIR = OUTDIR / "smoke_outputs"
DEFAULT_CELL_SNAPSHOTS_CSV = OUTPUT_DIR / "e18_large_smoke_cell_snapshots.csv"
DEFAULT_SIM_ROADS_CSV = OUTDIR / "e18_large_sim_roads.csv"
DEFAULT_SIM_NODES_CSV = OUTDIR / "e18_large_sim_nodes.csv"
DEFAULT_FRAMES_DIR = OUTPUT_DIR / "density_snapshot_frames"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


graph = load_module("build_e18_large_graph", OUTDIR / "build_e18_large_graph.py")
sim = load_module("build_e18_large_simulation_network", OUTDIR / "build_e18_large_simulation_network.py")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def base_node_id(node_id: str) -> str:
    return node_id.split(":", 1)[0]


def final_graph_objects() -> tuple[list[dict], list[dict], list[dict], set[str]]:
    objects, _ = graph.load_api_objects()
    simplified_objects, roundabouts = graph.apply_roundabout_simplification(objects)
    roundabout_nodes = {component["node_id"] for component in roundabouts}
    final_objects, final_roundabout_nodes, _ = graph.contract_short_connectors(
        simplified_objects, roundabout_nodes
    )
    final_objects, _ = graph.drop_model_roads(final_objects, graph.DROP_ROAD_IDS)
    final_objects, _ = graph.add_artificial_component_connectors(
        final_objects,
        final_roundabout_nodes,
        graph.FORCE_BOUNDARY_NODE_IDS,
    )
    junctions, boundaries, marker_nodes = graph.classify_nodes(
        final_objects,
        forced_junction_nodes=final_roundabout_nodes,
        forced_boundary_nodes=graph.FORCE_BOUNDARY_NODE_IDS,
    )
    return final_objects, junctions, boundaries, marker_nodes


def arc_geometries() -> dict[int, dict]:
    final_objects, _, _, _ = final_graph_objects()
    object_by_key = {obj["object_key"]: obj for obj in final_objects}
    graph_edges = read_csv(OUTDIR / "e18_large_graph_edges.csv")

    arcs = {}
    arc_id = 1
    for row in graph_edges:
        obj = object_by_key[row["object_key"]]
        for direction in sim.travel_directions(row):
            coords, _, _ = graph.directed_geometry(obj, direction)
            coords = graph.trim_polyline(coords, graph.NODE_TRIM_M, graph.NODE_TRIM_M)
            if row["lane_mode"].startswith("bidirectional"):
                coords = graph.offset_polyline(coords, graph.BIDIRECTIONAL_OFFSET_M)
            arcs[arc_id] = {
                "coords": coords,
                "road_type": row["typeVeg"],
            }
            arc_id += 1
    return arcs


def append_coords(base: list[tuple[float, float]], extra: list[tuple[float, float]]) -> None:
    if not extra:
        return
    if not base:
        base.extend(extra)
        return
    if math.dist(base[-1], extra[0]) < 1e-6:
        base.extend(extra[1:])
    else:
        base.extend(extra)


def road_geometries(arcs: dict[int, dict], sim_roads_csv: Path) -> dict[int, dict]:
    geometries = {}
    for row in read_csv(sim_roads_csv):
        road_id = int(row["id"])
        arc_ids = [int(value) for value in row["path_arc_ids"].split() if value]
        coords: list[tuple[float, float]] = []
        road_types = []
        for arc_id in arc_ids:
            arc = arcs.get(arc_id)
            if arc is None:
                continue
            append_coords(coords, arc["coords"])
            road_types.append(arc["road_type"])
        if len(coords) >= 2:
            geometries[road_id] = {
                "coords": coords,
                "road_type": row["road_type"] if not road_types else road_types[0],
                "start_node": row["start_node"],
                "end_node": row["end_node"],
            }
    return geometries


def graph_node_coordinates() -> dict[str, tuple[float, float]]:
    coords = {}
    for row in read_csv(OUTDIR / "e18_large_graph_nodes.csv"):
        coords[row["node_id"]] = (float(row["x"]), float(row["y"]))
    return coords


def road_endpoint_coordinates(road_geoms: dict[int, dict]) -> dict[str, tuple[float, float]]:
    coords = {}
    for geometry in road_geoms.values():
        road_coords = geometry["coords"]
        coords[geometry["start_node"]] = road_coords[0]
        coords[geometry["end_node"]] = road_coords[-1]
        coords.setdefault(base_node_id(geometry["start_node"]), road_coords[0])
        coords.setdefault(base_node_id(geometry["end_node"]), road_coords[-1])
    return coords


def active_sim_nodes(sim_nodes_csv: Path, road_geoms: dict[int, dict]) -> list[dict]:
    graph_coords = graph_node_coordinates()
    road_coords = road_endpoint_coordinates(road_geoms)
    nodes = []
    for row in read_csv(sim_nodes_csv):
        node_id = row["node_id"]
        source_node = base_node_id(node_id)
        xy = None
        if ":terminal_" not in node_id:
            xy = graph_coords.get(source_node)
        xy = xy or road_coords.get(node_id) or road_coords.get(source_node) or graph_coords.get(source_node)
        if xy is None:
            continue
        nodes.append(
            {
                "node_id": node_id,
                "node_label": row["node_label"],
                "node_type": row["node_type"],
                "x": xy[0],
                "y": xy[1],
            }
        )
    return nodes


def cumulative_lengths(coords: list[tuple[float, float]]) -> list[float]:
    lengths = [0.0]
    for start, end in zip(coords, coords[1:]):
        lengths.append(lengths[-1] + math.dist(start, end))
    return lengths


def point_at_distance(coords: list[tuple[float, float]], lengths: list[float], distance: float) -> tuple[float, float]:
    distance = min(max(distance, 0.0), lengths[-1])
    if lengths[-1] <= 0.0:
        return coords[0]
    for idx in range(1, len(lengths)):
        if lengths[idx] >= distance:
            segment_length = lengths[idx] - lengths[idx - 1]
            if segment_length <= 0.0:
                return coords[idx]
            ratio = (distance - lengths[idx - 1]) / segment_length
            x = coords[idx - 1][0] + ratio * (coords[idx][0] - coords[idx - 1][0])
            y = coords[idx - 1][1] + ratio * (coords[idx][1] - coords[idx - 1][1])
            return x, y
    return coords[-1]


def cell_segments(coords: list[tuple[float, float]], n_cells: int) -> list[list[tuple[float, float]]]:
    lengths = cumulative_lengths(coords)
    total = lengths[-1]
    if total <= 0.0 or n_cells <= 0:
        return []
    segments = []
    for cell_index in range(n_cells):
        start_distance = total * cell_index / n_cells
        end_distance = total * (cell_index + 1) / n_cells
        segments.append(
            [
                point_at_distance(coords, lengths, start_distance),
                point_at_distance(coords, lengths, end_distance),
            ]
        )
    return segments


def load_cell_snapshots(path: Path):
    density_by_time_and_road = defaultdict(lambda: defaultdict(list))
    time_seconds_by_index = {}
    for row in read_csv(path):
        time_index = int(row["time_index"])
        road_id = int(row["road_id"])
        time_seconds_by_index[time_index] = float(row["time_seconds"])
        density_by_time_and_road[time_index][road_id].append(float(row["density"]))
    return sorted(time_seconds_by_index), time_seconds_by_index, density_by_time_and_road


def draw_nodes(ax, nodes: list[dict], *, show_boundaries: bool) -> None:
    roundabouts = [row for row in nodes if row["node_type"] == "roundabout"]
    regular = [row for row in nodes if row["node_type"] == "junction"]
    boundaries = [row for row in nodes if row["node_type"] == "boundary"]
    if regular:
        ax.scatter(
            [row["x"] for row in regular],
            [row["y"] for row in regular],
            s=3.0,
            facecolor="white",
            edgecolor="#3a3a3a",
            linewidth=0.25,
            zorder=5,
        )
    if roundabouts:
        x = [row["x"] for row in roundabouts]
        y = [row["y"] for row in roundabouts]
        ax.scatter(x, y, s=14.0, facecolor="white", edgecolor="#252525", linewidth=0.4, zorder=6)
        ax.scatter(x, y, s=6.0, facecolor="none", edgecolor="#252525", linewidth=0.35, zorder=7)
    if show_boundaries and boundaries:
        ax.scatter(
            [row["x"] for row in boundaries],
            [row["y"] for row in boundaries],
            s=12.0,
            marker="s",
            facecolor="white",
            edgecolor="#6f6f6f",
            linewidth=0.55,
            zorder=5,
        )


def draw_topology(
    ax,
    road_geoms: dict[int, dict],
    nodes: list[dict],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    road_segments = []
    connector_segments = []
    for geometry in road_geoms.values():
        coords = geometry["coords"]
        target = connector_segments if "connector" in geometry["road_type"].lower() else road_segments
        target.append(coords)

    if connector_segments:
        ax.add_collection(
            LineCollection(
                connector_segments,
                colors="#cfcfcf",
                linewidths=0.55,
                capstyle="round",
                joinstyle="round",
                zorder=1,
            )
        )
    if road_segments:
        ax.add_collection(
            LineCollection(
                road_segments,
                colors="#464646",
                linewidths=0.95,
                capstyle="round",
                joinstyle="round",
                zorder=2,
            )
        )

    draw_nodes(ax, nodes, show_boundaries=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    handles = [
        Line2D([0], [0], color="#464646", lw=2.0, label="Road"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#3a3a3a", color="none", markersize=5.0, label="Junction"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#252525", color="none", markersize=8.0, label="Roundabout"),
        Line2D([0], [0], marker="s", markerfacecolor="white", markeredgecolor="#6f6f6f", color="none", markersize=7.0, label="Boundary"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=11)


def draw_snapshot(
    ax,
    time_index: int,
    road_geoms: dict[int, dict],
    density_by_road: dict[int, list[float]],
    nodes: list[dict],
    cmap,
    norm,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    segments = []
    colors = []
    line_widths = []
    for road_id, densities in density_by_road.items():
        geometry = road_geoms.get(road_id)
        if geometry is None:
            continue
        road_segments = cell_segments(geometry["coords"], len(densities))
        for segment, density in zip(road_segments, densities):
            segments.append(segment)
            colors.append(cmap(norm(min(max(density, 0.0), 1.0))))
            line_widths.append(0.65 if geometry["road_type"] == "Artificial connector" else 1.05)

    collection = LineCollection(
        segments,
        colors=colors,
        linewidths=line_widths,
        capstyle="butt",
        joinstyle="round",
        zorder=2,
    )
    ax.add_collection(collection)
    draw_nodes(ax, nodes, show_boundaries=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.text(
        0.012,
        0.965,
        f"t = {time_index:.0f} s",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.84},
        zorder=10,
    )


def figure_extent(road_geoms: dict[int, dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = [point for geometry in road_geoms.values() for point in geometry["coords"]]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    pad_x = 0.015 * (max(xs) - min(xs))
    pad_y = 0.06 * (max(ys) - min(ys))
    return (min(xs) - pad_x, max(xs) + pad_x), (min(ys) - pad_y, max(ys) + pad_y)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else OUTDIR / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-snapshots", type=Path, default=DEFAULT_CELL_SNAPSHOTS_CSV)
    parser.add_argument("--sim-roads", type=Path, default=DEFAULT_SIM_ROADS_CSV)
    parser.add_argument("--sim-nodes", type=Path, default=DEFAULT_SIM_NODES_CSV)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--combined-figure", type=Path, default=None)
    parser.add_argument("--topology-figure", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell_snapshots_csv = resolve_path(args.cell_snapshots)
    sim_roads_csv = resolve_path(args.sim_roads)
    sim_nodes_csv = resolve_path(args.sim_nodes)
    frames_dir = resolve_path(args.frames_dir)
    combined_figure = resolve_path(args.combined_figure) if args.combined_figure is not None else None
    topology_figure = resolve_path(args.topology_figure) if args.topology_figure is not None else None

    if not cell_snapshots_csv.exists():
        raise FileNotFoundError(f"Run simulate_e18_large_smoke.jl first: {cell_snapshots_csv}")

    arcs = arc_geometries()
    road_geoms = road_geometries(arcs, sim_roads_csv)
    nodes = active_sim_nodes(sim_nodes_csv, road_geoms)
    time_indices, time_seconds_by_index, density_by_time_and_road = load_cell_snapshots(cell_snapshots_csv)
    xlim, ylim = figure_extent(road_geoms)

    cmap = LinearSegmentedColormap.from_list("density_white_to_red", ["#f7fbff", "#fee08b", "#f46d43", "#a50026"])
    norm = Normalize(vmin=0.0, vmax=1.0)

    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("snapshot_*.png"):
        old_frame.unlink()

    for frame_number, time_index in enumerate(time_indices):
        fig, ax = plt.subplots(figsize=(13.5, 6.2))
        draw_snapshot(
            ax,
            time_seconds_by_index[time_index],
            road_geoms,
            density_by_time_and_road[time_index],
            nodes,
            cmap,
            norm,
            xlim,
            ylim,
        )
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.027, pad=0.01)
        colorbar.set_label("density", fontsize=12)
        colorbar.ax.tick_params(labelsize=10)
        fig.tight_layout(pad=0.05)
        fig.savefig(frames_dir / f"snapshot_{frame_number:02d}_{time_seconds_by_index[time_index]:.0f}s.png", dpi=180)
        plt.close(fig)

    if combined_figure is not None:
        nrows = len(time_indices)
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(13.5, 3.0 * nrows), constrained_layout=True)
        if nrows == 1:
            axes = [axes]
        for ax, time_index in zip(axes, time_indices):
            draw_snapshot(
                ax,
                time_seconds_by_index[time_index],
                road_geoms,
                density_by_time_and_road[time_index],
                nodes,
                cmap,
                norm,
                xlim,
                ylim,
            )
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.018, pad=0.01)
        colorbar.set_label("density", fontsize=12)
        colorbar.ax.tick_params(labelsize=10)
        fig.savefig(combined_figure, dpi=180)
        plt.close(fig)

    if topology_figure is not None:
        fig, ax = plt.subplots(figsize=(13.5, 6.2))
        draw_topology(ax, road_geoms, nodes, xlim, ylim)
        fig.tight_layout(pad=0.05)
        fig.savefig(topology_figure, dpi=180)
        plt.close(fig)

    print(f"Wrote {len(time_indices)} frames to {frames_dir}")
    if combined_figure is not None:
        print(f"Wrote {combined_figure}")
    if topology_figure is not None:
        print(f"Wrote {topology_figure}")


if __name__ == "__main__":
    main()
