from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patheffects import Stroke, Normal

import build_e18_large_graph as graph
import plot_e18_large_smoke_snapshots as smoke_plot


OUTDIR = Path(__file__).resolve().parent
FIGURE_DIR = OUTDIR / "pruned_selection_figures"
SIM_ROADS_CSV = OUTDIR / "e18_large_pruned_sim_roads.csv"
SIM_NODES_CSV = OUTDIR / "e18_large_pruned_sim_nodes.csv"

ROAD_COLORS = {
    "Enkel bilveg": "#8d98a8",
    "Kanalisert veg": "#8d98a8",
    "Rampe": "#8d98a8",
    "Artificial connector": "#b8b8b8",
    "Terminal boundary connector": "#c6c6c6",
}
E18_COLOR = "#c7362f"
SELECTED_ROAD_COLOR = "#586879"
BASE_ROAD_COLOR = "#b9c1cc"
BASE_CONNECTOR_COLOR = "#d0d0d0"
VISUAL_NODE_OFFSETS_M = {
    "CN149": (5.0, 4.0),
    "CN151": (5.0, 4.0),
}

TARGET_NODE_LABELS = ["RB08", "RB10", "RB14", "J168"]
SENSOR_LOCATIONS = [
    ("S1", 1047, 0.55),
    ("S2", 1052, 0.52),
    ("S4", 931, 0.72),
    ("S5", 1036, 0.55),
    ("S7", 986, 0.52),
]
SENSOR_ROAD_IDS = [road_id for _, road_id, _ in SENSOR_LOCATIONS]

SELECTION_WINDOWS = [
    {
        "name": "target_area_west",
        "center_label": "RB10",
        "radius_m": 210.0,
        "targets": TARGET_NODE_LABELS,
        "context": [],
        "sensors": SENSOR_LOCATIONS,
        "figsize": (14.0, 10.0),
        "show_legend": False,
    },
    {
        "name": "target_area_rb18_rb20",
        "labels": ["RB18", "RB19", "RB20", "J245", "J246"],
        "pad_m": 230.0,
        "targets": ["RB18", "RB19", "RB20", "J245", "J246"],
        "context": [],
        "sensors": [
            ("S8", 1002, 0.96),
            ("S9", 791, 0.52),
            ("S10", 798, 0.50),
        ],
        "figsize": (13.5, 9.5),
        "show_legend": False,
    },
    {
        "name": "target_area_rb23_rb24",
        "labels": ["RB23", "RB24", "J411", "J416"],
        "pad_m": 250.0,
        "targets": ["RB23", "RB24", "J411", "J416"],
        "context": [],
        "sensors": [
            ("S11", 544, 0.28),
            ("S12", 393, 0.68),
            ("S13", 551, 0.20),
        ],
        "figsize": (13.5, 9.5),
        "show_legend": False,
    },
    {
        "name": "target_area_east",
        "labels": ["RB28", "RB29", "RB30", "J686", "J644"],
        "pad_m": 230.0,
        "targets": ["RB28", "RB29", "RB30", "J686", "J635", "J641", "J644"],
        "context": [],
        "sensors": [
            ("S14", 137, 0.58),
            ("S15", 113, 0.50),
            ("S16", 235, 0.65),
        ],
        "figsize": (13.5, 9.5),
        "show_legend": False,
    },
]


def geometry_intersects_extent(coords, extent, margin=0.0) -> bool:
    x_min, x_max, y_min, y_max = extent
    for x, y in coords:
        if x_min - margin <= x <= x_max + margin and y_min - margin <= y <= y_max + margin:
            return True
    return False


def point_in_extent(x, y, extent, margin=0.0) -> bool:
    x_min, x_max, y_min, y_max = extent
    return x_min - margin <= x <= x_max + margin and y_min - margin <= y <= y_max + margin


def visual_node_xy(node):
    dx, dy = VISUAL_NODE_OFFSETS_M.get(node["node_id"], (0.0, 0.0))
    return node["x"] + dx, node["y"] + dy


def sensor_parts(sensor_location):
    if len(sensor_location) == 3:
        return sensor_location
    road_id, fraction = sensor_location
    return "", road_id, fraction


def figure_extent(road_geoms):
    points = [point for geometry in road_geoms.values() for point in geometry["coords"]]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def section_extents(road_geoms, count=4, overlap_fraction=0.08):
    x_min, x_max, y_min, y_max = figure_extent(road_geoms)
    width = (x_max - x_min) / count
    overlap = overlap_fraction * width
    extents = []
    for idx in range(count):
        left = x_min + idx * width - (overlap if idx > 0 else 0.0)
        right = x_min + (idx + 1) * width + (overlap if idx < count - 1 else 0.0)
        section_points = [
            point
            for geometry in road_geoms.values()
            if geometry_intersects_extent(geometry["coords"], (left, right, y_min, y_max), margin=0.0)
            for point in geometry["coords"]
        ]
        if section_points:
            local_y_min = min(point[1] for point in section_points)
            local_y_max = max(point[1] for point in section_points)
            pad_y = max(40.0, 0.12 * (local_y_max - local_y_min))
            local_y_min -= pad_y
            local_y_max += pad_y
        else:
            local_y_min, local_y_max = y_min, y_max
        extents.append((left, right, local_y_min, local_y_max))
    return extents


def node_extent(nodes, label: str, radius_m: float):
    matches = [node for node in nodes if node["node_label"] == label]
    if not matches:
        raise ValueError(f"Could not find node label {label!r}")
    node = matches[0]
    x, y = node["x"], node["y"]
    return x - radius_m, x + radius_m, y - radius_m, y + radius_m


def labels_extent(nodes, labels, pad_m: float):
    label_set = set(labels)
    matches = [node for node in nodes if node["node_label"] in label_set]
    missing = label_set - {node["node_label"] for node in matches}
    if missing:
        raise ValueError(f"Could not find node labels {sorted(missing)!r}")
    xs = [node["x"] for node in matches]
    ys = [node["y"] for node in matches]
    return min(xs) - pad_m, max(xs) + pad_m, min(ys) - pad_m, max(ys) + pad_m


def visible_label_position(coords, extent):
    x_min, x_max, y_min, y_max = extent
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    best = None
    for index in range(101):
        fraction = index / 100
        (x, y), (tx, ty) = graph.point_and_tangent(coords, fraction)
        inside = x_min <= x <= x_max and y_min <= y <= y_max
        penalty = 0.0 if inside else 1e9
        score = penalty + (x - center_x) ** 2 + (y - center_y) ** 2
        if best is None or score < best[0]:
            best = (score, x, y, tx, ty)
    _, x, y, tx, ty = best
    nx, ny = -ty, tx
    return x, y, nx, ny


def draw_road_label(ax, road_id, geometry, extent, fontsize=5.0, offset_m=7.0):
    coords = geometry["coords"]
    x, y, nx, ny = visible_label_position(coords, extent)
    side = -1 if road_id % 2 == 0 else 1
    ax.text(
        x + side * offset_m * nx,
        y + side * offset_m * ny,
        f"R{road_id:03d}",
        fontsize=fontsize,
        ha="center",
        va="center",
        color="#1f1f1f",
        bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.80},
        zorder=6,
    )


def draw_direction_arrow(
    ax,
    coords,
    color,
    alpha,
    *,
    arrow_length=15.0,
    mutation_scale=9.0,
    linewidth=1.15,
):
    if len(coords) < 2 or sum(graph.segment_lengths(coords)) < 10:
        return
    (x, y), (tx, ty) = graph.point_and_tangent(coords, 0.68)
    ax.annotate(
        "",
        xy=(x + tx * arrow_length * 0.55, y + ty * arrow_length * 0.55),
        xytext=(x - tx * arrow_length * 0.45, y - ty * arrow_length * 0.45),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "alpha": alpha,
            "lw": linewidth,
            "mutation_scale": mutation_scale,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=4,
    )


def road_color(geometry, *, quiet=False, highlighted=False):
    road_type = geometry["road_type"]
    if geometry.get("is_e18_highway", False):
        return E18_COLOR
    if quiet:
        if highlighted:
            return SELECTED_ROAD_COLOR
        if "connector" in road_type.lower():
            return BASE_CONNECTOR_COLOR
        return BASE_ROAD_COLOR
    return ROAD_COLORS.get(road_type, "#4f4f4f")


def draw_roads(
    ax,
    road_geoms,
    extent,
    *,
    road_label_fontsize=5.0,
    road_label_offset_m=7.0,
    labeled_road_ids=None,
    highlighted_road_ids=None,
    quiet=False,
    arrow_length=15.0,
    arrow_mutation_scale=9.0,
    arrow_linewidth=1.15,
):
    label_all_roads = labeled_road_ids is None
    labeled_road_ids = set(labeled_road_ids or [])
    highlighted_road_ids = set(highlighted_road_ids or labeled_road_ids)
    plotted_points = []
    for road_id, geometry in sorted(road_geoms.items()):
        coords = geometry["coords"]
        if not geometry_intersects_extent(coords, extent, margin=30.0):
            continue
        road_type = geometry["road_type"]
        highlighted = road_id in highlighted_road_ids
        color = road_color(geometry, quiet=quiet, highlighted=highlighted)
        linestyle = "--" if "connector" in road_type.lower() else "-"
        is_main = road_type == "Kanalisert veg" and int(float(geometry.get("speed_limit", 0))) >= 60
        if quiet:
            if geometry.get("is_e18_highway", False):
                linewidth = 2.75
                alpha = 0.95
            elif highlighted:
                linewidth = 2.15 if is_main else 1.75
                alpha = 0.90
            else:
                linewidth = 0.95 if "connector" in road_type.lower() else 1.15
                alpha = 0.36 if "connector" in road_type.lower() else 0.56
        else:
            linewidth = 1.2 if "connector" in road_type.lower() else (2.1 if is_main else 1.55)
            alpha = 0.62 if "connector" in road_type.lower() else 0.92
        xs, ys = zip(*coords)
        ax.plot(xs, ys, color=color, lw=linewidth, alpha=alpha, linestyle=linestyle, solid_capstyle="round", zorder=2)
        draw_direction_arrow(
            ax,
            coords,
            color,
            alpha,
            arrow_length=arrow_length,
            mutation_scale=arrow_mutation_scale,
            linewidth=arrow_linewidth,
        )
        if label_all_roads or road_id in labeled_road_ids:
            draw_road_label(ax, road_id, geometry, extent, fontsize=road_label_fontsize, offset_m=road_label_offset_m)
        plotted_points.extend(coords)
    return plotted_points


def draw_target_nodes(ax, nodes, target_labels, *, show_labels=True):
    node_by_label = {node["node_label"]: node for node in nodes}
    for label in target_labels:
        node = node_by_label.get(label)
        if node is None:
            continue
        if node["node_type"] == "roundabout":
            ax.scatter(
                [node["x"]],
                [node["y"]],
                s=320,
                marker="o",
                facecolor="white",
                edgecolor="#f03b8d",
                linewidth=2.4,
                zorder=20,
            )
            ax.scatter(
                [node["x"]],
                [node["y"]],
                s=130,
                marker="o",
                facecolor="none",
                edgecolor="#f03b8d",
                linewidth=2.0,
                zorder=21,
            )
        else:
            ax.scatter(
                [node["x"]],
                [node["y"]],
                s=82,
                marker="o",
                facecolor="#f03b8d",
                edgecolor="#222222",
                linewidth=1.0,
                zorder=20,
            )
        if show_labels:
            ax.text(
                node["x"],
                node["y"] + (20.0 if node["node_type"] == "roundabout" else 13.0),
                label,
                fontsize=12.4,
                weight="bold",
                ha="center",
                va="bottom",
                color="#111111",
                bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.85},
                path_effects=[Stroke(linewidth=1.0, foreground="white"), Normal()],
                zorder=21,
            )


def draw_context_nodes(ax, nodes, context_labels, *, show_labels=True):
    node_by_label = {node["node_label"]: node for node in nodes}
    for label in context_labels:
        node = node_by_label.get(label)
        if node is None:
            continue
        ax.scatter(
            [node["x"]],
            [node["y"]],
            s=72,
            marker="o",
            facecolor="white",
            edgecolor="#4f4f4f",
            linewidth=1.2,
            zorder=19,
        )
        if show_labels:
            ax.text(
                node["x"],
                node["y"] + 12.0,
                label,
                fontsize=11.2,
                weight="bold",
                ha="center",
                va="bottom",
                color="#333333",
                bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.82},
                zorder=20,
            )


def draw_sensor_locations(ax, road_geoms, sensor_locations, *, show_labels=True):
    for index, sensor_location in enumerate(sensor_locations, start=1):
        label, road_id, fraction = sensor_parts(sensor_location)
        label = label or f"S{index}"
        geometry = road_geoms.get(road_id)
        if geometry is None:
            continue
        (x, y), _ = graph.point_and_tangent(geometry["coords"], fraction)
        ax.scatter(
            [x],
            [y],
            s=100,
            marker="o",
            facecolor="#0072b2",
            edgecolor="white",
            linewidth=1.1,
            zorder=21,
        )
        if show_labels:
            ax.text(
                x + 5.0,
                y + 5.0,
                label,
                fontsize=11.6,
                weight="bold",
                ha="left",
                va="bottom",
                color="#00507c",
                bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.82},
                zorder=22,
            )


def incident_road_ids_for_targets(nodes, target_labels):
    target_set = set(target_labels)
    result = set()
    for node in smoke_plot.read_csv(SIM_NODES_CSV):
        if node["node_label"] not in target_set:
            continue
        for field in ("incoming_road_ids", "outgoing_road_ids"):
            if not node[field].strip():
                continue
            result.update(int(road_id) for road_id in node[field].split())
    return result


def draw_nodes(
    ax,
    nodes,
    extent,
    *,
    node_label_fontsize=3.8,
    roundabout_label_fontsize=5.5,
    labeled_node_labels=None,
    quiet=False,
):
    labeled_node_labels = set(labeled_node_labels or [])
    visible = [node for node in nodes if point_in_extent(node["x"], node["y"], extent, margin=30.0)]
    boundaries = [node for node in visible if node["node_type"] == "boundary"]
    regular = [node for node in visible if node["node_type"] == "junction"]
    roundabouts = [node for node in visible if node["node_type"] == "roundabout"]

    for node in boundaries:
        x, y = visual_node_xy(node)
        ax.scatter(
            [x],
            [y],
            s=28 if quiet else 48,
            marker="s",
            facecolor="white",
            edgecolor="#6f6f6f",
            linewidth=0.65 if quiet else 1.0,
            zorder=8,
        )
        if not quiet or node["node_label"] in labeled_node_labels:
            ax.text(
                x,
                y,
                node["node_label"],
                fontsize=max(3.4, node_label_fontsize - 0.1),
                weight="bold",
                ha="center",
                va="center",
                color="#303030",
                zorder=9,
            )

    for node in regular:
        x, y = visual_node_xy(node)
        ax.scatter(
            [x],
            [y],
            s=82 if quiet else 82,
            marker="o",
            facecolor="white",
            edgecolor="#111111",
            linewidth=0.75 if quiet else 1.0,
            zorder=10,
        )
        if not quiet or node["node_label"] in labeled_node_labels:
            ax.text(
                x,
                y,
                node["node_label"],
                fontsize=node_label_fontsize,
                weight="bold",
                ha="center",
                va="center",
                color="#111111",
                zorder=11,
            )

    for node in roundabouts:
        x, y = visual_node_xy(node)
        ax.scatter(
            [x],
            [y],
            s=320 if quiet else 320,
            marker="o",
            facecolor="white",
            edgecolor="#111111",
            linewidth=0.9 if quiet else 1.2,
            zorder=12,
        )
        ax.scatter(
            [x],
            [y],
            s=130 if quiet else 130,
            marker="o",
            facecolor="none",
            edgecolor="#111111",
            linewidth=0.65 if quiet else 0.9,
            zorder=13,
        )
        if not quiet or node["node_label"] in labeled_node_labels:
            ax.text(
                x + 8.0,
                y + 8.0,
                node["node_label"],
                fontsize=roundabout_label_fontsize,
                weight="bold",
                ha="left",
                va="bottom",
                color="#111111",
                bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.84},
                path_effects=[Stroke(linewidth=1.0, foreground="white"), Normal()],
                zorder=14,
            )

    return visible


def draw_legend(ax, road_geoms, nodes, extent, *, fontsize=14, loc="upper left"):
    used_road_types = {
        geometry["road_type"]
        for geometry in road_geoms.values()
        if geometry_intersects_extent(geometry["coords"], extent, margin=30.0)
    }
    handles = [
        Line2D([0], [0], color=BASE_ROAD_COLOR, lw=4.0, label="Directed road segment"),
    ]
    if any("connector" in road_type.lower() for road_type in used_road_types):
        handles.append(
            Line2D([0], [0], color=BASE_CONNECTOR_COLOR, lw=4.0, linestyle="--", label="Artificial connector")
        )

    visible_node_types = {
        node["node_type"]
        for node in nodes
        if point_in_extent(node["x"], node["y"], extent, margin=30.0)
    }
    if "junction" in visible_node_types:
        handles.append(
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#111111", color="none", markersize=10.0, label="Junction")
        )
    if "roundabout" in visible_node_types:
        handles.append(
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#111111", color="none", markersize=13.0, label="Roundabout")
        )
    if "boundary" in visible_node_types:
        handles.append(
            Line2D([0], [0], marker="s", markerfacecolor="white", markeredgecolor="#6f6f6f", color="none", markersize=10.0, label="Boundary")
        )
    handles.extend(
        [
            Line2D([0], [0], color=E18_COLOR, lw=4.0, label="E18"),
            Line2D([0], [0], color=SELECTED_ROAD_COLOR, lw=4.0, label="Target/sensor road"),
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#f03b8d", markeredgewidth=2.0, color="none", markersize=12.0, label="Inferred roundabout"),
            Line2D([0], [0], marker="o", markerfacecolor="#f03b8d", markeredgecolor="#222222", color="none", markersize=8.0, label="Inferred junction"),
            Line2D([0], [0], marker="o", markerfacecolor="#0072b2", markeredgecolor="white", color="none", markersize=9.0, label="Sensor"),
        ]
    )

    ax.legend(handles=handles, loc=loc, frameon=True, fontsize=fontsize, handlelength=2.2, borderpad=0.55)


def plot_section(
    road_geoms,
    nodes,
    extent,
    output_path,
    *,
    figsize=(17.0, 10.0),
    show_legend=True,
    road_label_fontsize=5.0,
    road_label_offset_m=7.0,
    node_label_fontsize=3.8,
    roundabout_label_fontsize=5.5,
    target_node_labels=None,
    context_node_labels=None,
    sensor_road_ids=None,
    labeled_road_ids=None,
    highlighted_road_ids=None,
    quiet=False,
    show_annotation_labels=True,
    legend_fontsize=14,
    legend_loc="upper left",
    arrow_length=15.0,
    arrow_mutation_scale=9.0,
    arrow_linewidth=1.15,
):
    fig, ax = plt.subplots(figsize=figsize)
    draw_roads(
        ax,
        road_geoms,
        extent,
        road_label_fontsize=road_label_fontsize,
        road_label_offset_m=road_label_offset_m,
        labeled_road_ids=labeled_road_ids,
        highlighted_road_ids=highlighted_road_ids,
        quiet=quiet,
        arrow_length=arrow_length,
        arrow_mutation_scale=arrow_mutation_scale,
        arrow_linewidth=arrow_linewidth,
    )
    draw_nodes(
        ax,
        nodes,
        extent,
        node_label_fontsize=node_label_fontsize,
        roundabout_label_fontsize=roundabout_label_fontsize,
        labeled_node_labels=[],
        quiet=quiet,
    )
    if target_node_labels:
        draw_target_nodes(ax, nodes, target_node_labels, show_labels=show_annotation_labels)
    if context_node_labels:
        draw_context_nodes(ax, nodes, context_node_labels, show_labels=show_annotation_labels)
    if sensor_road_ids:
        draw_sensor_locations(ax, road_geoms, sensor_road_ids, show_labels=show_annotation_labels)
    if show_legend:
        draw_legend(ax, road_geoms, nodes, extent, fontsize=legend_fontsize, loc=legend_loc)

    x_min, x_max, y_min, y_max = extent
    pad = 30
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def enrich_road_metadata(road_geoms):
    road_rows = {int(row["id"]): row for row in smoke_plot.read_csv(SIM_ROADS_CSV)}
    graph_edge_rows = smoke_plot.read_csv(OUTDIR / "e18_large_graph_edges.csv")
    e18_mainline_source_roads = {
        row["road_id"]
        for row in graph_edge_rows
        if row["kortform"].startswith("EV18")
        and " KD" not in row["kortform"]
        and " SD" not in row["kortform"]
    }
    for road_id, geometry in road_geoms.items():
        row = road_rows.get(road_id)
        if row is None:
            continue
        geometry["speed_limit"] = row["speed_limit"]
        geometry["lanes"] = row["lanes"]
        geometry["length_m"] = row["length_m"]
        source_road_ids = set(row["source_road_ids"].split())
        geometry["is_e18_highway"] = bool(source_road_ids & e18_mainline_source_roads)


def write_selection_tables(nodes, road_geoms):
    node_by_label = {node["node_label"]: node for node in smoke_plot.read_csv(SIM_NODES_CSV)}
    target_labels = []
    for window in SELECTION_WINDOWS:
        for label in window["targets"]:
            if label not in target_labels:
                target_labels.append(label)
    target_rows = []
    for label in target_labels:
        node = node_by_label.get(label)
        if node is None:
            continue
        target_rows.append(
            {
                "target_label": label,
                "node_id": node["node_id"],
                "node_type": node["node_type"],
                "incoming_road_ids": node["incoming_road_ids"],
                "outgoing_road_ids": node["outgoing_road_ids"],
                "degree_directed": node["degree_directed"],
            }
        )

    road_rows = {int(row["id"]): row for row in smoke_plot.read_csv(SIM_ROADS_CSV)}
    sensor_locations = []
    for window in SELECTION_WINDOWS:
        for sensor_location in window["sensors"]:
            if sensor_location not in sensor_locations:
                sensor_locations.append(sensor_location)
    sensor_rows = []
    sensor_by_road = {road_id: (label, fraction) for label, road_id, fraction in sensor_locations}
    for _, road_id, _ in sensor_locations:
        row = road_rows.get(road_id)
        if row is None:
            continue
        label, fraction = sensor_by_road[road_id]
        sensor_rows.append(
            {
                "sensor_label": label,
                "road_id": road_id,
                "road_fraction": fraction,
                "start_node": row["start_node"],
                "end_node": row["end_node"],
                "length_m": row["length_m"],
                "road_type": row["road_type"],
                "lanes": row["lanes"],
                "speed_limit": row["speed_limit"],
            }
        )

    def write_tsv(path, rows, columns):
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("\t".join(columns) + "\n")
            for row in rows:
                handle.write("\t".join(str(row[column]) for column in columns) + "\n")

    write_tsv(
        FIGURE_DIR / "proposed_inference_targets.tsv",
        target_rows,
        ["target_label", "node_id", "node_type", "incoming_road_ids", "outgoing_road_ids", "degree_directed"],
    )
    write_tsv(
        FIGURE_DIR / "proposed_sensor_locations.tsv",
        sensor_rows,
        ["sensor_label", "road_id", "road_fraction", "start_node", "end_node", "length_m", "road_type", "lanes", "speed_limit"],
    )


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    arcs = smoke_plot.arc_geometries()
    road_geoms = smoke_plot.road_geometries(arcs, SIM_ROADS_CSV)
    enrich_road_metadata(road_geoms)
    nodes = smoke_plot.active_sim_nodes(SIM_NODES_CSV, road_geoms)

    for section_index, extent in enumerate(section_extents(road_geoms), start=1):
        plot_section(
            road_geoms,
            nodes,
            extent,
            FIGURE_DIR / f"e18_large_pruned_section_{section_index:02d}_with_ids.png",
        )

    all_targets = []
    all_context = []
    all_sensors = []
    all_highlighted_roads = set()
    for window in SELECTION_WINDOWS:
        targets = window["targets"]
        context = window.get("context", [])
        sensors = window["sensors"]
        sensor_road_ids = {road_id for _, road_id, _ in sensors}
        highlighted_roads = incident_road_ids_for_targets(nodes, targets) | sensor_road_ids
        all_highlighted_roads.update(highlighted_roads)
        all_targets.extend(label for label in targets if label not in all_targets)
        all_context.extend(label for label in context if label not in all_context)
        all_sensors.extend(sensors)
        if "center_label" in window:
            extent = node_extent(nodes, window["center_label"], window["radius_m"])
        else:
            extent = labels_extent(nodes, window["labels"], window["pad_m"])
        plot_section(
            road_geoms,
            nodes,
            extent,
            FIGURE_DIR / f"e18_large_pruned_{window['name']}.png",
            figsize=window["figsize"],
            show_legend=window["show_legend"],
            road_label_fontsize=7.6,
            road_label_offset_m=7.0,
            node_label_fontsize=5.4,
            roundabout_label_fontsize=7.8,
            target_node_labels=targets,
            context_node_labels=context,
            sensor_road_ids=sensors,
            labeled_road_ids=set(),
            highlighted_road_ids=highlighted_roads,
            quiet=True,
            arrow_length=23.0,
            arrow_mutation_scale=16.0,
            arrow_linewidth=1.45,
        )

    plot_section(
        road_geoms,
        nodes,
        figure_extent(road_geoms),
        FIGURE_DIR / "e18_large_pruned_inference_overview.png",
        figsize=(18.0, 9.0),
        show_legend=True,
        road_label_fontsize=5.4,
        road_label_offset_m=7.0,
        node_label_fontsize=4.4,
        roundabout_label_fontsize=6.2,
        target_node_labels=all_targets,
        context_node_labels=all_context,
        sensor_road_ids=all_sensors,
        labeled_road_ids=set(),
        highlighted_road_ids=all_highlighted_roads,
        quiet=True,
        show_annotation_labels=False,
        legend_fontsize=22,
        legend_loc="lower right",
    )
    plot_section(
        road_geoms,
        nodes,
        figure_extent(road_geoms),
        FIGURE_DIR / "e18_large_pruned_inference_overview_no_legend.png",
        figsize=(18.0, 9.0),
        show_legend=False,
        road_label_fontsize=5.4,
        road_label_offset_m=7.0,
        node_label_fontsize=4.4,
        roundabout_label_fontsize=6.2,
        target_node_labels=all_targets,
        context_node_labels=all_context,
        sensor_road_ids=all_sensors,
        labeled_road_ids=set(),
        highlighted_road_ids=all_highlighted_roads,
        quiet=True,
        show_annotation_labels=False,
    )

    write_selection_tables(nodes, road_geoms)


if __name__ == "__main__":
    main()
