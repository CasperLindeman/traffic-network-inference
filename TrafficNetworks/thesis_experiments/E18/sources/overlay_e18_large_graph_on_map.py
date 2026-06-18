from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import plot_e18_large_pruned_sections as selection


OUTDIR = Path(__file__).resolve().parent
MAP_IMAGE = OUTDIR / "baerum_E18_skjermbilde.png"
GRAPH_BUILDER = OUTDIR / "build_e18_large_graph.py"
OUTPUT = OUTDIR / "e18_large_graph_map_overlay.png"
PRUNED_OUTPUT = OUTDIR / "pruned_selection_figures" / "e18_large_pruned_graph_map_overlay.png"
PRUNED_SIM_ROADS_CSV = OUTDIR / "e18_large_pruned_sim_roads.csv"
PRUNED_SIM_NODES_CSV = OUTDIR / "e18_large_pruned_sim_nodes.csv"
MAP_EXTENT_SCALE = 0.955
MAP_CENTER_SHIFT_M = (0.0, 0.0)
GRAPH_SHIFT_M = (-140.0, 0.0)

ROAD_OVERLAY_COLORS = {
    "Enkel bilveg": "#006d77",
    "Kanalisert veg": "#e85d04",
    "Rampe": "#7b2cbf",
    "Artificial connector": "#333333",
}


def load_graph_builder():
    spec = importlib.util.spec_from_file_location("e18_large_graph", GRAPH_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def final_graph_objects(builder):
    objects, _ = builder.load_api_objects()
    simplified_objects, roundabouts = builder.apply_roundabout_simplification(objects)
    roundabout_nodes = {component["node_id"] for component in roundabouts}
    final_objects, final_roundabout_nodes, _ = builder.contract_short_connectors(
        simplified_objects, roundabout_nodes
    )
    final_objects, _ = builder.drop_model_roads(final_objects, builder.DROP_ROAD_IDS)
    final_objects, _ = builder.add_artificial_component_connectors(
        final_objects,
        final_roundabout_nodes,
        builder.FORCE_BOUNDARY_NODE_IDS,
    )
    junctions, boundaries, marker_nodes = builder.classify_nodes(
        final_objects,
        forced_junction_nodes=final_roundabout_nodes,
        forced_boundary_nodes=builder.FORCE_BOUNDARY_NODE_IDS,
    )
    return final_objects, junctions, boundaries, marker_nodes


def map_extent_for_graph(builder, objects, image_shape, scale=MAP_EXTENT_SCALE, center_shift=MAP_CENTER_SHIFT_M):
    x_min, x_max, y_min, y_max = builder.graph_extent(objects)
    image_height, image_width = image_shape[:2]
    image_aspect = image_width / image_height
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    graph_width = x_max - x_min
    graph_height = y_max - y_min
    width_from_height = graph_height * image_aspect
    if width_from_height > graph_width:
        graph_width = width_from_height
    else:
        graph_height = graph_width / image_aspect

    graph_width *= scale
    graph_height *= scale
    x_mid += center_shift[0]
    y_mid += center_shift[1]
    return (
        x_mid - 0.5 * graph_width,
        x_mid + 0.5 * graph_width,
        y_mid - 0.5 * graph_height,
        y_mid + 0.5 * graph_height,
    )


def draw_overlay(builder, objects, junctions, boundaries, marker_nodes, output=OUTPUT, scale=MAP_EXTENT_SCALE):
    image = mpimg.imread(MAP_IMAGE)
    extent = map_extent_for_graph(builder, objects, image.shape, scale=scale)
    graph_dx, graph_dy = GRAPH_SHIFT_M

    fig, ax = plt.subplots(figsize=(16.68, 7.05))
    ax.imshow(image, extent=extent, origin="upper", zorder=0)

    for obj in objects:
        color = ROAD_OVERLAY_COLORS.get(obj["typeVeg"], "#005f73")
        alpha = 0.86 if obj["typeVeg"] != "Artificial connector" else 0.55
        linewidth = 1.25 if obj["typeVeg"] != "Artificial connector" else 0.9
        linestyle = "--" if obj["typeVeg"] == "Artificial connector" else "-"
        for direction in obj["travel_directions"]:
            coords, start_node, end_node = builder.directed_geometry(obj, direction)
            trim_start = builder.NODE_TRIM_M if start_node in marker_nodes else 0.0
            trim_end = builder.NODE_TRIM_M if end_node in marker_nodes else 0.0
            coords = builder.trim_polyline(coords, trim_start, trim_end)
            if len(obj["travel_directions"]) > 1:
                coords = builder.offset_polyline(coords, builder.BIDIRECTIONAL_OFFSET_M)
            coords = [(x + graph_dx, y + graph_dy) for x, y in coords]
            xs, ys = zip(*coords)
            ax.plot(
                xs,
                ys,
                color=color,
                lw=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                solid_capstyle="round",
                zorder=3,
            )

    regular_junctions = [row for row in junctions if row["node_type"] == "junction"]
    roundabouts = [row for row in junctions if row["node_type"] == "roundabout"]
    ax.scatter(
        [row["x"] + graph_dx for row in regular_junctions],
        [row["y"] + graph_dy for row in regular_junctions],
        s=11,
        marker="o",
        facecolor="white",
        edgecolor="#111111",
        linewidth=0.45,
        alpha=0.82,
        zorder=5,
    )
    ax.scatter(
        [row["x"] + graph_dx for row in roundabouts],
        [row["y"] + graph_dy for row in roundabouts],
        s=36,
        marker="o",
        facecolor="none",
        edgecolor="#111111",
        linewidth=0.85,
        alpha=0.9,
        zorder=6,
    )
    ax.scatter(
        [row["x"] + graph_dx for row in boundaries],
        [row["y"] + graph_dy for row in boundaries],
        s=10,
        marker="s",
        facecolor="white",
        edgecolor="#555555",
        linewidth=0.35,
        alpha=0.72,
        zorder=4,
    )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def draw_pruned_overlay(builder, objects, output=PRUNED_OUTPUT, scale=MAP_EXTENT_SCALE):
    output.parent.mkdir(parents=True, exist_ok=True)
    image = mpimg.imread(MAP_IMAGE)
    extent = map_extent_for_graph(builder, objects, image.shape, scale=scale)
    graph_dx, graph_dy = GRAPH_SHIFT_M

    arcs = selection.smoke_plot.arc_geometries()
    road_geoms = selection.smoke_plot.road_geometries(arcs, PRUNED_SIM_ROADS_CSV)
    selection.enrich_road_metadata(road_geoms)
    nodes = selection.smoke_plot.active_sim_nodes(PRUNED_SIM_NODES_CSV, road_geoms)

    fig, ax = plt.subplots(figsize=(16.68, 7.05))
    ax.imshow(image, extent=extent, origin="upper", zorder=0)

    for geometry in road_geoms.values():
        road_type = geometry["road_type"]
        if geometry.get("is_e18_highway", False):
            color = selection.E18_COLOR
            linewidth = 2.35
            alpha = 0.96
        elif "connector" in road_type.lower():
            color = "#4d4d4d"
            linewidth = 0.85
            alpha = 0.62
        else:
            color = "#145a42"
            linewidth = 1.18
            alpha = 0.84
        linestyle = "--" if "connector" in road_type.lower() else "-"
        coords = [(x + graph_dx, y + graph_dy) for x, y in geometry["coords"]]
        xs, ys = zip(*coords)
        ax.plot(
            xs,
            ys,
            color=color,
            lw=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=3,
        )

    regular_junctions = [row for row in nodes if row["node_type"] == "junction"]
    roundabouts = [row for row in nodes if row["node_type"] == "roundabout"]
    boundaries = [row for row in nodes if row["node_type"] == "boundary"]
    ax.scatter(
        [row["x"] + graph_dx for row in regular_junctions],
        [row["y"] + graph_dy for row in regular_junctions],
        s=14,
        marker="o",
        facecolor="white",
        edgecolor="#111111",
        linewidth=0.55,
        alpha=0.9,
        zorder=5,
    )
    ax.scatter(
        [row["x"] + graph_dx for row in roundabouts],
        [row["y"] + graph_dy for row in roundabouts],
        s=48,
        marker="o",
        facecolor="none",
        edgecolor="#111111",
        linewidth=1.0,
        alpha=0.94,
        zorder=6,
    )
    ax.scatter(
        [row["x"] + graph_dx for row in boundaries],
        [row["y"] + graph_dy for row in boundaries],
        s=13,
        marker="s",
        facecolor="white",
        edgecolor="#333333",
        linewidth=0.45,
        alpha=0.82,
        zorder=4,
    )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    if not MAP_IMAGE.exists():
        raise FileNotFoundError(f"Missing map screenshot: {MAP_IMAGE}")
    builder = load_graph_builder()
    objects, junctions, boundaries, marker_nodes = final_graph_objects(builder)
    draw_overlay(builder, objects, junctions, boundaries, marker_nodes)
    draw_pruned_overlay(builder, objects)
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {PRUNED_OUTPUT}")


if __name__ == "__main__":
    main()
