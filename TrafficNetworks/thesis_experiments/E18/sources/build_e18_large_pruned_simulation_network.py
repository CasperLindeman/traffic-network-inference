from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import build_e18_large_simulation_network as base


OUTDIR = Path(__file__).resolve().parent
OUTPUT_PREFIX = "e18_large_pruned_sim"
NETWORK_NAME = "e18_large_pruned_simulation_topology"

TURNING_MATRIX_OVERRIDES = {
    "RB28": [
        [0.10, 0.10, 0.00, 0.10, 0.70],
        [0.10, 0.10, 0.10, 0.00, 0.70],
        [0.20, 0.10, 0.10, 0.10, 0.50],
        [0.10, 0.10, 0.10, 0.00, 0.70],
    ],
    "RB29": [
        [0.30, 0.70],
        [0.70, 0.30],
        [0.70, 0.30],
    ],
}


def is_prunable_leaf_road(road: dict) -> bool:
    if str(road["road_type"]) != "Enkel bilveg":
        return False
    if int(road["speed_limit"]) > 40:
        return False
    if float(road["lanes"]) > 1.0:
        return False
    return True


def node_degrees(sim_roads: list[dict], keep_ids: set[int]) -> dict[str, int]:
    degrees = defaultdict(int)
    for road in sim_roads:
        if int(road["id"]) not in keep_ids:
            continue
        degrees[road["start_node"]] += 1
        degrees[road["end_node"]] += 1
    return degrees


def prune_peripheral_local_roads(sim_roads: list[dict]) -> tuple[list[dict], list[dict]]:
    keep_ids = {int(road["id"]) for road in sim_roads}
    dropped = []
    iteration = 1

    while True:
        degrees = node_degrees(sim_roads, keep_ids)
        to_drop = []
        for road in sim_roads:
            road_id = int(road["id"])
            if road_id not in keep_ids:
                continue
            if not is_prunable_leaf_road(road):
                continue
            if degrees[road["start_node"]] == 1 or degrees[road["end_node"]] == 1:
                to_drop.append(road_id)

        if not to_drop:
            break

        drop_set = set(to_drop)
        for road in sim_roads:
            if int(road["id"]) in drop_set:
                dropped.append(
                    dict(
                        road,
                        prune_iteration=iteration,
                        dropped_reason="peripheral_low_speed_single_lane_leaf",
                    )
                )
        keep_ids -= drop_set
        iteration += 1

    kept = [road for road in sim_roads if int(road["id"]) in keep_ids]
    return kept, dropped


def split_words(value) -> list[str]:
    return [part for part in str(value or "").split() if part]


def merge_words(left, right) -> str:
    return " ".join(sorted(set(split_words(left)) | set(split_words(right))))


def append_words(left, right) -> str:
    return " ".join(split_words(left) + split_words(right))


def is_connector_road(road: dict) -> bool:
    road_type = str(road["road_type"]).lower()
    return "connector" in road_type


def merged_road_type(left: dict, right: dict) -> str:
    left_type = str(left["road_type"])
    right_type = str(right["road_type"])
    if left_type == right_type:
        return left_type
    if left_type == "Rampe" or right_type == "Rampe":
        return "Rampe"
    if left_type == "Kanalisert veg" or right_type == "Kanalisert veg":
        return "Kanalisert veg"
    return left_type


def can_contract_pass_through(left: dict, right: dict) -> bool:
    if is_connector_road(left) or is_connector_road(right):
        return False
    return (
        str(left["road_type"]) == str(right["road_type"])
        and float(left["lanes"]) == float(right["lanes"])
        and int(left["speed_limit"]) == int(right["speed_limit"])
    )


def pass_through_node_map(sim_roads: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for road in sim_roads:
        outgoing[road["start_node"]].append(road)
        incoming[road["end_node"]].append(road)
    return incoming, outgoing


def collapse_pass_through_nodes(sim_roads: list[dict], graph_nodes: list[dict]) -> tuple[list[dict], list[dict]]:
    node_meta = {row["node_id"]: row for row in graph_nodes}
    protected_nodes = {row["node_id"] for row in graph_nodes if row.get("node_type") in ("boundary", "roundabout")}
    current = list(sim_roads)
    contractions = []
    iteration = 1

    while True:
        incoming, outgoing = pass_through_node_map(current)
        road_by_id = {int(road["id"]): road for road in current}
        contracted_node = None
        for node in sorted(set(incoming) | set(outgoing)):
            if node in protected_nodes or ":" in node:
                continue
            if len(incoming[node]) != 1 or len(outgoing[node]) != 1:
                continue
            left = incoming[node][0]
            right = outgoing[node][0]
            if int(left["id"]) == int(right["id"]):
                continue
            if left["start_node"] == right["end_node"]:
                continue
            if not can_contract_pass_through(left, right):
                continue
            contracted_node = node
            break

        if contracted_node is None:
            break

        left = incoming[contracted_node][0]
        right = outgoing[contracted_node][0]
        length_m = float(left["length_m"]) + float(right["length_m"])
        blocks, rounded_length_m = base.rounded_blocks(length_m)
        merged = dict(left)
        merged["end_node"] = right["end_node"]
        merged["length_m"] = f"{length_m:.3f}"
        merged["rounded_length_m"] = rounded_length_m
        merged["blocks_12m"] = blocks
        merged["lanes"] = min(float(left["lanes"]), float(right["lanes"]))
        merged["speed_limit"] = min(int(left["speed_limit"]), int(right["speed_limit"]))
        merged["road_type"] = merged_road_type(left, right)
        merged["source_road_ids"] = merge_words(left["source_road_ids"], right["source_road_ids"])
        merged["source_sequences"] = merge_words(left["source_sequences"], right["source_sequences"])
        merged["source_object_keys"] = append_words(left["source_object_keys"], right["source_object_keys"])
        merged["path_arc_ids"] = append_words(left["path_arc_ids"], right["path_arc_ids"])

        contractions.append(
            {
                "iteration": iteration,
                "node_id": contracted_node,
                "node_label": node_meta.get(contracted_node, {}).get("node_label", ""),
                "incoming_road_id": left["id"],
                "outgoing_road_id": right["id"],
                "merged_road_id": merged["id"],
                "length_m": f"{length_m:.3f}",
                "incoming_road_type": left["road_type"],
                "outgoing_road_type": right["road_type"],
                "road_type": merged["road_type"],
                "incoming_lanes": left["lanes"],
                "outgoing_lanes": right["lanes"],
                "lanes": merged["lanes"],
                "incoming_speed_limit": left["speed_limit"],
                "outgoing_speed_limit": right["speed_limit"],
                "speed_limit": merged["speed_limit"],
            }
        )

        remove_ids = {int(left["id"]), int(right["id"])}
        current = [road for road in current if int(road["id"]) not in remove_ids]
        road_by_id[int(merged["id"])] = merged
        current.append(merged)
        iteration += 1

    return current, contractions


def collapse_bidirectional_pass_through_nodes(sim_roads: list[dict], graph_nodes: list[dict]) -> tuple[list[dict], list[dict]]:
    node_meta = {row["node_id"]: row for row in graph_nodes}
    protected_nodes = {row["node_id"] for row in graph_nodes if row.get("node_type") in ("boundary", "roundabout")}
    current = list(sim_roads)
    contractions = []
    iteration = 1

    while True:
        incoming, outgoing = pass_through_node_map(current)
        contracted_node = None
        contraction_roads = None
        for node in sorted(set(incoming) | set(outgoing)):
            if node in protected_nodes or ":" in node:
                continue
            if len(incoming[node]) != 2 or len(outgoing[node]) != 2:
                continue
            incident = incoming[node] + outgoing[node]
            if any(is_connector_road(road) for road in incident):
                continue

            neighbours = sorted(
                set([road["start_node"] for road in incoming[node]] + [road["end_node"] for road in outgoing[node]])
            )
            if len(neighbours) != 2:
                continue
            left_neighbour, right_neighbour = neighbours

            try:
                left_to_node = next(road for road in incoming[node] if road["start_node"] == left_neighbour)
                node_to_left = next(road for road in outgoing[node] if road["end_node"] == left_neighbour)
                right_to_node = next(road for road in incoming[node] if road["start_node"] == right_neighbour)
                node_to_right = next(road for road in outgoing[node] if road["end_node"] == right_neighbour)
            except StopIteration:
                continue

            if not can_contract_pass_through(left_to_node, node_to_right):
                continue
            if not can_contract_pass_through(right_to_node, node_to_left):
                continue

            contracted_node = node
            contraction_roads = (left_to_node, node_to_right, right_to_node, node_to_left)
            break

        if contracted_node is None or contraction_roads is None:
            break

        left_to_node, node_to_right, right_to_node, node_to_left = contraction_roads

        def merged_direction(first: dict, second: dict) -> dict:
            length_m = float(first["length_m"]) + float(second["length_m"])
            blocks, rounded_length_m = base.rounded_blocks(length_m)
            merged = dict(first)
            merged["end_node"] = second["end_node"]
            merged["length_m"] = f"{length_m:.3f}"
            merged["rounded_length_m"] = rounded_length_m
            merged["blocks_12m"] = blocks
            merged["source_road_ids"] = merge_words(first["source_road_ids"], second["source_road_ids"])
            merged["source_sequences"] = merge_words(first["source_sequences"], second["source_sequences"])
            merged["source_object_keys"] = append_words(first["source_object_keys"], second["source_object_keys"])
            merged["path_arc_ids"] = append_words(first["path_arc_ids"], second["path_arc_ids"])
            return merged

        merged_forward = merged_direction(left_to_node, node_to_right)
        merged_reverse = merged_direction(right_to_node, node_to_left)

        contractions.append(
            {
                "iteration": iteration,
                "node_id": contracted_node,
                "node_label": node_meta.get(contracted_node, {}).get("node_label", ""),
                "forward_incoming_road_id": left_to_node["id"],
                "forward_outgoing_road_id": node_to_right["id"],
                "forward_merged_road_id": merged_forward["id"],
                "reverse_incoming_road_id": right_to_node["id"],
                "reverse_outgoing_road_id": node_to_left["id"],
                "reverse_merged_road_id": merged_reverse["id"],
                "road_type": merged_forward["road_type"],
                "lanes": merged_forward["lanes"],
                "speed_limit": merged_forward["speed_limit"],
            }
        )

        remove_ids = {
            int(left_to_node["id"]),
            int(node_to_right["id"]),
            int(right_to_node["id"]),
            int(node_to_left["id"]),
        }
        current = [road for road in current if int(road["id"]) not in remove_ids]
        current.extend([merged_forward, merged_reverse])
        iteration += 1

    return current, contractions


def build_pruned_topology() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    edge_rows = base.read_csv(base.GRAPH_EDGES)
    graph_nodes = base.read_csv(base.GRAPH_NODES)
    base_model_nodes = {row["node_id"] for row in graph_nodes}
    arcs = base.build_directed_arcs(edge_rows)
    model_nodes = base.extend_model_nodes(arcs, base_model_nodes)
    sim_roads = base.collapse_to_sim_roads(arcs, model_nodes)
    graph_nodes = base.split_external_boundary_ports(sim_roads, graph_nodes)

    full_road_count = len(sim_roads)
    sim_roads, dropped_pruned_roads = prune_peripheral_local_roads(sim_roads)
    sim_roads, pass_through_contractions = collapse_pass_through_nodes(sim_roads, graph_nodes)
    sim_roads, bidirectional_pass_through_contractions = collapse_bidirectional_pass_through_nodes(sim_roads, graph_nodes)
    sim_roads, second_pass_through_contractions = collapse_pass_through_nodes(sim_roads, graph_nodes)
    pass_through_contractions.extend(second_pass_through_contractions)
    sim_roads, dropped_disconnected_roads = base.split_largest_road_component(sim_roads)
    graph_nodes = base.add_terminal_boundary_connectors(sim_roads, graph_nodes)
    base.reindex_sim_roads(sim_roads)
    graph_nodes = base.split_bidirectional_terminal_boundary_ports(sim_roads, graph_nodes)

    sim_nodes, junctions, boundaries = base.build_nodes_and_junctions(sim_roads, graph_nodes)
    apply_turning_matrix_overrides(junctions)
    boundary_ports = base.boundary_ports_from_sim_nodes(sim_nodes)
    road_connections = base.road_connection_rows(sim_roads, sim_nodes, junctions, boundaries)
    summary = base.validate_network(sim_roads, sim_nodes, junctions, boundaries, boundary_ports)
    summary["full_unpruned_candidate_roads"] = full_road_count
    summary["dropped_pruned_roads"] = len(dropped_pruned_roads)
    summary["contracted_pass_through_nodes"] = len(pass_through_contractions)
    summary["contracted_bidirectional_pass_through_nodes"] = len(bidirectional_pass_through_contractions)
    summary["dropped_disconnected_sim_roads"] = len(dropped_disconnected_roads)
    return (
        sim_roads,
        sim_nodes,
        junctions,
        boundaries,
        boundary_ports,
        road_connections,
        dropped_pruned_roads,
        pass_through_contractions,
        bidirectional_pass_through_contractions,
        dropped_disconnected_roads,
        summary,
    )


def apply_turning_matrix_overrides(junctions: list[dict]) -> None:
    for junction in junctions:
        override = TURNING_MATRIX_OVERRIDES.get(junction["node_label"])
        if override is None:
            continue
        if len(override) != len(junction["incoming"]):
            raise ValueError(
                f"Override for {junction['node_label']} has {len(override)} rows, "
                f"expected {len(junction['incoming'])}"
            )
        if override and len(override[0]) != len(junction["outgoing"]):
            raise ValueError(
                f"Override for {junction['node_label']} has {len(override[0])} columns, "
                f"expected {len(junction['outgoing'])}"
            )
        junction["turning_matrix"] = override
        junction["priority_ranks"] = base.priority_ranks(override, [{"lanes": 1.0, "id": road_id} for road_id in junction["incoming"]])


def write_outputs() -> None:
    (
        sim_roads,
        sim_nodes,
        junctions,
        boundaries,
        boundary_ports,
        road_connections,
        dropped_pruned_roads,
        pass_through_contractions,
        bidirectional_pass_through_contractions,
        dropped_disconnected_roads,
        summary,
    ) = build_pruned_topology()

    base.write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_roads.csv",
        sim_roads,
        [
            "id",
            "original_sim_road_id",
            "label",
            "start_node",
            "end_node",
            "length_m",
            "rounded_length_m",
            "blocks_12m",
            "lanes",
            "speed_limit",
            "road_type",
            "source_road_ids",
            "source_sequences",
            "source_object_keys",
            "path_arc_ids",
        ],
    )
    base.write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_nodes.csv",
        sim_nodes,
        ["node_id", "node_label", "node_type", "incoming_road_ids", "outgoing_road_ids", "degree_directed"],
    )
    base.write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_junctions.csv",
        [
            {
                "id": row["id"],
                "node_id": row["node_id"],
                "node_label": row["node_label"],
                "node_type": row["node_type"],
                "rule_type": row["rule_type"],
                "incoming": " ".join(str(value) for value in row["incoming"]),
                "outgoing": " ".join(str(value) for value in row["outgoing"]),
                "turning_matrix": row["turning_matrix"],
                "priority_ranks": row["priority_ranks"],
                "alpha": row["alpha"],
            }
            for row in junctions
        ],
        ["id", "node_id", "node_label", "node_type", "rule_type", "incoming", "outgoing", "turning_matrix", "priority_ranks", "alpha"],
    )
    base.write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_boundaries.csv",
        boundaries,
        ["boundary_id", "node_id", "node_label", "road_id", "side"],
    )
    base.write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_boundary_ports.csv",
        boundary_ports,
        ["boundary_port_id", "node_id", "node_label", "source_node", "road_id", "side"],
    )
    base.write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_road_connections.csv",
        road_connections,
        [
            "road_id",
            "original_sim_road_id",
            "road_label",
            "upstream_node_id",
            "upstream_node_label",
            "upstream_kind",
            "downstream_node_id",
            "downstream_node_label",
            "downstream_kind",
            "lanes",
            "speed_limit",
            "blocks_12m",
            "length_m",
            "rounded_length_m",
            "source_road_ids",
        ],
    )
    base.write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_dropped_pruned_roads.csv",
        dropped_pruned_roads,
        [
            "id",
            "label",
            "start_node",
            "end_node",
            "length_m",
            "rounded_length_m",
            "blocks_12m",
            "lanes",
            "speed_limit",
            "road_type",
            "source_road_ids",
            "source_sequences",
            "source_object_keys",
            "path_arc_ids",
            "prune_iteration",
            "dropped_reason",
        ],
    )
    base.write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_dropped_disconnected_roads.csv",
        dropped_disconnected_roads,
        [
            "id",
            "label",
            "start_node",
            "end_node",
            "length_m",
            "rounded_length_m",
            "blocks_12m",
            "lanes",
            "speed_limit",
            "road_type",
            "source_road_ids",
            "source_sequences",
            "source_object_keys",
            "path_arc_ids",
            "dropped_reason",
        ],
    )
    base.write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_contracted_pass_through_nodes.csv",
        pass_through_contractions,
        [
            "iteration",
            "node_id",
            "node_label",
            "incoming_road_id",
            "outgoing_road_id",
            "merged_road_id",
            "length_m",
            "incoming_road_type",
            "outgoing_road_type",
            "road_type",
            "incoming_lanes",
            "outgoing_lanes",
            "lanes",
            "incoming_speed_limit",
            "outgoing_speed_limit",
            "speed_limit",
        ],
    )
    base.write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_contracted_bidirectional_pass_through_nodes.csv",
        bidirectional_pass_through_contractions,
        [
            "iteration",
            "node_id",
            "node_label",
            "forward_incoming_road_id",
            "forward_outgoing_road_id",
            "forward_merged_road_id",
            "reverse_incoming_road_id",
            "reverse_outgoing_road_id",
            "reverse_merged_road_id",
            "road_type",
            "lanes",
            "speed_limit",
        ],
    )

    old_name = base.NETWORK_NAME
    base.NETWORK_NAME = NETWORK_NAME
    try:
        base.write_network_toml(OUTDIR / f"{OUTPUT_PREFIX}_network.toml", sim_roads, junctions, boundaries)
    finally:
        base.NETWORK_NAME = old_name

    base.write_csv(OUTDIR / f"{OUTPUT_PREFIX}_summary.csv", [summary], list(summary.keys()))


def main() -> None:
    write_outputs()


if __name__ == "__main__":
    main()
