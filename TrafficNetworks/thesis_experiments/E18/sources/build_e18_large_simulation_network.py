from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict, deque
from pathlib import Path


OUTDIR = Path(__file__).resolve().parent
GRAPH_EDGES = OUTDIR / "e18_large_graph_edges.csv"
GRAPH_NODES = OUTDIR / "e18_large_graph_nodes.csv"
OUTPUT_PREFIX = "e18_large_sim"
NETWORK_NAME = "e18_large_simulation_topology"

BASIS_LENGTH_M = 12
CELLS_PER_BLOCK = 1
DEFAULT_INITIAL_DENSITY = 0.05
BOUNDARY_INFLOW_DEFAULT = 0.0
BOUNDARY_INFLOW_ACTIVE = 0.06
BOUNDARY_ACTIVE_START_SECONDS = 60
BOUNDARY_ACTIVE_STOP_SECONDS = 180
HORIZON_SECONDS = 240
CONTROL_STEP_SECONDS = 15


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def toml_string(value: object) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def toml_array(values: list[object]) -> str:
    return "[" + ", ".join(toml_value(value) for value in values) + "]"


def toml_matrix(rows: list[list[object]]) -> str:
    return "[" + ", ".join(toml_array(row) for row in rows) + "]"


def toml_value(value: object) -> str:
    if isinstance(value, str):
        return toml_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return toml_matrix(value)
        return toml_array(value)
    raise TypeError(f"Unsupported TOML value: {value!r}")


def parse_lanes(feltoversikt: str) -> list[int]:
    lanes = []
    for item in str(feltoversikt or "").split("#"):
        if not item:
            continue
        digits = ""
        for char in item:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            lanes.append(int(digits))
    return lanes


def travel_directions(row: dict[str, str]) -> list[str]:
    mode = row["lane_mode"]
    road_type = row["typeVeg"]
    if mode.startswith("bidirectional"):
        return ["forward", "reverse"]
    if mode.startswith("against_link"):
        return ["reverse"]
    if mode.startswith("with_link"):
        return ["forward"]
    if mode == "unknown" and road_type == "Enkel bilveg":
        return ["forward", "reverse"]
    return ["forward"]


def lane_count(row: dict[str, str], direction: str) -> int:
    lanes = parse_lanes(row["feltoversikt"])
    if not lanes:
        return 1
    count = sum(
        1
        for lane in lanes
        if ("forward" if lane % 2 == 1 else "reverse") == direction
    )
    return max(count, 1)


def speed_limit(row: dict[str, str]) -> int:
    road_type = row["typeVeg"]
    kortform = row.get("kortform", "")
    if road_type == "Artificial connector":
        return 30
    if road_type == "Rampe":
        return 50
    if kortform.startswith(("EV", "RV")):
        return 80
    if kortform.startswith("FV"):
        return 60
    return 40


def rounded_blocks(length_m: float) -> tuple[int, int]:
    blocks = max(1, int(math.floor(length_m / BASIS_LENGTH_M + 0.5)))
    return blocks, blocks * BASIS_LENGTH_M


def build_directed_arcs(edge_rows: list[dict[str, str]]) -> list[dict]:
    arcs = []
    for row in edge_rows:
        for direction in travel_directions(row):
            if direction == "forward":
                start, end = row["startnode"], row["sluttnode"]
            else:
                start, end = row["sluttnode"], row["startnode"]

            arcs.append(
                {
                    "arc_id": len(arcs) + 1,
                    "start": start,
                    "end": end,
                    "direction": direction,
                    "length_m": float(row["length_m"]),
                    "lanes": lane_count(row, direction),
                    "speed_limit": speed_limit(row),
                    "source_road_id": row["road_id"],
                    "source_object_key": row["object_key"],
                    "source_sequence": row["veglenkesekvensid"],
                    "road_type": row["typeVeg"],
                }
            )
    return arcs


def extend_model_nodes(arcs: list[dict], base_model_nodes: set[str]) -> set[str]:
    neighbours = defaultdict(set)
    for arc in arcs:
        neighbours[arc["start"]].add(arc["end"])
        neighbours[arc["end"]].add(arc["start"])
    extra = {node for node, node_neighbours in neighbours.items() if len(node_neighbours) != 2}
    return set(base_model_nodes) | extra


def choose_continuation(candidates: list[dict], previous_node: str, previous_arc: dict) -> dict | None:
    forward_candidates = [arc for arc in candidates if arc["end"] != previous_node]
    if not forward_candidates:
        return None
    same_road = [arc for arc in forward_candidates if arc["source_road_id"] == previous_arc["source_road_id"]]
    if same_road:
        return sorted(same_road, key=lambda arc: arc["arc_id"])[0]
    return sorted(forward_candidates, key=lambda arc: arc["arc_id"])[0]


def collapse_to_sim_roads(arcs: list[dict], model_nodes: set[str]) -> list[dict]:
    outgoing = defaultdict(list)
    for arc in arcs:
        outgoing[arc["start"]].append(arc)

    used = set()
    sim_roads = []
    for arc in sorted(arcs, key=lambda item: item["arc_id"]):
        if arc["arc_id"] in used or arc["start"] not in model_nodes:
            continue

        path = [arc]
        used.add(arc["arc_id"])
        start_node = arc["start"]
        previous_node = arc["start"]
        current_node = arc["end"]

        while current_node not in model_nodes:
            continuation = choose_continuation(
                [candidate for candidate in outgoing[current_node] if candidate["arc_id"] not in used],
                previous_node,
                path[-1],
            )
            if continuation is None:
                model_nodes.add(current_node)
                break
            path.append(continuation)
            used.add(continuation["arc_id"])
            previous_node, current_node = current_node, continuation["end"]

        length_m = sum(part["length_m"] for part in path)
        blocks, rounded_length_m = rounded_blocks(length_m)
        road_types = [part["road_type"] for part in path]
        source_road_ids = sorted({part["source_road_id"] for part in path})
        sim_roads.append(
            {
                "id": len(sim_roads) + 1,
                "label": f"R{len(sim_roads) + 1:03d}: {start_node} -> {current_node}",
                "start_node": start_node,
                "end_node": current_node,
                "length_m": f"{length_m:.3f}",
                "rounded_length_m": rounded_length_m,
                "blocks_12m": blocks,
                "lanes": min(part["lanes"] for part in path),
                "speed_limit": min(part["speed_limit"] for part in path),
                "road_type": Counter(road_types).most_common(1)[0][0],
                "source_road_ids": " ".join(source_road_ids),
                "source_sequences": " ".join(sorted({part["source_sequence"] for part in path})),
                "source_object_keys": " ".join(part["source_object_key"] for part in path),
                "path_arc_ids": " ".join(str(part["arc_id"]) for part in path),
            }
        )

    return sim_roads


def source_ids(road: dict) -> set[str]:
    return set(str(road["source_road_ids"]).split())


def normalize_turning_row(row: list[float]) -> list[float]:
    row_sum = sum(row)
    if row_sum > 1.0:
        return [value / row_sum for value in row]
    return row


def turning_matrix(incoming: list[dict], outgoing: list[dict]) -> list[list[float]]:
    matrix = []
    for road_in in incoming:
        if not outgoing:
            matrix.append([])
            continue
        if len(outgoing) == 1:
            matrix.append([1.0])
            continue

        continuation = [
            idx for idx, road_out in enumerate(outgoing)
            if source_ids(road_in) & source_ids(road_out)
        ]
        row = [0.0 for _ in outgoing]
        if continuation:
            continuation_share = 0.8 / len(continuation)
            for idx in continuation:
                row[idx] = continuation_share
            remaining = [idx for idx in range(len(outgoing)) if idx not in continuation]
            if remaining:
                for idx in remaining:
                    row[idx] = 0.2 / len(remaining)
            else:
                row = [value / sum(row) for value in row]
        else:
            row = [1.0 / len(outgoing) for _ in outgoing]
        matrix.append(normalize_turning_row(row))
    return matrix


def priority_ranks(P: list[list[float]], incoming: list[dict]) -> list[list[int]]:
    if not P:
        return []
    nout = len(P[0])
    ranks = [[0 for _ in range(nout)] for _ in incoming]
    for col in range(nout):
        legal = [idx for idx, row in enumerate(P) if row[col] > 0.0]
        ordered = sorted(legal, key=lambda idx: (-incoming[idx]["lanes"], incoming[idx]["id"]))
        for rank, idx in enumerate(ordered, start=1):
            ranks[idx][col] = rank
    return ranks


def build_nodes_and_junctions(sim_roads: list[dict], graph_nodes: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    node_meta = {
        row["node_id"]: row
        for row in graph_nodes
    }
    all_nodes = sorted({road["start_node"] for road in sim_roads} | {road["end_node"] for road in sim_roads})
    incoming_by_node = defaultdict(list)
    outgoing_by_node = defaultdict(list)
    for road in sim_roads:
        outgoing_by_node[road["start_node"]].append(road)
        incoming_by_node[road["end_node"]].append(road)

    sim_nodes = []
    junctions = []
    boundaries = []
    junction_index = 1
    boundary_index = 1

    for node in all_nodes:
        incoming = sorted(incoming_by_node[node], key=lambda road: road["id"])
        outgoing = sorted(outgoing_by_node[node], key=lambda road: road["id"])
        original_type = node_meta.get(node, {}).get("node_type", "")
        original_label = node_meta.get(node, {}).get("node_label", "")
        bidirectional_terminal = (
            original_type != "roundabout"
            and len(incoming) == 1
            and len(outgoing) == 1
            and incoming[0]["start_node"] == outgoing[0]["end_node"]
        )

        if (
            original_type == "boundary"
            or bidirectional_terminal
            or (original_type != "roundabout" and (len(incoming) == 0 or len(outgoing) == 0))
        ):
            node_type = "boundary"
            node_label = original_label or f"B{boundary_index:02d}"
            boundary_index += 1
        else:
            node_type = "roundabout" if original_type == "roundabout" else "junction"
            node_label = original_label or f"J{junction_index:02d}"
            junction_index += 1

        sim_nodes.append(
            {
                "node_id": node,
                "node_label": node_label,
                "node_type": node_type,
                "incoming_road_ids": " ".join(str(road["id"]) for road in incoming),
                "outgoing_road_ids": " ".join(str(road["id"]) for road in outgoing),
                "degree_directed": len(incoming) + len(outgoing),
            }
        )

        if node_type == "boundary":
            for road in outgoing:
                boundaries.append(
                    {
                        "boundary_id": len(boundaries) + 1,
                        "node_id": node,
                        "node_label": node_label,
                        "road_id": road["id"],
                        "side": "upstream",
                    }
                )
        else:
            P = turning_matrix(incoming, outgoing)
            ranks = priority_ranks(P, incoming)
            junctions.append(
                {
                    "id": len(junctions) + 1,
                    "node_id": node,
                    "node_label": node_label,
                    "node_type": node_type,
                    "rule_type": "turning",
                    "incoming": [road["id"] for road in incoming],
                    "outgoing": [road["id"] for road in outgoing],
                    "turning_matrix": P,
                    "priority_ranks": ranks,
                    "alpha": 0.3,
                }
            )

    return sim_nodes, junctions, boundaries


def boundary_ports_from_sim_nodes(sim_nodes: list[dict]) -> list[dict]:
    ports = []
    for node in sim_nodes:
        if node["node_type"] != "boundary":
            continue

        source_node = str(node["node_id"]).split(":", 1)[0]
        for road_id in str(node["incoming_road_ids"]).split():
            ports.append(
                {
                    "boundary_port_id": len(ports) + 1,
                    "node_id": node["node_id"],
                    "node_label": node["node_label"],
                    "source_node": source_node,
                    "road_id": road_id,
                    "side": "downstream",
                }
            )
        for road_id in str(node["outgoing_road_ids"]).split():
            ports.append(
                {
                    "boundary_port_id": len(ports) + 1,
                    "node_id": node["node_id"],
                    "node_label": node["node_label"],
                    "source_node": source_node,
                    "road_id": road_id,
                    "side": "upstream",
                }
            )
    return ports


def split_external_boundary_ports(sim_roads: list[dict], graph_nodes: list[dict]) -> list[dict]:
    node_meta = {row["node_id"]: row for row in graph_nodes}
    boundary_nodes = {row["node_id"] for row in graph_nodes if row["node_type"] == "boundary"}
    extended_nodes = list(graph_nodes)

    for road in sim_roads:
        original_start = road["start_node"]
        original_end = road["end_node"]
        if original_start in boundary_nodes:
            meta = node_meta[original_start]
            port_id = f"{original_start}:upstream:{road['id']}"
            port_label = f"{meta['node_label']}u"
            road["start_node"] = port_id
            road["upstream_boundary_source_node"] = original_start
            extended_nodes.append(
                {
                    "node_id": port_id,
                    "node_label": port_label,
                    "node_type": "boundary",
                    "x": meta.get("x", ""),
                    "y": meta.get("y", ""),
                    "incident_road_ids": road["id"],
                    "incident_sequences": "",
                }
            )
        if original_end in boundary_nodes:
            meta = node_meta[original_end]
            port_id = f"{original_end}:downstream:{road['id']}"
            port_label = f"{meta['node_label']}d"
            road["end_node"] = port_id
            road["downstream_boundary_source_node"] = original_end
            extended_nodes.append(
                {
                    "node_id": port_id,
                    "node_label": port_label,
                    "node_type": "boundary",
                    "x": meta.get("x", ""),
                    "y": meta.get("y", ""),
                    "incident_road_ids": road["id"],
                    "incident_sequences": "",
                }
            )

    return extended_nodes


def add_terminal_boundary_connectors(sim_roads: list[dict], graph_nodes: list[dict]) -> list[dict]:
    node_meta = {row["node_id"]: row for row in graph_nodes}
    incoming_by_node = defaultdict(list)
    outgoing_by_node = defaultdict(list)
    for road in sim_roads:
        outgoing_by_node[road["start_node"]].append(road)
        incoming_by_node[road["end_node"]].append(road)

    extended_nodes = list(graph_nodes)
    next_road_id = max(road["id"] for road in sim_roads) + 1 if sim_roads else 1
    terminal_index = 1

    all_nodes = sorted(set(incoming_by_node) | set(outgoing_by_node))
    for node in all_nodes:
        meta = node_meta.get(node, {})
        original_type = meta.get("node_type", "")
        if original_type == "boundary":
            continue

        incoming = incoming_by_node[node]
        outgoing = outgoing_by_node[node]
        if incoming and outgoing:
            continue

        if not incoming and len(outgoing) > 1:
            port_id = f"{node}:terminal_upstream"
            port_label = f"{meta.get('node_label', 'T')}u"
            lanes = max(road["lanes"] for road in outgoing)
            connector = {
                "id": next_road_id,
                "label": f"R{next_road_id:03d}: {port_id} -> {node}",
                "start_node": port_id,
                "end_node": node,
                "length_m": f"{BASIS_LENGTH_M:.3f}",
                "rounded_length_m": BASIS_LENGTH_M,
                "blocks_12m": 1,
                "lanes": lanes,
                "speed_limit": 30,
                "road_type": "Terminal boundary connector",
                "source_road_ids": f"terminal_{terminal_index:03d}",
                "source_sequences": "",
                "source_object_keys": "",
                "path_arc_ids": "",
            }
            sim_roads.append(connector)
            next_road_id += 1
            terminal_index += 1
            extended_nodes.append(
                {
                    "node_id": port_id,
                    "node_label": port_label,
                    "node_type": "boundary",
                    "x": meta.get("x", ""),
                    "y": meta.get("y", ""),
                    "incident_road_ids": connector["id"],
                    "incident_sequences": "",
                }
            )

        if not outgoing and len(incoming) > 1:
            port_id = f"{node}:terminal_downstream"
            port_label = f"{meta.get('node_label', 'T')}d"
            lanes = max(road["lanes"] for road in incoming)
            connector = {
                "id": next_road_id,
                "label": f"R{next_road_id:03d}: {node} -> {port_id}",
                "start_node": node,
                "end_node": port_id,
                "length_m": f"{BASIS_LENGTH_M:.3f}",
                "rounded_length_m": BASIS_LENGTH_M,
                "blocks_12m": 1,
                "lanes": lanes,
                "speed_limit": 30,
                "road_type": "Terminal boundary connector",
                "source_road_ids": f"terminal_{terminal_index:03d}",
                "source_sequences": "",
                "source_object_keys": "",
                "path_arc_ids": "",
            }
            sim_roads.append(connector)
            next_road_id += 1
            terminal_index += 1
            extended_nodes.append(
                {
                    "node_id": port_id,
                    "node_label": port_label,
                    "node_type": "boundary",
                    "x": meta.get("x", ""),
                    "y": meta.get("y", ""),
                    "incident_road_ids": connector["id"],
                    "incident_sequences": "",
                }
            )

    return extended_nodes


def split_bidirectional_terminal_boundary_ports(sim_roads: list[dict], graph_nodes: list[dict]) -> list[dict]:
    node_meta = {row["node_id"]: row for row in graph_nodes}
    incoming_by_node = defaultdict(list)
    outgoing_by_node = defaultdict(list)
    for road in sim_roads:
        outgoing_by_node[road["start_node"]].append(road)
        incoming_by_node[road["end_node"]].append(road)

    extended_nodes = list(graph_nodes)
    all_nodes = sorted(set(incoming_by_node) | set(outgoing_by_node))
    for node in all_nodes:
        if ":" in str(node):
            continue
        meta = node_meta.get(node, {})
        if meta.get("node_type") in ("boundary", "roundabout"):
            continue

        incoming = incoming_by_node[node]
        outgoing = outgoing_by_node[node]
        if len(incoming) != 1 or len(outgoing) != 1:
            continue
        if incoming[0]["start_node"] != outgoing[0]["end_node"]:
            continue

        incoming_road = incoming[0]
        outgoing_road = outgoing[0]
        downstream_port_id = f"{node}:terminal_downstream:{incoming_road['id']}"
        upstream_port_id = f"{node}:terminal_upstream:{outgoing_road['id']}"
        base_label = meta.get("node_label", "T")

        incoming_road["end_node"] = downstream_port_id
        incoming_road["downstream_boundary_source_node"] = node
        outgoing_road["start_node"] = upstream_port_id
        outgoing_road["upstream_boundary_source_node"] = node

        extended_nodes.extend(
            [
                {
                    "node_id": downstream_port_id,
                    "node_label": f"{base_label}d",
                    "node_type": "boundary",
                    "x": meta.get("x", ""),
                    "y": meta.get("y", ""),
                    "incident_road_ids": incoming_road["id"],
                    "incident_sequences": "",
                },
                {
                    "node_id": upstream_port_id,
                    "node_label": f"{base_label}u",
                    "node_type": "boundary",
                    "x": meta.get("x", ""),
                    "y": meta.get("y", ""),
                    "incident_road_ids": outgoing_road["id"],
                    "incident_sequences": "",
                },
            ]
        )

    return extended_nodes


def write_network_toml(path: Path, sim_roads: list[dict], junctions: list[dict], boundaries: list[dict]) -> None:
    lines = [
        f"name = {toml_value(NETWORK_NAME)}",
        "",
        "[simulation]",
        f"horizon_seconds = {HORIZON_SECONDS}",
        f"control_step_seconds = {CONTROL_STEP_SECONDS}",
        "cfl = 0.5",
        "",
        "[discretization]",
        f"basis_length_m = {BASIS_LENGTH_M}",
        f"cells_per_block = {CELLS_PER_BLOCK}",
        "",
        "[initial_condition]",
        f"density = {DEFAULT_INITIAL_DENSITY}",
        "",
        "[boundary_signal]",
        f"default = {BOUNDARY_INFLOW_DEFAULT}",
        f"active_value = {BOUNDARY_INFLOW_ACTIVE}",
        f"active_start_seconds = {BOUNDARY_ACTIVE_START_SECONDS}",
        f"active_stop_seconds = {BOUNDARY_ACTIVE_STOP_SECONDS}",
    ]

    for road in sim_roads:
        lines.extend(
            [
                "",
                "[[roads]]",
                f"id = {road['id']}",
                f"label = {toml_value(road['label'])}",
                f"start_node = {toml_value(road['start_node'])}",
                f"end_node = {toml_value(road['end_node'])}",
                f"blocks = {road['blocks_12m']}",
                f"length_m = {road['rounded_length_m']}",
                f"original_length_m = {road['length_m']}",
                f"lanes = {road['lanes']}",
                f"speed_limit = {road['speed_limit']}",
                f"road_type = {toml_value(road['road_type'])}",
                f"source_road_ids = {toml_value(road['source_road_ids'])}",
            ]
        )

    for junction in junctions:
        lines.extend(
            [
                "",
                "[[junctions]]",
                f"id = {junction['id']}",
                f"node_id = {toml_value(junction['node_id'])}",
                f"node_label = {toml_value(junction['node_label'])}",
                f"node_type = {toml_value(junction['node_type'])}",
                f"rule_type = {toml_value(junction['rule_type'])}",
                f"incoming = {toml_value(junction['incoming'])}",
                f"outgoing = {toml_value(junction['outgoing'])}",
                f"turning_matrix = {toml_value(junction['turning_matrix'])}",
                f"priority_ranks = {toml_value(junction['priority_ranks'])}",
                f"alpha = {junction['alpha']}",
            ]
        )

    for boundary in boundaries:
        lines.extend(
            [
                "",
                "[[boundaries]]",
                f"id = {boundary['boundary_id']}",
                f"node_id = {toml_value(boundary['node_id'])}",
                f"node_label = {toml_value(boundary['node_label'])}",
                f"road_id = {boundary['road_id']}",
                f"side = {toml_value(boundary['side'])}",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sim_weak_components(sim_roads: list[dict]) -> list[set[str]]:
    adjacency = defaultdict(set)
    road_nodes = set()
    for road in sim_roads:
        start = road["start_node"]
        end = road["end_node"]
        adjacency[start].add(end)
        adjacency[end].add(start)
        road_nodes.add(start)
        road_nodes.add(end)

    components = []
    remaining = set(road_nodes)
    while remaining:
        start = remaining.pop()
        component = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(component)

    components.sort(key=len, reverse=True)
    return components


def component_road_counts(sim_roads: list[dict], components: list[set[str]]) -> list[int]:
    counts = []
    for component in components:
        count = 0
        for road in sim_roads:
            if road["start_node"] in component or road["end_node"] in component:
                count += 1
        counts.append(count)
    return counts


def split_largest_road_component(sim_roads: list[dict]) -> tuple[list[dict], list[dict]]:
    components = sim_weak_components(sim_roads)
    if len(components) <= 1:
        return sim_roads, []

    component_roads = []
    for component in components:
        rows = [
            road
            for road in sim_roads
            if road["start_node"] in component or road["end_node"] in component
        ]
        component_roads.append(rows)

    keep_index = max(range(len(component_roads)), key=lambda idx: len(component_roads[idx]))
    kept = component_roads[keep_index]
    dropped = [
        dict(road, dropped_reason="outside_largest_simulation_component")
        for idx, rows in enumerate(component_roads)
        if idx != keep_index
        for road in rows
    ]
    kept_ids = {road["id"] for road in kept}
    return [road for road in sim_roads if road["id"] in kept_ids], dropped


def reindex_sim_roads(sim_roads: list[dict]) -> dict[int, int]:
    old_to_new = {}
    for new_id, road in enumerate(sorted(sim_roads, key=lambda item: item["id"]), start=1):
        old_id = int(road["id"])
        old_to_new[old_id] = new_id
        road["original_sim_road_id"] = old_id
        road["id"] = new_id
        road["label"] = f"R{new_id:03d}: {road['start_node']} -> {road['end_node']}"
    return old_to_new


def road_connection_rows(sim_roads: list[dict], sim_nodes: list[dict], junctions: list[dict], boundaries: list[dict]) -> list[dict]:
    node_meta = {node["node_id"]: node for node in sim_nodes}
    junction_nodes = {junction["node_id"] for junction in junctions}
    upstream_boundary_roads = {str(boundary["road_id"]) for boundary in boundaries}
    rows = []
    for road in sim_roads:
        road_id = str(road["id"])
        start_node = road["start_node"]
        end_node = road["end_node"]
        start_meta = node_meta.get(start_node, {})
        end_meta = node_meta.get(end_node, {})

        if road_id in upstream_boundary_roads:
            upstream_kind = "boundary"
        elif start_node in junction_nodes:
            upstream_kind = start_meta.get("node_type", "junction")
        else:
            upstream_kind = "missing"

        if end_node in junction_nodes:
            downstream_kind = end_meta.get("node_type", "junction")
        elif end_meta.get("node_type") == "boundary":
            downstream_kind = "boundary"
        else:
            downstream_kind = "missing"

        rows.append(
            {
                "road_id": road["id"],
                "original_sim_road_id": road.get("original_sim_road_id", road["id"]),
                "road_label": road["label"],
                "upstream_node_id": start_node,
                "upstream_node_label": start_meta.get("node_label", ""),
                "upstream_kind": upstream_kind,
                "downstream_node_id": end_node,
                "downstream_node_label": end_meta.get("node_label", ""),
                "downstream_kind": downstream_kind,
                "lanes": road["lanes"],
                "speed_limit": road["speed_limit"],
                "blocks_12m": road["blocks_12m"],
                "length_m": road["length_m"],
                "rounded_length_m": road["rounded_length_m"],
                "source_road_ids": road["source_road_ids"],
            }
        )
    return rows


def validate_network(
    sim_roads: list[dict],
    sim_nodes: list[dict],
    junctions: list[dict],
    boundaries: list[dict],
    boundary_ports: list[dict],
) -> dict:
    junction_nodes = {junction["node_id"] for junction in junctions}
    boundary_nodes = {node["node_id"] for node in sim_nodes if node["node_type"] == "boundary"}
    upstream_boundary_roads = {boundary["road_id"] for boundary in boundaries}
    road_errors = []
    for road in sim_roads:
        start_ok = road["start_node"] in junction_nodes or road["id"] in upstream_boundary_roads
        end_ok = road["end_node"] in junction_nodes or road["end_node"] in boundary_nodes
        if not start_ok or not end_ok:
            road_errors.append(str(road["id"]))

    boundary_incident_counts = []
    for node in sim_nodes:
        if node["node_type"] != "boundary":
            continue
        incident = len(node["incoming_road_ids"].split()) + len(node["outgoing_road_ids"].split())
        boundary_incident_counts.append(incident)

    components = sim_weak_components(sim_roads)
    road_counts = component_road_counts(sim_roads, components)

    return {
        "sim_roads": len(sim_roads),
        "sim_weak_components": len(components),
        "sim_component_road_counts": " ".join(str(value) for value in road_counts),
        "junctions": len(junctions),
        "roundabout_junctions": sum(1 for junction in junctions if junction["node_type"] == "roundabout"),
        "regular_junctions": sum(1 for junction in junctions if junction["node_type"] == "junction"),
        "inflow_boundaries": len(boundaries),
        "boundary_ports": len(boundary_ports),
        "upstream_boundary_ports": sum(1 for port in boundary_ports if port["side"] == "upstream"),
        "downstream_boundary_ports": sum(1 for port in boundary_ports if port["side"] == "downstream"),
        "boundary_nodes": len(boundary_nodes),
        "invalid_road_connections": len(road_errors),
        "invalid_road_ids": " ".join(road_errors),
        "boundary_nodes_with_one_directed_road": sum(1 for count in boundary_incident_counts if count == 1),
        "boundary_nodes_with_multiple_directed_roads": sum(1 for count in boundary_incident_counts if count > 1),
    }


def main() -> None:
    edge_rows = read_csv(GRAPH_EDGES)
    graph_nodes = read_csv(GRAPH_NODES)
    base_model_nodes = {row["node_id"] for row in graph_nodes}
    arcs = build_directed_arcs(edge_rows)
    model_nodes = extend_model_nodes(arcs, base_model_nodes)
    sim_roads = collapse_to_sim_roads(arcs, model_nodes)
    graph_nodes = split_external_boundary_ports(sim_roads, graph_nodes)
    graph_nodes = add_terminal_boundary_connectors(sim_roads, graph_nodes)
    sim_roads, dropped_disconnected_roads = split_largest_road_component(sim_roads)
    reindex_sim_roads(sim_roads)
    graph_nodes = split_bidirectional_terminal_boundary_ports(sim_roads, graph_nodes)
    sim_nodes, junctions, boundaries = build_nodes_and_junctions(sim_roads, graph_nodes)
    boundary_ports = boundary_ports_from_sim_nodes(sim_nodes)
    road_connections = road_connection_rows(sim_roads, sim_nodes, junctions, boundaries)

    write_csv(
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
    write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_nodes.csv",
        sim_nodes,
        ["node_id", "node_label", "node_type", "incoming_road_ids", "outgoing_road_ids", "degree_directed"],
    )
    write_csv(
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
    write_csv(
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
    write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_boundaries.csv",
        boundaries,
        ["boundary_id", "node_id", "node_label", "road_id", "side"],
    )
    write_csv(
        OUTDIR / f"{OUTPUT_PREFIX}_boundary_ports.csv",
        boundary_ports,
        ["boundary_port_id", "node_id", "node_label", "source_node", "road_id", "side"],
    )
    write_csv(
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
    write_network_toml(OUTDIR / f"{OUTPUT_PREFIX}_network.toml", sim_roads, junctions, boundaries)
    summary = validate_network(sim_roads, sim_nodes, junctions, boundaries, boundary_ports)
    summary["dropped_disconnected_sim_roads"] = len(dropped_disconnected_roads)
    write_csv(OUTDIR / f"{OUTPUT_PREFIX}_summary.csv", [summary], list(summary.keys()))


if __name__ == "__main__":
    main()
