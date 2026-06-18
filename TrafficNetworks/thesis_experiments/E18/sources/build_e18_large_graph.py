from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUTDIR = Path(__file__).resolve().parent
SOURCE_DIR = OUTDIR / "source_json"


def source_sort_key(path: Path) -> tuple[str, int | str]:
    match = re.search(r"(\d+)$", path.stem)
    return (path.stem[: match.start(1)] if match else path.stem, int(match.group(1)) if match else path.stem)


API_JSON_FILES = sorted(SOURCE_DIR.glob("E18_*.json"), key=source_sort_key)

NODE_TRIM_M = 8.0
BIDIRECTIONAL_OFFSET_M = 3.4
SHORT_CONNECTOR_M = 30.0
ARTIFICIAL_CONNECTOR_MAX_M = 100.0
FINAL_PREFIX = "e18_large_graph"
SUPER_LANE_KEY = "kj\u00f8refelt"
DROP_ROAD_IDS = {
    # Local branches removed after inspecting the first large E18 draft.
    "r1183",
    "r1199",
    "r1194",
    "r1207",
    "r1208",
    "r1067",
    "r892",
    "r888",
    "r891",
    "r700",
    "r709",
    "r206",
    "r207",
    "r200",
}
FORCE_BOUNDARY_NODE_IDS = {
    # J655 and J659 in the labelled graph. These are model boundary endpoints,
    # not an internal artificial connection.
    "2814044",
    "457714",
}
BLOCK_ARTIFICIAL_CONNECTOR_NODE_PAIRS = {
    frozenset(("2814044", "457714")),
}

ROAD_COLORS = {
    "Enkel bilveg": "#287271",
    "Kanalisert veg": "#d98c24",
    "Rampe": "#7a5195",
    "Rundkjøring": "#5f6f52",
    "Artificial connector": "#3f3f3f",
}


def parse_linestring(wkt: str) -> list[tuple[float, float]]:
    match = re.search(r"\((.*)\)", wkt)
    if not match:
        return []
    coords = []
    for part in match.group(1).split(","):
        x, y = part.strip().split()[:2]
        coords.append((float(x), float(y)))
    return coords


def lane_mode(feltoversikt) -> str:
    numbers = []
    has_special = False
    for lane in feltoversikt or []:
        lane_text = str(lane)
        match = re.match(r"^(\d+)", lane_text)
        if match:
            numbers.append(int(match.group(1)))
        if not lane_text.isdigit():
            has_special = True

    odd = any(number % 2 == 1 for number in numbers)
    even = any(number % 2 == 0 for number in numbers)
    if odd and even:
        base = "bidirectional"
    elif odd:
        base = "with_link"
    elif even:
        base = "against_link"
    else:
        base = "unknown"
    return f"{base}_special" if has_special and base != "unknown" else base


def driving_lanes(obj: dict) -> list:
    lanes = obj.get("feltoversikt") or []
    if lanes:
        return lanes
    return (obj.get("superstedfesting") or {}).get(SUPER_LANE_KEY) or []


def travel_directions(obj: dict) -> list[str]:
    mode = lane_mode(driving_lanes(obj))
    road_type = obj.get("typeVeg", "")
    if mode.startswith("bidirectional"):
        return ["forward", "reverse"]
    if mode.startswith("against_link"):
        return ["reverse"]
    if mode.startswith("with_link"):
        return ["forward"]
    if mode == "unknown" and road_type == "Enkel bilveg":
        return ["forward", "reverse"]
    return ["forward"]


def segment_lengths(coords: list[tuple[float, float]]) -> list[float]:
    return [
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(coords, coords[1:])
    ]


def polyline_length(coords: list[tuple[float, float]]) -> float:
    return sum(segment_lengths(coords))


def point_at_distance(coords: list[tuple[float, float]], distance: float) -> tuple[float, float]:
    lengths = segment_lengths(coords)
    total = sum(lengths)
    if distance <= 0:
        return coords[0]
    if distance >= total:
        return coords[-1]

    travelled = 0.0
    for (x1, y1), (x2, y2), length in zip(coords, coords[1:], lengths):
        if length == 0:
            continue
        if travelled + length >= distance:
            local = (distance - travelled) / length
            return (x1 + local * (x2 - x1), y1 + local * (y2 - y1))
        travelled += length
    return coords[-1]


def trim_polyline(
    coords: list[tuple[float, float]], trim_start: float, trim_end: float
) -> list[tuple[float, float]]:
    total = sum(segment_lengths(coords))
    if total <= trim_start + trim_end + 1e-6:
        return coords

    start_point = point_at_distance(coords, trim_start)
    end_point = point_at_distance(coords, total - trim_end)
    trimmed = [start_point]

    travelled = 0.0
    for point, next_point, length in zip(coords, coords[1:], segment_lengths(coords)):
        next_travelled = travelled + length
        if trim_start < travelled < total - trim_end:
            trimmed.append(point)
        if trim_start < next_travelled < total - trim_end:
            trimmed.append(next_point)
        travelled = next_travelled

    trimmed.append(end_point)
    cleaned = [trimmed[0]]
    for point in trimmed[1:]:
        if math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) > 1e-6:
            cleaned.append(point)
    return cleaned


def right_normals(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    normals = []
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        normals.append((dy / length, -dx / length) if length else (0.0, 0.0))

    point_normals = []
    for index in range(len(coords)):
        candidates = []
        if index > 0:
            candidates.append(normals[index - 1])
        if index < len(normals):
            candidates.append(normals[index])
        nx = sum(normal[0] for normal in candidates)
        ny = sum(normal[1] for normal in candidates)
        length = math.hypot(nx, ny)
        point_normals.append((nx / length, ny / length) if length else (0.0, 0.0))
    return point_normals


def offset_polyline(coords: list[tuple[float, float]], distance: float) -> list[tuple[float, float]]:
    return [
        (x + distance * nx, y + distance * ny)
        for (x, y), (nx, ny) in zip(coords, right_normals(coords))
    ]


def point_and_tangent(
    coords: list[tuple[float, float]], fraction: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    total = sum(segment_lengths(coords))
    target = total * fraction
    travelled = 0.0
    for (x1, y1), (x2, y2), length in zip(coords, coords[1:], segment_lengths(coords)):
        if length == 0:
            continue
        if travelled + length >= target:
            local = (target - travelled) / length
            point = (x1 + local * (x2 - x1), y1 + local * (y2 - y1))
            tangent = ((x2 - x1) / length, (y2 - y1) / length)
            return point, tangent
        travelled += length
    x1, y1 = coords[-2]
    x2, y2 = coords[-1]
    length = math.hypot(x2 - x1, y2 - y1)
    return coords[-1], ((x2 - x1) / length, (y2 - y1) / length)


def draw_arrow(ax, coords: list[tuple[float, float]], color: str, alpha: float) -> None:
    if len(coords) < 2 or sum(segment_lengths(coords)) < 10:
        return
    (x, y), (tx, ty) = point_and_tangent(coords, 0.68)
    arrow_length = 15.0
    ax.annotate(
        "",
        xy=(x + tx * arrow_length * 0.55, y + ty * arrow_length * 0.55),
        xytext=(x - tx * arrow_length * 0.45, y - ty * arrow_length * 0.45),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "alpha": alpha,
            "lw": 1.15,
            "mutation_scale": 9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=4,
    )


def object_key(obj: dict) -> str:
    return (
        f"{obj.get('veglenkesekvensid')}-{obj.get('veglenkenummer')}-"
        f"{obj.get('segmentnummer')}:{obj.get('startnode')}>{obj.get('sluttnode')}"
    )


def load_api_objects() -> tuple[list[dict], dict]:
    raw_objects = []
    source_object_counts = {}
    next_page_flags = {}
    for path in API_JSON_FILES:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        objects = data.get("objekter", [])
        source_object_counts[path.name] = len(objects)
        next_page_flags[path.name] = bool((data.get("metadata") or {}).get("neste"))
        for obj in objects:
            raw_objects.append((path.name, obj))

    deduplicated_objects = {}
    source_files_by_key = defaultdict(list)
    for source_file, obj in raw_objects:
        key = object_key(obj)
        if key not in deduplicated_objects:
            deduplicated_objects[key] = obj
        source_files_by_key[key].append(source_file)

    objects = list(deduplicated_objects.values())

    road_id_by_sequence = {}
    for obj in objects:
        sequence_id = str(obj.get("veglenkesekvensid", ""))
        if sequence_id not in road_id_by_sequence:
            road_id_by_sequence[sequence_id] = f"r{len(road_id_by_sequence) + 1:03d}"

    rows = []
    for obj in objects:
        coords = parse_linestring((obj.get("geometri") or {}).get("wkt", ""))
        if len(coords) < 2:
            continue
        sequence_id = str(obj.get("veglenkesekvensid", ""))
        vegref = obj.get("vegsystemreferanse") or {}
        strekning = vegref.get("strekning") or {}
        rows.append(
            {
                "object_key": object_key(obj),
                "road_id": road_id_by_sequence[sequence_id],
                "veglenkesekvensid": sequence_id,
                "veglenkenummer": str(obj.get("veglenkenummer", "")),
                "segmentnummer": str(obj.get("segmentnummer", "")),
                "startnode": str(obj.get("startnode", "")),
                "sluttnode": str(obj.get("sluttnode", "")),
                "typeVeg": obj.get("typeVeg", ""),
                "street_name": (obj.get("adresse") or {}).get("navn", ""),
                "kortform": vegref.get("kortform", obj.get("kortform", "")),
                "lenkeretning": strekning.get("retning", ""),
                "length_m": obj.get("lengde", ""),
                "feltoversikt": "#".join(str(item) for item in driving_lanes(obj)),
                "lane_mode": lane_mode(driving_lanes(obj)),
                "travel_directions": travel_directions(obj),
                "source_files": " ".join(sorted(set(source_files_by_key[object_key(obj)]), key=lambda name: source_sort_key(Path(name)))),
                "coords": coords,
            }
        )
    metadata = {
        "source_files": " ".join(path.name for path in API_JSON_FILES),
        "source_file_count": len(API_JSON_FILES),
        "raw_objects_returned": len(raw_objects),
        "unique_objects_returned": len(objects),
        "duplicate_objects_removed": len(raw_objects) - len(objects),
        "source_object_counts": source_object_counts,
        "source_has_next_page": next_page_flags,
    }
    return rows, metadata


def roundabout_components(objects: list[dict]) -> list[dict]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    points_by_node: dict[str, list[tuple[float, float]]] = defaultdict(list)
    roundabout_count_by_node: dict[str, int] = defaultdict(int)

    for obj in objects:
        if obj["typeVeg"] != "Rundkjøring":
            continue
        start = obj["startnode"]
        end = obj["sluttnode"]
        adjacency[start].add(end)
        adjacency[end].add(start)
        points_by_node[start].append(obj["coords"][0])
        points_by_node[end].append(obj["coords"][-1])
        roundabout_count_by_node[start] += 1
        roundabout_count_by_node[end] += 1

    components = []
    seen = set()
    for seed in adjacency:
        if seed in seen:
            continue
        stack = [seed]
        nodes = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            nodes.add(node)
            stack.extend(adjacency[node] - seen)

        points = [point for node in nodes for point in points_by_node[node]]
        if not points:
            continue
        centroid = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        components.append(
            {
                "nodes": nodes,
                "centroid": centroid,
                "roundabout_segments": sum(roundabout_count_by_node[node] for node in nodes) // 2,
            }
        )

    components.sort(key=lambda component: (component["centroid"][0], -component["centroid"][1]))
    for index, component in enumerate(components, start=1):
        component["node_id"] = f"RB{index:02d}"
    return components


def apply_roundabout_simplification(objects: list[dict]) -> tuple[list[dict], list[dict]]:
    components = roundabout_components(objects)
    node_to_component = {}
    for component in components:
        for node in component["nodes"]:
            node_to_component[node] = component

    simplified = []
    for obj in objects:
        if obj["typeVeg"] == "Rundkjøring":
            continue

        new_obj = {key: value for key, value in obj.items() if key != "coords"}
        coords = list(obj["coords"])

        start_component = node_to_component.get(obj["startnode"])
        end_component = node_to_component.get(obj["sluttnode"])
        if start_component is not None:
            new_obj["startnode"] = start_component["node_id"]
            coords[0] = start_component["centroid"]
        if end_component is not None:
            new_obj["sluttnode"] = end_component["node_id"]
            coords[-1] = end_component["centroid"]

        if new_obj["startnode"] == new_obj["sluttnode"]:
            continue

        new_obj["coords"] = coords
        simplified.append(new_obj)

    return simplified, components


class DisjointSet:
    def __init__(self, nodes: set[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def model_node_ids(objects: list[dict], forced_junction_nodes: set[str]) -> set[str]:
    sequences_by_node = defaultdict(set)
    for obj in objects:
        sequences_by_node[obj["startnode"]].add(obj["veglenkesekvensid"])
        sequences_by_node[obj["sluttnode"]].add(obj["veglenkesekvensid"])
    return {node for node, sequences in sequences_by_node.items() if len(sequences) >= 2} | forced_junction_nodes


def endpoint_points_by_node(objects: list[dict]) -> dict[str, list[tuple[float, float]]]:
    points_by_node = defaultdict(list)
    for obj in objects:
        points_by_node[obj["startnode"]].append(obj["coords"][0])
        points_by_node[obj["sluttnode"]].append(obj["coords"][-1])
    return points_by_node


def average_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def contract_short_connectors(
    objects: list[dict],
    forced_junction_nodes: set[str],
    threshold_m: float = SHORT_CONNECTOR_M,
) -> tuple[list[dict], set[str], list[dict]]:
    current = [{key: value for key, value in obj.items() if key != "coords"} | {"coords": list(obj["coords"])} for obj in objects]
    forced = set(forced_junction_nodes)
    contractions = []
    contraction_index = 1

    while True:
        model_nodes = model_node_ids(current, forced)
        candidates = [
            obj
            for obj in current
            if obj["startnode"] in model_nodes
            and obj["sluttnode"] in model_nodes
            and obj["startnode"] != obj["sluttnode"]
            and polyline_length(obj["coords"]) <= threshold_m
        ]
        if not candidates:
            break

        all_nodes = {obj["startnode"] for obj in current} | {obj["sluttnode"] for obj in current}
        dsu = DisjointSet(all_nodes)
        for obj in candidates:
            dsu.union(obj["startnode"], obj["sluttnode"])

        groups = defaultdict(set)
        for node in all_nodes:
            groups[dsu.find(node)].add(node)

        points_by_node = endpoint_points_by_node(current)
        group_id_by_root = {}
        group_point_by_root = {}
        new_contraction_nodes = []
        next_connector_index = 1
        for root, nodes in sorted(groups.items(), key=lambda item: sorted(item[1])):
            if len(nodes) == 1:
                group_id = next(iter(nodes))
            else:
                roundabout_nodes = sorted(node for node in nodes if node in forced or node.startswith("RB"))
                connector_nodes = sorted(node for node in nodes if node.startswith("CN"))
                if roundabout_nodes:
                    group_id = roundabout_nodes[0]
                elif connector_nodes:
                    group_id = connector_nodes[0]
                else:
                    while f"CN{next_connector_index:02d}" in all_nodes:
                        next_connector_index += 1
                    group_id = f"CN{next_connector_index:02d}"
                    next_connector_index += 1
                new_contraction_nodes.append((group_id, nodes))

            group_id_by_root[root] = group_id
            if group_id in forced and points_by_node.get(group_id):
                group_point_by_root[root] = average_point(points_by_node[group_id])
            else:
                points = [point for node in nodes for point in points_by_node[node]]
                group_point_by_root[root] = average_point(points)

        candidate_keys = {obj["object_key"] for obj in candidates}
        for obj in candidates:
            contractions.append(
                {
                    "contraction_id": f"SC{contraction_index:02d}",
                    "road_id": obj["road_id"],
                    "object_key": obj["object_key"],
                    "startnode": obj["startnode"],
                    "sluttnode": obj["sluttnode"],
                    "merged_node": group_id_by_root[dsu.find(obj["startnode"])],
                    "typeVeg": obj["typeVeg"],
                    "length_m": f"{polyline_length(obj['coords']):.2f}",
                }
            )
            contraction_index += 1

        updated = []
        for obj in current:
            if obj["object_key"] in candidate_keys:
                continue

            start_root = dsu.find(obj["startnode"])
            end_root = dsu.find(obj["sluttnode"])
            new_start = group_id_by_root[start_root]
            new_end = group_id_by_root[end_root]
            if new_start == new_end:
                continue

            new_obj = {key: value for key, value in obj.items() if key != "coords"}
            coords = list(obj["coords"])
            if new_start != obj["startnode"]:
                coords[0] = group_point_by_root[start_root]
            if new_end != obj["sluttnode"]:
                coords[-1] = group_point_by_root[end_root]
            new_obj["startnode"] = new_start
            new_obj["sluttnode"] = new_end
            new_obj["coords"] = coords
            updated.append(new_obj)

        current = updated
        forced = {group_id_by_root[dsu.find(node)] for node in forced if node in all_nodes}

    return current, forced, contractions


def drop_model_roads(objects: list[dict], road_ids: set[str]) -> tuple[list[dict], list[dict]]:
    dropped = [obj for obj in objects if obj["road_id"] in road_ids]
    kept = [obj for obj in objects if obj["road_id"] not in road_ids]
    return kept, dropped


def weak_components(objects: list[dict]) -> list[set[str]]:
    adjacency = defaultdict(set)
    for obj in objects:
        start = obj["startnode"]
        end = obj["sluttnode"]
        adjacency[start].add(end)
        adjacency[end].add(start)

    seen = set()
    components = []
    for node in adjacency:
        if node in seen:
            continue
        stack = [node]
        component = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(adjacency[current] - seen)
        components.append(component)
    components.sort(key=len, reverse=True)
    return components


def node_positions(objects: list[dict]) -> dict[str, tuple[float, float]]:
    points_by_node = endpoint_points_by_node(objects)
    return {node: average_point(points) for node, points in points_by_node.items()}


def candidate_boundary_nodes(
    objects: list[dict],
    forced_junction_nodes: set[str],
    forced_boundary_nodes: set[str],
) -> set[str]:
    _, boundaries, _ = classify_nodes(
        objects,
        forced_junction_nodes=forced_junction_nodes,
        forced_boundary_nodes=forced_boundary_nodes,
    )
    return {row["node_id"] for row in boundaries}


def add_artificial_component_connectors(
    objects: list[dict],
    forced_junction_nodes: set[str],
    forced_boundary_nodes: set[str],
    max_distance_m: float = ARTIFICIAL_CONNECTOR_MAX_M,
) -> tuple[list[dict], list[dict]]:
    current = [{key: value for key, value in obj.items() if key != "coords"} | {"coords": list(obj["coords"])} for obj in objects]
    connectors = []
    connector_nodes = set()
    connector_index = 1

    while True:
        components = weak_components(current)
        if len(components) <= 1:
            break

        positions = node_positions(current)
        boundaries = candidate_boundary_nodes(current, forced_junction_nodes, forced_boundary_nodes) | connector_nodes
        component_index = {
            node: index
            for index, component in enumerate(components)
            for node in component
        }
        best = None
        boundary_list = sorted(boundaries)
        for left_index, left in enumerate(boundary_list):
            if left not in positions:
                continue
            for right in boundary_list[left_index + 1 :]:
                if right not in positions:
                    continue
                if component_index.get(left) == component_index.get(right):
                    continue
                if left in forced_boundary_nodes or right in forced_boundary_nodes:
                    continue
                if frozenset((left, right)) in BLOCK_ARTIFICIAL_CONNECTOR_NODE_PAIRS:
                    continue
                distance = math.hypot(
                    positions[left][0] - positions[right][0],
                    positions[left][1] - positions[right][1],
                )
                if distance > max_distance_m:
                    continue
                if best is None or distance < best[0]:
                    best = (distance, left, right)

        if best is None:
            break

        distance, start, end = best
        road_id = f"a{connector_index:03d}"
        obj = {
            "object_key": f"artificial-{road_id}:{start}>{end}",
            "road_id": road_id,
            "veglenkesekvensid": f"ART{connector_index:03d}",
            "veglenkenummer": "",
            "segmentnummer": "",
            "startnode": start,
            "sluttnode": end,
            "typeVeg": "Artificial connector",
            "street_name": "",
            "kortform": "",
            "lenkeretning": "MANUAL",
            "length_m": f"{distance:.3f}",
            "feltoversikt": "1#2",
            "lane_mode": "bidirectional",
            "travel_directions": ["forward", "reverse"],
            "source_files": "artificial",
            "coords": [positions[start], positions[end]],
        }
        current.append(obj)
        connectors.append(
            {
                "connector_id": road_id,
                "startnode": start,
                "sluttnode": end,
                "length_m": f"{distance:.2f}",
                "max_distance_m": f"{max_distance_m:.1f}",
            }
        )
        connector_nodes.update({start, end})
        connector_index += 1

    return current, connectors


def classify_nodes(
    objects: list[dict],
    forced_junction_nodes: set[str] | None = None,
    forced_boundary_nodes: set[str] | None = None,
) -> tuple[list[dict], list[dict], set[str]]:
    forced_junction_nodes = forced_junction_nodes or set()
    forced_boundary_nodes = forced_boundary_nodes or set()
    points_by_node = defaultdict(list)
    sequences_by_node = defaultdict(set)
    neighbours_by_node = defaultdict(set)

    for obj in objects:
        start = obj["startnode"]
        end = obj["sluttnode"]
        sequences_by_node[start].add(obj["veglenkesekvensid"])
        sequences_by_node[end].add(obj["veglenkesekvensid"])
        neighbours_by_node[start].add(end)
        neighbours_by_node[end].add(start)
        points_by_node[start].append(obj["coords"][0])
        points_by_node[end].append(obj["coords"][-1])

    marker_nodes = set()
    junctions = []
    boundaries = []
    for node, sequences in sequences_by_node.items():
        points = points_by_node[node]
        row = {
            "node_id": node,
            "x": sum(point[0] for point in points) / len(points),
            "y": sum(point[1] for point in points) / len(points),
            "incident_road_ids": " ".join(
                sorted({obj["road_id"] for obj in objects if obj["startnode"] == node or obj["sluttnode"] == node})
            ),
            "incident_sequences": " ".join(sorted(sequences)),
        }
        if node in forced_boundary_nodes:
            boundaries.append(row)
            marker_nodes.add(node)
        elif node in forced_junction_nodes or len(sequences) >= 2:
            junctions.append(row)
            marker_nodes.add(node)
        elif len(neighbours_by_node[node]) == 1:
            boundaries.append(row)
            marker_nodes.add(node)

    junctions.sort(key=lambda row: (row["x"], -row["y"]))
    boundaries.sort(key=lambda row: (row["x"], -row["y"]))
    junction_index = 1
    for row in junctions:
        if row["node_id"] in forced_junction_nodes:
            row["node_label"] = row["node_id"]
            row["node_type"] = "roundabout"
        else:
            row["node_label"] = f"J{junction_index:02d}"
            row["node_type"] = "junction"
            junction_index += 1
    for index, row in enumerate(boundaries, start=1):
        row["node_label"] = f"B{index:02d}"
        row["node_type"] = "boundary"
    return junctions, boundaries, marker_nodes


def directed_geometry(obj: dict, direction: str) -> list[tuple[float, float]]:
    coords = obj["coords"] if direction == "forward" else list(reversed(obj["coords"]))
    start_node = obj["startnode"] if direction == "forward" else obj["sluttnode"]
    end_node = obj["sluttnode"] if direction == "forward" else obj["startnode"]
    return coords, start_node, end_node


def geometry_intersects_extent(
    coords: list[tuple[float, float]],
    extent: tuple[float, float, float, float],
    margin: float = 40.0,
) -> bool:
    x_min, x_max, y_min, y_max = extent
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    return (
        max(xs) >= x_min - margin
        and min(xs) <= x_max + margin
        and max(ys) >= y_min - margin
        and min(ys) <= y_max + margin
    )


def point_in_extent(
    x: float,
    y: float,
    extent: tuple[float, float, float, float],
    margin: float = 25.0,
) -> bool:
    x_min, x_max, y_min, y_max = extent
    return x_min - margin <= x <= x_max + margin and y_min - margin <= y <= y_max + margin


def graph_extent(objects: list[dict]) -> tuple[float, float, float, float]:
    xs = [point[0] for obj in objects for point in obj["coords"]]
    ys = [point[1] for obj in objects for point in obj["coords"]]
    return min(xs), max(xs), min(ys), max(ys)


def section_extents(objects: list[dict], count: int = 4, overlap_fraction: float = 0.08) -> list[tuple[float, float, float, float]]:
    x_min, x_max, y_min, y_max = graph_extent(objects)
    width = (x_max - x_min) / count
    overlap = width * overlap_fraction
    extents = []
    for i in range(count):
        left = x_min + i * width - (overlap if i else 0.0)
        right = x_min + (i + 1) * width + (overlap if i < count - 1 else 0.0)
        section_points = [
            point
            for obj in objects
            if geometry_intersects_extent(obj["coords"], (left, right, y_min, y_max), margin=0.0)
            for point in obj["coords"]
            if left - 40.0 <= point[0] <= right + 40.0
        ]
        if section_points:
            local_y_min = min(point[1] for point in section_points)
            local_y_max = max(point[1] for point in section_points)
        else:
            local_y_min, local_y_max = y_min, y_max
        extents.append((left, right, local_y_min, local_y_max))
    return extents


def plot(
    objects: list[dict],
    junctions: list[dict],
    boundaries: list[dict],
    marker_nodes: set[str],
    labelled: bool,
    name: str,
    draw_arrows_on_roads: bool = True,
    extent: tuple[float, float, float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(17.0, 10.0))
    plotted_points = []
    labelled_roads = set()

    for obj in objects:
        if extent is not None and not geometry_intersects_extent(obj["coords"], extent):
            continue
        directions = obj["travel_directions"]
        for direction in directions:
            coords, start_node, end_node = directed_geometry(obj, direction)
            trim_start = NODE_TRIM_M if start_node in marker_nodes else 0.0
            trim_end = NODE_TRIM_M if end_node in marker_nodes else 0.0
            coords = trim_polyline(coords, trim_start, trim_end)
            if len(directions) > 1:
                coords = offset_polyline(coords, BIDIRECTIONAL_OFFSET_M)

            color = ROAD_COLORS.get(obj["typeVeg"], "#4f4f4f")
            alpha = 0.62 if obj["typeVeg"] == "Artificial connector" else (0.84 if obj["typeVeg"] != "Rampe" else 0.72)
            linewidth = 1.25 if obj["typeVeg"] == "Artificial connector" else 1.55
            linestyle = "--" if obj["typeVeg"] == "Artificial connector" else "-"
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=color, lw=linewidth, alpha=alpha, linestyle=linestyle, solid_capstyle="round", zorder=2)
            if draw_arrows_on_roads:
                draw_arrow(ax, coords, color, alpha)
            plotted_points.extend(coords)

        if labelled and obj["road_id"] not in labelled_roads:
            labelled_roads.add(obj["road_id"])
            label_coords = obj["coords"]
            (x, y), (tx, ty) = point_and_tangent(label_coords, 0.50)
            nx, ny = -ty, tx
            road_number = int(obj["road_id"][1:])
            side = -1 if road_number % 2 == 0 else 1
            ax.text(
                x + side * 8 * nx,
                y + side * 8 * ny,
                obj["road_id"],
                fontsize=3.4,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.78},
                zorder=5,
            )

    visible_boundaries = [
        row for row in boundaries if extent is None or point_in_extent(row["x"], row["y"], extent)
    ]
    visible_junctions = [
        row for row in junctions if extent is None or point_in_extent(row["x"], row["y"], extent)
    ]

    for row in visible_boundaries:
        ax.scatter(
            [row["x"]],
            [row["y"]],
            s=42,
            marker="s",
            facecolor="white",
            edgecolor="#6f6f6f",
            linewidth=1.0,
            zorder=7,
        )
        if labelled:
            ax.text(row["x"], row["y"], row["node_label"], fontsize=3.2, weight="bold", ha="center", va="center", zorder=8)

    regular_junctions = [row for row in visible_junctions if row["node_type"] == "junction"]
    roundabout_junctions = [row for row in visible_junctions if row["node_type"] == "roundabout"]

    for row in regular_junctions:
        ax.scatter(
            [row["x"]],
            [row["y"]],
            s=44,
            marker="o",
            facecolor="white",
            edgecolor="#111111",
            linewidth=1.0,
            zorder=9,
        )
        if labelled:
            ax.text(row["x"], row["y"], row["node_label"], fontsize=3.1, weight="bold", ha="center", va="center", zorder=10)

    for row in roundabout_junctions:
        ax.scatter(
            [row["x"]],
            [row["y"]],
            s=100,
            marker="o",
            facecolor="white",
            edgecolor="#111111",
            linewidth=1.2,
            zorder=11,
        )
        ax.scatter(
            [row["x"]],
            [row["y"]],
            s=46,
            marker="o",
            facecolor="none",
            edgecolor="#111111",
            linewidth=0.9,
            zorder=12,
        )
        if labelled:
            ax.text(
                row["x"] + 7.5,
                row["y"] + 7.5,
                row["node_label"],
                fontsize=4.2,
                weight="bold",
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.82},
                zorder=13,
            )

    used_road_types = {obj["typeVeg"] for obj in objects}
    road_legend = [
        ("Enkel bilveg", "Simple road", "-"),
        ("Kanalisert veg", "Channelized road", "-"),
        ("Rampe", "Ramp", "-"),
        ("Rundkjøring", "Roundabout", "-"),
        ("Artificial connector", "Artificial connector", "--"),
    ]
    legend = [
        Line2D([0], [0], color=ROAD_COLORS[road_type], lw=4.8, linestyle=linestyle, label=label)
        for road_type, label, linestyle in road_legend
        if road_type in used_road_types
    ]
    legend.append(
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#111111", color="none", markersize=13.0, label="Junction")
    )
    if roundabout_junctions:
        legend.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#111111",
                markeredgewidth=1.5,
                color="none",
                markersize=16.0,
                label="Roundabout",
            )
        )
    if visible_boundaries:
        legend.append(
            Line2D([0], [0], marker="s", markerfacecolor="white", markeredgecolor="#6f6f6f", color="none", markersize=13.0, label="Boundary")
        )
    ax.legend(
        handles=legend,
        loc="upper left",
        frameon=True,
        fontsize=20,
        handlelength=2.4,
        borderpad=0.6,
        labelspacing=0.42,
    )

    xs = [point[0] for point in plotted_points]
    ys = [point[1] for point in plotted_points]
    pad = 30
    if extent is None:
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    else:
        x_min, x_max, y_min, y_max = extent
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUTDIR / f"{name}.png", dpi=300)
    plt.close(fig)


def write_tables(
    objects: list[dict],
    junctions: list[dict],
    boundaries: list[dict],
    metadata: dict,
    prefix: str,
    extra_summary: dict | None = None,
) -> None:
    segment_columns = [
        "object_key",
        "road_id",
        "veglenkesekvensid",
        "veglenkenummer",
        "segmentnummer",
        "startnode",
        "sluttnode",
        "typeVeg",
        "street_name",
        "kortform",
        "lenkeretning",
        "length_m",
        "feltoversikt",
        "lane_mode",
        "source_files",
    ]
    with (OUTDIR / f"{prefix}_edges.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=segment_columns)
        writer.writeheader()
        for obj in objects:
            writer.writerow({column: obj[column] for column in segment_columns})

    node_columns = ["node_label", "node_type", "node_id", "x", "y", "incident_road_ids", "incident_sequences"]
    with (OUTDIR / f"{prefix}_nodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=node_columns)
        writer.writeheader()
        for row in junctions + boundaries:
            writer.writerow({column: row[column] for column in node_columns})

    summary = {
        "objects_returned": len(objects),
        "road_sequences": len({obj["veglenkesekvensid"] for obj in objects}),
        "junction_nodes": sum(row["node_type"] == "junction" for row in junctions),
        "roundabout_nodes": sum(row["node_type"] == "roundabout" for row in junctions),
        "boundary_nodes": len(boundaries),
        "source_file_count": metadata.get("source_file_count", 0),
        "raw_api_objects": metadata.get("raw_objects_returned", 0),
        "unique_api_objects": metadata.get("unique_objects_returned", 0),
        "duplicate_api_objects_removed": metadata.get("duplicate_objects_removed", 0),
        "any_source_has_next_page": any(metadata.get("source_has_next_page", {}).values()),
    }
    if extra_summary:
        summary.update(extra_summary)

    with (OUTDIR / f"{prefix}_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def write_roundabout_table(components: list[dict], prefix: str) -> None:
    columns = [
        "roundabout_id",
        "x",
        "y",
        "source_node_count",
        "roundabout_segments",
        "source_nodes",
    ]
    with (OUTDIR / f"{prefix}_roundabouts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for component in components:
            x, y = component["centroid"]
            writer.writerow(
                {
                    "roundabout_id": component["node_id"],
                    "x": x,
                    "y": y,
                    "source_node_count": len(component["nodes"]),
                    "roundabout_segments": component["roundabout_segments"],
                    "source_nodes": " ".join(sorted(component["nodes"])),
                }
            )


def write_contraction_table(contractions: list[dict], prefix: str) -> None:
    columns = [
        "contraction_id",
        "road_id",
        "object_key",
        "startnode",
        "sluttnode",
        "merged_node",
        "typeVeg",
        "length_m",
    ]
    with (OUTDIR / f"{prefix}_short_connectors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in contractions:
            writer.writerow({column: row[column] for column in columns})


def write_artificial_connector_table(connectors: list[dict], prefix: str) -> None:
    columns = ["connector_id", "startnode", "sluttnode", "length_m", "max_distance_m"]
    with (OUTDIR / f"{prefix}_artificial_connectors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in connectors:
            writer.writerow({column: row[column] for column in columns})


def write_dropped_roads_table(dropped_roads: list[dict], prefix: str) -> None:
    columns = [
        "road_id",
        "object_key",
        "startnode",
        "sluttnode",
        "typeVeg",
        "street_name",
        "length_m",
    ]
    with (OUTDIR / f"{prefix}_dropped_roads.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for obj in dropped_roads:
            writer.writerow({column: obj[column] for column in columns})


def main() -> None:
    objects, metadata = load_api_objects()
    simplified_objects, roundabouts = apply_roundabout_simplification(objects)
    roundabout_nodes = {component["node_id"] for component in roundabouts}
    final_objects, final_roundabout_nodes, contractions = contract_short_connectors(
        simplified_objects, roundabout_nodes
    )
    final_objects, dropped_roads = drop_model_roads(final_objects, DROP_ROAD_IDS)
    final_objects, artificial_connectors = add_artificial_component_connectors(
        final_objects, final_roundabout_nodes, FORCE_BOUNDARY_NODE_IDS
    )
    junctions, boundaries, marker_nodes = classify_nodes(
        final_objects,
        forced_junction_nodes=final_roundabout_nodes,
        forced_boundary_nodes=FORCE_BOUNDARY_NODE_IDS,
    )
    write_tables(
        final_objects,
        junctions,
        boundaries,
        metadata,
        FINAL_PREFIX,
        extra_summary={
            "source_objects_after_deduplication": len(objects),
            "roundabout_components": len(roundabouts),
            "roundabout_segments_removed": sum(component["roundabout_segments"] for component in roundabouts),
            "short_connector_threshold_m": SHORT_CONNECTOR_M,
            "short_connectors_contracted": len(contractions),
            "dropped_road_ids": " ".join(sorted(DROP_ROAD_IDS)),
            "forced_boundary_node_ids": " ".join(sorted(FORCE_BOUNDARY_NODE_IDS)),
            "dropped_road_objects": len(dropped_roads),
            "artificial_connector_max_m": ARTIFICIAL_CONNECTOR_MAX_M,
            "artificial_connectors_added": len(artificial_connectors),
            "weak_components": len(weak_components(final_objects)),
        },
    )
    write_roundabout_table(roundabouts, FINAL_PREFIX)
    write_contraction_table(contractions, FINAL_PREFIX)
    write_artificial_connector_table(artificial_connectors, FINAL_PREFIX)
    write_dropped_roads_table(dropped_roads, FINAL_PREFIX)
    plot(
        final_objects,
        junctions,
        boundaries,
        marker_nodes,
        labelled=False,
        name=FINAL_PREFIX,
        draw_arrows_on_roads=True,
    )
    plot(
        final_objects,
        junctions,
        boundaries,
        marker_nodes,
        labelled=True,
        name=f"{FINAL_PREFIX}_with_ids",
        draw_arrows_on_roads=False,
    )
    for section_index, extent in enumerate(section_extents(final_objects), start=1):
        plot(
            final_objects,
            junctions,
            boundaries,
            marker_nodes,
            labelled=True,
            name=f"{FINAL_PREFIX}_section_{section_index:02d}_with_ids",
            draw_arrows_on_roads=False,
            extent=extent,
        )


if __name__ == "__main__":
    main()
