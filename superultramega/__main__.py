"""Program entry point."""

import json
from pathlib import Path
from typing import Any, Final

from .map import Room
from .model import GeneticSimulation
from .traversal_graph import Graph

ITERATIONS: Final[int] = 2 ^ 16

with Path("map.json").open(encoding="utf-8", errors="replace") as room_file:
    room_dict: dict[str, Any] = json.load(room_file)
    room: Room = Room.model_validate(room_dict)

with Path("graphs.json").open(encoding="utf-8", errors="replace") as graphs_file:
    graphs_list: list[dict[str, Any]] = json.load(graphs_file)
    graphs: list[Graph] = [Graph.model_validate(graph_dict) for graph_dict in graphs_list]

simulation: GeneticSimulation = GeneticSimulation(room, 2.0, 10, 3, graphs)

highest: float = 0
highest_scoring_index: int = 0

for i in range(ITERATIONS):
    average: float
    median: float
    highest, average, median, highest_scoring_index = simulation.iterate()

    print(f"#{i} Average: {average}\tMedian: {median}\tHighest: {median}")

print(f"Highest scoring room: {highest}", simulation.output_rooms[highest_scoring_index].model_dump_json(), sep="\n\n")
