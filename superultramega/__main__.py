"""Program entry point."""

import json
from pathlib import Path
from typing import Any, Final

from .map import Room
from .model import GeneticSimulation
from .traversal_graph import Graph

ITERATIONS: Final[int] = 2**6

with Path("map.json").open(encoding="utf-8", errors="replace") as room_file:
    room_dict: dict[str, Any] = json.load(room_file)
    room: Room = Room.model_validate(room_dict)

with Path("graphs.json").open(encoding="utf-8", errors="replace") as graphs_file:
    graphs_list: list[dict[str, Any]] = json.load(graphs_file)
    graphs: list[Graph] = [Graph.model_validate(graph_dict) for graph_dict in graphs_list]

simulation: GeneticSimulation = GeneticSimulation(room, 10.0, 64, 57, graphs)

best: float = 0
best_scoring_index: int = 0

for i in range(ITERATIONS):
    average: float
    median: float
    best, average, median, best_scoring_index = simulation.iterate()

    print(f"#{i} Average: {average}\tMedian: {median}\tBest: {best}")

print(f"Highest scoring room: {best}", simulation.output_rooms[best_scoring_index].model_dump_json(), sep="\n\n")
