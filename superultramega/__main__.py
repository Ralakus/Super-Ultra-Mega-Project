"""Program entry point."""

import json
from contextlib import suppress
from pathlib import Path
from typing import Any, Final

from .map import Room
from .model import GeneticSimulation
from .traversal_graph import Graph

ITERATIONS: Final[int] = 2**6
MAX_SAME_ITERATIONS: Final[int] = 5

with Path("map.json").open(encoding="utf-8", errors="replace") as room_file:
    room_dict: dict[str, Any] = json.load(room_file)
    room: Room = Room.model_validate(room_dict)

with Path("graphs.json").open(encoding="utf-8", errors="replace") as graphs_file:
    graphs_list: list[dict[str, Any]] = json.load(graphs_file)
    graphs: list[Graph] = [Graph.model_validate(graph_dict) for graph_dict in graphs_list]

simulation: GeneticSimulation = GeneticSimulation(room, 1.0, 64, 57, graphs)

same_iterations: int = 0
previous_average: float = 0

best: float = 0
best_scoring_index: int = 0

with suppress(KeyboardInterrupt):
    for i in range(ITERATIONS):
        average: float
        median: float
        best, average, median, best_scoring_index = simulation.iterate()

        if average == previous_average:
            same_iterations += 1
        else:
            same_iterations = 0

        previous_average = average

        print(f"#{i} Average: {average}\tMedian: {median}\tBest: {best}")

        if same_iterations >= MAX_SAME_ITERATIONS:
            print(f"No improvement shown in the past {MAX_SAME_ITERATIONS} iterations. Ending simulation.")
            break

print(f"Highest scoring room: {best}", simulation.output_rooms[best_scoring_index].model_dump_json(), sep="\n\n")

with Path("output_room.json").open("w", encoding="utf-8") as output_room_file:
    output_room_file.write(simulation.output_rooms[best_scoring_index].model_dump_json())
