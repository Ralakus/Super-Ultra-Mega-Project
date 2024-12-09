"""Program entry point."""

import json
from contextlib import suppress
from pathlib import Path
from typing import Any, Final

from .map import Room
from .model import GeneticSimulation
from .traversal_graph import Graph

ITERATIONS: Final[int] = 2**6
MAX_UNIMPROVED_ITERATIONS: Final[int] = 10

with Path("map.json").open(encoding="utf-8", errors="replace") as room_file:
    room_dict: dict[str, Any] = json.load(room_file)
    room: Room = Room.model_validate(room_dict)

print("Room vector span:", room.vector_span(), "\nRoom devector span:", room.devector_span())

with Path("graphs.json").open(encoding="utf-8", errors="replace") as graphs_file:
    graphs_list: list[dict[str, Any]] = json.load(graphs_file)
    graphs: list[Graph] = [Graph.model_validate(graph_dict) for graph_dict in graphs_list]

print("Preparing simulation...")
simulation: GeneticSimulation = GeneticSimulation(
    room=room,
    entropy=1.0,
    number_of_models=192,
    model_hidden_layers=64,
    model_hidden_layer_span_multiplier=8,
    traversal_graphs=graphs,
)

unimproved_iterations: int = 0
previous_best: float = 0
best: float = 0

print("Running iterations...")
with suppress(KeyboardInterrupt):
    for i in range(ITERATIONS):
        average: float
        median: float
        best, average, median = simulation.iterate()

        if best == previous_best:
            unimproved_iterations += 1
        else:
            unimproved_iterations = 0

        previous_best = best

        print(f"#{i}\tAverage: {average}\tMedian: {median}\tBest: {best}")

        with Path("output_room.json").open("w", encoding="utf-8") as output_room_file:
            output_room_file.write(simulation.output_rooms[0].model_dump_json())

        if unimproved_iterations >= MAX_UNIMPROVED_ITERATIONS:
            print(f"No improvement shown in the past {MAX_UNIMPROVED_ITERATIONS} iterations. Ending simulation.")
            break

print(f"Best scoring room: {best}", simulation.output_rooms[0].model_dump_json(), sep="\n\n")
