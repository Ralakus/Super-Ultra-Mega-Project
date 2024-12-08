"""Torch model and evolutionary training."""

import multiprocessing
import statistics
from copy import deepcopy
from itertools import count, repeat
from typing import Any, Final

import torch
from torch import Tensor, nn

from .map import Room
from .nudge import nudge
from .score import score_room
from .traversal_graph import Graph

HIDDEN_LAYER_SPAN_MULTIPLIER: Final[int] = 4
"""Mulplier for number of hidden layers relative to the vector span of the room map."""


class Model:
    """Torch model used to perform gradient descent optimization."""

    linear_layers: list[nn.Linear]
    activation: nn.ReLU
    hidden_parameter_count: int

    def __init__(self, input_count: int, output_count: int, hidden_layers: int, hidden_parameter_count: int) -> None:
        """Create randomly assigned model.

        Args:
            input_count (int): number of input parameters
            output_count (int): number of output parameters
            hidden_layers (int): number of hidden layers
            hidden_parameter_count (int): number of parameters in hidden layers
        """
        self.linear_layers = [
            nn.Linear(input_count, hidden_parameter_count),
            *[nn.Linear(hidden_parameter_count, hidden_parameter_count) for _i in range(max(hidden_layers, 1))],
            nn.Linear(hidden_parameter_count, output_count),
        ]
        self.activation = nn.ReLU()

    def reproduce(self, entropy: float, other: "Model") -> "Model":
        """Create changed model from other model.

        Args:
            entropy (float): entropy value to adjust parameters.
                0 will be a direct copy of other while higher values create more
                variance.
            other (Model): model to base from

        Returns:
            Model: new derived model
        """
        derived_model: Model = deepcopy(other)

        for derived_layer, self_layer in zip(derived_model.linear_layers, self.linear_layers, strict=True):
            derived_layer.weight = nn.Parameter(
                derived_layer.weight + ((derived_layer.weight - self_layer.weight) * entropy),
            )

        return derived_model

    def forward(self, x: Tensor) -> Tensor:
        """Model forward pass.

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        for layer in self.linear_layers[:-2]:
            x = layer(x)
            x = self.activation(x)

        x = self.linear_layers[-1](x)

        return nn.functional.normalize(x, dim=0)


def genetic_evaluation(
    packet: tuple[int, list[Graph], Model | None, Tensor, Room],
) -> tuple[int, float, Room]:
    """Run simulation score on room.

    Args:
        packet (tuple[int, list[Graph], Model | None, Tensor, Room]): input

    Returns:
        float: score.
    """
    index: int
    traversal_graphs: list[Graph]
    model: Model | None
    input_tensor: Tensor
    output_room: Room
    index, traversal_graphs, model, input_tensor, output_room = packet

    if model is None:
        return (index, float("inf"), output_room)

    output_room.devectorize(model.forward(input_tensor))
    for item in output_room.items:
        if not item.fixed:
            # Move object back into global coordinates from normalized coordinates
            item.origin.x *= output_room.bounds.x
            item.origin.y *= output_room.bounds.y

        nudge(output_room, item.name)

    return (
        index,
        sum(score_room(output_room, graph) for graph in traversal_graphs)
        / len(
            traversal_graphs,
        ),
        output_room,
    )


class GeneticSimulation:
    """Wrapper to hold simulation state."""

    models: list[Model | None]
    input_tensor: Tensor
    output_rooms: list[Room]
    scores: list[float]
    traversal_graphs: list[Graph]
    entropy: float
    iterations: int
    room: Room
    pool: Any

    def __init__(  # noqa: PLR0913
        self,
        room: Room,
        entropy: float,
        number_of_models: int,
        model_hidden_layers: int,
        model_hidden_layer_span_multiplier: int | None,
        traversal_graphs: list[Graph],
    ) -> None:
        """Create new simulation of room.

        Args:
            room (Room): room
            entropy (float): starting entropy
            number_of_models (int): number of models to run simulation with
            model_hidden_layers (int): number of hidden layers per model
            model_hidden_layer_span_multiplier (int): number of parameters in hidden layers relative to span
            traversal_graphs (list[Graph]): traversal graphs to score models
        """
        # Ensure at least 4 models are in simulation
        number_of_models = max(number_of_models, 4)

        # Create input tensor from vectorized room data
        span: int = room.vector_span()
        devector_span: int = room.devector_span()
        input_tensor: Tensor = torch.zeros(span)
        room.vectorize(input_tensor)
        self.input_tensor = input_tensor

        if model_hidden_layer_span_multiplier is None:
            model_hidden_layer_span_multiplier = HIDDEN_LAYER_SPAN_MULTIPLIER

        self.traversal_graphs = traversal_graphs
        self.entropy = entropy
        self.models = [
            Model(span, devector_span, model_hidden_layers, span * model_hidden_layer_span_multiplier)
            for _i in range(number_of_models)
        ]
        self.output_rooms = [deepcopy(room) for _i in range(number_of_models)]
        self.scores = [0.0] * number_of_models
        self.iterations = 0
        self.pool = multiprocessing.Pool()

    def iterate(self) -> tuple[float, float, float]:
        """Perform one iteration of the simulation.

        Returns:
            tuple[float, float, float]: The best, average, and median scores
        """
        self.iterations += 1

        try:
            for i, score, output_room in self.pool.imap(
                genetic_evaluation,
                zip(
                    count(start=0, step=1),
                    repeat(self.traversal_graphs),
                    self.models,
                    repeat(self.input_tensor),
                    self.output_rooms,
                    strict=False,
                ),
            ):
                self.scores[i] = score
                self.output_rooms[i] = output_room

        except (ValueError, OverflowError):
            return (0.0, 0.0, 0.0)

        # In place sort by scores from least to greatest score
        self.models, self.output_rooms, self.scores = zip(
            *sorted(
                zip(self.models, self.output_rooms, self.scores, strict=True),
                key=lambda x: x[2],
                reverse=False,
            ),
            strict=True,
        )

        self.models = list(self.models)
        self.output_rooms = list(self.output_rooms)
        self.scores = list(self.scores)

        # Reproduce models based off of best scoring one.
        for i in range(1, len(self.models)):
            survior: Model | None = self.models[i]
            if survior is None or self.models[0] is None:
                continue

            positional_entropy: float = 1.0 + float(i) / float(len(self.models))
            local_entropy: float = abs(self.scores[i] - self.scores[0]) / self.scores[0]

            self.models[i] = survior.reproduce(self.entropy * positional_entropy * local_entropy, self.models[0])

        return (
            self.scores[0],
            statistics.mean(self.scores),
            statistics.median(self.scores),
        )
