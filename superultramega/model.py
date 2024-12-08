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

    def reproduce(self, entropy: float) -> "Model":
        """Create changed new model from current model.

        Args:
            entropy (float): entropy value to adjust parameters.
                0 will be a direct copy while higher values create more
                variance.

        Returns:
            Model: new derived model
        """
        derived_model: Model = deepcopy(self)

        for layer in derived_model.linear_layers:
            layer.weight = nn.Parameter((torch.rand_like(layer.weight) * entropy) + layer.weight)

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

        return self.linear_layers[-1](x)


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
    input_tensors: list[Tensor]
    output_rooms: list[Room]
    scores: list[float]
    keep_mask: list[bool]
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
        # Create input tensor from vectorized room data
        span: int = sum(item.vector_span() for item in room.items)
        input_tensor: Tensor = torch.zeros(span)
        room.vectorize(input_tensor)

        if model_hidden_layer_span_multiplier is None:
            model_hidden_layer_span_multiplier = HIDDEN_LAYER_SPAN_MULTIPLIER

        self.traversal_graphs = traversal_graphs
        self.entropy = entropy
        self.models = [
            Model(span, span, model_hidden_layers, span * model_hidden_layer_span_multiplier)
            for _i in range(number_of_models)
        ]
        self.input_tensors: list[Tensor] = [input_tensor.clone().detach() for _i in range(number_of_models)]
        self.output_rooms = [deepcopy(room) for _i in range(number_of_models)]
        self.scores = [0.0] * number_of_models
        self.keep_mask = [False] * number_of_models
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
                    self.input_tensors,
                    self.output_rooms,
                    strict=False,
                ),
            ):
                self.scores[i] = score
                self.output_rooms[i] = output_room

        except (ValueError, OverflowError):
            return (0.0, 0.0, 0.0)

        # In place sort by scores from least to greatest score
        self.models, self.input_tensors, self.output_rooms, self.scores = zip(
            *sorted(
                zip(self.models, self.input_tensors, self.output_rooms, self.scores, strict=True),
                key=lambda x: x[3],
                reverse=False,
            ),
            strict=True,
        )

        self.models = list(self.models)
        self.input_tensors = list(self.input_tensors)
        self.output_rooms = list(self.output_rooms)
        self.scores = list(self.scores)

        for i in range(1, len(self.models)):
            survior: Model | None = self.models[i]
            if survior is None:
                continue

            self.models[i] = survior.reproduce(self.entropy)

        return (
            self.scores[0],
            statistics.mean(self.scores),
            statistics.median(self.scores),
        )
