"""Torch model and evolutionary training."""

import statistics
from copy import deepcopy
from typing import Final

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
            layer.weight += torch.rand_like(layer.weight) * entropy

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

    def __init__(
        self,
        room: Room,
        entropy: float,
        number_of_models: int,
        model_hidden_layers: int,
        traversal_graphs: list[Graph],
    ) -> None:
        """Create new simulation of room.

        Args:
            room (Room): room
            entropy (float): starting entropy
            number_of_models (int): number of models to run simulation with
            model_hidden_layers (int): number of hidden layers per model
            traversal_graphs (list[Graph]): traversal graphs to score models
        """
        # Create input tensor from vectorized room data
        span: int = sum(item.vector_span() for item in room.items)
        input_tensor: Tensor = torch.zeros(span)
        room.vectorize(input_tensor)

        self.traversal_graphs = traversal_graphs
        self.entropy = entropy
        self.models = [
            Model(span, span, model_hidden_layers, span * HIDDEN_LAYER_SPAN_MULTIPLIER)
            for _i in range(number_of_models)
        ]
        self.input_tensors: list[Tensor] = [input_tensor.clone().detach() for _i in range(number_of_models)]
        self.output_rooms = [deepcopy(room) for _i in range(number_of_models)]
        self.scores = [0.0] * number_of_models
        self.keep_mask = [False] * number_of_models

    def iterate(self) -> tuple[float, float, float]:
        """Perform one iteration of the simulation.

        Returns:
            tuple[float, float, float]: The highest, average, and median scores
        """
        self.iterations += 1

        # First pass to score models
        for i, (model, input_tensor, output_room) in enumerate(
            zip(
                self.models,
                self.input_tensors,
                self.output_rooms,
                strict=True,
            ),
        ):
            if model is None:
                continue

            output_room.devectorize(model.forward(input_tensor))
            for item in output_room.items:
                item.origin = nudge(output_room, item.name)

            self.scores[i] = sum(score_room(output_room, graph) for graph in self.traversal_graphs) / len(
                self.traversal_graphs,
            )

        # In place sort by scores from greatest to least score
        self.models, self.input_tensors, self.output_rooms, self.scores = zip(
            *sorted(
                zip(self.models, self.input_tensors, self.output_rooms, self.scores, strict=True),
                key=lambda x: x[3],
                reverse=True,
            ),
            strict=True,
        )

        # Replace the second half of the models with derivatives of the highest scoring half
        midpoint: int = len(self.models) // 2
        for i in range(midpoint):
            survior: Model | None = self.models[i]
            if survior is None:
                continue

            self.models[midpoint + i] = survior.reproduce(self.entropy)

        # Replace the lastmost model if there is an odd amount of number of models
        if len(self.models) % 2 != 0 and self.models[0] is not None:
            self.models[-1] = self.models[0].reproduce(self.entropy)

        return (max(self.scores), statistics.mean(self.scores), statistics.median(self.scores))
