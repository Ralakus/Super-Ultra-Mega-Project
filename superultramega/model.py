"""Torch model and evolutionary training."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class Model:
    """Torch model used to perform gradient descent optimization."""

    linear1: nn.Linear
    activation: nn.ReLU
    linear2: nn.Linear
    softmax: nn.Softmax

    def __init__(self, input_count: int, output_count: int) -> None:
        """Create randomly assigned model.

        Args:
            input_count (int): number of input parameters
            output_count (int): number of output parameters
        """
        self.linear1 = nn.Linear(input_count, input_count)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(input_count, output_count)
        self.softmax = nn.Softmax(output_count)

    def reproduce(self, entropy: float) -> "Model":
        """Create new model variant of current model.

        Args:
            entropy (float): entropy value to adjust parameters.
                0 will be a direct copy while higher values create more
                variance.

        Returns:
            Model: new derived model
        """
        derived_model: Model = deepcopy(self)

        self.linear1.weight = self.linear1.weight + (torch.rand_like(self.linear1.weight) * entropy)
        self.linear2.weight = self.linear2.weight + (torch.rand_like(self.linear2.weight) * entropy)

        return derived_model

    def forward(self, x: Tensor) -> Tensor:
        """Model forward pass.

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return self.softmax(x)


@dataclass
class GeneticSimulation:
    """Wrapper to hold simulation state."""

    models: list[Model]
    entropy: float
    iterations: int

    def iterate(self) -> list[Tensor]:
        """Perform one iteration of the simulation.

        Returns:
            list[Tensor]: list of outputs from models
        """
        self.iterations += 1
        return []
