"""Fully connected ReLU network used by the density-ratio classifier."""

from typing import Optional

import torch
from torch import nn


class DeepReLUNetwork(nn.Module):
    """A constant-width network with ``depth`` hidden ReLU layers."""

    def __init__(
        self,
        input_dimension: int,
        depth: int = 3,
        width: int = 64,
        output_clip: Optional[float] = None,
    ) -> None:
        super().__init__()
        if input_dimension < 1:
            raise ValueError("input_dimension must be positive")
        if depth < 1:
            raise ValueError("depth must be positive")
        if width < 1:
            raise ValueError("width must be positive")

        layers = []
        current_dimension = input_dimension
        for _ in range(depth):
            layers.extend((nn.Linear(current_dimension, width), nn.ReLU()))
            current_dimension = width
        layers.append(nn.Linear(current_dimension, 1))
        self.network = nn.Sequential(*layers)
        self.output_clip = output_clip

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.network(inputs).squeeze(-1)
        if self.output_clip is not None:
            logits = torch.clamp(logits, -self.output_clip, self.output_clip)
        return logits
