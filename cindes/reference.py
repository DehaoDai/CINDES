"""Reference distributions used to generate fake responses."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class UniformBoxReference:
    """Uniform distribution over a coordinate-aligned box."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.ndim != 1 or upper.ndim != 1:
            raise ValueError("lower and upper must be one-dimensional")
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(upper <= lower):
            raise ValueError("every upper bound must exceed its lower bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def from_data(cls, y: np.ndarray, minimum_width: float = 1e-6) -> "UniformBoxReference":
        """Fit the coordinate-wise min/max reference box."""

        values = np.asarray(y, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("y must have shape (n_samples, response_dimension)")
        lower = values.min(axis=0)
        upper = values.max(axis=0)
        narrow = (upper - lower) < minimum_width
        if np.any(narrow):
            center = 0.5 * (lower[narrow] + upper[narrow])
            lower[narrow] = center - 0.5 * minimum_width
            upper[narrow] = center + 0.5 * minimum_width
        return cls(lower=lower, upper=upper)

    @property
    def dimension(self) -> int:
        return int(self.lower.size)

    @property
    def log_density_value(self) -> float:
        return float(-np.log(self.upper - self.lower).sum())

    def sample(
        self,
        size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw reference samples using an explicit NumPy generator."""

        if size < 1:
            raise ValueError("size must be positive")
        generator = rng if rng is not None else np.random.default_rng()
        return generator.uniform(self.lower, self.upper, size=(size, self.dimension))

    def torch_log_prob(
        self,
        y: torch.Tensor,
        enforce_support: bool = True,
    ) -> torch.Tensor:
        """Evaluate log q(y).

        ``enforce_support=False`` applies the box-density constant outside the
        observed min/max range. This is the numerical convention needed for
        finite validation NLL when the data distribution itself is unbounded.
        """

        lower = torch.as_tensor(self.lower, dtype=y.dtype, device=y.device)
        upper = torch.as_tensor(self.upper, dtype=y.dtype, device=y.device)
        log_density = torch.as_tensor(self.log_density_value, dtype=y.dtype, device=y.device)
        if not enforce_support:
            return log_density.expand(y.shape[0])
        inside = torch.logical_and(y >= lower, y <= upper).all(dim=-1)
        negative_infinity = torch.full_like(inside, -torch.inf, dtype=y.dtype)
        return torch.where(inside, log_density.expand_as(negative_infinity), negative_infinity)
