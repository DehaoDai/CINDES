"""Typed configuration for CINDES experiments."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NetworkConfig:
    """Architecture described in Section 5."""

    depth: int = 3
    width: int = 64
    output_clip: Optional[float] = None

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if self.width < 1:
            raise ValueError("width must be positive")
        if self.output_clip is not None and self.output_clip <= 0:
            raise ValueError("output_clip must be positive when provided")


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimization settings, including explicitly documented choices."""

    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 50
    min_delta: float = 1e-5
    scheduler_gamma: Optional[float] = 0.999
    fake_response_strategy: str = "fixed_global"
    reset_seed_per_candidate: bool = True
    weight_decay_candidates: tuple[float, ...] = (
        1e-3,
        5e-4,
        2e-4,
        1e-4,
        0.0,
    )

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.scheduler_gamma is not None and not 0 < self.scheduler_gamma <= 1:
            raise ValueError("scheduler_gamma must lie in (0, 1]")
        if self.fake_response_strategy not in ("fixed_global", "batch_resampled"):
            raise ValueError("fake_response_strategy must be 'fixed_global' or 'batch_resampled'")
        if not self.weight_decay_candidates:
            raise ValueError("weight_decay_candidates cannot be empty")
        if any(value < 0 for value in self.weight_decay_candidates):
            raise ValueError("weight decay cannot be negative")


@dataclass(frozen=True)
class ValidationConfig:
    """Monte Carlo settings for validation NLL normalization."""

    reference_samples: int = 256
    batch_size: int = 128

    def __post_init__(self) -> None:
        if self.reference_samples < 1:
            raise ValueError("reference_samples must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
