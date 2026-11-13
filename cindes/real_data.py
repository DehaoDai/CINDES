"""Result aggregation for the Light Chamber experiment."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LIGHT_CHAMBER_TARGETS = (
    "red",
    "green",
    "blue",
    "current",
    "pol_1",
    "pol_2",
    "ir_1",
    "vis_1",
    "ir_2",
    "vis_2",
    "ir_3",
    "vis_3",
)


@dataclass(frozen=True)
class FilteredSummary:
    """Statistics after applying the row-wise Tukey outlier rule."""

    mean: float
    standard_deviation: float
    first_quartile: float
    third_quartile: float
    lower_bound: float
    upper_bound: float
    total_count: int
    retained_indices: tuple[int, ...]
    removed_indices: tuple[int, ...]

    @property
    def retained_count(self) -> int:
        return len(self.retained_indices)

    @property
    def removed_count(self) -> int:
        return len(self.removed_indices)


def summarize_after_outlier_removal(values: np.ndarray) -> FilteredSummary:
    """Remove Tukey outliers, then compute the mean and sample standard deviation."""

    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or observations.size < 2:
        raise ValueError("values must be a one-dimensional array with at least two entries")
    if not np.isfinite(observations).all():
        raise ValueError("values must contain only finite entries")

    first_quartile, third_quartile = np.quantile(
        observations,
        (0.25, 0.75),
        method="linear",
    )
    interquartile_range = third_quartile - first_quartile
    lower_bound = first_quartile - 1.5 * interquartile_range
    upper_bound = third_quartile + 1.5 * interquartile_range
    retained_mask = (observations >= lower_bound) & (observations <= upper_bound)
    retained_indices = np.flatnonzero(retained_mask)
    removed_indices = np.flatnonzero(~retained_mask)
    if retained_indices.size < 2:
        raise ValueError("outlier removal retained fewer than two entries")

    retained = observations[retained_indices]
    return FilteredSummary(
        mean=float(retained.mean()),
        standard_deviation=float(retained.std(ddof=1)),
        first_quartile=float(first_quartile),
        third_quartile=float(third_quartile),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        total_count=int(observations.size),
        retained_indices=tuple(int(index) for index in retained_indices),
        removed_indices=tuple(int(index) for index in removed_indices),
    )


def load_result_matrix(
    path: Path,
    expected_rows: int = len(LIGHT_CHAMBER_TARGETS),
) -> np.ndarray:
    """Load a target-by-replication result matrix."""

    values = np.loadtxt(path, delimiter=",", ndmin=2)
    if values.shape[0] != expected_rows:
        raise ValueError(
            f"{path} has {values.shape[0]} rows; expected {expected_rows}"
        )
    if values.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two replications")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    return values
