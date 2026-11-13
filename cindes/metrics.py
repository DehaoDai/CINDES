"""Density-estimation metrics."""

import numpy as np


def mean_absolute_density_error(
    estimated_density: np.ndarray,
    true_density: np.ndarray,
) -> float:
    """Compute the mean absolute difference between estimated and true density.

    This is the empirical TV metric used in Table 1 and does not include a
    support-volume multiplier.
    """

    estimated = np.asarray(estimated_density, dtype=np.float64)
    truth = np.asarray(true_density, dtype=np.float64)
    if estimated.shape != truth.shape:
        raise ValueError("estimated_density and true_density must have matching shapes")
    if estimated.size < 1:
        raise ValueError("density arrays cannot be empty")
    if not np.isfinite(estimated).all() or not np.isfinite(truth).all():
        raise ValueError("density arrays must be finite")
    return float(np.mean(np.abs(estimated - truth)))


def box_integrated_total_variation(
    estimated_density: np.ndarray,
    true_density: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Approximate total variation by uniform Monte Carlo over a box."""

    lower_bound = np.asarray(lower, dtype=np.float64)
    upper_bound = np.asarray(upper, dtype=np.float64)
    if lower_bound.shape != upper_bound.shape or lower_bound.ndim != 1:
        raise ValueError("lower and upper must be one-dimensional arrays with matching shapes")
    widths = upper_bound - lower_bound
    if (widths <= 0.0).any() or not np.isfinite(widths).all():
        raise ValueError("upper must be finite and strictly greater than lower")
    volume = float(np.prod(widths))
    return 0.5 * volume * mean_absolute_density_error(estimated_density, true_density)
