"""Unconditional Gaussian-mixture experiments from Section 5.1."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianMixtureScenario:
    """Equal-weight multivariate Gaussian mixture."""

    name: str
    means: np.ndarray
    covariances: np.ndarray

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=np.float64)
        covariances = np.asarray(self.covariances, dtype=np.float64)
        if means.ndim != 2:
            raise ValueError("means must have shape (components, dimension)")
        expected_shape = (means.shape[0], means.shape[1], means.shape[1])
        if covariances.shape != expected_shape:
            raise ValueError(f"covariances must have shape {expected_shape}")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "covariances", covariances)

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    @property
    def components(self) -> int:
        return int(self.means.shape[0])

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if size < 1:
            raise ValueError("size must be positive")
        component_ids = rng.integers(0, self.components, size=size)
        output = np.empty((size, self.dimension), dtype=np.float64)
        for component in range(self.components):
            selected = component_ids == component
            count = int(selected.sum())
            if count:
                output[selected] = rng.multivariate_normal(
                    self.means[component],
                    self.covariances[component],
                    size=count,
                )
        return output

    def density(self, y: np.ndarray) -> np.ndarray:
        points = np.asarray(y, dtype=np.float64)
        if points.ndim == 1:
            points = points[:, None]
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("y has the wrong response dimension")

        mixture_density = np.zeros(points.shape[0], dtype=np.float64)
        normalizing_power = (2.0 * np.pi) ** self.dimension
        for mean, covariance in zip(self.means, self.covariances):
            difference = points - mean
            inverse = np.linalg.inv(covariance)
            exponent = np.einsum("ni,ij,nj->n", difference, inverse, difference)
            normalizer = np.sqrt(normalizing_power * np.linalg.det(covariance))
            mixture_density += np.exp(-0.5 * exponent) / normalizer
        return mixture_density / self.components


def spherical_gaussian_mixture() -> GaussianMixtureScenario:
    """Six components on a radius-0.5 circle with covariance 0.01 I."""

    indices = np.arange(1, 7)
    angles = 2.0 * np.pi * indices / 6.0
    means = 0.5 * np.column_stack((np.cos(angles), np.sin(angles)))
    covariances = np.repeat((0.01 * np.eye(2))[None, :, :], repeats=6, axis=0)
    return GaussianMixtureScenario("spherical", means, covariances)


def elliptical_gaussian_mixture() -> GaussianMixtureScenario:
    """Eight rotated elliptical components on a radius-3 circle."""

    indices = np.arange(1, 9)
    angles = np.pi * indices / 4.0
    means = 3.0 * np.column_stack((np.cos(angles), np.sin(angles)))
    minor_variance = 0.16**2
    covariances = []
    for angle in angles:
        cosine = np.cos(angle)
        sine = np.sin(angle)
        covariance = np.array(
            [
                [
                    cosine**2 + minor_variance * sine**2,
                    (1.0 - minor_variance) * sine * cosine,
                ],
                [
                    (1.0 - minor_variance) * sine * cosine,
                    sine**2 + minor_variance * cosine**2,
                ],
            ]
        )
        covariances.append(covariance)
    return GaussianMixtureScenario("elliptical", means, np.stack(covariances))


def evaluation_grid(
    lower: np.ndarray,
    upper: np.ndarray,
    grid_size: int = 100,
) -> np.ndarray:
    """Create the two-dimensional regular evaluation grid."""

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != (2,) or upper.shape != (2,):
        raise ValueError("the Table 1 grid is defined only for two-dimensional responses")
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    axes = [np.linspace(lower[dimension], upper[dimension], grid_size) for dimension in range(2)]
    first, second = np.meshgrid(*axes, indexing="xy")
    return np.column_stack((first.ravel(), second.ravel()))
