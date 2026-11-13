"""Conditional density scenarios used in Tables 2 and 3."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import truncnorm


@dataclass(frozen=True)
class MultivariateConditionalScenario:
    """Truncated-normal conditional density with a multivariate response."""

    weight_matrix: np.ndarray
    covariate_dimension: int = 16
    response_dimension: int = 4
    response_lower: float = -1.0
    response_upper: float = 1.0

    def __post_init__(self) -> None:
        weights = np.asarray(self.weight_matrix, dtype=np.float64)
        expected_shape = (self.response_dimension, self.covariate_dimension)
        if weights.shape != expected_shape:
            raise ValueError(f"weight_matrix must have shape {expected_shape}")
        if not np.isfinite(weights).all() or (weights < 0.0).any():
            raise ValueError("weight_matrix must contain finite non-negative values")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("each row of weight_matrix must sum to one")
        object.__setattr__(self, "weight_matrix", weights.copy())

    def sample_covariates(self, size: int, rng) -> np.ndarray:
        if size < 1:
            raise ValueError("size must be positive")
        return rng.uniform(0.0, 1.0, size=(size, self.covariate_dimension))

    def sample_responses(self, x: np.ndarray, rng) -> np.ndarray:
        covariates = self._validate_covariates(x)
        means = covariates @ self.weight_matrix.T
        responses = np.empty_like(means)
        for coordinate in range(self.response_dimension):
            location = means[:, coordinate]
            lower = self.response_lower - location
            upper = self.response_upper - location
            responses[:, coordinate] = truncnorm.rvs(
                lower,
                upper,
                loc=location,
                scale=1.0,
                random_state=rng,
            )
        return responses

    def sample(self, size: int, rng) -> tuple[np.ndarray, np.ndarray]:
        x = self.sample_covariates(size, rng)
        y = self.sample_responses(x, rng)
        return x.astype(np.float32), y.astype(np.float32)

    def density(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        covariates = self._validate_covariates(x)
        responses = self._validate_responses(y)
        if covariates.shape[0] != responses.shape[0]:
            raise ValueError("x and y must contain the same number of rows")
        means = covariates @ self.weight_matrix.T
        lower = self.response_lower - means
        upper = self.response_upper - means
        component_density = truncnorm.pdf(
            responses,
            a=lower,
            b=upper,
            loc=means,
            scale=1.0,
        )
        return np.prod(component_density, axis=1)

    def density_grid(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        covariates = self._validate_covariates(x)
        responses = self._validate_responses(y)
        means = covariates @ self.weight_matrix.T
        expanded_means = means[:, None, :]
        lower = self.response_lower - expanded_means
        upper = self.response_upper - expanded_means
        component_density = truncnorm.pdf(
            responses[None, :, :],
            a=lower,
            b=upper,
            loc=expanded_means,
            scale=1.0,
        )
        return np.prod(component_density, axis=2)

    def _validate_covariates(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.covariate_dimension:
            raise ValueError("x has the wrong shape")
        if values.shape[0] < 1 or not np.isfinite(values).all():
            raise ValueError("x must contain finite rows")
        return values

    def _validate_responses(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.response_dimension:
            raise ValueError("y has the wrong shape")
        if values.shape[0] < 1 or not np.isfinite(values).all():
            raise ValueError("y must contain finite rows")
        return values


def multivariate_conditional_scenario(
    rng,
    covariate_dimension: int = 16,
    response_dimension: int = 4,
) -> MultivariateConditionalScenario:
    """Draw the Dirichlet coefficient matrix used by the multivariate experiment."""

    if covariate_dimension < 1 or response_dimension < 1:
        raise ValueError("dimensions must be positive")
    weights = rng.dirichlet(
        np.ones(covariate_dimension),
        size=response_dimension,
    )
    return MultivariateConditionalScenario(
        weight_matrix=weights,
        covariate_dimension=covariate_dimension,
        response_dimension=response_dimension,
    )


@dataclass(frozen=True)
class ConditionalDensityScenario:
    """A seeded conditional density with one-dimensional response."""

    name: str
    covariate_dimension: int
    additive_function_indices: tuple[int, ...] = ()
    response_lower: float = -1.0
    response_upper: float = 1.0

    def sample_covariates(
        self,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if size < 1:
            raise ValueError("size must be positive")
        lower = -1.0 if self.name == "nonlinear" else 0.0
        return rng.uniform(lower, 1.0, size=(size, self.covariate_dimension))

    def sample(
        self,
        size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self.sample_covariates(size, rng)
        y = self.sample_responses(x, rng)
        return x.astype(np.float32), y[:, None].astype(np.float32)

    def sample_responses(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample responses for an existing covariate matrix."""

        covariates = np.asarray(x, dtype=np.float64)
        if (
            covariates.ndim != 2
            or covariates.shape[0] < 1
            or covariates.shape[1] != self.covariate_dimension
        ):
            raise ValueError("x has the wrong shape")
        if self.name == "nonlinear":
            y = self._sample_nonlinear(covariates, rng)
        elif self.name == "additive":
            mean = self._additive_mean(covariates)
            y = _sample_truncated_normal(
                mean,
                standard_deviation=2.0,
                lower=self.response_lower,
                upper=self.response_upper,
                rng=rng,
            )
        elif self.name == "mixture":
            weight, mean_one, mean_two = _mixture_parameters(covariates)
            component_one = rng.binomial(1, weight).astype(bool)
            means = np.where(component_one, mean_one, mean_two)
            standard_deviations = np.where(component_one, 0.15, 0.12)
            y = _sample_truncated_normal(
                means,
                standard_deviation=standard_deviations,
                lower=self.response_lower,
                upper=self.response_upper,
                rng=rng,
            )
        else:
            raise ValueError(f"unknown scenario: {self.name}")
        return np.asarray(y, dtype=np.float64)

    def density(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        covariates = np.asarray(x, dtype=np.float64)
        responses = np.asarray(y, dtype=np.float64)
        if responses.ndim == 2 and responses.shape[1] == 1:
            responses = responses[:, 0]
        if covariates.ndim != 2 or covariates.shape[1] != self.covariate_dimension:
            raise ValueError("x has the wrong covariate dimension")
        if responses.ndim != 1 or responses.shape[0] != covariates.shape[0]:
            raise ValueError("x and y must contain the same number of rows")

        if self.name == "nonlinear":
            signal = np.tanh(
                np.sin(covariates[:, 0]) + covariates[:, 1] ** 2 - 0.5 * covariates[:, 2]
            )
            density = 0.5 * (1.0 - responses * signal)
            return np.where(self._inside_support(responses), density, 0.0)
        if self.name == "additive":
            return _truncated_normal_density(
                responses,
                self._additive_mean(covariates),
                standard_deviation=2.0,
                lower=self.response_lower,
                upper=self.response_upper,
            )
        if self.name == "mixture":
            weight, mean_one, mean_two = _mixture_parameters(covariates)
            density_one = _truncated_normal_density(
                responses,
                mean_one,
                standard_deviation=0.15,
                lower=self.response_lower,
                upper=self.response_upper,
            )
            density_two = _truncated_normal_density(
                responses,
                mean_two,
                standard_deviation=0.12,
                lower=self.response_lower,
                upper=self.response_upper,
            )
            return weight * density_one + (1.0 - weight) * density_two
        raise ValueError(f"unknown scenario: {self.name}")

    def _inside_support(self, y: np.ndarray) -> np.ndarray:
        return (y >= self.response_lower) & (y <= self.response_upper)

    def _sample_nonlinear(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        signal = np.tanh(np.sin(x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 2])
        uniform = rng.uniform(0.0, 1.0, size=x.shape[0])
        response = np.empty(x.shape[0], dtype=np.float64)
        nearly_uniform = np.abs(signal) < 1e-10
        response[nearly_uniform] = 2.0 * uniform[nearly_uniform] - 1.0

        active = ~nearly_uniform
        coefficient_a = -0.25 * signal[active]
        coefficient_b = 0.5
        coefficient_c = 0.5 + 0.25 * signal[active] - uniform[active]
        discriminant = coefficient_b**2 - 4.0 * coefficient_a * coefficient_c
        root_one = (-coefficient_b + np.sqrt(discriminant)) / (2.0 * coefficient_a)
        root_two = (-coefficient_b - np.sqrt(discriminant)) / (2.0 * coefficient_a)
        response[active] = np.where(
            (root_one >= self.response_lower) & (root_one <= self.response_upper),
            root_one,
            root_two,
        )
        return response

    def _additive_mean(self, x: np.ndarray) -> np.ndarray:
        if len(self.additive_function_indices) != 5:
            raise ValueError("additive scenario requires five function indices")
        values = np.zeros(x.shape[0], dtype=np.float64)
        for coordinate, function_index in enumerate(self.additive_function_indices):
            values += _additive_function(function_index, x[:, coordinate])
        return values


def conditional_scenario(
    name: str,
    rng: np.random.Generator,
) -> ConditionalDensityScenario:
    """Construct one Table 2 scenario with replication-specific parameters."""

    if name == "nonlinear":
        return ConditionalDensityScenario(name=name, covariate_dimension=4)
    if name == "additive":
        if isinstance(rng, np.random.RandomState):
            selected = rng.randint(0, 5, size=5)
        else:
            selected = rng.integers(0, 5, size=5)
        indices = tuple(int(value) for value in selected)
        return ConditionalDensityScenario(
            name=name,
            covariate_dimension=20,
            additive_function_indices=indices,
        )
    if name == "mixture":
        return ConditionalDensityScenario(name=name, covariate_dimension=4)
    raise ValueError(f"unknown scenario: {name}")


def conditional_covariates(
    name: str,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample covariates without drawing scenario-specific parameters."""

    dimension = 20 if name == "additive" else 4
    if name not in ("nonlinear", "additive", "mixture"):
        raise ValueError(f"unknown scenario: {name}")
    template = ConditionalDensityScenario(
        name=name,
        covariate_dimension=dimension,
    )
    return template.sample_covariates(size, rng)


def _additive_function(index: int, values: np.ndarray) -> np.ndarray:
    if index == 0:
        return np.cos(np.pi * values)
    if index == 1:
        return np.sin(values)
    if index == 2:
        return (1.0 - np.abs(values)) ** 2
    if index == 3:
        return expit(values)
    if index == 4:
        return 2.0 * np.sqrt(np.abs(values)) - 1.0
    raise ValueError(f"unknown additive function index: {index}")


def _mixture_parameters(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight = expit(0.2 + x @ np.array([1.2, -0.8, 0.6, -0.4]))
    mean_one = (
        0.6 * x[:, 0]
        - 0.3 * x[:, 1]
        + 0.2 * x[:, 2]
        + 0.4 * np.sin(2.0 * np.pi * x[:, 0])
        + 0.2 * np.cos(2.0 * np.pi * x[:, 1])
    )
    mean_two = (
        -0.5 * x[:, 0]
        + 0.2 * x[:, 1]
        - 0.25 * x[:, 2]
        + 0.1 * x[:, 3]
        - 0.35 * np.sin(2.0 * np.pi * x[:, 0])
        + 0.25 * np.cos(2.0 * np.pi * x[:, 2])
    )
    return weight, np.clip(mean_one, -0.85, 0.85), np.clip(mean_two, -0.85, 0.85)


def _sample_truncated_normal(
    mean: np.ndarray,
    standard_deviation,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    deviation = np.asarray(standard_deviation, dtype=np.float64)
    lower_standardized = (lower - mean) / deviation
    upper_standardized = (upper - mean) / deviation
    return np.asarray(
        truncnorm.rvs(
            lower_standardized,
            upper_standardized,
            loc=mean,
            scale=deviation,
            random_state=rng,
        ),
        dtype=np.float64,
    )


def _truncated_normal_density(
    y: np.ndarray,
    mean: np.ndarray,
    standard_deviation: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    lower_standardized = (lower - mean) / standard_deviation
    upper_standardized = (upper - mean) / standard_deviation
    return truncnorm.pdf(
        y,
        lower_standardized,
        upper_standardized,
        loc=mean,
        scale=standard_deviation,
    )
