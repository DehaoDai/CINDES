"""Random-forest classifier density estimator used in the comparisons."""

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


class RandomForestDensityEstimator:
    """Estimate a conditional density through random-forest density ratios."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 12,
        seed: int = 42,
        n_jobs: int = -1,
    ) -> None:
        if n_estimators < 1 or max_depth < 1:
            raise ValueError("forest size and depth must be positive")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.n_jobs = n_jobs
        self.model_: Optional[RandomForestClassifier] = None
        self.reference_lower_: Optional[np.ndarray] = None
        self.reference_upper_: Optional[np.ndarray] = None
        self.reference_density_: Optional[float] = None
        self.covariate_dimension_: Optional[int] = None
        self.response_dimension_: Optional[int] = None

    def get_config(self) -> dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_forest_seed": self.seed,
            "n_jobs": self.n_jobs,
        }

    def fit(
        self,
        y: np.ndarray,
        x: np.ndarray,
        rng=None,
    ) -> "RandomForestDensityEstimator":
        responses = _matrix(y, "y")
        covariates = _matrix(x, "x")
        if covariates.shape[0] != responses.shape[0]:
            raise ValueError("x and y must contain the same number of rows")
        generator = np.random.RandomState(self.seed) if rng is None else rng

        lower = responses.min(axis=0)
        upper = responses.max(axis=0)
        widths = upper - lower
        if (widths <= 0.0).any():
            raise ValueError("every response coordinate must have a positive observed range")
        fake_responses = generator.uniform(lower, upper, size=responses.shape)

        observed = np.concatenate((covariates, responses), axis=1)
        reference = np.concatenate((covariates, fake_responses), axis=1)
        features = np.concatenate((observed, reference), axis=0)
        labels = np.concatenate((np.ones(responses.shape[0]), np.zeros(responses.shape[0])))
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )
        model.fit(features, labels)

        self.model_ = model
        self.reference_lower_ = lower
        self.reference_upper_ = upper
        self.reference_density_ = float(1.0 / np.prod(widths))
        self.covariate_dimension_ = covariates.shape[1]
        self.response_dimension_ = responses.shape[1]
        return self

    def conditional_density_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        normalization_samples: int = 1_000,
        seed: Optional[int] = None,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        covariate_batch_size: int = 32,
    ) -> np.ndarray:
        model, reference_density = self._require_fitted()
        covariates = _matrix(x, "x")
        responses = _matrix(y, "y")
        self._check_dimensions(covariates, responses)
        if normalization_samples < 1 or covariate_batch_size < 1:
            raise ValueError("sample and batch sizes must be positive")

        integration_lower = responses.min(axis=0) if lower is None else np.asarray(lower)
        integration_upper = responses.max(axis=0) if upper is None else np.asarray(upper)
        if integration_lower.shape != (responses.shape[1],):
            raise ValueError("lower has the wrong shape")
        if integration_upper.shape != (responses.shape[1],):
            raise ValueError("upper has the wrong shape")
        widths = integration_upper - integration_lower
        if (widths <= 0.0).any():
            raise ValueError("upper must be strictly greater than lower")
        volume = float(np.prod(widths))
        generator = np.random.RandomState(self.seed if seed is None else seed)

        output = []
        for start in range(0, covariates.shape[0], covariate_batch_size):
            stop = min(start + covariate_batch_size, covariates.shape[0])
            batch_x = covariates[start:stop]
            batch_size = batch_x.shape[0]

            query_x = np.repeat(batch_x, responses.shape[0], axis=0)
            query_y = np.tile(responses, (batch_size, 1))
            query_ratio = self._density_ratio(
                model,
                np.concatenate((query_x, query_y), axis=1),
            ).reshape(batch_size, responses.shape[0])

            normalization_y = generator.uniform(
                integration_lower,
                integration_upper,
                size=(batch_size * normalization_samples, responses.shape[1]),
            )
            normalization_x = np.repeat(batch_x, normalization_samples, axis=0)
            normalization_ratio = self._density_ratio(
                model,
                np.concatenate((normalization_x, normalization_y), axis=1),
            ).reshape(batch_size, normalization_samples)
            normalizer = reference_density * volume * normalization_ratio.mean(axis=1)
            normalizer = np.maximum(normalizer, np.finfo(np.float64).tiny)
            output.append(reference_density * query_ratio / normalizer[:, None])
        return np.concatenate(output, axis=0)

    @staticmethod
    def _density_ratio(model: RandomForestClassifier, features: np.ndarray) -> np.ndarray:
        probability = np.clip(model.predict_proba(features)[:, 1], 1e-6, 1.0 - 1e-6)
        return probability / (1.0 - probability)

    def _check_dimensions(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.shape[1] != self.covariate_dimension_:
            raise ValueError("covariate dimension differs from fitted data")
        if y.shape[1] != self.response_dimension_:
            raise ValueError("response dimension differs from fitted data")

    def _require_fitted(self) -> tuple[RandomForestClassifier, float]:
        if self.model_ is None or self.reference_density_ is None:
            raise RuntimeError("fit must be called before density evaluation")
        return self.model_, self.reference_density_
