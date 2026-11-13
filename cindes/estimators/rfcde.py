"""Random-forest classifier density estimator used in the comparisons."""

from typing import Optional

import numpy as np
from scipy.stats import norm
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
        reference: str = "uniform",
    ) -> None:
        if n_estimators < 1 or max_depth < 1:
            raise ValueError("forest size and depth must be positive")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.n_jobs = n_jobs
        if reference not in ("uniform", "gaussian"):
            raise ValueError("reference must be uniform or gaussian")
        self.reference = reference
        self.model_: Optional[RandomForestClassifier] = None
        self.reference_lower_: Optional[np.ndarray] = None
        self.reference_upper_: Optional[np.ndarray] = None
        self.reference_density_: Optional[float] = None
        self.reference_mean_: Optional[np.ndarray] = None
        self.reference_std_: Optional[np.ndarray] = None
        self.covariate_dimension_: Optional[int] = None
        self.response_dimension_: Optional[int] = None

    def get_config(self) -> dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_forest_seed": self.seed,
            "n_jobs": self.n_jobs,
            "reference": self.reference,
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
        if self.reference == "gaussian":
            self.reference_mean_ = responses.mean(axis=0)
            self.reference_std_ = responses.std(axis=0)
            if (self.reference_std_ <= 0).any():
                raise ValueError("Gaussian reference requires nonzero response variance")
            fake_responses = generator.normal(self.reference_mean_, self.reference_std_,
                                              size=responses.shape)
        else:
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

    def paired_log_density(self, x: np.ndarray, y: np.ndarray, *,
                           normalization_samples: int = 1_000, seed: int = 0,
                           batch_size: int = 16) -> np.ndarray:
        """Evaluate matched (x_i, y_i) pairs using the archived query-box normalizer."""
        model, _ = self._require_fitted()
        covariates, responses = _matrix(x, "x"), _matrix(y, "y")
        if covariates.shape[0] != responses.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        self._check_dimensions(covariates, responses)
        lower, upper = responses.min(axis=0), responses.max(axis=0)
        widths = upper - lower
        if (widths <= 0).any():
            raise ValueError("test response range must be positive in every dimension")
        volume = float(np.prod(widths))
        rng = np.random.RandomState(seed)
        output = []
        for start in range(0, len(responses), batch_size):
            end = min(start + batch_size, len(responses))
            batch_x, batch_y = covariates[start:end], responses[start:end]
            observed_ratio = self._density_ratio(
                model, np.concatenate((batch_x, batch_y), axis=1))
            fake_y = rng.uniform(lower, upper,
                                 size=(len(batch_y) * normalization_samples, responses.shape[1]))
            fake_x = np.repeat(batch_x, normalization_samples, axis=0)
            fake_ratio = self._density_ratio(
                model, np.concatenate((fake_x, fake_y), axis=1))
            observed_log_q = self._reference_log_density(batch_y)
            fake_q = np.exp(self._reference_log_density(fake_y))
            normalizer = (fake_ratio * fake_q).reshape(len(batch_y), normalization_samples)
            normalizer = np.maximum(normalizer.mean(axis=1) * volume,
                                    np.finfo(np.float64).tiny)
            output.append(np.log(observed_ratio) + observed_log_q - np.log(normalizer))
        return np.concatenate(output)

    def _reference_log_density(self, y: np.ndarray) -> np.ndarray:
        if self.reference == "gaussian":
            return np.sum(norm.logpdf(y, loc=self.reference_mean_,
                                      scale=self.reference_std_), axis=1)
        return np.full(len(y), np.log(self.reference_density_))

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
        model, _ = self._require_fitted()
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
            query_reference = np.exp(self._reference_log_density(query_y)).reshape(
                batch_size, responses.shape[0])
            normalization_reference = np.exp(self._reference_log_density(
                normalization_y)).reshape(batch_size, normalization_samples)
            normalizer = volume * (normalization_reference * normalization_ratio).mean(axis=1)
            normalizer = np.maximum(normalizer, np.finfo(np.float64).tiny)
            output.append(query_reference * query_ratio / normalizer[:, None])
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
