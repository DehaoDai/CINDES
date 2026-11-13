"""Python interface to the official LinCDE R package."""

from typing import Optional

import numpy as np


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


class LinCDEEstimator:
    """Fit and evaluate univariate conditional densities with LinCDE 0.2.2."""

    def __init__(self, n_trees: int = 100, seed: int = 0) -> None:
        if n_trees < 1:
            raise ValueError("n_trees must be positive")
        self.n_trees = n_trees
        self.seed = seed
        self.model_ = None
        self.covariate_dimension_: Optional[int] = None
        self.package_version_: Optional[str] = None

    @staticmethod
    def is_available() -> bool:
        try:
            from rpy2 import robjects

            available = robjects.r["requireNamespace"]("LinCDE", quietly=True)
            return bool(available[0])
        except Exception:
            return False

    def get_config(self) -> dict[str, object]:
        return {
            "r_package": "LinCDE",
            "r_package_version": self.package_version_,
            "n_trees": self.n_trees,
            "seed": self.seed,
            "other_hyperparameters": "LinCDE.boost defaults",
        }

    def fit(self, y: np.ndarray, x: np.ndarray) -> "LinCDEEstimator":
        responses = _matrix(y, "y")
        covariates = _matrix(x, "x")
        if responses.shape[1] != 1:
            raise ValueError("LinCDE supports only a one-dimensional response")
        if covariates.shape[0] != responses.shape[0]:
            raise ValueError("x and y must contain the same number of rows")

        robjects, numpy2ri, localconverter = self._r_interface()
        robjects.r('suppressPackageStartupMessages(library("LinCDE"))')
        robjects.r["set.seed"](self.seed)
        with localconverter(robjects.default_converter + numpy2ri.converter):
            r_x = robjects.conversion.get_conversion().py2rpy(covariates)
            r_y = robjects.conversion.get_conversion().py2rpy(responses[:, 0])
        fit_function = robjects.r(
            "function(y, X, n_trees) {LinCDE.boost(y=y, X=X, n.trees=n_trees, verbose=FALSE)}"
        )
        self.model_ = fit_function(r_y, r_x, self.n_trees)
        self.covariate_dimension_ = covariates.shape[1]
        self.package_version_ = str(robjects.r('as.character(packageVersion("LinCDE"))')[0])
        return self

    def raw_conditional_density_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        covariate_batch_size: int = 32,
    ) -> np.ndarray:
        """Return the conditional densities produced by ``predict.LinCDE``."""

        self._require_fitted()
        covariates = _matrix(x, "x")
        responses = _matrix(y, "y")
        if covariates.shape[1] != self.covariate_dimension_:
            raise ValueError("covariate dimension differs from fitted data")
        if responses.shape[1] != 1:
            raise ValueError("LinCDE supports only a one-dimensional response")
        if covariate_batch_size < 1:
            raise ValueError("covariate_batch_size must be positive")

        output = []
        for start in range(0, covariates.shape[0], covariate_batch_size):
            stop = min(start + covariate_batch_size, covariates.shape[0])
            batch_x = covariates[start:stop]
            paired_x = np.repeat(batch_x, responses.shape[0], axis=0)
            paired_y = np.tile(responses[:, 0], batch_x.shape[0])
            density = self._predict_paired(paired_x, paired_y)
            output.append(density.reshape(batch_x.shape[0], responses.shape[0]))
        return np.concatenate(output, axis=0)

    def conditional_density_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        normalization_samples: int = 1_000,
        seed: Optional[int] = None,
        covariate_batch_size: int = 32,
    ) -> np.ndarray:
        """Apply the query-range normalization used by the archived experiments."""

        responses = _matrix(y, "y")
        if responses.shape[1] != 1:
            raise ValueError("LinCDE supports only a one-dimensional response")
        if normalization_samples < 1:
            raise ValueError("normalization_samples must be positive")
        lower = float(responses[:, 0].min())
        upper = float(responses[:, 0].max())
        if upper <= lower:
            raise ValueError("the response query range must have positive width")
        generator = np.random.RandomState(self.seed if seed is None else seed)
        normalization_y = generator.uniform(lower, upper, size=(normalization_samples, 1))

        query_density = self.raw_conditional_density_grid(
            x,
            responses,
            covariate_batch_size=covariate_batch_size,
        )
        normalization_density = self.raw_conditional_density_grid(
            x,
            normalization_y,
            covariate_batch_size=covariate_batch_size,
        )
        normalizer = (upper - lower) * normalization_density.mean(axis=1)
        normalizer = np.maximum(normalizer, np.finfo(np.float64).tiny)
        return query_density / normalizer[:, None]

    def _predict_paired(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        model = self._require_fitted()
        robjects, numpy2ri, localconverter = self._r_interface()
        with localconverter(robjects.default_converter + numpy2ri.converter):
            r_x = robjects.conversion.get_conversion().py2rpy(x)
            r_y = robjects.conversion.get_conversion().py2rpy(y)
        prediction = robjects.r["predict"](model, X=r_x, y=r_y)
        with localconverter(robjects.default_converter + numpy2ri.converter):
            density = robjects.conversion.get_conversion().rpy2py(prediction)
        values = np.asarray(density, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise RuntimeError("LinCDE returned invalid density values")
        return values

    @staticmethod
    def _r_interface():
        try:
            from rpy2 import robjects
            from rpy2.robjects import numpy2ri
            from rpy2.robjects.conversion import localconverter
        except ImportError as error:
            raise RuntimeError(
                "LinCDE requires R, the LinCDE R package, and the optional rpy2 dependency"
            ) from error
        available = robjects.r["requireNamespace"]("LinCDE", quietly=True)
        if not bool(available[0]):
            raise RuntimeError("the LinCDE R package is not installed")
        return robjects, numpy2ri, localconverter

    def _require_fitted(self):
        if self.model_ is None:
            raise RuntimeError("fit must be called before density evaluation")
        return self.model_
