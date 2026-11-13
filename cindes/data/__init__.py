"""Data-generating processes for the CINDES numerical examples."""

from cindes.data.conditional import (
    ConditionalDensityScenario,
    MultivariateConditionalScenario,
    conditional_covariates,
    conditional_scenario,
    multivariate_conditional_scenario,
)
from cindes.data.unconditional import (
    GaussianMixtureScenario,
    elliptical_gaussian_mixture,
    evaluation_grid,
    spherical_gaussian_mixture,
)

__all__ = [
    "ConditionalDensityScenario",
    "MultivariateConditionalScenario",
    "GaussianMixtureScenario",
    "conditional_covariates",
    "conditional_scenario",
    "multivariate_conditional_scenario",
    "elliptical_gaussian_mixture",
    "evaluation_grid",
    "spherical_gaussian_mixture",
]
