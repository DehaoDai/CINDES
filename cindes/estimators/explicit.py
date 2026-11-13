"""Algorithm 1: explicit classification-induced density estimation."""

import copy
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from cindes.config import NetworkConfig, OptimizationConfig, ValidationConfig
from cindes.models import DeepReLUNetwork
from cindes.reference import UniformBoxReference
from cindes.reproducibility import seed_everything


def _as_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, dimension)")
    if matrix.shape[0] < 1:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _covariates_or_empty(x: Optional[np.ndarray], sample_size: int) -> np.ndarray:
    if x is None:
        return np.empty((sample_size, 0), dtype=np.float32)
    matrix = _as_matrix(x, "x")
    if matrix.shape[0] != sample_size:
        raise ValueError("x and y must contain the same number of samples")
    return matrix


class ExplicitCINDES:
    """Neural explicit density estimator from Algorithm 1.

    The network learns ``log p(y | x) - log q(y)`` by classifying observed
    responses against fake responses from a known reference density ``q``.
    """

    def __init__(
        self,
        network: NetworkConfig = NetworkConfig(),
        optimization: OptimizationConfig = OptimizationConfig(),
        validation: ValidationConfig = ValidationConfig(),
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.network_config = network
        self.optimization_config = optimization
        self.validation_config = validation
        self.seed = seed
        self.device = torch.device(device)

        self.model_: Optional[DeepReLUNetwork] = None
        self.reference_: Optional[UniformBoxReference] = None
        self.response_dimension_: Optional[int] = None
        self.covariate_dimension_: Optional[int] = None
        self.best_weight_decay_: Optional[float] = None
        self.best_validation_nll_: Optional[float] = None
        self.selection_history_: list[dict[str, float]] = []

    def get_config(self) -> dict[str, object]:
        return {
            "network": asdict(self.network_config),
            "optimization": asdict(self.optimization_config),
            "validation": asdict(self.validation_config),
            "seed": self.seed,
            "device": str(self.device),
        }

    def fit(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        x_valid: Optional[np.ndarray] = None,
    ) -> "ExplicitCINDES":
        """Fit all weight-decay candidates and retain the best validation NLL."""

        responses = _as_matrix(y, "y")
        covariates = _covariates_or_empty(x, responses.shape[0])
        if y_valid is None:
            raise ValueError("y_valid is required for validation NLL model selection")
        validation_responses = _as_matrix(y_valid, "y_valid")
        validation_covariates = _covariates_or_empty(x_valid, validation_responses.shape[0])

        if validation_responses.shape[1] != responses.shape[1]:
            raise ValueError("training and validation response dimensions differ")
        if validation_covariates.shape[1] != covariates.shape[1]:
            raise ValueError("training and validation covariate dimensions differ")

        seed_everything(self.seed)
        rng = np.random.default_rng(self.seed)
        reference = UniformBoxReference.from_data(responses)
        fake_responses = reference.sample(responses.shape[0], rng=rng).astype(np.float32)
        normalization_samples = reference.sample(
            self.validation_config.reference_samples,
            rng=rng,
        ).astype(np.float32)

        self.reference_ = reference
        self.response_dimension_ = responses.shape[1]
        self.covariate_dimension_ = covariates.shape[1]
        self.selection_history_ = []

        best_state = None
        best_nll = np.inf
        best_weight_decay = None

        for candidate_index, weight_decay in enumerate(
            self.optimization_config.weight_decay_candidates
        ):
            candidate_state, candidate_nll, best_epoch = self._fit_candidate(
                responses=responses,
                covariates=covariates,
                fake_responses=fake_responses,
                y_valid=validation_responses,
                x_valid=validation_covariates,
                normalization_samples=normalization_samples,
                weight_decay=weight_decay,
                candidate_index=candidate_index,
            )
            self.selection_history_.append(
                {
                    "weight_decay": float(weight_decay),
                    "validation_nll": float(candidate_nll),
                    "best_epoch": float(best_epoch),
                }
            )
            if best_state is None or candidate_nll < best_nll:
                best_nll = candidate_nll
                best_weight_decay = weight_decay
                best_state = candidate_state

        if best_state is None or best_weight_decay is None:
            raise RuntimeError("model selection failed to produce a fitted model")

        self.model_ = self._new_model()
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_weight_decay_ = float(best_weight_decay)
        self.best_validation_nll_ = float(best_nll)
        return self

    def _new_model(self) -> DeepReLUNetwork:
        if self.response_dimension_ is None or self.covariate_dimension_ is None:
            raise RuntimeError("dimensions are unavailable before fit")
        return DeepReLUNetwork(
            input_dimension=self.response_dimension_ + self.covariate_dimension_,
            depth=self.network_config.depth,
            width=self.network_config.width,
            output_clip=self.network_config.output_clip,
        ).to(self.device)

    def _fit_candidate(
        self,
        responses: np.ndarray,
        covariates: np.ndarray,
        fake_responses: np.ndarray,
        y_valid: np.ndarray,
        x_valid: np.ndarray,
        normalization_samples: np.ndarray,
        weight_decay: float,
        candidate_index: int,
    ) -> tuple[dict[str, torch.Tensor], float, int]:
        del candidate_index
        if self.optimization_config.reset_seed_per_candidate:
            seed_everything(self.seed)
        model = self._new_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.optimization_config.learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = None
        if self.optimization_config.scheduler_gamma is not None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.optimization_config.scheduler_gamma,
            )
        dataset = TensorDataset(
            torch.from_numpy(responses),
            torch.from_numpy(covariates),
            torch.from_numpy(fake_responses),
        )
        loader_generator = None
        if self.optimization_config.reset_seed_per_candidate:
            loader_generator = torch.Generator()
            loader_generator.manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=min(self.optimization_config.batch_size, len(dataset)),
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
        )

        best_state = copy.deepcopy(model.state_dict())
        best_nll = np.inf
        best_epoch = 0
        stale_epochs = 0

        for epoch in range(1, self.optimization_config.max_epochs + 1):
            model.train()
            for batch_y, batch_x, batch_fake_y in loader:
                batch_y = batch_y.to(self.device)
                batch_x = batch_x.to(self.device)
                batch_fake_y = batch_fake_y.to(self.device)
                if self.optimization_config.fake_response_strategy == "batch_resampled":
                    lower = batch_y.amin(dim=0)
                    upper = batch_y.amax(dim=0)
                    batch_fake_y = lower + torch.rand_like(batch_y) * (upper - lower)

                real_features = torch.cat((batch_y, batch_x), dim=1)
                fake_features = torch.cat((batch_fake_y, batch_x), dim=1)
                optimizer.zero_grad(set_to_none=True)
                loss = F.softplus(-model(real_features)).mean()
                loss = loss + F.softplus(model(fake_features)).mean()
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            validation_nll = self._validation_nll(
                model=model,
                y=y_valid,
                x=x_valid,
                normalization_samples=normalization_samples,
            )
            if validation_nll < best_nll - self.optimization_config.min_delta:
                best_nll = validation_nll
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.optimization_config.patience:
                    break

        return best_state, float(best_nll), best_epoch

    @torch.no_grad()
    def _validation_nll(
        self,
        model: DeepReLUNetwork,
        y: np.ndarray,
        x: np.ndarray,
        normalization_samples: np.ndarray,
    ) -> float:
        model.eval()
        batch_size = self.validation_config.batch_size
        losses = []
        for start in range(0, y.shape[0], batch_size):
            stop = min(start + batch_size, y.shape[0])
            batch_y = torch.from_numpy(y[start:stop]).to(self.device)
            batch_x = torch.from_numpy(x[start:stop]).to(self.device)
            log_density = self._normalized_log_density_tensor(
                model=model,
                y=batch_y,
                x=batch_x,
                normalization_samples=normalization_samples,
            )
            losses.append(-log_density.cpu())
        return float(torch.cat(losses).mean().item())

    def _normalized_log_density_tensor(
        self,
        model: DeepReLUNetwork,
        y: torch.Tensor,
        x: torch.Tensor,
        normalization_samples: np.ndarray,
    ) -> torch.Tensor:
        if self.reference_ is None:
            raise RuntimeError("reference distribution is unavailable before fit")

        observed_features = torch.cat((y, x), dim=1)
        observed_logit = model(observed_features)
        observed_log_q = self.reference_.torch_log_prob(y, enforce_support=False)

        reference_y = torch.from_numpy(normalization_samples).to(self.device)
        batch_size = y.shape[0]
        sample_size = reference_y.shape[0]
        if x.shape[1] == 0:
            reference_logits = model(reference_y)
            log_normalizer = torch.logsumexp(reference_logits, dim=0) - np.log(sample_size)
            return observed_logit + observed_log_q - log_normalizer

        expanded_y = reference_y.unsqueeze(0).expand(batch_size, sample_size, -1)
        expanded_x = x.unsqueeze(1).expand(batch_size, sample_size, -1)
        reference_features = torch.cat((expanded_y, expanded_x), dim=2)
        reference_logits = model(reference_features.reshape(batch_size * sample_size, -1))
        reference_logits = reference_logits.reshape(batch_size, sample_size)
        log_normalizer = torch.logsumexp(reference_logits, dim=1) - np.log(sample_size)
        return observed_logit + observed_log_q - log_normalizer

    @torch.no_grad()
    def log_density(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate the unnormalized Algorithm 1 density estimate."""

        model, reference = self._require_fitted()
        responses = _as_matrix(y, "y")
        covariates = _covariates_or_empty(x, responses.shape[0])
        self._check_dimensions(responses, covariates)

        output = []
        batch_size = self.validation_config.batch_size
        model.eval()
        for start in range(0, responses.shape[0], batch_size):
            stop = min(start + batch_size, responses.shape[0])
            batch_y = torch.from_numpy(responses[start:stop]).to(self.device)
            batch_x = torch.from_numpy(covariates[start:stop]).to(self.device)
            features = torch.cat((batch_y, batch_x), dim=1)
            log_density = model(features) + reference.torch_log_prob(
                batch_y,
                enforce_support=False,
            )
            output.append(log_density.cpu().numpy())
        return np.concatenate(output)

    @torch.no_grad()
    def normalized_log_density(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        reference_samples: int = 1_000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Evaluate a Monte Carlo normalized density as in Remark 4."""

        model, reference = self._require_fitted()
        responses = _as_matrix(y, "y")
        covariates = _covariates_or_empty(x, responses.shape[0])
        self._check_dimensions(responses, covariates)
        if reference_samples < 1:
            raise ValueError("reference_samples must be positive")

        evaluation_seed = self.seed + 10_000 if seed is None else seed
        rng = np.random.default_rng(evaluation_seed)
        normalization_samples = reference.sample(reference_samples, rng=rng).astype(np.float32)

        output = []
        batch_size = self.validation_config.batch_size
        model.eval()
        for start in range(0, responses.shape[0], batch_size):
            stop = min(start + batch_size, responses.shape[0])
            batch_y = torch.from_numpy(responses[start:stop]).to(self.device)
            batch_x = torch.from_numpy(covariates[start:stop]).to(self.device)
            log_density = self._normalized_log_density_tensor(
                model=model,
                y=batch_y,
                x=batch_x,
                normalization_samples=normalization_samples,
            )
            output.append(log_density.cpu().numpy())
        return np.concatenate(output)

    def density(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        normalize: bool = False,
        reference_samples: int = 1_000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Evaluate ``exp(logit) * q(y)``, optionally normalized."""

        if normalize:
            log_density = self.normalized_log_density(
                y=y,
                x=x,
                reference_samples=reference_samples,
                seed=seed,
            )
        else:
            log_density = self.log_density(y=y, x=x)
        return np.exp(log_density)

    @torch.no_grad()
    def conditional_density_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        reference_samples: int = 1_000,
        seed: Optional[int] = None,
        covariate_batch_size: int = 32,
    ) -> np.ndarray:
        """Evaluate normalized densities on the Cartesian product of x and y."""

        model, reference = self._require_fitted()
        covariates = _as_matrix(x, "x")
        responses = _as_matrix(y, "y")
        if covariates.shape[1] != self.covariate_dimension_:
            raise ValueError("covariate dimension differs from fitted data")
        if responses.shape[1] != self.response_dimension_:
            raise ValueError("response dimension differs from fitted data")
        if self.covariate_dimension_ == 0:
            raise ValueError("conditional_density_grid requires fitted covariates")
        if reference_samples < 1:
            raise ValueError("reference_samples must be positive")
        if covariate_batch_size < 1:
            raise ValueError("covariate_batch_size must be positive")

        evaluation_seed = self.seed + 10_000 if seed is None else seed
        rng = np.random.default_rng(evaluation_seed)
        reference_values = reference.sample(reference_samples, rng=rng).astype(np.float32)
        reference_y = torch.from_numpy(reference_values).to(self.device)
        response_y = torch.from_numpy(responses).to(self.device)
        response_log_q = reference.torch_log_prob(response_y, enforce_support=False)

        model.eval()
        output = []
        for start in range(0, covariates.shape[0], covariate_batch_size):
            stop = min(start + covariate_batch_size, covariates.shape[0])
            batch_x = torch.from_numpy(covariates[start:stop]).to(self.device)
            batch_size = batch_x.shape[0]

            observed_x = batch_x[:, None, :].expand(
                batch_size,
                responses.shape[0],
                -1,
            )
            observed_y = response_y[None, :, :].expand(batch_size, -1, -1)
            observed_features = torch.cat((observed_y, observed_x), dim=2)
            observed_logits = model(
                observed_features.reshape(batch_size * responses.shape[0], -1)
            ).reshape(batch_size, responses.shape[0])

            normalization_x = batch_x[:, None, :].expand(
                batch_size,
                reference_samples,
                -1,
            )
            normalization_y = reference_y[None, :, :].expand(batch_size, -1, -1)
            normalization_features = torch.cat(
                (normalization_y, normalization_x),
                dim=2,
            )
            normalization_logits = model(
                normalization_features.reshape(batch_size * reference_samples, -1)
            ).reshape(batch_size, reference_samples)
            log_normalizer = torch.logsumexp(normalization_logits, dim=1) - np.log(
                reference_samples
            )

            log_density = observed_logits + response_log_q[None, :] - log_normalizer[:, None]
            output.append(torch.exp(log_density.clamp(-60.0, 60.0)).cpu().numpy())
        return np.concatenate(output, axis=0)

    def _check_dimensions(self, y: np.ndarray, x: np.ndarray) -> None:
        if y.shape[1] != self.response_dimension_:
            raise ValueError("response dimension differs from fitted data")
        if x.shape[1] != self.covariate_dimension_:
            raise ValueError("covariate dimension differs from fitted data")

    def _require_fitted(self) -> tuple[DeepReLUNetwork, UniformBoxReference]:
        if self.model_ is None or self.reference_ is None:
            raise RuntimeError("fit must be called before density evaluation")
        return self.model_, self.reference_
