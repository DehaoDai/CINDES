"""Conditional masked autoregressive flow used in the comparisons."""

import copy
from typing import Optional

import numpy as np
import torch
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import (
    CompositeTransform,
    MaskedAffineAutoregressiveTransform,
    ReversePermutation,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cindes.reproducibility import seed_everything


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


class ConditionalFlowNetwork(nn.Module):
    """Three-stage conditional MAF with two masked blocks per stage."""

    def __init__(
        self,
        covariate_dimension: int,
        response_dimension: int,
        depth: int,
        width: int,
    ) -> None:
        super().__init__()
        transforms = []
        for _ in range(depth):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=response_dimension,
                    hidden_features=width,
                    context_features=covariate_dimension,
                    num_blocks=2,
                    use_residual_blocks=False,
                )
            )
            transforms.append(ReversePermutation(features=response_dimension))
        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal([response_dimension]),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=y, context=x)


class ConditionalMAF:
    """Conditional MAF with validation-NLL weight-decay selection."""

    def __init__(
        self,
        depth: int = 3,
        width: int = 64,
        learning_rate: float = 1e-3,
        max_epochs: int = 200,
        patience: int = 2,
        batch_size: int = 500,
        weight_decay_candidates: tuple[float, ...] = (
            1e-3,
            5e-4,
            2e-4,
            1e-4,
            0.0,
        ),
        scheduler_gamma: float = 0.999,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        if depth < 1 or width < 1 or max_epochs < 1 or patience < 1 or batch_size < 1:
            raise ValueError("architecture and optimization counts must be positive")
        if learning_rate <= 0.0 or not weight_decay_candidates:
            raise ValueError("learning rate and weight-decay candidates must be valid")
        self.depth = depth
        self.width = width
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.weight_decay_candidates = weight_decay_candidates
        self.scheduler_gamma = scheduler_gamma
        self.seed = seed
        self.device = torch.device(device)

        self.model_: Optional[ConditionalFlowNetwork] = None
        self.covariate_dimension_: Optional[int] = None
        self.response_dimension_: Optional[int] = None
        self.best_weight_decay_: Optional[float] = None
        self.best_validation_nll_: Optional[float] = None
        self.selection_history_: list[dict[str, float]] = []

    def get_config(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "width": self.width,
            "masked_blocks_per_stage": 2,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "weight_decay_candidates": list(self.weight_decay_candidates),
            "scheduler_gamma": self.scheduler_gamma,
            "seed": self.seed,
            "device": str(self.device),
        }

    def fit(
        self,
        y: np.ndarray,
        x: np.ndarray,
        y_valid: np.ndarray,
        x_valid: np.ndarray,
    ) -> "ConditionalMAF":
        responses = _matrix(y, "y")
        covariates = _matrix(x, "x")
        validation_responses = _matrix(y_valid, "y_valid")
        validation_covariates = _matrix(x_valid, "x_valid")
        if covariates.shape[0] != responses.shape[0]:
            raise ValueError("x and y must contain the same number of rows")
        if validation_covariates.shape[0] != validation_responses.shape[0]:
            raise ValueError("x_valid and y_valid must contain the same number of rows")
        if covariates.shape[1] != validation_covariates.shape[1]:
            raise ValueError("training and validation covariate dimensions differ")
        if responses.shape[1] != validation_responses.shape[1]:
            raise ValueError("training and validation response dimensions differ")

        self.covariate_dimension_ = covariates.shape[1]
        self.response_dimension_ = responses.shape[1]
        self.selection_history_ = []
        seed_everything(self.seed)

        best_state = None
        best_nll = np.inf
        best_weight_decay = None
        for weight_decay in self.weight_decay_candidates:
            state, validation_nll, best_epoch = self._fit_candidate(
                covariates,
                responses,
                validation_covariates,
                validation_responses,
                weight_decay,
            )
            self.selection_history_.append(
                {
                    "weight_decay": float(weight_decay),
                    "validation_nll": float(validation_nll),
                    "best_epoch": float(best_epoch),
                }
            )
            if best_state is None or validation_nll < best_nll:
                best_state = state
                best_nll = validation_nll
                best_weight_decay = weight_decay

        if best_state is None or best_weight_decay is None:
            raise RuntimeError("model selection failed to produce a fitted flow")
        self.model_ = self._new_model()
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_validation_nll_ = float(best_nll)
        self.best_weight_decay_ = float(best_weight_decay)
        return self

    def _new_model(self) -> ConditionalFlowNetwork:
        if self.covariate_dimension_ is None or self.response_dimension_ is None:
            raise RuntimeError("dimensions are unavailable before fit")
        return ConditionalFlowNetwork(
            covariate_dimension=self.covariate_dimension_,
            response_dimension=self.response_dimension_,
            depth=self.depth,
            width=self.width,
        ).to(self.device)

    def _fit_candidate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        weight_decay: float,
    ) -> tuple[dict[str, torch.Tensor], float, int]:
        model = self._new_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.scheduler_gamma,
        )
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            num_workers=0,
        )

        best_state = copy.deepcopy(model.state_dict())
        best_nll = np.inf
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = -model(batch_x, batch_y).mean()
                loss.backward()
                optimizer.step()
            scheduler.step()

            validation_nll = self._negative_log_likelihood(model, x_valid, y_valid)
            if np.isfinite(validation_nll) and validation_nll < best_nll:
                best_state = copy.deepcopy(model.state_dict())
                best_nll = validation_nll
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if not np.isfinite(best_nll):
            raise RuntimeError("MAF training did not produce a finite validation NLL")
        return best_state, float(best_nll), best_epoch

    @torch.no_grad()
    def _negative_log_likelihood(
        self,
        model: ConditionalFlowNetwork,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        model.eval()
        log_density = model(
            torch.from_numpy(x).to(self.device),
            torch.from_numpy(y).to(self.device),
        )
        return float((-log_density.mean()).cpu().item())

    @torch.no_grad()
    def conditional_density_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        covariate_batch_size: int = 32,
    ) -> np.ndarray:
        model = self._require_fitted()
        covariates = _matrix(x, "x")
        responses = _matrix(y, "y")
        if covariates.shape[1] != self.covariate_dimension_:
            raise ValueError("covariate dimension differs from fitted data")
        if responses.shape[1] != self.response_dimension_:
            raise ValueError("response dimension differs from fitted data")
        if covariate_batch_size < 1:
            raise ValueError("covariate_batch_size must be positive")

        model.eval()
        response_tensor = torch.from_numpy(responses).to(self.device)
        output = []
        for start in range(0, covariates.shape[0], covariate_batch_size):
            stop = min(start + covariate_batch_size, covariates.shape[0])
            batch_x = torch.from_numpy(covariates[start:stop]).to(self.device)
            batch_size = batch_x.shape[0]
            grid_x = batch_x[:, None, :].expand(batch_size, responses.shape[0], -1)
            grid_y = response_tensor[None, :, :].expand(batch_size, -1, -1)
            log_density = model(
                grid_x.reshape(batch_size * responses.shape[0], -1),
                grid_y.reshape(batch_size * responses.shape[0], -1),
            ).reshape(batch_size, responses.shape[0])
            output.append(torch.exp(log_density.clamp(-60.0, 60.0)).cpu().numpy())
        return np.concatenate(output, axis=0)

    def _require_fitted(self) -> ConditionalFlowNetwork:
        if self.model_ is None:
            raise RuntimeError("fit must be called before density evaluation")
        return self.model_

    @torch.no_grad()
    def paired_log_density(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Log density for matched covariate-response pairs."""
        model = self._require_fitted()
        covariates, responses = _matrix(x, "x"), _matrix(y, "y")
        if covariates.shape[0] != responses.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        if covariates.shape[1] != self.covariate_dimension_ or responses.shape[1] != self.response_dimension_:
            raise ValueError("test dimensions differ from fitted data")
        model.eval()
        return model(torch.from_numpy(covariates).to(self.device),
                     torch.from_numpy(responses).to(self.device)).cpu().numpy()
