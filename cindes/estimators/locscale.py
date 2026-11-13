"""Location-scale neural baseline for the Light Chamber experiment."""

import copy
from typing import Optional

import numpy as np
import torch
from torch import nn

from cindes.reproducibility import seed_everything


class _MeanVarianceNet(nn.Module):
    def __init__(self, input_dim: int, depth: int = 3, width: int = 32):
        super().__init__()
        layers = []
        for layer in range(depth):
            layers.extend((nn.Linear(input_dim if layer == 0 else width, width), nn.ReLU()))
        self.shared = nn.Sequential(*layers)
        self.mean = nn.Linear(width, 1)
        self.variance = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(x)
        return self.mean(hidden), self.variance(hidden)


class LocScaleNN:
    """Fit the archived mean/variance loss and evaluate a factorized Gaussian NLL."""

    def __init__(self, *, max_epochs: int = 1000, patience: int = 100, seed: int = 0,
                 device: str = "cpu") -> None:
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.device = torch.device(device)
        self.model_: Optional[_MeanVarianceNet] = None

    def fit(self, x: np.ndarray, y: np.ndarray, x_valid: np.ndarray,
            y_valid: np.ndarray) -> "LocScaleNN":
        seed_everything(self.seed)
        x_train = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        x_val = torch.as_tensor(x_valid, dtype=torch.float32, device=self.device)
        y_val = torch.as_tensor(y_valid, dtype=torch.float32, device=self.device)
        model = _MeanVarianceNet(x_train.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        best_nll = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        stale = 0
        for _ in range(self.max_epochs):
            model.train()
            mean, variance = model(x_train)
            residual_sq = (y_train - mean).square()
            loss = (residual_sq + (residual_sq - variance).square()).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            nll = self._nll(model, x_val, y_val)
            if np.isfinite(nll) and nll < best_nll:
                best_nll = nll
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        model.load_state_dict(best_state)
        self.model_ = model.eval()
        return self

    @staticmethod
    @torch.no_grad()
    def _nll(model: _MeanVarianceNet, x: torch.Tensor, y: torch.Tensor) -> float:
        model.eval()
        mean, variance = model(x)
        variance = variance.abs().clamp_min(1e-6)
        log_density = -0.5 * (np.log(2.0 * np.pi) + variance.log()
                              + (y - mean).square() / variance)
        return float(-log_density.sum(dim=1).mean().item())

    def negative_log_likelihood(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.model_ is None:
            raise RuntimeError("fit must be called before evaluation")
        return self._nll(
            self.model_,
            torch.as_tensor(x, dtype=torch.float32, device=self.device),
            torch.as_tensor(y, dtype=torch.float32, device=self.device),
        )
