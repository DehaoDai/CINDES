"""Numerics for the known product log-Exponential(1) reference.

No target potential, hidden graph, evaluation sample, or target score is used.
The reference is the law of log E for E~Exponential(rate=1), so its one-
coordinate density is exp(z-exp(z)).  Diffusion acts in these log coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time
import hashlib

import numpy as np
import torch


def sample_reference(rng, shape):
    """Independent log exponentials; redraw exact floating-point endpoints."""
    values = rng.exponential(size=shape)
    while np.any(values == 0):
        mask = values == 0
        values[mask] = rng.exponential(size=int(mask.sum()))
    return np.log(values).astype(np.float32)


def log_reference(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - np.exp(x)).sum(axis=-1)


def log_reference_torch(x):
    return (x - torch.exp(x)).sum(dim=-1)


@dataclass(frozen=True)
class QuadratureConfig:
    nodes: int = 128
    log_density_drop: float = 36.0
    mode_iterations: int = 24
    boundary_iterations: int = 22


class LogExponentialReference:
    """Adaptive Gauss--Legendre quadrature for q_t and Z_0|Z_t.

    All reference arithmetic is double precision.  The posterior is strictly
    log concave; integration endpoints are selected by a fixed log-density
    drop from its mode.  This is a numerical quadrature, not an exact sampler.
    """
    def __init__(self, config=None, device="cpu"):
        self.config = config if isinstance(config, QuadratureConfig) else QuadratureConfig(**(config or {}))
        if self.config.nodes < 32 or self.config.log_density_drop < 20:
            raise ValueError("Insufficient quadrature resolution or tail interval")
        self.device = torch.device(device)
        locations, weights = np.polynomial.legendre.leggauss(self.config.nodes)
        self.locations = torch.as_tensor(locations, dtype=torch.float64, device=self.device)
        self.weights = torch.as_tensor(weights, dtype=torch.float64, device=self.device)

    def _time(self, y, t):
        value = torch.as_tensor(t, dtype=torch.float64, device=self.device)
        if value.numel() == 1:
            value = value.reshape(1, 1)
        elif value.shape in ((len(y),), (len(y), 1)):
            value = value.reshape(len(y), 1)
        else:
            raise ValueError("t must be scalar or one time per row")
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError("Positive finite forward times are required")
        return torch.exp(-value), -torch.expm1(-2 * value)

    @torch.no_grad()
    def components(self, y, t):
        if y.ndim != 2 or y.device != self.device or not bool(torch.isfinite(y).all()):
            raise ValueError("Finite (batch,d) points on the reference device required")
        yd = y.double()
        m, sigma2 = self._time(y, t)
        a = m.square() / sigma2
        b = m * yd / sigma2
        zero = torch.zeros_like(b)
        lower = torch.where(b >= 0, zero, b / a)
        upper = torch.where(b >= 0,
                            torch.minimum(b / a, torch.log1p(b.clamp_min(0))),
                            torch.minimum(zero, (1 + b) / a))
        c = upper.clone()
        for _ in range(self.config.mode_iterations):
            ec = torch.exp(c)
            f = ec + a * c - 1 - b
            stationary = f.abs() <= 8 * torch.finfo(torch.float64).eps * (1 + b.abs() + (a * c).abs())
            lower = torch.where(f < 0, c, lower)
            upper = torch.where(f > 0, c, upper)
            candidate = c - f / (ec + a)
            outside = (candidate < lower) | (candidate > upper) | ~torch.isfinite(candidate)
            candidate = torch.where(outside, (lower + upper) / 2, candidate)
            c = torch.where(stationary, c, candidate)
        ec = torch.exp(c)
        residual = ec + a * c - 1 - b
        relative_residual = residual.abs() / (1 + b.abs() + (a * c).abs())
        if float(relative_residual.max()) > 2e-11:
            raise ArithmeticError("Posterior mode solver did not converge")
        cutoff = self.config.log_density_drop

        def loss(u):
            # The residual term keeps this exactly equal to log p(c)-log p(c+u)
            # even at the finite-precision mode returned by Newton's method.
            return ec * (torch.expm1(u) - u) + .5 * a * u.square() + residual * u

        width = torch.sqrt(2 * cutoff / (a + ec))
        left_high = width.clone()
        for _ in range(16):
            too_near = loss(-left_high) < cutoff
            if not bool(too_near.any()):
                break
            left_high = torch.where(too_near, 2 * left_high, left_high)
        if bool((loss(-left_high) < cutoff).any()):
            raise ArithmeticError("Failed to bracket left posterior boundary")
        left_low, right_low = torch.zeros_like(width), torch.zeros_like(width)
        right_high = width.clone()
        for _ in range(self.config.boundary_iterations):
            mid = (left_low + left_high) / 2
            inside = loss(-mid) < cutoff
            left_low = torch.where(inside, mid, left_low)
            left_high = torch.where(inside, left_high, mid)
            mid = (right_low + right_high) / 2
            inside = loss(mid) < cutoff
            right_low = torch.where(inside, mid, right_low)
            right_high = torch.where(inside, right_high, mid)
        left = -left_high
        right = right_high
        half = (right - left) / 2
        offsets = half[..., None] * self.locations + ((right + left) / 2)[..., None]
        nodes = c[..., None] + offsets
        drop = ec[..., None] * (torch.expm1(offsets) - offsets)
        drop += .5 * a[..., None] * offsets.square() + residual[..., None] * offsets
        log_mass = torch.log(half[..., None] * self.weights) - drop
        log_integral = torch.logsumexp(log_mass, dim=-1)
        probabilities = torch.softmax(log_mass, dim=-1)
        log_at_mode = c - ec - .5 * (yd - m * c).square() / sigma2 - .5 * torch.log(2 * math.pi * sigma2)
        log_density = log_at_mode + log_integral
        score = -(probabilities * torch.expm1(nodes)).sum(-1) / m
        edge_mass = torch.maximum(probabilities[..., :2].sum(-1), probabilities[..., -2:].sum(-1)).max()
        return {"nodes": nodes, "probabilities": probabilities, "score": score,
                "log_density": log_density, "mode": c, "m": m, "sigma2": sigma2,
                "edge_mass": float(edge_mass), "mode_relative_residual": float(relative_residual.max())}

    @torch.no_grad()
    def score(self, y, t):
        return self.components(y, t)["score"].to(y.dtype)

    @torch.no_grad()
    def log_density(self, y, t):
        return self.components(y, t)["log_density"].sum(dim=1)

    @torch.no_grad()
    def antithetic(self, y, t, half_particles, generator):
        comp = self.components(y, t)
        b, d = y.shape
        cdf = comp["probabilities"].cumsum(-1)
        cdf[..., -1] = 1.0
        uniforms = torch.rand((b, d, half_particles), dtype=torch.float64, device=y.device, generator=generator)
        uniforms = torch.cat((uniforms, 1 - uniforms), dim=-1)
        indices = torch.searchsorted(cdf.contiguous(), uniforms.contiguous()).clamp_max(self.config.nodes - 1)
        points = torch.gather(comp["nodes"], -1, indices).transpose(1, 2)
        return comp, points


DEFAULT_TABLE_PATH = Path(__file__).with_name("reference_score_table.npz")


def reference_asset_hash(path=DEFAULT_TABLE_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_score_table(path=DEFAULT_TABLE_PATH, *, n_times=801, n_points=1601,
                        t_min=.001, t_max=3., y_min=-24., y_max=8., chunk=2048):
    """Build a fixed interpolation table from the reference only.

    Reproducible numerical acceleration, unrelated to fitting or test data.
    Cubic interpolation is independently checked away from its table nodes.
    """
    path = Path(path)
    if path.exists():
        return path
    log_times = np.linspace(math.log(t_min), math.log(t_max), n_times)
    points = np.linspace(y_min, y_max, n_points)
    flat_times = np.repeat(np.exp(log_times), n_points)
    flat_points = np.tile(points, n_times)
    scores = np.empty(len(flat_points), dtype=np.float64)
    posterior = LogExponentialReference()
    started = time.perf_counter()
    for start in range(0, len(scores), chunk):
        stop = min(start + chunk, len(scores))
        values = torch.as_tensor(flat_points[start:stop, None])
        times = torch.as_tensor(flat_times[start:stop, None])
        scores[start:stop] = posterior.score(values, times).numpy().reshape(-1)
    np.savez_compressed(path, log_times=log_times, points=points,
                        scores=scores.reshape(n_times, n_points),
                        preparation_seconds=time.perf_counter() - started,
                        quadrature_nodes=posterior.config.nodes,
                        log_density_drop=posterior.config.log_density_drop)
    return path


class ReferenceScore:
    """Cubic interpolation of the fixed q_t score with quadrature fallback.

    SciPy's tensor-product cubic spline is evaluated on the CPU; tensors are
    returned to the input device. GPU use therefore includes CPU transfers.
    This object has no trainable parameters.
    """
    def __init__(self, device="cpu", table_path=DEFAULT_TABLE_PATH):
        from scipy.interpolate import RectBivariateSpline
        self.device = torch.device(device)
        if not Path(table_path).exists():
            raise FileNotFoundError(f"Reference score table missing: run python reference.py --prepare-table ({table_path})")
        with np.load(table_path) as saved:
            self.log_times = saved["log_times"].copy()
            self.points = saved["points"].copy()
            self.values = saved["scores"].copy()
        self.spline = RectBivariateSpline(self.log_times, self.points, self.values, kx=3, ky=3, s=0)
        self.posterior = LogExponentialReference(device=device)

    @torch.no_grad()
    def score(self, y, t):
        # Reference evaluation is detached: no model parameters enter q_t.
        td = torch.as_tensor(t, dtype=torch.float64, device=y.device)
        if td.numel() == 1:
            td = td.expand(len(y), 1)
        else:
            td = td.reshape(len(y), 1)
        yy = y.detach().double().cpu().numpy()
        lt = np.broadcast_to(td.log().cpu().numpy(), yy.shape)
        outside = ((lt < self.log_times[0] - 1e-12) | (lt > self.log_times[-1] + 1e-12)
                   | (yy < self.points[0]) | (yy > self.points[-1]))
        values = self.spline.ev(lt.reshape(-1), yy.reshape(-1)).reshape(yy.shape)
        if np.any(outside):
            # Compute whole affected rows so the exact routine's shape contract
            # and per-row time convention remain simple and auditable.
            rows = np.flatnonzero(outside.any(axis=1))
            idx = torch.as_tensor(rows, device=y.device)
            exact = self.posterior.score(y[idx].double(), td[idx]).cpu().numpy()
            values[rows] = exact
        return torch.as_tensor(values, dtype=y.dtype, device=y.device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-table", action="store_true")
    parser.add_argument("--path", default=str(DEFAULT_TABLE_PATH))
    args = parser.parse_args()
    if not args.prepare_table:
        parser.error("Pass --prepare-table")
    print(prepare_score_table(args.path), flush=True)
