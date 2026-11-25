"""Paired reverse-OU sampling in common Z=log(Y) coordinates.

The functions return Z, never clipped or exponentiated values.  The caller
applies exp once, after generation, to obtain positive observations Y.
Only the known reference and learned model enter the sampling algorithms.
"""
from __future__ import annotations

import math
import time
from typing import Callable
import numpy as np
import torch
from reference import LogExponentialReference, QuadratureConfig, sample_reference


class ScoreDiagnostics:
    def __init__(self):
        self.by_time = {}

    def update(self, t, weights, edge_mass):
        key = f"{float(t):.12g}"
        ess = 1.0 / weights.double().square().sum(dim=1)
        maximum = weights.max(dim=1).values
        row = self.by_time.setdefault(key, {
            "forward_time": float(t), "observations": 0, "score_calls": 0,
            "sum_weight_ess": 0.0, "min_weight_ess": float("inf"),
            "sum_max_normalized_weight": 0.0, "max_normalized_weight": 0.0,
            "max_quadrature_endpoint_mass": 0.0})
        row["observations"] += len(weights)
        row["score_calls"] += 1
        row["sum_weight_ess"] += float(ess.sum())
        row["min_weight_ess"] = min(row["min_weight_ess"], float(ess.min()))
        row["sum_max_normalized_weight"] += float(maximum.sum())
        row["max_normalized_weight"] = max(row["max_normalized_weight"], float(maximum.max()))
        row["max_quadrature_endpoint_mass"] = max(row["max_quadrature_endpoint_mass"], edge_mass)

    def summary(self, K):
        rows = []
        for row in sorted(self.by_time.values(), key=lambda r: -r["forward_time"]):
            value = dict(row)
            value["mean_weight_ess"] = value.pop("sum_weight_ess") / value["observations"]
            value["mean_weight_ess_fraction"] = value["mean_weight_ess"] / K
            value["mean_max_normalized_weight"] = value.pop("sum_max_normalized_weight") / value["observations"]
            rows.append(value)
        return {"weight_ess_interpretation": "Weight concentration diagnostic; antithetic particles are dependent.",
                "by_forward_time": rows,
                "minimum_weight_ess": min((r["min_weight_ess"] for r in rows), default=None),
                "maximum_normalized_weight": max((r["max_normalized_weight"] for r in rows), default=None),
                "maximum_quadrature_endpoint_mass": max((r["max_quadrature_endpoint_mass"] for r in rows), default=None)}


@torch.no_grad()
def cindes_score(model, y, t, K, generator, posterior, diagnostics=None):
    """Reference-posterior importance score with a centered control variate.

    q_t score + m/sigma² * sum_k (w_k-1/K)(Z_k-mode).
    Self-normalization has finite-K bias; quadrature is a separate numerical
    approximation. A constant classifier yields the reference score exactly.
    """
    if not isinstance(K, int) or K < 2 or K % 2:
        raise ValueError("K must be an even integer at least two")
    b, d = y.shape
    comp, points = posterior.antithetic(y, t, K // 2, generator)
    logits = model(points.to(y.dtype).reshape(b * K, d)).reshape(b, K).double()
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Nonfinite CINDES logits during score evaluation")
    weights = torch.softmax(logits, dim=1)
    centered = points - comp["mode"][:, None, :]
    # Subtracting the common mode is algebraically exact. Double accumulation
    # prevents small-time amplification of a difference of nearly equal means.
    correction = ((weights - 1.0 / K)[:, :, None] * centered).sum(1)
    m = math.exp(-float(t))
    sigma2 = -math.expm1(-2 * float(t))
    score = comp["score"] + (m / sigma2) * correction
    if not bool(torch.isfinite(score).all()):
        raise FloatingPointError("Nonfinite CINDES score")
    if diagnostics is not None:
        diagnostics.update(t, weights, comp["edge_mass"])
    return score.to(y.dtype)


def make_timesteps(T, delta, M):
    if T <= 1 or not 0 < delta < 1 or M < 2 or M % 2:
        raise ValueError("Require T>1, 0<delta<1, and even M>=2")
    return np.array([(T - 1) * j / (M / 2) if j <= M / 2
                     else T - delta ** ((2 * j - M) / M)
                     for j in range(M + 1)])


@torch.no_grad()
def sample_reverse(model, method, n, d=7, *, seed=0, particle_seed=None,
                   K=1000, M=200, T=3.0, delta=.001, batch=64,
                   device="cpu", quadrature=None, progress: Callable | None = None):
    """Return log-coordinate samples and metadata; noise on every step."""
    if method not in ("cindes", "dsm", "reference") or min(n, d, batch) < 1:
        raise ValueError("Invalid method, dimension, count, or batch size")
    device = torch.device(device)
    if model is not None:
        model = model.to(device).eval()
    reverse = torch.Generator(device=device).manual_seed(int(seed) + 999)
    particles = torch.Generator(device=device).manual_seed(int(particle_seed) if particle_seed is not None else int(seed) + 555)
    posterior = LogExponentialReference(quadrature, device=device)
    diagnostic = ScoreDiagnostics() if method == "cindes" else None
    times = make_timesteps(T, delta, M)
    outputs = []
    started = time.perf_counter()
    for start in range(0, n, batch):
        size = min(batch, n - start)
        y = torch.randn((size, d), generator=reverse, device=device, dtype=torch.float32)
        for j in range(M):
            tau, tau_next = T - times[j], T - times[j + 1]
            if method == "cindes":
                score = cindes_score(model, y, tau, K, particles, posterior, diagnostic)
            elif method == "dsm":
                score = model(y, torch.full((size, 1), tau, dtype=y.dtype, device=device))
            else:
                score = posterior.score(y, tau)
            h = tau - tau_next
            decay = math.exp(-h)
            mean = decay * y + 2 * (-math.expm1(-h)) * (score + y)
            noise = torch.randn(y.shape, generator=reverse, device=device, dtype=y.dtype)
            y = mean + math.sqrt(-math.expm1(-2 * h)) * noise
            if not bool(torch.isfinite(y).all()):
                raise FloatingPointError(f"Nonfinite reverse sample at forward time {tau_next}")
        outputs.append(y.cpu())
        if progress is not None:
            progress({"completed": start + size, "n": n, "elapsed_seconds": time.perf_counter() - started})
    values = torch.cat(outputs).numpy()
    result = {"generation_seconds": time.perf_counter() - started, "method": method,
              "n": n, "d": d, "T": T, "delta": delta, "M": M,
              "K": K if method == "cindes" else None, "generation_batch": batch,
              "reverse_seed": int(seed) + 999,
              "particle_seed": int(particle_seed) if particle_seed is not None else int(seed) + 555,
              "quadrature": posterior.config.__dict__,
              "sampler": "exponential Euler for reverse OU in common log coordinates",
              "returned_coordinates": "Z=log(Y); caller exponentiates once after generation",
              "last_reverse_step_noise": True, "terminal_correction": "none", "support_clipping": False,
              "maximum_absolute_log_coordinate": float(np.max(np.abs(values)))}
    if diagnostic is not None:
        result["score_diagnostics"] = diagnostic.summary(K)
    return values, result


@torch.no_grad()
def sample_cindes_rejection(model, n, d=7, *, seed=0, batch=8192,
                            device="cpu", max_proposals=10000000, progress: Callable | None = None):
    """Direct fitted-density diagnostic: return log-coordinate samples.

    Proposals are q_Z; accept with exp(logit-upper_bound). No target density,
    H function, or generating graph enters this operation.
    """
    if min(n, d, batch) < 1 or max_proposals < n:
        raise ValueError("Invalid rejection settings")
    model = model.to(device).eval()
    bound = float(model.logit_upper_bound())
    if not math.isfinite(bound):
        raise FloatingPointError("Nonfinite learned rejection envelope")
    rng = np.random.default_rng(seed)
    pieces = []
    proposed = accepted = retained = batches = 0
    maximum = -float("inf")
    started = time.perf_counter()
    while retained < n:
        size = min(batch, max_proposals - proposed)
        if size <= 0:
            raise RuntimeError("Rejection proposal cap reached; no fallback was used")
        proposals = sample_reference(rng, (size, d))
        logits = model(torch.as_tensor(proposals, device=device)).reshape(-1).double().cpu().numpy()
        log_acceptance = logits - bound
        if not np.isfinite(log_acceptance).all() or np.max(log_acceptance) > 2e-6 * max(1, abs(bound)):
            raise ArithmeticError("Invalid fitted-density rejection envelope")
        maximum = max(maximum, float(log_acceptance.max()))
        with np.errstate(divide="ignore"):
            kept = proposals[np.log(rng.random(size)) < log_acceptance]
        proposed += size
        accepted += len(kept)
        batches += 1
        count = min(n - retained, len(kept))
        if count:
            pieces.append(kept[:count].copy())
            retained += count
        if progress is not None:
            progress({"retained": retained, "n": n, "proposed": proposed, "accepted": accepted,
                      "elapsed_seconds": time.perf_counter() - started})
    return np.concatenate(pieces), {
        "method": "cindes_rejection", "generation_seconds": time.perf_counter() - started,
        "n": n, "d": d, "seed": seed, "proposal_batch": batch,
        "proposed": proposed, "accepted": accepted, "retained": retained,
        "batches": batches, "acceptance_rate": accepted / proposed,
        "learned_logit_upper_bound": bound, "max_observed_log_acceptance": maximum,
        "returned_coordinates": "Z=log(Y)", "support_clipping": False,
        "target_information_used": False}
