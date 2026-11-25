"""CINDES and DSM on R^d, using a fixed N(0,I) reference for CINDES.

The classifier estimates log(p/q) minus the known class-prior offset.
Its implicit density is q * exp(clipped_logit), up to normalization.
Both methods run the OU reverse discretization on real-valued coordinates,
without support clipping or a terminal posterior/mean correction.
"""

import math
import os

import numpy as np
import torch
from torch import nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_threads = int(os.environ.get("NSLOTS", os.environ.get("OMP_NUM_THREADS", "1")))
torch.set_num_threads(max(1, _threads))


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width=128, depth=3):
        super().__init__()
        layers = []
        dims = [d_in] + [width] * depth
        for i in range(depth):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(width, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ScoreNet(nn.Module):
    def __init__(self, d, width=128, depth=3, n_freq=8, gaussian_residual=False):
        super().__init__()
        self.gaussian_residual = gaussian_residual
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float())
        self.mlp = MLP(d + 2 * n_freq, d, width, depth)

    def forward(self, y, t):
        ang = t * self.freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        raw = self.mlp(torch.cat([y, emb], dim=1))
        return raw - y if self.gaussian_residual else raw


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train_cindes(Y, R=25.0, epochs=400, bs=256, lr=1e-3,
                 weight_decay=1e-5, fake_ratio=2, seed=0, verbose=False):
    """Train log(p/q)-log(fake_ratio) against q=N(0,I) negatives.

The additive class-prior correction cancels in normalized importance weights.
Do not interpret exp(logit) alone as a density on R^d.
"""
    torch.manual_seed(seed)
    Y = Y.detach().cpu()
    n, d = Y.shape
    nf = fake_ratio * n
    g = torch.Generator().manual_seed(seed + 1)
    Yf = torch.randn((nf, d), generator=g, dtype=Y.dtype)
    Z = torch.cat([Y, Yf]).to(DEVICE)
    labels = torch.cat([torch.ones(n), torch.zeros(nf)]).to(DEVICE)
    f = MLP(d, 1).to(DEVICE)
    opt = torch.optim.Adam(f.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        order = torch.randperm(n + nf, device=DEVICE)
        for start in range(0, n + nf, bs):
            j = order[start:start + bs]
            logits = f(Z[j]).squeeze(-1).clamp(-R, R)
            loss = loss_fn(logits, labels[j])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CINDES training loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if verbose and (epoch + 1) % 100 == 0:
            print(f"[cindes] epoch {epoch + 1}, loss={loss.item():.6g}", flush=True)
    return f.eval()


@torch.no_grad()
def cindes_score(f, R, y, t, K, gen):
    """Gaussian-reference posterior importance sampling, with K even.

For m=exp(-t), sigma^2=1-exp(-2t), q(Y0|Yt=y)=N(m*y,sigma^2 I).
Samples are paired as z_plus=m*y+sigma*eps, z_minus=m*y-sigma*eps.
Weights are exp(clipped classifier logit), normalized over all K samples.
The score is -y+(m/sigma)*sum(w_k*eps_k). The paired implementation
gives exactly -y for a constant classifier, avoiding small-time MC noise
in the known Gaussian component. Finite-K self-normalization is still biased.
"""
    if K < 2 or K % 2:
        raise ValueError("K must be a positive even integer for antithetic particles")
    if t <= 0:
        raise ValueError("The score is evaluated at positive forward times")
    b, d = y.shape
    half = K // 2
    m = math.exp(-t)
    sigma = math.sqrt(-math.expm1(-2.0 * t))
    eps = torch.randn((b, half, d), generator=gen, device=y.device, dtype=y.dtype)
    center = m * y[:, None, :]
    zp = center + sigma * eps
    zm = center - sigma * eps
    lp = f(zp.reshape(b * half, d)).squeeze(-1).clamp(-R, R).reshape(b, half)
    lm = f(zm.reshape(b * half, d)).squeeze(-1).clamp(-R, R).reshape(b, half)
    weights = torch.softmax(torch.cat([lp, lm], dim=1), dim=1)
    paired_weights = weights[:, :half] - weights[:, half:]
    residual = (paired_weights[:, :, None] * eps).sum(dim=1)
    return -y + (m / sigma) * residual


def train_dsm(Y, T=3.0, t_min=1e-3, epochs=800, bs=256, lr=1e-3,
              weight_decay=1e-5, seed=0, verbose=False, gaussian_residual=False):
    """Same DSM architecture, objective, and schedule as the bounded study."""
    torch.manual_seed(seed + 10)
    Y = Y.to(DEVICE)
    n, d = Y.shape
    net = ScoreNet(d, gaussian_residual=gaussian_residual).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for epoch in range(epochs):
        order = torch.randperm(n, device=DEVICE)
        for start in range(0, n, bs):
            y0 = Y[order[start:start + bs]]
            t = torch.exp(torch.rand(len(y0), 1, device=DEVICE)
                          * (math.log(T) - math.log(t_min)) + math.log(t_min))
            m = torch.exp(-t)
            sigma = torch.sqrt(-torch.expm1(-2.0 * t))
            eps = torch.randn_like(y0)
            yt = m * y0 + sigma * eps
            loss = ((sigma * net(yt, t) + eps) ** 2).sum(dim=1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite DSM training loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if verbose and (epoch + 1) % 200 == 0:
            print(f"[dsm] epoch {epoch + 1}, loss={loss.item():.6g}", flush=True)
    return net.eval()


def make_timesteps(T, delta, M):
    if T <= 1 or not 0 < delta < 1 or M < 2 or M % 2:
        raise ValueError("Require T>1, 0<delta<1, and a positive even M")
    return np.array([(T - 1) * j / (M / 2) if j <= M / 2
                     else T - delta ** ((2 * j - M) / M)
                     for j in range(M + 1)])


@torch.no_grad()
def backward_sample(score_fn, d, n_samples, T=3.0, delta=1e-3, M=200,
                    seed=0, batch=64, sampler="gaussian-reference"):
    """Shared reverse updates on R^d, with no terminal correction/clipping.

    Default: write the reverse OU drift as -y+2*r, r=s(y,t)+y. Freeze r
    for the current step, integrating the Gaussian -y drift and noise exactly:
      y_next = e^-h*y + 2*(1-e^-h)*r + sqrt(1-e^-2h)*xi.
    Noise is included in EVERY reverse step. For s=-y this preserves N(0,I)
    exactly on any grid. General scores still have finite-step error.

    sampler='legacy' retains the previous DDPM posterior variance and omitted
    last noise increment, for explicit compatibility comparisons. It shrinks
    Gaussian oracle variance to ~0.9578 at the default grid.
"""
    if sampler not in ("gaussian-reference", "legacy"):
        raise ValueError("Unknown reverse sampler")
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 999)
    times = make_timesteps(T, delta, M)
    outputs = []
    for start in range(0, n_samples, batch):
        size = min(batch, n_samples - start)
        y = torch.randn((size, d), generator=gen, device=DEVICE)
        for j in range(M):
            tau, tau_next = T - times[j], T - times[j + 1]
            score = score_fn(y, tau)
            h = tau - tau_next
            if sampler == "gaussian-reference":
                decay = math.exp(-h)
                one_minus_decay = -math.expm1(-h)
                mean = decay * y + 2.0 * one_minus_decay * (score + y)
                variance = -math.expm1(-2.0 * h)
                noise = torch.randn(y.shape, generator=gen, device=DEVICE)
                y = mean + math.sqrt(variance) * noise
            else:
                alpha = math.exp(-2.0 * h)
                abar = math.exp(-2.0 * tau)
                abar_next = math.exp(-2.0 * tau_next)
                mean = (y + (1.0 - alpha) * score) / math.sqrt(alpha)
                variance = (1.0 - alpha) * (1.0 - abar_next) / (1.0 - abar)
                if j < M - 1:
                    noise = torch.randn(y.shape, generator=gen, device=DEVICE)
                    y = mean + math.sqrt(max(variance, 0.0)) * noise
                else:
                    y = mean
        if not torch.isfinite(y).all():
            raise FloatingPointError("Reverse sampler produced non-finite observations")
        outputs.append(y.cpu())
    return torch.cat(outputs).numpy()


def energy_distance(X, Y, rng, max_n=2000):
    X = X[rng.choice(len(X), min(len(X), max_n), replace=False)].astype(np.float64)
    Y = Y[rng.choice(len(Y), min(len(Y), max_n), replace=False)].astype(np.float64)
    def mean_distance(A, B):
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.sqrt(np.maximum(d2, 0)).mean()
    return float(2 * mean_distance(X, Y) - mean_distance(X, X) - mean_distance(Y, Y))


def sliced_wasserstein(X, Y, rng, n_proj=500, max_n=2000):
    size = min(len(X), len(Y), max_n)
    X = X[rng.choice(len(X), size, replace=False)].astype(np.float64)
    Y = Y[rng.choice(len(Y), size, replace=False)].astype(np.float64)
    theta = rng.standard_normal((X.shape[1], n_proj))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True)
    return float(np.abs(np.sort(X @ theta, axis=0) - np.sort(Y @ theta, axis=0)).mean())
