"""
CINDES vs. Denoising Score Matching (DSM) diffusion: sample generation
under a Markov-random-field density.

Motivation (reviewer response experiment)
-----------------------------------------
Target density: pairwise chain MRF on [0,1]^d,

    log p0(y) = sum_{j=1}^{d-1} psi_j(y_j, y_{j+1}) + sum_j phi_j(y_j) - log Z.

By Hammersley-Clifford, p0 factorizes over cliques of size d* = 2, so
p0 in H_f(d, d*=2, beta, C) with beta = infinity (analytic potentials).
Theory (Corollary 4.2 of the manuscript) says the CINDES implicit density
estimator attains the rate n^{-beta/(2beta+d*)}, adapting to d* = 2, because
log p0 is a SUM of 2-dimensional clique potentials -- a compositional
structure that the classification network learns efficiently.

Denoising score matching must instead estimate the score of the *diffused*
density grad log p_t for all t > 0.  The OU noise destroys the Markov
factorization: for t > 0, p_t is no longer an MRF, and grad log p_t has no
low-dimensional additive structure to exploit.  DSM therefore faces the
d-dimensional (plus time) estimation problem and is expected to be less
sample-efficient at moderate n.

Both methods use the SAME backward sampler (exponential-integrator /
DDPM-style discretization of the reverse OU SDE, Eq. (10)-(12) of the
manuscript), the same discretization grid, and networks of comparable size,
so the comparison isolates the score-estimation quality.

Ground truth
------------
The chain structure allows (essentially) exact computations on a fine grid
via transfer matrices: exact ancestral sampling, exact normalizing constant,
and exact 1D/2D marginals.  Used for training data, held-out reference
samples, and evaluation.

Metrics (generated samples vs. ground truth)
--------------------------------------------
  * energy distance
  * sliced Wasserstein distance (500 random projections)
  * mean TV distance of all 1D marginals (vs. exact grid marginals)
  * mean TV distance of adjacent-pair 2D marginals (vs. exact grid marginals)
  * log-likelihood gap  E[log p0(Y_true)] - E[log p0(Y_hat)]  (exact log p0)

Usage
-----
  python cindes_vs_dsm_mrf.py --n 2000 --seed 0            # single run
  python cindes_vs_dsm_mrf.py --study                      # full grid
  python cindes_vs_dsm_mrf.py --study --n-list 500 1000 2000 4000 --seeds 3
Results are appended as JSON lines to results.jsonl.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count() or 4)


# ======================================================================
# 1. Ground truth: pairwise chain MRF on [0,1]^d  (d* = 2)
# ======================================================================

class ChainMRF:
    """log p0(y) = sum_j psi_j(y_j, y_{j+1}) + sum_j phi_j(y_j) - log Z.

    Smooth (analytic) potentials; coefficients drawn once from a fixed RNG so
    the density is heterogeneous across edges but reproducible.
    Exact operations via transfer matrices on a G-point grid per coordinate.
    """

    def __init__(self, d=8, G=200, coupling=1.0, unary=0.5, octave=0.0,
                 pot_seed=7):
        self.d, self.G = d, G
        rng = np.random.default_rng(pot_seed)
        # Multi-scale ("rough") clique potentials.  The theoretical
        # separation between the two methods is governed by the Hoelder
        # smoothness beta of the clique potentials relative to dimension:
        # CINDES converges at n^{-beta/(2beta+d*)} (Corollary 4.2, d*=2)
        # while structure-agnostic score estimation faces the ambient-
        # dimension rate n^{-beta/(2beta+d)}.  Analytic single-frequency
        # potentials (beta = infinity) would make both rates ~ n^{-1/2};
        # adding higher-frequency octaves emulates a low effective beta
        # while keeping all exact grid computations valid.
        self.a = coupling * rng.uniform(0.7, 1.3, size=d - 1)   # base coupling
        self.b = coupling * rng.uniform(0.4, 0.8, size=d - 1)   # product term
        self.sgn = rng.choice([-1.0, 1.0], size=d - 1)
        self.c = unary * rng.uniform(0.5, 1.5, size=d)          # unary base
        self.ph = rng.uniform(0, 2 * np.pi, size=d)
        # optional higher-frequency octave (0 = smooth potentials)
        self.a2 = octave * rng.uniform(0.5, 0.9, size=d - 1)
        self.ph2 = rng.uniform(0, 2 * np.pi, size=d - 1)
        self.c2 = octave * rng.uniform(0.5, 1.0, size=d)
        self.ph3 = rng.uniform(0, 2 * np.pi, size=d)
        # grid (bin midpoints)
        self.h = 1.0 / G
        self.u = (np.arange(G) + 0.5) * self.h
        self._build_messages()

    # ---- potentials -------------------------------------------------
    def psi(self, j, u, v):
        """Edge potential j between y_j = u and y_{j+1} = v (numpy arrays)."""
        return (self.a[j] * self.sgn[j] * np.cos(2 * np.pi * (u - v))
                + self.a2[j] * np.cos(6 * np.pi * (u - v) + self.ph2[j])
                + self.b[j] * np.sin(2 * np.pi * u) * np.sin(2 * np.pi * v))

    def phi(self, j, u):
        """Unary potential for coordinate j."""
        return (self.c[j] * np.sin(2 * np.pi * u + self.ph[j])
                + self.c2[j] * np.sin(6 * np.pi * u + self.ph3[j]))

    def log_p_unnorm(self, y):
        """Unnormalized log density, y: (n, d) array in [0,1]^d."""
        s = np.zeros(y.shape[0])
        for j in range(self.d):
            s += self.phi(j, y[:, j])
        for j in range(self.d - 1):
            s += self.psi(j, y[:, j], y[:, j + 1])
        return s

    def log_p(self, y):
        return self.log_p_unnorm(y) - self.logZ

    # ---- transfer matrices ------------------------------------------
    def _build_messages(self):
        G, d, u = self.G, self.d, self.u
        U, V = np.meshgrid(u, u, indexing="ij")
        # A_j[a,b] = exp(psi_j(u_a,u_b) + phi_{j+1}(u_b)),  j = 0..d-2
        self.A = [np.exp(self.psi(j, U, V) + self.phi(j + 1, u)[None, :])
                  for j in range(d - 1)]
        self.v0 = np.exp(self.phi(0, u))                     # initial vector
        # backward messages beta[j][a] = sum over y_{j+1..d-1} of prod A
        beta = [None] * d
        beta[d - 1] = np.ones(G)
        for j in range(d - 2, -1, -1):
            beta[j] = self.A[j] @ beta[j + 1]
            beta[j] /= beta[j].max()                          # stabilize
            self._log_beta_scale = getattr(self, "_log_beta_scale", 0.0)
        self.beta = beta
        # normalizing constant on the grid (exact for the discretized density,
        # excellent approximation of the continuous Z for smooth potentials)
        b = np.ones(G)
        logZ = 0.0
        for j in range(d - 2, -1, -1):
            b = self.A[j] @ b
            m = b.max()
            b /= m
            logZ += np.log(m)
        logZ += np.log(np.sum(self.v0 * b)) + d * np.log(self.h)
        self.logZ = logZ

    # ---- exact ancestral sampling -----------------------------------
    def sample(self, n, rng):
        G, d = self.G, self.d
        idx = np.zeros((n, d), dtype=np.int64)
        p1 = self.v0 * self.beta[0]
        p1 /= p1.sum()
        idx[:, 0] = rng.choice(G, size=n, p=p1)
        for j in range(1, d):
            # p(y_j = b | y_{j-1} = a) propto A_{j-1}[a,b] * beta_j[b]
            cond = self.A[j - 1] * self.beta[j][None, :]      # (G, G)
            cond = cond / cond.sum(axis=1, keepdims=True)
            cum = np.cumsum(cond, axis=1)
            r = rng.random(n)
            idx[:, j] = (r[:, None] > cum[idx[:, j - 1]]).sum(axis=1)
        # jitter within bins -> continuous samples
        y = (idx + rng.random((n, d))) * self.h
        return y

    # ---- exact marginals on the grid --------------------------------
    def marginals(self):
        """Exact 1D marginals (d, G) and adjacent 2D marginals (d-1, G, G),
        as probability masses per bin."""
        G, d = self.G, self.d
        # forward messages alpha[j][a] propto p(y_0..y_j-1 summed, y_j = a)
        alpha = [None] * d
        alpha[0] = self.v0.copy()
        alpha[0] /= alpha[0].max()
        for j in range(1, d):
            alpha[j] = alpha[j - 1] @ self.A[j - 1]
            alpha[j] /= alpha[j].max()
        m1 = np.zeros((d, G))
        for j in range(d):
            m = alpha[j] * self.beta[j]
            m1[j] = m / m.sum()
        m2 = np.zeros((d - 1, G, G))
        for j in range(d - 1):
            m = alpha[j][:, None] * self.A[j] * self.beta[j + 1][None, :]
            m2[j] = m / m.sum()
        return m1, m2


# ======================================================================
# 2. Networks
# ======================================================================

class MLP(nn.Module):
    def __init__(self, d_in, d_out, width=128, depth=3):
        super().__init__()
        layers, dims = [], [d_in] + [width] * depth
        for i in range(depth):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(width, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ScoreNet(nn.Module):
    """DSM score network s_phi(y, t) with Fourier time embedding."""

    def __init__(self, d, width=128, depth=3, n_freq=8):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float())
        self.mlp = MLP(d + 2 * n_freq, d, width, depth)

    def forward(self, y, t):
        # t: (batch, 1), in [0, T]
        ang = t * self.freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        return self.mlp(torch.cat([y, emb], dim=1))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ======================================================================
# 3. CINDES: explicit density (Algorithm 1) + implicit sampler (Algorithm 2)
# ======================================================================

def train_cindes(Y, R=8.0, width=128, depth=3, epochs=400, bs=256, lr=1e-3,
                 weight_decay=1e-5, fake_ratio=2, seed=0, verbose=False):
    """Algorithm 1: logistic regression of real samples vs. uniform fakes.
    Returns f_hat with p_hat(y) = exp(f_hat(y)) (unnormalized).

    fake_ratio: number of uniform reference samples per real sample.  The
    reference samples are simulated and therefore free; a ratio > 1 reduces
    the variance of the classification-based estimator (the population
    minimizer changes only by a known additive constant log(fake_ratio),
    which cancels in the self-normalized sampler weights).
    """
    torch.manual_seed(seed)
    n, d = Y.shape
    nf = fake_ratio * n
    g = torch.Generator().manual_seed(seed + 1)
    Yf = torch.rand((nf, d), generator=g)                    # fake ~ U([0,1]^d)
    Z = torch.cat([Y, Yf]).to(DEVICE)
    lab = torch.cat([torch.ones(n), torch.zeros(nf)]).to(DEVICE)

    f = MLP(d, 1, width, depth).to(DEVICE)
    opt = torch.optim.Adam(f.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss()
    N = n + nf
    for ep in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, bs):
            j = perm[i:i + bs]
            logits = f(Z[j]).squeeze(-1).clamp(-R, R)        # truncation T_R
            loss = bce(logits, lab[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if verbose and (ep + 1) % 100 == 0:
            print(f"  [cindes] epoch {ep+1}: loss {loss.item():.4f}")
    f.eval()
    return f


@torch.no_grad()
def cindes_score(f, R, y, t, K, gen):
    """Plug-in Monte-Carlo estimator of grad log p_t (Eq. (9) of the
    manuscript), in numerically stable self-normalized form.  Writing
    p_t(y) = int p0(y0) N(y; m_t y0, sigma_t^2 I) dy0, the score is

        s(y,t) = -(1/sigma_t^2) E[ y - m_t Y0 | Y_t = y ].

    The posterior over Y0 is propto p0(y0) N(y0; y/m_t, (sigma_t/m_t)^2 I)
    restricted to [0,1]^d.  We importance-sample with the proposal
    q(y0) = TruncNormal(y/m_t, (sigma_t/m_t)^2, [0,1]) per coordinate
    (inverse-CDF sampling).  The Gaussian factor and the truncation
    normalizers then cancel row-wise, leaving self-normalized weights
    w_k = softmax_k(f_hat(y0_k)).  This is Eq. (9) with a support-adapted
    proposal: it is a strictly better-conditioned but mathematically
    equivalent estimator (identical target expectation), avoiding the
    degeneracy of raw N(0,I) proposals at large t, where (y - sigma_t U)/m_t
    almost never lands in the support.  p_hat = exp(f_hat) 1{[0,1]^d} is
    unnormalized; the constant cancels in the softmax.
    """
    B, d = y.shape
    m_t = math.exp(-t)
    s_t = math.sqrt(1.0 - math.exp(-2.0 * t))
    sp = s_t / m_t                                           # posterior std of Y0
    c = y / m_t                                              # posterior center
    # sample y0 ~ TruncNormal(c, sp^2, [0,1]) per coordinate via inverse CDF
    lo = torch.special.ndtr((0.0 - c) / sp)                  # (B, d)
    hi = torch.special.ndtr((1.0 - c) / sp)
    u = torch.rand((B, K, d), generator=gen, device=DEVICE)
    p = (lo[:, None, :] + u * (hi - lo)[:, None, :]).clamp(1e-6, 1 - 1e-6)
    y0 = (c[:, None, :] + sp * torch.special.ndtri(p)).clamp(0.0, 1.0)
    logp = f(y0.reshape(B * K, d)).squeeze(-1).clamp(-R, R).reshape(B, K)
    w = torch.softmax(logp, dim=1)                           # (B, K)
    resid = y[:, None, :] - m_t * y0                         # (B, K, d)
    return -(w[:, :, None] * resid).sum(dim=1) / (s_t ** 2)


# ======================================================================
# 4. DSM baseline
# ======================================================================

def train_dsm(Y, T=3.0, t_min=1e-3, width=128, depth=3, epochs=800, bs=256,
              lr=1e-3, weight_decay=1e-5, seed=0, verbose=False):
    """Standard denoising score matching for the OU forward process
    (beta_t = 1):  Y_t = m_t Y_0 + sigma_t eps,  target score -eps/sigma_t,
    loss weighting lambda(t) = sigma_t^2 (the standard DDPM weighting):
        E || sigma_t s_phi(Y_t, t) + eps ||^2.
    """
    torch.manual_seed(seed + 10)
    n, d = Y.shape
    Y = Y.to(DEVICE)
    s_net = ScoreNet(d, width, depth).to(DEVICE)
    opt = torch.optim.Adam(s_net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            y0 = Y[perm[i:i + bs]]
            b = y0.shape[0]
            # log-uniform times emphasize small t (where the score is hardest)
            t = torch.exp(torch.rand(b, 1, device=DEVICE)
                          * (math.log(T) - math.log(t_min)) + math.log(t_min))
            m_t = torch.exp(-t)
            s_t = torch.sqrt(1 - torch.exp(-2 * t))
            eps = torch.randn_like(y0)
            yt = m_t * y0 + s_t * eps
            loss = ((s_t * s_net(yt, t) + eps) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if verbose and (ep + 1) % 200 == 0:
            print(f"  [dsm] epoch {ep+1}: loss {loss.item():.4f}")
    s_net.eval()
    return s_net


# ======================================================================
# 5. Shared backward sampler (Eq. (10)-(12), DDPM-style discretization)
# ======================================================================

def make_timesteps(T, delta, M):
    """Eq. (10): first half uniform on [0, T-1], second half exponential
    from T-1 to T-delta."""
    ts = []
    for m in range(M + 1):
        if m <= M / 2:
            ts.append((T - 1.0) * m / (M / 2))
        else:
            ts.append(T - delta ** ((2 * m - M) / M))
    return np.array(ts)


@torch.no_grad()
def cindes_posterior_sample(f, R, y, t, K, gen):
    """Draw Y0 ~ p_hat(y0 | Y_t = y) by sampling-importance-resampling over
    the same K-particle truncated-normal proposal used in cindes_score:
    particles y0_k, weights w_k = softmax(f_hat(y0_k)); return one particle
    per row drawn according to w.  As K grows this is an exact draw from the
    model posterior; unlike the posterior MEAN it neither contracts the
    distribution nor creates boundary atoms.  Only available for CINDES
    (requires the explicit density, not just the score)."""
    B, d = y.shape
    m_t = math.exp(-t)
    s_t = math.sqrt(1.0 - math.exp(-2.0 * t))
    sp = s_t / m_t
    c = y / m_t
    lo = torch.special.ndtr((0.0 - c) / sp)
    hi = torch.special.ndtr((1.0 - c) / sp)
    u = torch.rand((B, K, d), generator=gen, device=DEVICE)
    p = (lo[:, None, :] + u * (hi - lo)[:, None, :]).clamp(1e-6, 1 - 1e-6)
    y0 = (c[:, None, :] + sp * torch.special.ndtri(p)).clamp(0.0, 1.0)
    logp = f(y0.reshape(B * K, d)).squeeze(-1).clamp(-R, R).reshape(B, K)
    w = torch.softmax(logp, dim=1)
    idx = torch.multinomial(w, 1, generator=gen).squeeze(-1)      # (B,)
    return y0[torch.arange(B, device=DEVICE), idx]


@torch.no_grad()
def backward_sample(score_fn, d, n_samples, T=3.0, delta=1e-3, M=100,
                    seed=0, batch=500, denoise_final=False, final_fn=None):
    """Discretized reverse OU SDE shared by both methods.
    score_fn(y, t_forward) -> estimated score of p_{t_forward} at y.

    denoise_final: if True, apply the standard terminal denoising step
    (predict-x0 via Tweedie's formula) instead of returning Y_delta:
        Y_hat = (Y_delta + sigma_delta^2 * s(Y_delta, delta)) / m_delta
             = E[Y_0 | Y_delta]  under the estimated model.
    This removes the early-stopping blur and, on compact support, the
    boundary atoms created by clamping (the posterior mean lies in the
    convex hull of the support).  Applied identically to both methods.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 999)
    ts = make_timesteps(T, delta, M)
    out = []
    for start in range(0, n_samples, batch):
        b = min(batch, n_samples - start)
        y = torch.randn((b, d), generator=gen, device=DEVICE)
        for m in range(M):
            tau, tau_next = T - ts[m], T - ts[m + 1]         # forward times
            a_m = math.exp(-2.0 * (tau - tau_next))          # alpha_m
            abar_cur = math.exp(-2.0 * tau)
            abar_next = math.exp(-2.0 * tau_next)
            s = score_fn(y, tau)
            mean = (y + (1.0 - a_m) * s) / math.sqrt(a_m)
            var = (1.0 - a_m) * (1.0 - abar_next) / (1.0 - abar_cur)
            if m < M - 1:
                w = torch.randn((b, d), generator=gen, device=DEVICE)
                y = mean + math.sqrt(max(var, 0.0)) * w
            else:
                y = mean
        if final_fn is not None:
            y = final_fn(y, delta)          # e.g. posterior sampling (CINDES)
        elif denoise_final:
            m_d = math.exp(-delta)
            s2_d = 1.0 - math.exp(-2.0 * delta)
            y = (y + s2_d * score_fn(y, delta)) / m_d
        out.append(y.cpu())
    return torch.cat(out).clamp(0.0, 1.0).numpy()            # support is [0,1]^d


# ======================================================================
# 6. Metrics
# ======================================================================

def energy_distance(X, Y, rng, max_n=2000):
    X = X[rng.choice(len(X), min(len(X), max_n), replace=False)]
    Y = Y[rng.choice(len(Y), min(len(Y), max_n), replace=False)]

    def mean_pdist(A, B):
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.sqrt(np.maximum(d2, 0)).mean()

    return 2 * mean_pdist(X, Y) - mean_pdist(X, X) - mean_pdist(Y, Y)


def sliced_wasserstein(X, Y, rng, n_proj=500, max_n=2000):
    m = min(len(X), len(Y), max_n)          # equal sizes: valid quantile pairing
    X = X[rng.choice(len(X), m, replace=False)]
    Y = Y[rng.choice(len(Y), m, replace=False)]
    d = X.shape[1]
    theta = rng.standard_normal((d, n_proj))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True)
    Xp, Yp = np.sort(X @ theta, axis=0), np.sort(Y @ theta, axis=0)
    return np.abs(Xp - Yp).mean()


def marginal_tv(samples, mrf, bins=20):
    """Mean TV of 1D marginals (all coords) and 2D marginals (adjacent pairs)
    against exact grid marginals aggregated to `bins` bins."""
    m1, m2 = mrf.marginals()
    G, d = mrf.G, mrf.d
    fac = G // bins
    m1c = m1.reshape(d, bins, fac).sum(axis=2)
    m2c = m2.reshape(d - 1, bins, fac, bins, fac).sum(axis=(2, 4))
    edges = np.linspace(0, 1, bins + 1)
    tv1 = []
    for j in range(d):
        h, _ = np.histogram(samples[:, j], bins=edges)
        tv1.append(0.5 * np.abs(h / h.sum() - m1c[j]).sum())
    tv2 = []
    for j in range(d - 1):
        h, _, _ = np.histogram2d(samples[:, j], samples[:, j + 1],
                                 bins=[edges, edges])
        tv2.append(0.5 * np.abs(h / h.sum() - m2c[j]).sum())
    return float(np.mean(tv1)), float(np.mean(tv2))


def loglik_gap(samples, mrf, ref_ll):
    """E[log p0(Y_true)] - E[log p0(Y_gen)] (>= ~0; smaller is better)."""
    return float(ref_ll - mrf.log_p(samples).mean())


# ======================================================================
# 7. Experiment driver
# ======================================================================

def run_once(n, seed, d=16, methods=("cindes", "dsm"), coupling=1.0, unary=0.5,
             T=3.0, delta=1e-3, M=100, K=192, n_gen=1200, cindes_steps=10000,
             dsm_steps=24000, fake_ratio=2, denoise_final=False,
             verbose=True):
    """Train the requested method(s) on the same data, generate samples with
    the identical backward sampler, and compute all metrics.

    Optimization budgets are fixed in SGD steps across n (fairness), chosen
    generously past convergence for both methods -- DSM gets 3x the steps of
    CINDES since it must fit a time-indexed vector field.
    """
    bs = 256
    mrf = ChainMRF(d=d, coupling=coupling, unary=unary)
    rng = np.random.default_rng(seed)
    Y_train = torch.tensor(mrf.sample(n, rng), dtype=torch.float32)
    Y_ref = mrf.sample(20000, np.random.default_rng(12345))  # shared reference
    ref_ll = mrf.log_p(Y_ref).mean()

    res = {"n": n, "seed": seed, "d": d}
    R = 16.0   # covers the range of log p0 (Condition 3.2: R >= log c1)
    samples = {}

    if "cindes" in methods:
        ep = max(1, round(cindes_steps
                          / max(1, math.ceil((1 + fake_ratio) * n / bs))))
        t0 = time.time()
        f = train_cindes(Y_train, R=R, epochs=ep, fake_ratio=fake_ratio,
                         seed=seed, verbose=verbose)
        gen = torch.Generator(device=DEVICE).manual_seed(seed + 555)

        def cindes_fn(y, tau):
            return cindes_score(f, R, y, tau, K, gen)

        samples["cindes"] = backward_sample(cindes_fn, d, n_gen, T=T,
                                            delta=delta, M=M, seed=seed,
                                            denoise_final=denoise_final)
        res["cindes_time"] = time.time() - t0
        res["cindes_params"] = n_params(f)

    if "dsm" in methods:
        ep = max(1, round(dsm_steps / max(1, math.ceil(n / bs))))
        t0 = time.time()
        s_net = train_dsm(Y_train, T=T, epochs=ep, seed=seed, verbose=verbose)

        def dsm_fn(y, tau):
            t = torch.full((y.shape[0], 1), tau, device=DEVICE)
            return s_net(y, t)

        samples["dsm"] = backward_sample(dsm_fn, d, n_gen, T=T, delta=delta,
                                         M=M, seed=seed,
                                         denoise_final=denoise_final)
        res["dsm_time"] = time.time() - t0
        res["dsm_params"] = n_params(s_net)

    mrng = np.random.default_rng(seed + 777)
    for name, Yg in samples.items():
        tv1, tv2 = marginal_tv(Yg, mrf)
        res[f"{name}_energy"] = float(energy_distance(Yg, Y_ref, mrng))
        res[f"{name}_sw"] = float(sliced_wasserstein(Yg, Y_ref, mrng))
        res[f"{name}_tv1"] = tv1
        res[f"{name}_tv2"] = tv2
        res[f"{name}_llgap"] = loglik_gap(Yg, mrf, ref_ll)
    return res, (mrf, samples, Y_ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--study", action="store_true")
    ap.add_argument("--n-list", type=int, nargs="+",
                    default=[500, 1000, 2000, 4000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=str, default="results.jsonl")
    ap.add_argument("--max-runs", type=int, default=0,
                    help="stop after this many new runs (0 = no limit)")
    ap.add_argument("--n-gen", type=int, default=1200,
                    help="generated samples per run")
    ap.add_argument("--K", type=int, default=192,
                    help="Monte-Carlo particles for the plug-in score, Eq. (9)")
    ap.add_argument("--M", type=int, default=100,
                    help="backward discretization steps")
    ap.add_argument("--cindes-steps", type=int, default=10000)
    ap.add_argument("--dsm-steps", type=int, default=24000)
    ap.add_argument("--coupling", type=float, default=1.0,
                    help="edge potential strength (reduce for larger d to "
                         "keep KL(p0||unif) moderate, e.g. 0.55 at d=32)")
    ap.add_argument("--unary", type=float, default=0.5)
    args = ap.parse_args()

    # one unit of work = (n, seed, method); resumable via the output file
    configs = ([(n, s, m) for n in args.n_list for s in range(args.seeds)
                for m in ("cindes", "dsm")]
               if args.study else [(args.n, args.seed, m)
                                   for m in ("cindes", "dsm")])
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            if line.strip():
                r = json.loads(line)
                done.add((r["n"], r["seed"], r["method"]))
    new_runs = 0
    for n, seed, meth in configs:
        if (n, seed, meth) in done:
            continue
        if args.max_runs and new_runs >= args.max_runs:
            break
        print(f"=== n={n} seed={seed} method={meth} ===", flush=True)
        res, _ = run_once(n, seed, d=args.d, methods=(meth,),
                          coupling=args.coupling, unary=args.unary,
                          n_gen=args.n_gen, K=args.K, M=args.M,
                          cindes_steps=args.cindes_steps,
                          dsm_steps=args.dsm_steps, verbose=False)
        res["method"] = meth
        line = json.dumps(res)
        print(line, flush=True)
        with open(args.out, "a") as fh:
            fh.write(line + "\n")
        done.add((n, seed, meth))
        new_runs += 1
    print("ALL_DONE" if all(c in done for c in configs) else "PARTIAL",
          flush=True)


if __name__ == "__main__":
    main()
