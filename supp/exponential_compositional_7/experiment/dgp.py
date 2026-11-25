"""Simulation/evaluation-only positive exponential compositional target.

Estimator modules must not import this module. The reference remains product
Exp(1). The generating features are centered exponential CDF values, which
are independent Uniform(-1,1) under the reference. The signed edges are
unchanged from the preceding exponential pilot.
"""
from __future__ import annotations
import math
import numpy as np

EXPONENTIAL_CDF_FEATURE_SECOND_MOMENT = 1.0 / 3.0

def exponential_positive(rng, shape):
    """Exp(1) draws, redrawing machine endpoints instead of clipping them."""
    x = rng.exponential(size=shape)
    bad = x <= 0
    while bad.any():
        x[bad] = rng.exponential(size=int(bad.sum()))
        bad = x <= 0
    return x

class ExponentialCompositionalDGP:
    def __init__(self, d=7, pot_seed=11, amplitude=1.5, kappa=1.0):
        self.d = int(d)
        self.pot_seed = int(pot_seed)
        self.amplitude, self.kappa = float(amplitude), float(kappa)
        if self.d < 2 or self.amplitude < 0 or self.kappa <= 0:
            raise ValueError('Invalid DGP parameters')
        rng = np.random.default_rng(self.pot_seed)
        self.permutation = rng.permutation(self.d)
        self.signs = rng.choice(np.array([-1.0, 1.0]), size=self.d-1)
        self.edges = [(int(self.permutation[(j-1)//2]), int(self.permutation[j]))
                      for j in range(1, self.d)]
        self.h_scale = EXPONENTIAL_CDF_FEATURE_SECOND_MOMENT * math.sqrt(self.d-1)
        self.config = {
            'd': self.d, 'base': 'Product Exp(1)', 'pot_seed': self.pot_seed,
            'amplitude': self.amplitude, 'kappa': self.kappa,
            'generating_pairs': self.edges, 'edge_signs': self.signs.tolist(),
            'permutation': self.permutation.tolist(), 'h_scale': self.h_scale,
            'feature': 'psi(y)=1-2*exp(-y); Uniform(-1,1) under Exp(1)',
            'H_normalization': 'sqrt(d-1)/3; H has mean zero and variance one under q',
            'iid_sampler': 'Exp(1) proposal; accept exp(log_tilt-amplitude)',
            'support_clipping': False,
        }
        self.last_sample_stats = {}

    def H(self, y):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != self.d or not np.isfinite(y).all():
            raise ValueError('Expected finite (n,d) observations')
        z = 1.0 - 2.0 * np.exp(-y)
        h = np.zeros(len(y))
        for sign, (j,k) in zip(self.signs, self.edges):
            h += sign*z[:,j]*z[:,k]
        return h/self.h_scale

    def log_tilt(self, y):
        return self.amplitude*np.tanh(self.kappa*self.H(y))

    def log_p_unnorm(self, y):
        y = np.asarray(y, dtype=np.float64)
        if not (y > 0).all():
            raise ValueError('Target density is supported on strictly positive observations')
        return -y.sum(axis=1)+self.log_tilt(y)

    def log_pz_unnorm(self, z):
        z = np.asarray(z, dtype=np.float64)
        return self.log_p_unnorm(np.exp(z))+z.sum(axis=1)

    def sample(self, n, rng):
        if int(n) != n or n < 0 or not isinstance(rng, np.random.Generator):
            raise ValueError('n must be nonnegative and rng a NumPy Generator')
        n = int(n)
        pieces, retained, proposed, accepted_count = [], 0, 0, 0
        while retained < n:
            size = min(65536, max(1024, math.ceil(math.exp(self.amplitude)*(n-retained))))
            y = exponential_positive(rng, (size, self.d))
            log_accept = self.log_tilt(y)-self.amplitude
            if np.max(log_accept) > 1e-12:
                raise ArithmeticError('Invalid rejection envelope')
            with np.errstate(divide='ignore'):
                accepted = y[np.log(rng.random(size)) < log_accept]
            keep = min(n-retained, len(accepted))
            if keep:
                pieces.append(accepted[:keep].copy())
                retained += keep
            proposed += size
            accepted_count += len(accepted)
        self.last_sample_stats = {
            'proposed': proposed, 'accepted': accepted_count, 'retained': retained,
            'acceptance_rate': accepted_count/proposed if proposed else None,
        }
        return np.concatenate(pieces) if pieces else np.empty((0,self.d))
