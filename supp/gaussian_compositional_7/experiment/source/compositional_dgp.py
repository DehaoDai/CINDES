"""Smooth bounded compositional tilt of a standard Gaussian.

For a permuted balanced binary tree E, put

  H(x) = sum_{(j,k) in E} a_jk tanh(x_j) tanh(x_k) / (v sqrt(|E|)),
  v = E[tanh(Z)^2], Z ~ N(0,1),
  g(x) = A tanh(kappa H(x)),
  p(x) = phi_d(x) exp(g(x)) / Z_g.

Under the Gaussian reference, E[H]=0 and Var(H)=1: distinct edge products
are uncorrelated, even when their edges share a vertex. The tree is bipartite,
so flipping one partition changes H and g to their negatives. Therefore
Z_g=E[cosh(g)] >= 1. Gaussian-proposal rejection with acceptance exp(g-A)
is exact in real arithmetic, with acceptance probability at least exp(-A).
Since |g| <= A, the target has full support and a bounded Gaussian density
ratio. No support or coordinate clipping is used.

This module knows only the target distribution; it contains no estimators,
training criteria, or model-specific features.
"""

from functools import lru_cache
import math
from numbers import Integral

import numpy as np
from scipy.integrate import quad


@lru_cache(maxsize=1)
def gaussian_tanh_second_moment():
    """Accurate deterministic one-dimensional quadrature for E[tanh(Z)^2]."""
    factor = math.sqrt(2.0 / math.pi)

    def integrand(x):
        return factor * math.tanh(x) ** 2 * math.exp(-0.5 * x * x)

    value, error = quad(integrand, 0.0, np.inf, epsabs=1e-13,
                        epsrel=1e-13, limit=200)
    if not 0.0 < value < 1.0 or not math.isfinite(error):
        raise ArithmeticError("Gaussian tanh second-moment quadrature failed")
    return float(value), float(error)


class BoundedCompositionalDGP:
    """Bounded Gaussian tilt with deterministic hidden tree coefficients.

    ``permutation[j]`` is the observed-coordinate index of original tree
    vertex j (all indices are zero based). The RNG draws the permutation
    first, then the d-1 edge signs in increasing original child-node order.
    """

    _MAX_PROPOSAL_BATCH = 65536

    def __init__(self, d=7, pot_seed=11, amplitude=1.5, kappa=1.0):
        if isinstance(d, bool) or not isinstance(d, Integral) or d < 2:
            raise ValueError("d must be an integer at least two")
        if isinstance(pot_seed, bool) or not isinstance(pot_seed, Integral) or pot_seed < 0:
            raise ValueError("pot_seed must be a nonnegative integer")
        if not math.isfinite(amplitude) or amplitude < 0:
            raise ValueError("amplitude must be finite and nonnegative")
        if not math.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        self.d = int(d)
        self.pot_seed = int(pot_seed)
        self.amplitude = float(amplitude)
        self.kappa = float(kappa)
        rng = np.random.default_rng(self.pot_seed)
        self.permutation = rng.permutation(self.d)
        self.signs = rng.choice(np.array([-1.0, 1.0]), size=self.d - 1)
        self.original_edges = [((child - 1) // 2, child)
                               for child in range(1, self.d)]
        self.edges = [(int(self.permutation[parent]), int(self.permutation[child]))
                      for parent, child in self.original_edges]
        self.generating_pairs = list(self.edges)
        self.v, self.v_quadrature_abs_error = gaussian_tanh_second_moment()
        self.h_scale = self.v * math.sqrt(len(self.edges))
        self.bipartition = [[], []]
        for original in range(self.d):
            depth = (original + 1).bit_length() - 1
            self.bipartition[depth % 2].append(int(self.permutation[original]))
        self.config = {
            "protocol": "bounded_compositional_gaussian_tilt_v1",
            "d": self.d,
            "pot_seed": self.pot_seed,
            "amplitude": self.amplitude,
            "kappa": self.kappa,
            "permutation_original_to_observed_zero_based": self.permutation.tolist(),
            "edge_order": "increasing original child vertex, before permutation",
            "original_edges_zero_based": [list(e) for e in self.original_edges],
            "generating_pairs_zero_based": [list(e) for e in self.edges],
            "edge_signs": self.signs.astype(int).tolist(),
            "bipartition_observed_zero_based": self.bipartition,
            "v_gaussian_tanh_second_moment": self.v,
            "v_quadrature_reported_absolute_error": self.v_quadrature_abs_error,
            "h_scale": self.h_scale,
            "log_tilt_formula": "amplitude * tanh(kappa * H)",
            "H_formula": "sum(sign * tanh(x_parent) * tanh(x_child)) / (v * sqrt(d-1))",
            "reference_distribution": "fixed standard multivariate Gaussian",
            "iid_sampler": "Gaussian rejection with log acceptance = log_tilt - amplitude",
            "support_clipping": False,
        }
        self.last_sample_stats = self._empty_stats()

    @staticmethod
    def _empty_stats():
        return {"proposed": 0, "accepted": 0, "retained": 0,
                "batches": 0, "acceptance_rate": None}

    def _points(self, X):
        points = np.asarray(X, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != self.d:
            raise ValueError(f"Expected an (n, {self.d}) array")
        if not np.isfinite(points).all():
            raise ValueError("All input coordinates must be finite")
        return points

    def h_statistic(self, X):
        points = self._points(X)
        transformed = np.tanh(points)
        values = np.zeros(len(points), dtype=np.float64)
        for sign, (parent, child) in zip(self.signs, self.edges):
            values += sign * transformed[:, parent] * transformed[:, child]
        return values / self.h_scale

    def H(self, X):
        """Alias for h_statistic, for evaluation code."""
        return self.h_statistic(X)

    def log_tilt(self, X):
        return self.amplitude * np.tanh(self.kappa * self.h_statistic(X))

    def log_p_unnorm(self, X):
        points = self._points(X)
        gaussian = -0.5 * np.square(points).sum(axis=1)
        gaussian -= 0.5 * self.d * math.log(2.0 * math.pi)
        return gaussian + self.log_tilt(points)

    def sample(self, n, rng):
        """Return n independent target observations using the supplied RNG.

        ``accepted`` counts all accepted proposals, including any final-batch
        excess. ``retained`` counts the observations actually returned. The
        proposal and acceptance RNG is supplied entirely by the caller.
        """
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        n = int(n)
        stats = self._empty_stats()
        pieces = []
        try:
            while stats["retained"] < n:
                remaining = n - stats["retained"]
                # The default A=1.5 has acceptance >= exp(-A). This fixed,
                # deterministic batch rule is only a computational choice.
                multiplier = min(math.exp(min(self.amplitude, 10.0)), 10.0)
                size = min(self._MAX_PROPOSAL_BATCH,
                           max(1024, math.ceil(multiplier * remaining)))
                proposals = rng.standard_normal((size, self.d))
                log_accept = self.log_tilt(proposals) - self.amplitude
                if np.max(log_accept) > 32 * np.finfo(float).eps:
                    raise ArithmeticError("The rejection envelope was violated")
                with np.errstate(divide="ignore"):
                    accepted = proposals[np.log(rng.random(size)) < log_accept]
                stats["proposed"] += size
                stats["accepted"] += len(accepted)
                stats["batches"] += 1
                keep = min(remaining, len(accepted))
                if keep:
                    pieces.append(accepted[:keep].copy())
                    stats["retained"] += keep
            if n == 0:
                return np.empty((0, self.d), dtype=np.float64)
            return np.concatenate(pieces, axis=0)
        finally:
            if stats["proposed"]:
                stats["acceptance_rate"] = stats["accepted"] / stats["proposed"]
            self.last_sample_stats = stats
