"""Tree-structured MRF on [0,1]^d (balanced binary tree, d* = 2).

    log p0(y) = phi_1(y_1) + sum_{i=2}^{d} [ psi_i(y_{pi(i)}, y_i)
                + phi_i(y_i) ] - log Z,          pi(i) = floor(i/2),

with the same trigonometric potential family as the chain experiment
(per-edge random coefficients, fixed seed).  All population quantities are
exact via belief propagation on a G-point grid per coordinate: ancestral
sampling root-to-leaves, normalizing constant, 1D marginals for every
node, and 2D marginals for every EDGE (parent-child pair) -- the tree
analog of the chain's adjacent pairs.

Same interface as ChainMRF where it matters: sample(), log_p(),
log_p_unnorm(), marginals() -> (m1 [d,G], m2 [n_edges,G,G]), plus
.edges listing (parent, child) index pairs.
"""

import numpy as np


class TreeMRF:
    def __init__(self, d=31, G=200, coupling=0.55, unary=0.3, pot_seed=11):
        assert d >= 3
        self.d, self.G = d, G
        # 1-indexed tree in 0-indexed arrays: parent of node i (>=1) is
        # (i-1)//2 in 0-indexed terms
        self.parent = np.array([-1] + [(i - 1) // 2 for i in range(1, d)])
        self.children = [[] for _ in range(d)]
        for i in range(1, d):
            self.children[self.parent[i]].append(i)
        self.edges = [(self.parent[i], i) for i in range(1, d)]

        rng = np.random.default_rng(pot_seed)
        self.a = coupling * rng.uniform(0.7, 1.3, size=d)    # per-node edge
        self.b = coupling * rng.uniform(0.4, 0.8, size=d)
        self.sgn = rng.choice([-1.0, 1.0], size=d)
        self.c = unary * rng.uniform(0.5, 1.5, size=d)
        self.ph = rng.uniform(0, 2 * np.pi, size=d)

        self.h = 1.0 / G
        self.u = (np.arange(G) + 0.5) * self.h
        self._build_messages()

    # ---- potentials (edge i connects parent(i) -> i, for i >= 1) -----
    def psi(self, i, u, v):
        """u = parent value, v = child (node i) value."""
        return (self.a[i] * self.sgn[i] * np.cos(2 * np.pi * (u - v))
                + self.b[i] * np.sin(2 * np.pi * u) * np.sin(2 * np.pi * v))

    def phi(self, j, u):
        return self.c[j] * np.sin(2 * np.pi * u + self.ph[j])

    def log_p_unnorm(self, y):
        s = np.zeros(y.shape[0])
        for j in range(self.d):
            s += self.phi(j, y[:, j])
        for i in range(1, self.d):
            s += self.psi(i, y[:, self.parent[i]], y[:, i])
        return s

    def log_p(self, y):
        return self.log_p_unnorm(y) - self.logZ

    # ---- belief propagation ------------------------------------------
    def _build_messages(self):
        G, d, u = self.G, self.d, self.u
        U, V = np.meshgrid(u, u, indexing="ij")
        # A[i][a,b] = exp(psi_i(u_a, u_b) + phi_i(u_b)) for i >= 1
        self.A = [None] * d
        for i in range(1, d):
            self.A[i] = np.exp(self.psi(i, U, V) + self.phi(i, u)[None, :])
        self.v0 = np.exp(self.phi(0, u))                     # root unary

        # upward pass (leaves -> root), with per-node rescaling
        self.up = [np.ones(G) for _ in range(d)]             # prod child msgs
        self.beta = [None] * d                               # msg node->parent
        logscale = 0.0
        for i in range(d - 1, 0, -1):                        # children first
            self.beta[i] = self.A[i] @ self.up[i]
            m = self.beta[i].max()
            self.beta[i] /= m
            logscale += np.log(m)
            self.up[self.parent[i]] = self.up[self.parent[i]] * self.beta[i]
        root = self.v0 * self.up[0]
        self.logZ = logscale + np.log(root.sum()) + d * np.log(self.h)

        # downward pass (root -> leaves)
        self.down = [None] * d
        self.down[0] = self.v0.copy()
        self.down[0] /= self.down[0].max()
        for i in range(1, d):
            p = self.parent[i]
            ctx = self.down[p].copy()
            for c in self.children[p]:
                if c != i:
                    ctx = ctx * self.beta[c]
            self.down[i] = self.A[i].T @ ctx
            self.down[i] /= self.down[i].max()

    # ---- exact ancestral sampling (root -> leaves) -------------------
    def sample(self, n, rng):
        G, d = self.G, self.d
        idx = np.zeros((n, d), dtype=np.int64)
        p0 = self.v0 * self.up[0]
        p0 = p0 / p0.sum()
        idx[:, 0] = rng.choice(G, size=n, p=p0)
        for i in range(1, d):                                # parents first
            cond = self.A[i] * self.up[i][None, :]           # (G_par, G_child)
            cond = cond / cond.sum(axis=1, keepdims=True)
            cum = np.cumsum(cond, axis=1)
            r = rng.random(n)
            idx[:, i] = (r[:, None] > cum[idx[:, self.parent[i]]]).sum(axis=1)
        return (idx + rng.random((n, d))) * self.h

    # ---- exact marginals ---------------------------------------------
    def marginals(self):
        G, d = self.G, self.d
        m1 = np.zeros((d, G))
        for j in range(d):
            m = self.down[j] * self.up[j]
            m1[j] = m / m.sum()
        m2 = np.zeros((len(self.edges), G, G))
        for k, (p, i) in enumerate(self.edges):
            ctx = self.down[p].copy()
            for c in self.children[p]:
                if c != i:
                    ctx = ctx * self.beta[c]
            m = ctx[:, None] * self.A[i] * self.up[i][None, :]
            m2[k] = m / m.sum()
        return m1, m2


def marginal_tv_tree(samples, mrf, bins=20):
    """Mean TV of 1D marginals (all nodes) and 2D marginals (all EDGES)."""
    m1, m2 = mrf.marginals()
    G, d = mrf.G, mrf.d
    fac = G // bins
    m1c = m1.reshape(d, bins, fac).sum(axis=2)
    m2c = m2.reshape(len(mrf.edges), bins, fac, bins, fac).sum(axis=(2, 4))
    edges_ = np.linspace(0, 1, bins + 1)
    tv1 = []
    for j in range(d):
        h, _ = np.histogram(samples[:, j], bins=edges_)
        tv1.append(0.5 * np.abs(h / h.sum() - m1c[j]).sum())
    tv2 = []
    for k, (p, i) in enumerate(mrf.edges):
        h, _, _ = np.histogram2d(samples[:, p], samples[:, i],
                                 bins=[edges_, edges_])
        tv2.append(0.5 * np.abs(h / h.sum() - m2c[k]).sum())
    return float(np.mean(tv1)), float(np.mean(tv2))


if __name__ == "__main__":
    # ---- validation ---------------------------------------------------
    mrf = TreeMRF(d=31)
    rng = np.random.default_rng(0)
    Y = mrf.sample(50000, rng)
    tv1, tv2 = marginal_tv_tree(Y, mrf)
    print(f"sampler self-check (50k samples): tv1={tv1:.4f} tv2={tv2:.4f}")
    U = rng.random((300000, mrf.d))
    lp = mrf.log_p(U)
    print(f"MC integral of p0 (want ~1): {np.exp(lp).mean():.3f}")
    lpY = mrf.log_p(Y)
    print(f"KL(p0||unif) = E[log p0] = {lpY.mean():.2f}; "
          f"range [{np.quantile(lp, 1e-5):.1f}, {lpY.max():.1f}]")
    # brute-force check of log Z and marginals on a tiny tree
    m = TreeMRF(d=3, G=40)
    g = m.u
    Yg = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    w = np.exp(m.log_p_unnorm(Yg)) * m.h ** 3
    print(f"tiny-tree logZ: BP {m.logZ:.6f} vs brute {np.log(w.sum()):.6f}")
    m1, m2 = m.marginals()
    bf1 = w.reshape(40, 40, 40).sum(axis=(1, 2))
    print("tiny-tree 1D marginal max err:",
          f"{np.abs(bf1 / bf1.sum() - m1[0]).max():.2e}")
    bf2 = w.reshape(40, 40, 40).sum(axis=2)                  # edge (0,1)
    print("tiny-tree edge-(0,1) marginal max err:",
          f"{np.abs(bf2 / bf2.sum() - m2[0]).max():.2e}")
