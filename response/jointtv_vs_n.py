"""Joint TV as a function of n (d = 32 chain MRF, 3 seeds):

  * CINDES: exact MC estimate of TV(p_hat_n, p0) -- retrain f_hat at each
    (n, seed) with the study configuration and evaluate
        Z  = E_p0[ p_hat/p0 ],   TV = (1/2) E_p0 | p_hat/(Z p0) - 1 |
    over 20k fresh exact samples.  Possible only because p_hat is explicit.
  * DSM: rigorous lower bound on the joint TV of its sample law at each
    (n, seed): the boundary-atom mass (fraction of generated samples with
    a coordinate exactly on a face, a p0-null event).  Requires training
    DSM and generating samples at each cell.

Resumable via jointtv_results.jsonl.  GPU job, ~45 min total.
Afterwards, rerun make_fig_tv_hierarchy.py (it upgrades panel (c) to
curves automatically when this file is present).
"""

import json
import math
import os

import numpy as np
import torch

from cindes_vs_dsm_mrf import (ChainMRF, train_cindes, train_dsm,
                               backward_sample, DEVICE)

D, CPL, UN, R = 32, 0.55, 0.3, 16.0
N_LIST = [500, 1000, 2000, 4000, 8000, 16000]
SEEDS = 3
OUT = "jointtv_results.jsonl"


def cindes_joint_tv(mrf, n, seed, n_mc=20000):
    rng = np.random.default_rng(seed)
    Y = torch.tensor(mrf.sample(n, rng), dtype=torch.float32)
    ep = max(1, round(20000 / max(1, math.ceil(3 * n / 256))))
    f = train_cindes(Y, R=R, epochs=ep, fake_ratio=2, seed=seed)
    Ymc = mrf.sample(n_mc, np.random.default_rng(999 + seed))
    with torch.no_grad():
        fh = f(torch.tensor(Ymc, dtype=torch.float32,
                            device=DEVICE)).squeeze(-1).cpu().numpy()
    log_r = np.clip(fh, -R, R) - mrf.log_p(Ymc)
    log_r -= log_r.max()
    r = np.exp(log_r)
    return float(0.5 * np.abs(r / r.mean() - 1.0).mean())


def dsm_atom_bound(mrf, n, seed, n_gen=4000):
    rng = np.random.default_rng(seed)
    Y = torch.tensor(mrf.sample(n, rng), dtype=torch.float32)
    ep = max(1, round(60000 / max(1, math.ceil(n / 256))))
    s_net = train_dsm(Y, T=3.0, epochs=ep, seed=seed)

    def d_fn(y, tau):
        t = torch.full((y.shape[0], 1), tau, device=DEVICE)
        return s_net(y, t)

    Yg = backward_sample(d_fn, D, n_gen, T=3.0, delta=1e-3, M=200, seed=seed)
    return float((np.minimum(Yg, 1 - Yg) < 1e-6).any(axis=1).mean())


if __name__ == "__main__":
    mrf = ChainMRF(d=D, coupling=CPL, unary=UN)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                r = json.loads(line)
                done.add((r["n"], r["seed"], r["method"]))
    for n in N_LIST:
        for seed in range(SEEDS):
            for meth in ("cindes", "dsm"):
                if (n, seed, meth) in done:
                    continue
                if meth == "cindes":
                    val = cindes_joint_tv(mrf, n, seed)
                    rec = {"n": n, "seed": seed, "method": "cindes",
                           "joint_tv": val}
                else:
                    val = dsm_atom_bound(mrf, n, seed)
                    rec = {"n": n, "seed": seed, "method": "dsm",
                           "atom_lb": val}
                print(json.dumps(rec), flush=True)
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
    print("ALL_DONE", flush=True)
