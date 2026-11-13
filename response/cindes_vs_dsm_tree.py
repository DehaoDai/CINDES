"""CINDES vs. DSM on a TREE-structured MRF (balanced binary tree,
d = 31, d* = 2) on [0,1]^31.

Second bounded-domain experiment for the reviewer response: same potential
family and conditioning as the chain studies, different graph topology --
demonstrating that CINDES's adaptation is graph-agnostic (it never sees
the graph), while DSM again faces the ambient dimension.

Reuses training, samplers, and global metrics from cindes_vs_dsm_mrf.py;
ground truth and clique-marginal metrics from tree_mrf.py.  TV-2D is over
the 30 tree EDGES (the cliques), the analog of the chain's adjacent pairs.

Usage (same pattern as before; resumable):
  python cindes_vs_dsm_tree.py --study --seeds 10 \
      --n-list 500 1000 2000 4000 8000 16000 \
      --n-gen 10000 --K 1024 --M 200 \
      --cindes-steps 20000 --dsm-steps 60000
Single run (saves qual_{method}_tree.npz):
  python cindes_vs_dsm_tree.py --n 16000 --seed 0 ...
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from cindes_vs_dsm_mrf import (DEVICE, n_params, train_cindes, train_dsm,
                               cindes_score, cindes_posterior_sample,
                               backward_sample, energy_distance,
                               sliced_wasserstein)
from tree_mrf import TreeMRF, marginal_tv_tree

D_DEFAULT = 31


def run_once(n, seed, d=D_DEFAULT, methods=("cindes", "dsm"), coupling=0.55,
             unary=0.3, T=3.0, delta=1e-3, M=100, K=192, n_gen=1200,
             cindes_steps=10000, dsm_steps=24000, fake_ratio=2,
             cindes_final="sample", verbose=True):
    """cindes_final: 'sample' (terminal posterior sampling, recommended),
    'none' (raw Algorithm 2 output, as in the chain study tables)."""
    bs = 256
    mrf = TreeMRF(d=d, coupling=coupling, unary=unary)
    rng = np.random.default_rng(seed)
    Y_train = torch.tensor(mrf.sample(n, rng), dtype=torch.float32)
    Y_ref = mrf.sample(20000, np.random.default_rng(12345))
    ref_ll = mrf.log_p(Y_ref).mean()

    res = {"n": n, "seed": seed, "d": d, "graph": "btree"}
    R = 16.0
    samples = {}

    if "cindes" in methods:
        ep = max(1, round(cindes_steps
                          / max(1, math.ceil((1 + fake_ratio) * n / bs))))
        t0 = time.time()
        f = train_cindes(Y_train, R=R, epochs=ep, fake_ratio=fake_ratio,
                         seed=seed, verbose=verbose)
        gen = torch.Generator(device=DEVICE).manual_seed(seed + 555)

        def c_fn(y, tau):
            return cindes_score(f, R, y, tau, K, gen)

        fin = None
        if cindes_final == "sample":
            def fin(y, delta_):
                return cindes_posterior_sample(f, R, y, delta_, K, gen)

        samples["cindes"] = backward_sample(c_fn, d, n_gen, T=T, delta=delta,
                                            M=M, seed=seed, final_fn=fin)
        res["cindes_time"] = time.time() - t0
        res["cindes_params"] = n_params(f)

    if "dsm" in methods:
        ep = max(1, round(dsm_steps / max(1, math.ceil(n / bs))))
        t0 = time.time()
        s_net = train_dsm(Y_train, T=T, epochs=ep, seed=seed, verbose=verbose)

        def d_fn(y, tau):
            t = torch.full((y.shape[0], 1), tau, device=DEVICE)
            return s_net(y, t)

        samples["dsm"] = backward_sample(d_fn, d, n_gen, T=T, delta=delta,
                                         M=M, seed=seed)
        res["dsm_time"] = time.time() - t0
        res["dsm_params"] = n_params(s_net)

    mrng = np.random.default_rng(seed + 777)
    for name, Yg in samples.items():
        tv1, tv2 = marginal_tv_tree(Yg, mrf)
        res[f"{name}_energy"] = float(energy_distance(Yg, Y_ref, mrng))
        res[f"{name}_sw"] = float(sliced_wasserstein(Yg, Y_ref, mrng))
        res[f"{name}_tv1"] = tv1
        res[f"{name}_tv2"] = tv2
        res[f"{name}_llgap"] = float(ref_ll - mrf.log_p(Yg).mean())
        res[f"{name}_clip"] = float(
            (np.minimum(Yg, 1 - Yg) < 1e-6).any(axis=1).mean())
    return res, (mrf, samples, Y_ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=D_DEFAULT)
    ap.add_argument("--study", action="store_true")
    ap.add_argument("--n-list", type=int, nargs="+",
                    default=[500, 1000, 2000, 4000, 8000, 16000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=str, default="results_tree.jsonl")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--n-gen", type=int, default=1200)
    ap.add_argument("--K", type=int, default=192)
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--cindes-steps", type=int, default=10000)
    ap.add_argument("--dsm-steps", type=int, default=24000)
    ap.add_argument("--coupling", type=float, default=0.55)
    ap.add_argument("--unary", type=float, default=0.3)
    ap.add_argument("--cindes-final", type=str, default="sample",
                    choices=["sample", "none"])
    args = ap.parse_args()

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
        print(f"=== tree n={n} seed={seed} method={meth} ===", flush=True)
        res, (mrf, samples, Y_ref) = run_once(
            n, seed, d=args.d, methods=(meth,), coupling=args.coupling,
            unary=args.unary, n_gen=args.n_gen, K=args.K, M=args.M,
            cindes_steps=args.cindes_steps, dsm_steps=args.dsm_steps,
            cindes_final=args.cindes_final, verbose=False)
        res["method"] = meth
        line = json.dumps(res)
        print(line, flush=True)
        with open(args.out, "a") as fh:
            fh.write(line + "\n")
        if not args.study:
            np.savez(f"qual_{meth}_tree.npz", Y=samples[meth],
                     Yref=Y_ref[:8000])
        done.add((n, seed, meth))
        new_runs += 1
    print("ALL_DONE" if all(c in done for c in configs) else "PARTIAL",
          flush=True)


if __name__ == "__main__":
    main()
