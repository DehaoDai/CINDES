"""CINDES versus DSM on the original 31-node binary-tree MRF.

Default experiment requested for the referee-response comparison:
  * n = 8000 training observations; 20 replications (seeds 0,...,19).
  * CINDES truncation R = 25 and K = 5000 score particles at every step.
  * No terminal posterior sampling or terminal mean denoising for CINDES.
  * Both methods: 10000 generated observations and M = 200 reverse steps.
  * Optimization budgets: approximately 20000 CINDES / 60000 DSM updates.

Run the complete comparison (40 fits, resumable):
    python cindes_vs_dsm_tree.py

Inspect the configuration without importing NumPy/PyTorch or fitting models:
    python cindes_vs_dsm_tree.py --dry-run

Run a single paired replication:
    python cindes_vs_dsm_tree.py --single-run --n 8000 --seed 0

Keep cindes_vs_dsm_mrf.py and tree_mrf.py in the same directory. The original
helpers, tree potentials, grid sampler, and shared reverse updates are reused.
The shared reverse sampler still performs its original final support clipping.
"""

import argparse
import json
import math
from pathlib import Path
import time


D_DEFAULT = 31
DEFAULT_OUT = "results_tree_n8000_R25_K5000_no_final_20reps.jsonl"


def experiment_config(*, d, coupling, unary, R, T, delta, M, K, n_gen,
                      cindes_steps, dsm_steps, fake_ratio, generation_batch):
    """Settings saved with every result and checked before resuming."""
    return {
        "protocol": "tree_no_terminal_v1",
        "d": d,
        "coupling": coupling,
        "unary": unary,
        "pot_seed": 11,
        "G": 200,
        "R": R,
        "T": T,
        "delta": delta,
        "M": M,
        "K": K,
        "n_gen": n_gen,
        "n_ref": 20000,
        "reference_seed": 12345,
        "cindes_steps_requested": cindes_steps,
        "dsm_steps_requested": dsm_steps,
        "training_batch": 256,
        "fake_ratio": fake_ratio,
        "generation_batch": generation_batch,
        "cindes_final": "none",
        "denoise_final": False,
    }


def run_once(n, seed, d=D_DEFAULT, methods=("cindes", "dsm"), coupling=0.55,
             unary=0.3, R=25.0, T=3.0, delta=1e-3, M=200, K=5000,
             n_gen=10000, cindes_steps=20000, dsm_steps=60000, fake_ratio=2,
             generation_batch=64, verbose=True):
    """Fit the requested methods; generate without a terminal correction."""
    # Deferred imports also make --help and --dry-run usable without PyTorch.
    import numpy as np
    import torch
    from cindes_vs_dsm_mrf import (
        DEVICE, n_params, train_cindes, train_dsm, cindes_score,
        backward_sample, energy_distance, sliced_wasserstein,
    )
    from tree_mrf import TreeMRF, marginal_tv_tree

    bs = 256
    mrf = TreeMRF(d=d, G=200, coupling=coupling, unary=unary, pot_seed=11)
    rng = np.random.default_rng(seed)
    Y_train = torch.tensor(mrf.sample(n, rng), dtype=torch.float32)
    Y_ref = mrf.sample(20000, np.random.default_rng(12345))
    ref_ll = mrf.log_p(Y_ref).mean()

    config = experiment_config(
        d=d, coupling=coupling, unary=unary, R=R, T=T, delta=delta, M=M,
        K=K, n_gen=n_gen, cindes_steps=cindes_steps, dsm_steps=dsm_steps,
        fake_ratio=fake_ratio, generation_batch=generation_batch,
    )
    res = {"n": n, "seed": seed, "d": d, "graph": "btree",
           "experiment_config": config}
    samples = {}

    if "cindes" in methods:
        batches = max(1, math.ceil((1 + fake_ratio) * n / bs))
        ep = max(1, round(cindes_steps / batches))
        t0 = time.time()
        f = train_cindes(Y_train, R=R, epochs=ep, fake_ratio=fake_ratio,
                         seed=seed, verbose=verbose)
        gen = torch.Generator(device=DEVICE).manual_seed(seed + 555)

        def c_fn(y, tau):
            return cindes_score(f, R, y, tau, K, gen)

        samples["cindes"] = backward_sample(
            c_fn, d, n_gen, T=T, delta=delta, M=M, seed=seed,
            batch=generation_batch, denoise_final=False, final_fn=None,
        )
        res["cindes_time"] = time.time() - t0
        res["cindes_params"] = n_params(f)
        res["cindes_updates"] = ep * batches

    if "dsm" in methods:
        batches = max(1, math.ceil(n / bs))
        ep = max(1, round(dsm_steps / batches))
        t0 = time.time()
        s_net = train_dsm(Y_train, T=T, epochs=ep, seed=seed, verbose=verbose)

        def d_fn(y, tau):
            t = torch.full((y.shape[0], 1), tau, device=DEVICE)
            return s_net(y, t)

        samples["dsm"] = backward_sample(
            d_fn, d, n_gen, T=T, delta=delta, M=M, seed=seed,
            batch=generation_batch, denoise_final=False, final_fn=None,
        )
        res["dsm_time"] = time.time() - t0
        res["dsm_params"] = n_params(s_net)
        res["dsm_updates"] = ep * batches

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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--study", dest="study", action="store_true",
                      help="run the replication study (default)")
    mode.add_argument("--single-run", dest="study", action="store_false",
                      help="run both methods once using --n and --seed")
    ap.set_defaults(study=True)
    ap.add_argument("--n", type=int, default=8000,
                    help="training sample size in --single-run mode")
    ap.add_argument("--seed", type=int, default=0,
                    help="replication seed in --single-run mode")
    ap.add_argument("--d", type=int, default=D_DEFAULT)
    ap.add_argument("--n-list", type=int, nargs="+", default=[8000],
                    help="training sample sizes in study mode (default: 8000)")
    ap.add_argument("--seeds", type=int, default=20,
                    help="number of replications, using seeds 0,...,seeds-1")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--max-runs", type=int, default=0,
                    help="stop after this many new method fits; 0 means all")
    ap.add_argument("--n-gen", type=int, default=10000)
    ap.add_argument("--R", type=float, default=25.0)
    ap.add_argument("--K", type=int, default=5000)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--cindes-steps", type=int, default=20000)
    ap.add_argument("--dsm-steps", type=int, default=60000)
    ap.add_argument("--coupling", type=float, default=0.55)
    ap.add_argument("--unary", type=float, default=0.3)
    ap.add_argument("--generation-batch", type=int, default=64,
                    help="generated samples per batch; use 32 for less memory")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the experiment configuration without fitting")
    args = ap.parse_args()

    positive = (args.n, *args.n_list, args.seeds, args.n_gen, args.R, args.K,
                args.M, args.cindes_steps, args.dsm_steps, args.generation_batch)
    if any(value <= 0 for value in positive) or args.d < 3:
        ap.error("sample sizes, budgets, R, K and M must be positive; d must be >= 3")
    if args.seed < 0 or args.max_runs < 0:
        ap.error("--seed and --max-runs must be nonnegative")

    settings = dict(
        d=args.d, coupling=args.coupling, unary=args.unary, R=args.R,
        T=3.0, delta=1e-3, M=args.M, K=args.K, n_gen=args.n_gen,
        cindes_steps=args.cindes_steps, dsm_steps=args.dsm_steps,
        fake_ratio=2, generation_batch=args.generation_batch,
    )
    expected_config = experiment_config(**settings)
    configs = ([(n, seed, method) for n in args.n_list
                for seed in range(args.seeds) for method in ("cindes", "dsm")]
               if args.study else [(args.n, args.seed, method)
                                   for method in ("cindes", "dsm")])

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("experiment_config") != expected_config:
                    ap.error(
                        f"{out_path}, line {line_number}: saved settings do not "
                        "match this experiment. Choose a new --out filename."
                    )
                done.add((r["n"], r["seed"], r["method"]))

    pending = [config for config in configs if config not in done]
    print(json.dumps({
        "training_sample_sizes": args.n_list if args.study else [args.n],
        "replications_per_size": args.seeds if args.study else 1,
        "replication_seeds": list(range(args.seeds)) if args.study else [args.seed],
        "methods": ["cindes", "dsm"],
        "total_method_fits": len(configs),
        "pending_method_fits": len(pending),
        "experiment_config": expected_config,
        "output": str(out_path),
    }, indent=2), flush=True)
    if args.dry_run:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_runs = 0
    for n, seed, method in pending:
        if args.max_runs and new_runs >= args.max_runs:
            break
        print(f"=== tree n={n} seed={seed} method={method} ===", flush=True)
        res, (mrf, samples, Y_ref) = run_once(
            n, seed, methods=(method,), **settings, verbose=False,
        )
        res["method"] = method
        line = json.dumps(res)
        print(line, flush=True)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if not args.study:
            import numpy as np
            np.savez(out_path.with_name(f"{out_path.stem}_qual_{method}.npz"),
                     Y=samples[method], Yref=Y_ref[:8000])
        done.add((n, seed, method))
        new_runs += 1
    print("ALL_DONE" if all(config in done for config in configs) else "PARTIAL",
          flush=True)


if __name__ == "__main__":
    main()
