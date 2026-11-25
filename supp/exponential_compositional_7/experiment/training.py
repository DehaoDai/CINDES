"""Three-fold selection and all-observation refitting for the exponential pilot.

The module imports no target DGP, graph, potential, evaluation reference, or
target score. All selection uses held-out observations and the common known
log-exponential reference (CINDES), or held-out denoising loss (DSM).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import models
import reference


DEFAULT_LAMBDAS = (0.0, 0.01, 0.1, 0.3, 1.0, 3.0)
T, T_MIN = 3.0, 0.001


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def source_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in ("models.py", "reference.py", "reference_score_table.npz", "training.py")}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


@contextmanager
def locked(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".training.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float32).copy()
        fold = np.asarray(data["fold_id"], dtype=np.int64).copy()
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a finite nonempty matrix")
    if fold.shape != (len(X),) or set(fold.tolist()) != {0, 1, 2}:
        raise ValueError("fold_id must assign all observations to three nonempty folds")
    return X, fold


def rng(seed):
    return torch.Generator(device="cpu").manual_seed(int(seed))


def seeds_for(method, seed):
    namespace = 110000000 + 1000 * int(seed)
    values = {"permutation": namespace + 2}
    if method == "cindes":
        values.update(initialization=namespace+1, negatives=namespace+3,
                      validation_negatives=namespace+4, normalizer=namespace+5)
    else:
        values.update(initialization=namespace+11, times=namespace+13, noise=namespace+14,
                      validation_times=namespace+15, validation_noise=namespace+16)
    return values


def reference_sample(generator, shape):
    return torch.from_numpy(reference.sample_reference(generator, shape))


def reference_log_density(x):
    return reference.log_reference_torch(x.double())


def initialize(spec, seed, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = models.build_model(spec)
    return model.to(device)


def state_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def make_training_loss(train, method, seeds, coefficient, target):
    train = torch.as_tensor(train, dtype=torch.float32, device="cpu")
    if method == "cindes":
        negative_rng = np.random.default_rng(seeds["negatives"])
        def loss(model, indices):
            positive = train[indices]
            negative = reference_sample(negative_rng, (2*len(indices), train.shape[1]))
            points = torch.cat((positive, negative)).to(target)
            labels = torch.cat((torch.ones(len(indices)), torch.zeros(2*len(indices)))).to(target)
            raw = model(points).squeeze(-1)
            value = F.binary_cross_entropy_with_logits(raw, labels)
            if coefficient:
                value = value + coefficient*models.mean_absolute_weight(model)
            return value
    else:
        train = train.to(target)
        time_rng, noise_rng = rng(seeds["times"]), rng(seeds["noise"])
        def loss(model, indices):
            clean = train[indices.to(target)]
            times = torch.exp(torch.rand((len(indices), 1), generator=time_rng)
                              *(math.log(T)-math.log(T_MIN))+math.log(T_MIN)).to(target)
            epsilon = torch.randn((len(indices), train.shape[1]), generator=noise_rng).to(target)
            sigma = torch.sqrt(-torch.expm1(-2*times))
            noisy = torch.exp(-times)*clean + sigma*epsilon
            value = ((sigma*model(noisy, times)+epsilon)**2).mean()
            if coefficient:
                value = value + coefficient*models.mean_absolute_weight(model)
            return value
    return loss


def make_validator(validation, method, seeds, normalizer_size, target):
    validation = torch.as_tensor(validation, dtype=torch.float32, device="cpu")
    if method == "cindes":
        bank = reference_sample(np.random.default_rng(seeds["normalizer"]),
                              (normalizer_size, validation.shape[1]))
        negatives = reference_sample(np.random.default_rng(seeds["validation_negatives"]),
                                   (2*len(validation), validation.shape[1]))
        constant_nll = -float(reference_log_density(validation).mean())
        metadata = {"criterion": "normalized_density_nll_omitting_common_log_q",
                    "q": "independent log(Exp(rate=1)); common to both methods",
                    "normalizer_size": normalizer_size,
                    "normalizer_bank_sha256": fingerprint(bank.numpy()),
                    "validation_negative_sha256": fingerprint(negatives.numpy())}
        @torch.no_grad()
        def validate(model):
            positive = torch.cat([model(x.to(target)).squeeze(-1).double().cpu()
                                  for x in validation.split(1024)])
            negative = torch.cat([model(x.to(target)).squeeze(-1).double().cpu()
                                  for x in negatives.split(1024)])
            values = torch.cat([model(x.to(target)).squeeze(-1).double().cpu()
                                for x in bank.split(8192)])
            if not bool(torch.isfinite(values).all() and torch.isfinite(positive).all()
                        and torch.isfinite(negative).all()):
                raise FloatingPointError("Nonfinite untruncated classifier logits")
            logsum = torch.logsumexp(values, 0)
            logz = float(logsum - math.log(len(values)))
            loss = logz-float(positive.mean())
            result = {"validation_loss": loss,
                      "validation_density_nll_full": loss+constant_nll,
                      "validation_log_normalizer": logz,
                      "validation_positive_mean_logit": float(positive.mean()),
                      "validation_bce": float((F.softplus(-positive).sum()+F.softplus(negative).sum())/(3*len(positive))),
                      "normalizer_ess": float(torch.exp(2*logsum-torch.logsumexp(2*values,0))),
                      "normalizer_max_normalized_weight": float(torch.exp(values.max()-logsum)),
                      "normalizer_logit_min": float(values.min()),
                      "normalizer_logit_max": float(values.max()),
                      **models.diagnostics(model)}
            return result
    else:
        clean = validation.repeat_interleave(8, dim=0)
        times = torch.exp(torch.rand((len(clean), 1), generator=rng(seeds["validation_times"]))
                          *(math.log(T)-math.log(T_MIN))+math.log(T_MIN))
        epsilon = torch.randn(clean.shape, generator=rng(seeds["validation_noise"]))
        sigma = torch.sqrt(-torch.expm1(-2*times))
        noisy = torch.exp(-times)*clean+sigma*epsilon
        metadata = {"criterion": "unpenalized_denoising_mse_per_coordinate", "draws_per_point": 8,
                    "time_bank_sha256": fingerprint(times.numpy()),
                    "noise_bank_sha256": fingerprint(epsilon.numpy()),
                    "fixed_reference_scores_cached": True}
        times, epsilon, sigma, noisy = (x.to(target) for x in (times, epsilon, sigma, noisy))
        cached_reference = None
        @torch.no_grad()
        def validate(model):
            nonlocal cached_reference
            if cached_reference is None:
                cached_reference = torch.cat([
                    model.reference_score(noisy[start:start+1024], times[start:start+1024])
                    for start in range(0, len(noisy), 1024)
                ])
            total = 0.0
            for start in range(0, len(noisy), 1024):
                sl = slice(start, start+1024)
                prediction = cached_reference[sl]+model.correction(noisy[sl], times[sl])
                errors = ((sigma[sl]*prediction+epsilon[sl])**2).mean(1)
                total += float(errors.double().sum())
            return {"validation_loss": total/len(noisy), **models.diagnostics(model)}
    return validate, metadata


def common_config(args, X, fold_id, train, validation, seed):
    return {"protocol": "log_exponential_common_reference_training_v1", "method": args.method,
            "kind": args.command, "d": X.shape[1], "n": len(X), "n_train": len(train),
            "n_validation": len(validation), "fit_seed": args.seed, "training_seed": seed,
            "fold": args.fold if args.command == "cv" else None,
            "data_sha256": fingerprint(X), "fold_partition_sha256": fingerprint(fold_id),
            "train_sha256": fingerprint(train), "validation_sha256": fingerprint(validation),
            "max_positive_epochs": args.epochs, "positive_batch": args.batch_size,
            "l1_lambda": args.coefficient, "l1_definition": "mean absolute Linear weights; biases excluded",
            "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0,
            "fixed_logit_truncation_R": None, "reference_base": "Exp(rate=1)",
            "common_transform": "Z=log(Y)", "reference_coordinates_independent": True,
            "fake_ratio": 2, "fresh_negatives": True,
            "positive_schedule_shared_across_methods_and_lambdas": True,
            "selection_uses_target_truth": False,
            "cindes_variant": args.cindes_variant,
            "source_sha256": source_hashes(), "torch": str(torch.__version__),
            "numpy": np.__version__, "device": args.device, "torch_threads": args.threads,
            "T": T, "t_min": T_MIN,
            "time_distribution": "log-uniform", "dsm_loss_reduction": "mean over examples and coordinates"}


def train_job(args):
    torch.set_num_threads(args.threads)
    target = torch.device(args.device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    X, fold_id = load_data(args.data)
    selected = None
    if args.command == "cv":
        train, validation = X[fold_id != args.fold], X[fold_id == args.fold]
        train_seed = 100*args.seed + args.fold
    else:
        selected = json.loads(args.selection.read_text())
        if selected["method"] != args.method or selected["fit_seed"] != args.seed:
            raise ValueError("Selection belongs to another method or fit seed")
        if selected["data_sha256"] != fingerprint(X) or selected["fold_partition_sha256"] != fingerprint(fold_id):
            raise ValueError("Selection data or folds changed")
        if selected.get("source_sha256") != source_hashes():
            raise ValueError("Selection was made with different fitting sources or reference table")
        args.coefficient = selected["selected_lambda"]
        args.epochs = selected["max_positive_epochs"]
        args.batch_size = selected["positive_batch"]
        args.dsm_parameterization = selected["dsm_parameterization"]
        args.cindes_variant = selected.get("cindes_variant", "baseline")
        train, validation, train_seed = X, X[:0], args.seed
    n_batch = math.ceil(len(train)/args.batch_size)
    horizon = args.epochs*n_batch
    stop = horizon if selected is None else int(round(selected["selected_fraction"]*horizon))
    config = common_config(args, X, fold_id, train, validation, train_seed)
    config.update({"horizon_updates": horizon, "stop_updates": stop,
                   "scheduler": "CosineAnnealingLR per update with full horizon retained"})
    spec = models.model_spec(args.method, X.shape[1], args.dsm_parameterization, cindes_variant=args.cindes_variant)
    config["model_spec"] = spec
    seeds = seeds_for(args.method, train_seed)
    config["rng_seeds"] = seeds
    if selected is not None:
        config["selection_sha256"] = file_hash(args.selection)
    else:
        config["validation_checks_requested"] = args.checks
        config["normalizer_size"] = args.normalizer_size
    directory = args.out
    with locked(directory):
        metadata_path = directory/"metadata.json"
        model = initialize(spec, seeds["initialization"], target)
        config["parameter_count"] = sum(p.numel() for p in model.parameters())
        config["penalized_weight_count"] = sum(w.numel() for w in models.linear_weights(model))
        if args.command == "cv":
            validate, validation_config = make_validator(validation, args.method, seeds, args.normalizer_size, target)
            config["validation"] = validation_config
        else:
            validate = None
        if metadata_path.exists():
            old = json.loads(metadata_path.read_text())
            if old.get("status") != "complete" or old.get("config") != config:
                raise ValueError("Output already contains incomplete or incompatible work; choose a new directory")
            if old["checkpoint_sha256"] != file_hash(directory/"model.pt"):
                raise ValueError("Saved model checkpoint changed")
            print(json.dumps({"status": "reused", "out": str(directory), "method": args.method}), flush=True)
            return old
        objective = make_training_loss(train, args.method, seeds, args.coefficient, target)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=horizon)
        permutation = rng(seeds["permutation"])
        checks_count = min(args.checks, horizon) if validate is not None else 0
        check_updates = {int(round(horizon*k/checks_count)): k for k in range(checks_count+1)} if checks_count else {}
        history = []
        training_seconds = validation_seconds = serialization_seconds = 0.0
        total_started = time.perf_counter()
        best_loss, best_update = float("inf"), None
        positive_seen = 0
        batch_digest = hashlib.sha256()

        def save_payload(path, update, extra=None):
            payload = {"format_version": 1, "model_config": spec, "state_dict": state_cpu(model),
                       "config": config, "update": update, "epoch": update/n_batch}
            if extra:
                payload.update(extra)
            atomic_torch(path, payload)
            return payload

        def checkpoint(update, train_loss=None):
            nonlocal validation_seconds, serialization_seconds, best_loss, best_update
            if validate is None:
                return
            before = time.perf_counter()
            model.eval()
            metrics = validate(model)
            if not all(math.isfinite(float(v)) for v in metrics.values()):
                raise FloatingPointError("Nonfinite validation metric")
            validation_seconds += time.perf_counter()-before
            row = {"check": check_updates[update], "update": update,
                   "fraction": check_updates[update]/checks_count,
                   "epoch": update/n_batch, "n_validation": len(validation),
                   "training_loss_since_previous_check": train_loss, **metrics}
            history.append(row)
            before = time.perf_counter()
            if metrics["validation_loss"] < best_loss:
                best_loss, best_update = metrics["validation_loss"], update
                save_payload(directory/"best.pt", update, {"validation_metrics": metrics})
            atomic_json(directory/"history.json", history)
            serialization_seconds += time.perf_counter()-before
            model.train()
            if args.verbose:
                print(f"{args.method} fold={getattr(args, 'fold', None)} lambda={args.coefficient:g} update={update}/{horizon} val={metrics['validation_loss']:.7g}", flush=True)

        atomic_json(metadata_path, {"status": "in_progress", "config": config})
        checkpoint(0)
        update = 0
        accumulated, occurrences = 0.0, 0
        segment_started = time.perf_counter()
        while update < stop:
            order = torch.randperm(len(train), generator=permutation)
            for start in range(0, len(train), args.batch_size):
                indices = order[start:start+args.batch_size]
                batch_digest.update(indices.numpy().tobytes())
                loss = objective(model, indices)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                scheduler.step()
                update += 1
                positive_seen += len(indices)
                accumulated += float(loss.detach())*len(indices)
                occurrences += len(indices)
                if update in check_updates:
                    training_seconds += time.perf_counter()-segment_started
                    checkpoint(update, accumulated/occurrences)
                    accumulated, occurrences = 0.0, 0
                    segment_started = time.perf_counter()
                if update >= stop:
                    break
        training_seconds += time.perf_counter()-segment_started
        before = time.perf_counter()
        model.eval()
        save_payload(directory/"model.pt", update)
        serialization_seconds += time.perf_counter()-before
        record = {"status": "complete", "config": config,
                  "actual_updates": update, "actual_positive_epochs": update/n_batch,
                  "positive_occurrences": positive_seen,
                  "negative_occurrences": 2*positive_seen if args.method == "cindes" else 0,
                  "positive_batch_schedule_sha256": batch_digest.hexdigest(),
                  "best_validation_loss": best_loss if best_update is not None else None,
                  "best_validation_update": best_update,
                  "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                  "serialization_seconds": serialization_seconds,
                  "wall_seconds": time.perf_counter()-total_started,
                  "checkpoint_sha256": file_hash(directory/"model.pt"),
                  "final_weight_diagnostics": models.diagnostics(model),
                  "timing_scope": "training excludes validation and checkpoint serialization"}
        atomic_json(metadata_path, record)
        print(json.dumps({"status": "complete", "method": args.method, "kind": args.command,
                          "out": str(directory), "updates": update,
                          "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                          "best_validation_loss": record["best_validation_loss"]}), flush=True)
        return record


def select_job(args):
    X, fold_id = load_data(args.data)
    expected = {(float(coefficient), fold) for coefficient in args.lambdas for fold in range(3)}
    candidates = {}
    records = []
    for path in sorted(args.root.rglob("metadata.json")):
        row = json.loads(path.read_text())
        conf = row.get("config", {})
        if conf.get("kind") != "cv" or conf.get("method") != args.method or conf.get("fit_seed") != args.seed:
            continue
        key = (float(conf["l1_lambda"]), conf["fold"])
        if key not in expected:
            continue
        if key in candidates:
            raise ValueError(f"Duplicate candidate {key}")
        if row.get("status") != "complete" or conf["data_sha256"] != fingerprint(X) or conf["fold_partition_sha256"] != fingerprint(fold_id):
            raise ValueError("CV candidate incomplete or data differs")
        if conf["source_sha256"] != source_hashes():
            raise ValueError("CV candidate was fitted with different source bytes")
        if file_hash(path.parent/"model.pt") != row["checkpoint_sha256"]:
            raise ValueError("CV checkpoint changed")
        history = json.loads((path.parent/"history.json").read_text())
        candidates[key] = (row, history)
        records.append({"lambda": key[0], "fold": key[1], "metadata": str(path),
                        "metadata_sha256": file_hash(path), "history_sha256": file_hash(path.parent/"history.json")})
    if set(candidates) != expected:
        raise ValueError(f"Missing CV candidates: {sorted(expected-set(candidates))}")
    configurations = [row["config"] for row, _ in candidates.values()]
    for field in ("max_positive_epochs", "positive_batch", "model_spec", "normalizer_size", "validation_checks_requested"):
        if any(c[field] != configurations[0][field] for c in configurations):
            raise ValueError(f"Incompatible CV setting: {field}")
    choices = []
    for coefficient in args.lambdas:
        histories = [candidates[float(coefficient), fold][1] for fold in range(3)]
        lengths = {len(h) for h in histories}
        if len(lengths) != 1:
            raise ValueError("Different CV evaluation counts across folds")
        for k in range(len(histories[0])):
            check = [h[k] for h in histories]
            if any(r["check"] != check[0]["check"] or r["fraction"] != check[0]["fraction"] for r in check):
                raise ValueError("CV fractions do not align")
            count = sum(r["n_validation"] for r in check)
            value = sum(r["n_validation"]*r["validation_loss"] for r in check)/count
            choices.append({"lambda": float(coefficient), "check": k,
                            "fraction": check[0]["fraction"], "weighted_validation_loss": value,
                            "fold_validation_losses": [r["validation_loss"] for r in check]})
    best = min(choices, key=lambda r: (r["weighted_validation_loss"], r["lambda"], r["check"]))
    config = configurations[0]
    result = {"protocol": "three_fold_joint_lambda_common_duration_v1", "method": args.method,
              "fit_seed": args.seed, "data_sha256": fingerprint(X),
              "fold_partition_sha256": fingerprint(fold_id),
              "max_positive_epochs": config["max_positive_epochs"], "positive_batch": config["positive_batch"],
              "dsm_parameterization": config["model_spec"]["parameterization"] if args.method == "dsm" else "reference_residual",
              "cindes_variant": config["cindes_variant"],
              "selected_lambda": best["lambda"], "selected_check": best["check"],
              "selected_fraction": best["fraction"], "selected_positive_epochs": best["fraction"]*config["max_positive_epochs"],
              "selected_validation_loss": best["weighted_validation_loss"],
              "tie_rule": "smaller loss, then smaller lambda, then earlier common check",
              "target_truth_used": False, "candidates": choices, "inputs": records,
              "cv_training_seconds_sum": sum(row["training_seconds"] for row, _ in candidates.values()),
              "cv_validation_seconds_sum": sum(row["validation_seconds"] for row, _ in candidates.values()),
              "source_sha256": source_hashes()}
    atomic_json(args.out, result)
    print(json.dumps({key: result[key] for key in ("method", "fit_seed", "selected_lambda", "selected_check", "selected_positive_epochs", "selected_validation_loss")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("cv", "refit"):
        sub = subs.add_parser(command)
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--method", choices=("cindes", "dsm"), required=True)
        sub.add_argument("--seed", type=int, required=True)
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--lambda", dest="coefficient", type=float, default=0.0)
        sub.add_argument("--epochs", type=int, default=2000)
        sub.add_argument("--batch-size", type=int, default=256)
        sub.add_argument("--checks", type=int, default=100)
        sub.add_argument("--normalizer-size", type=int, default=65536)
        sub.add_argument("--dsm-parameterization", choices=("reference_residual",), default="reference_residual")
        sub.add_argument("--cindes-variant", choices=("baseline",), default="baseline")
        sub.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        sub.add_argument("--threads", type=int, default=1)
        sub.add_argument("--verbose", action="store_true")
        if command == "cv":
            sub.add_argument("--fold", type=int, choices=(0,1,2), required=True)
        else:
            sub.add_argument("--selection", type=Path, required=True)
    select = subs.add_parser("select")
    select.add_argument("--data", type=Path, required=True)
    select.add_argument("--method", choices=("cindes", "dsm"), required=True)
    select.add_argument("--seed", type=int, required=True)
    select.add_argument("--root", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    select.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    args = parser.parse_args()
    if args.command == "select":
        select_job(args)
        return
    if min(args.epochs,args.batch_size,args.checks,args.normalizer_size,args.threads) < 1:
        parser.error("Budgets, batch size, normalizer size, and threads must be positive")
    if not math.isfinite(args.coefficient) or args.coefficient < 0 or args.seed < 0:
        parser.error("Nonnegative finite lambda and nonnegative seed required")
    train_job(args)


if __name__ == "__main__":
    main()
