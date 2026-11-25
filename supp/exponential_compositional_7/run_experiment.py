"""Run one paired, resumable SCC replication under the frozen study protocol.

The statistical implementation lives unchanged in experiment/.  This driver
handles data, scheduler metadata, complete-stage reuse, and output validation.
It never changes a model or setting in response to generated-sample metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "experiment"))

from common import (atomic_npz, ensure_study, lock, read_json, resolve_root,
                    sha256, validate_reference, validate_task_record, write_json)
from study_config import profile_config, task_plan


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(path, task_dir):
    return Path(path).resolve().relative_to(task_dir.resolve()).as_posix()


def announce(stage, **values):
    print(json.dumps({"utc": now(), "stage": stage, **values}, allow_nan=False), flush=True)


def archive_partial(paths, task_dir, label, reason):
    """Retain interrupted work outside the directory searched by CV selection."""
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        return None
    destination = task_dir / "interrupted" / f"{label}_{uuid.uuid4().hex}"
    destination.mkdir(parents=True, exist_ok=False)
    moved = []
    for path in existing:
        target = destination / path.name
        if target.exists():
            raise RuntimeError("Duplicate name in interrupted-stage archive")
        shutil.move(str(path), str(target))
        moved.append({"original": relative(path, task_dir),
                      "archived": relative(target, task_dir)})
    write_json(destination / "interruption.json",
               {"archived_utc": now(), "reason": reason, "files": moved})
    announce("interrupted_stage_preserved", label=label, destination=str(destination))
    return destination


def prepare_data(task_dir, task, config, study_hash):
    import numpy as np
    from dgp import ExponentialCompositionalDGP

    path, metadata = task_dir / "data.npz", task_dir / "data.json"
    expected = {"study_identity_sha256": study_hash, "task_identity": task,
                "d": config["d"], "folds": config["folds"]}
    if metadata.exists():
        old = read_json(metadata)
        if old.get("status") == "complete":
            if old.get("identity") != expected or not path.is_file() or sha256(path) != old.get("sha256"):
                raise RuntimeError("Completed paired training data changed or belong to another study")
            with np.load(path, allow_pickle=False) as data:
                if data["X"].shape != (task["n"], config["d"]):
                    raise RuntimeError("Unexpected training data shape")
                if set(data["fold_id"].tolist()) != {0, 1, 2}:
                    raise RuntimeError("Training data do not contain the three paired folds")
            return path, old
    archive_partial([path, metadata], task_dir, "data", "No complete data record")
    dgp = ExponentialCompositionalDGP(d=config["d"], **{
        name: config["dgp"][name] for name in ("pot_seed", "amplitude", "kappa")})
    observations = dgp.sample(task["n"], np.random.default_rng(task["data_seed"]))
    folds = np.empty(len(observations), dtype=np.int64)
    order = np.random.default_rng(task["fold_seed"]).permutation(len(observations))
    for fold, indices in enumerate(np.array_split(order, config["folds"])):
        folds[indices] = fold
    with np.errstate(divide="raise", invalid="raise"):
        x = np.log(observations).astype(np.float32)
    if not np.isfinite(x).all() or not (observations > 0).all():
        raise FloatingPointError("Invalid paired training observations")
    atomic_npz(path, X=x, Y=observations, fold_id=folds)
    record = {"status": "complete", "identity": expected, "sha256": sha256(path),
              "created_utc": now(), "dgp": dgp.config,
              "training_sampler_stats": dict(dgp.last_sample_stats)}
    write_json(metadata, record)
    return path, record


def fitting_stage(args, directory, task_dir, stage_identity, environment_path):
    """Run or validate one CV/refit stage without altering the core trainer."""
    import training

    metadata = directory / "metadata.json"
    intent = directory / "scc_stage_started.json"
    receipt = directory / "scc_stage.json"
    complete = metadata.exists() and read_json(metadata).get("status") == "complete"
    if intent.exists() and read_json(intent).get("identity") != stage_identity:
        raise RuntimeError(f"Incompatible fitting-stage identity: {directory}")
    if not complete and directory.exists():
        archive_partial([directory], task_dir, "fit", "Fitting stage did not finish; restart its deterministic schedule")
    directory.mkdir(parents=True, exist_ok=True)
    if not intent.exists():
        if complete:
            raise RuntimeError("Completed stage lacks its original SCC provenance record")
        write_json(intent, {"identity": stage_identity, "started_utc": now(),
                            "environment_path": relative(environment_path, task_dir),
                            "environment_sha256": sha256(environment_path)})
    provenance = read_json(intent)
    environment = task_dir / provenance["environment_path"]
    if not environment.is_file() or sha256(environment) != provenance["environment_sha256"]:
        raise RuntimeError("Fitting-stage hardware/environment record changed")
    if receipt.exists():
        old = read_json(receipt)
        if old.get("identity") != stage_identity:
            raise RuntimeError("Completed fitting-stage receipt is incompatible")
        for name, digest in old.get("artifact_sha256", {}).items():
            if sha256(directory / name) != digest:
                raise RuntimeError(f"Completed fitting artifact changed: {directory / name}")
    record = training.train_job(args)
    files = [directory / "metadata.json", directory / "model.pt"]
    if args.command == "cv":
        files += [directory / "history.json", directory / "best.pt"]
    hashes = {path.name: sha256(path) for path in files}
    receipt_value = {"status": "complete", "identity": stage_identity,
                     "environment_path": provenance["environment_path"],
                     "environment_sha256": provenance["environment_sha256"],
                     "artifact_sha256": hashes}
    if receipt.exists() and read_json(receipt) != receipt_value:
        raise RuntimeError("Completed fitting-stage artifacts changed during reuse")
    if not receipt.exists():
        write_json(receipt, receipt_value)
    return record, receipt_value


def fit_method(method, task_dir, task, config, manifest, data_path, device,
               threads, environment_path, progress):
    import training

    method_dir = task_dir / "methods" / method
    cv_root = method_dir / "cv"
    stage_environments, cv_records = [], []
    for coefficient in config["lambdas"]:
        for fold in range(config["folds"]):
            directory = cv_root / f"lambda_{coefficient:g}" / f"fold_{fold}"
            label = f"{method}/cv/lambda_{coefficient:g}/fold_{fold}"
            progress(label)
            args = SimpleNamespace(command="cv", data=data_path, method=method,
                seed=task["fit_seed"], out=directory, coefficient=float(coefficient),
                epochs=config["epochs"], batch_size=config["batch_size"],
                checks=config["checks"], normalizer_size=config["normalizer_size"],
                dsm_parameterization="reference_residual", cindes_variant="baseline",
                device=device, threads=threads, verbose=False, fold=fold)
            identity = {"study_identity_sha256": manifest["identity_sha256"],
                        "task_identity": task, "method": method, "stage": "cv",
                        "coefficient": float(coefficient), "fold": fold,
                        "data_sha256": sha256(data_path), "device": device,
                        "threads": threads}
            record, receipt = fitting_stage(args, directory, task_dir, identity, environment_path)
            cv_records.append(record)
            stage_environments.append(receipt["environment_path"])
    selection_path = method_dir / "selection.json"
    progress(f"{method}/selection")
    # Core selection deterministically rechecks every complete CV checkpoint,
    # data fingerprint, fold history, source, and common evaluation fraction.
    old_selection = read_json(selection_path) if selection_path.exists() else None
    candidate_path = method_dir / f"selection_candidate_{uuid.uuid4().hex}.json"
    selected = training.select_job(SimpleNamespace(
        data=data_path, method=method, seed=task["fit_seed"], root=cv_root,
        out=candidate_path, lambdas=list(config["lambdas"])))
    if old_selection is not None and old_selection != selected:
        raise RuntimeError(f"A previously recorded CV selection changed; original retained, candidate saved at {candidate_path}")
    if old_selection is None:
        os.replace(candidate_path, selection_path)
    else:
        candidate_path.unlink()
    directory = method_dir / "refit"
    progress(f"{method}/refit")
    args = SimpleNamespace(command="refit", data=data_path, method=method,
        seed=task["fit_seed"], out=directory, coefficient=selected["selected_lambda"],
        epochs=config["epochs"], batch_size=config["batch_size"],
        checks=config["checks"], normalizer_size=config["normalizer_size"],
        dsm_parameterization="reference_residual", cindes_variant="baseline",
        device=device, threads=threads, verbose=False, selection=selection_path)
    identity = {"study_identity_sha256": manifest["identity_sha256"],
                "task_identity": task, "method": method, "stage": "refit",
                "data_sha256": sha256(data_path), "selection_sha256": sha256(selection_path),
                "device": device, "threads": threads}
    refit, receipt = fitting_stage(args, directory, task_dir, identity, environment_path)
    stage_environments.append(receipt["environment_path"])
    return {"selection": selected, "refit": refit, "cv": cv_records,
            "environment_paths": stage_environments,
            "model_path": directory / "model.pt", "selection_path": selection_path}


def generation_method(method, fitted, task_dir, task, config, manifest,
                      device, environment_path, progress):
    import numpy as np
    import torch
    import models
    from sampling_variants import sample_variant

    method_dir = task_dir / "methods" / method
    model_hash = sha256(fitted["model_path"])
    common = {"study_identity_sha256": manifest["identity_sha256"],
              "task_identity": task, "method": method, "model_sha256": model_hash,
              "device": device, "initialization": config["initialization"],
              "M": config["M"], "K": config["K"], "T": config["T"],
              "delta": config["delta"], "generation_batch": config["generation_batch"]}
    model = None
    # The unchanged posterior checks device equality.  A concrete device is
    # needed because tensors allocated on generic 'cuda' report 'cuda:0'.
    sampling_device = str(torch.device("cuda", torch.cuda.current_device())) if device == "cuda" else device
    parts, receipts = [], []
    shard_size = config["n_gen"] // config["generation_shards"]
    for shard in range(config["generation_shards"]):
        progress(f"{method}/generation/shard_{shard:02d}")
        path = method_dir / "generation" / f"shard_{shard:02d}.npz"
        metadata = path.with_suffix(".json")
        seeds = {"seed": 180000000 + 10000 * task["fit_seed"] + shard,
                 "particle_seed": 190000000 + 10000 * task["fit_seed"] + shard,
                 "reference_seed": 330000000 + 10000 * task["fit_seed"] + shard,
                 "bridge_seed": 340000000 + 10000 * task["fit_seed"] + shard,
                 "fine_particle_seed": 350000000 + 10000 * task["fit_seed"] + shard}
        expected = {**common, "shard": shard, "n": shard_size, "rng": seeds}
        record = read_json(metadata) if metadata.exists() else None
        if record is not None and record.get("status") == "complete":
            if record.get("identity") != expected or not path.is_file() or sha256(path) != record.get("sha256"):
                raise RuntimeError(f"Completed sampling chunk changed: {path}")
            env_path = task_dir / record["environment_path"]
            if not env_path.is_file() or sha256(env_path) != record["environment_sha256"]:
                raise RuntimeError("Sampling-chunk environment record changed")
            announce("generation_reused", method=method, shard=shard)
        else:
            archive_partial([path, metadata], task_dir, "generation",
                            "Sampling chunk lacks a complete record")
            if model is None:
                model, _ = models.load_model(fitted["model_path"], device=device)
            z, sampling = sample_variant(model, method, shard_size, d=config["d"],
                initialization=config["initialization"], M=config["M"],
                K=config["K"], T=config["T"], delta=config["delta"],
                batch=config["generation_batch"], device=sampling_device, **seeds)
            with np.errstate(over="raise", under="raise", invalid="raise"):
                y = np.exp(np.asarray(z, dtype=np.float64))
            if z.shape != (shard_size, config["d"]) or not np.isfinite(z).all() or not np.isfinite(y).all() or not (y > 0).all():
                raise FloatingPointError("Invalid generated sample")
            atomic_npz(path, Z=z, Y=y)
            record = {"status": "complete", "identity": expected, "sha256": sha256(path),
                      "sampling": sampling, "environment_path": relative(environment_path, task_dir),
                      "environment_sha256": sha256(environment_path), "completed_utc": now()}
            write_json(metadata, record)
            announce("generation_complete", method=method, shard=shard,
                     seconds=sampling["generation_seconds"])
        with np.load(path, allow_pickle=False) as data:
            z = np.asarray(data["Z"])
            if z.shape != (shard_size, config["d"]) or not np.isfinite(z).all():
                raise RuntimeError("Invalid saved generation chunk")
            parts.append(z.copy())
        receipts.append({"path": relative(path, task_dir), "sha256": record["sha256"],
                         "metadata_path": relative(metadata, task_dir),
                         "metadata_sha256": sha256(metadata),
                         "generation_seconds": record["sampling"]["generation_seconds"],
                         "environment_path": record["environment_path"]})
    del model
    gc.collect()
    z = np.concatenate(parts)
    path = method_dir / "generated.npz"
    metadata = method_dir / "generated.json"
    expected = {**common, "n": config["n_gen"], "shards": receipts}
    if metadata.exists() and read_json(metadata).get("status") == "complete":
        record = read_json(metadata)
        if record.get("identity") != expected or not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError("Completed generated sample aggregate changed")
    else:
        archive_partial([path, metadata], task_dir, "aggregate", "No complete aggregate record")
        with np.errstate(over="raise", under="raise", invalid="raise"):
            y = np.exp(np.asarray(z, dtype=np.float64))
        if z.shape != (config["n_gen"], config["d"]) or not np.isfinite(y).all() or not (y > 0).all():
            raise FloatingPointError("Invalid generated aggregate")
        atomic_npz(path, Z=z, Y=y)
        record = {"status": "complete", "identity": expected, "sha256": sha256(path),
                  "generation_seconds": sum(r["generation_seconds"] for r in receipts),
                  "timing_scope": "Sum of completed sampling chunks; training and evaluation excluded"}
        write_json(metadata, record)
    return {"path": path, "metadata_path": metadata, "metadata": record,
            "environment_paths": [r["environment_path"] for r in receipts]}


def artifact_map(task_dir):
    """Hash stable scientific artifacts; attempt logs are retained separately."""
    paths = [task_dir / "data.npz", task_dir / "data.json", task_dir / "task_manifest.json"]
    for method in ("cindes", "dsm"):
        for path in sorted((task_dir / "methods" / method).rglob("*")):
            if path.is_file() and path.suffix not in (".lock", ".tmp", ".pyc"):
                paths.append(path)
    paths += sorted((task_dir / "environment").glob("*.json"))
    return {relative(path, task_dir): sha256(path) for path in paths}


def run(args, root, task, config, manifest):
    task_dir = root / "runs" / f"n_{task['n']}" / f"seed_{task['seed']}"
    task_dir.mkdir(parents=True, exist_ok=True)
    with lock(task_dir / ".task.lock"):
        try:
            reference_record = validate_reference(root, manifest)
            completed = validate_task_record(task_dir, manifest, task)
            if completed is not None:
                if any(row.get('reference_sha256') != reference_record['reference_sha256']
                       for row in completed['methods'].values()):
                    raise RuntimeError('Completed task metrics refer to a different evaluation sample')
                announce("paired_replication_reused", n=task["n"], seed=task["seed"],
                         result=str(task_dir / "result.json"))
                return completed
            reference_path = root / reference_record.get("target_path", "reference/target.npz")
            from preflight import check_environment
            environment = check_environment(args.device, args.threads)
        except BaseException as error:
            failure = {"status": "failed", "stage": "preflight", "failed_utc": now(),
                       "task_identity": task, "study_identity_sha256": manifest["identity_sha256"],
                       "requested_device": args.device, "job_id": os.environ.get("JOB_ID"),
                       "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc()}
            write_json(task_dir / "attempts" / f"preflight_{uuid.uuid4().hex}.json", failure)
            write_json(task_dir / "progress.json", failure)
            raise
        runtime_identity = {"device": args.device, "threads": args.threads,
                            "torch": environment["torch"], "numpy": environment["numpy"],
                            "scipy": environment["scipy"]}
        task_manifest = task_dir / "task_manifest.json"
        identity = {"study_identity_sha256": manifest["identity_sha256"],
                    "task_identity": task, "numerical_environment": runtime_identity}
        if task_manifest.exists():
            if read_json(task_manifest)["identity"] != identity:
                raise RuntimeError("Partially completed task has a different protocol or numerical environment")
        else:
            write_json(task_manifest, {"identity": identity, "created_utc": now()})
        attempt = uuid.uuid4().hex
        environment_path = task_dir / "environment" / f"attempt_{attempt}.json"
        write_json(environment_path, {**environment, "attempt_id": attempt, "created_utc": now(),
                                      "study_identity_sha256": manifest["identity_sha256"]})
        attempt_path = task_dir / "attempts" / f"{attempt}.json"
        began = time.perf_counter()
        attempt_record = {"status": "running", "attempt_id": attempt, "started_utc": now(),
                          "task_identity": task, "study_identity_sha256": manifest["identity_sha256"],
                          "environment_path": relative(environment_path, task_dir)}
        write_json(attempt_path, attempt_record)

        def progress(stage):
            attempt_record.update(stage=stage, updated_utc=now(),
                                  current_attempt_seconds=time.perf_counter()-began)
            write_json(attempt_path, attempt_record)
            write_json(task_dir / "progress.json", attempt_record)
            announce(stage, n=task["n"], seed=task["seed"])

        try:
            progress("data")
            data_path, data_record = prepare_data(task_dir, task, config, manifest["identity_sha256"])
            method_order = ("cindes", "dsm") if task["seed"] % 2 == 0 else ("dsm", "cindes")
            fits, generated = {}, {}
            for method in method_order:
                fits[method] = fit_method(method, task_dir, task, config, manifest,
                    data_path, args.device, args.threads, environment_path, progress)
                generated[method] = generation_method(method, fits[method], task_dir,
                    task, config, manifest, args.device, environment_path, progress)
            # The target evaluator is constructed only after both methods have
            # selected, refitted, and generated their observations.
            import numpy as np
            from evaluation import Evaluator
            progress("evaluation_reference")
            reference_started = time.perf_counter()
            with np.load(reference_path, allow_pickle=False) as data:
                evaluator = Evaluator(data["Y"])
            reference_setup_seconds = time.perf_counter()-reference_started
            results = {}
            for method in ("cindes", "dsm"):
                progress(f"{method}/evaluation")
                fit, generation = fits[method], generated[method]
                with np.load(generation["path"], allow_pickle=False) as data:
                    y = data["Y"]
                evaluation_started = time.perf_counter()
                metrics = evaluator.evaluate(y)
                evaluation_seconds = time.perf_counter()-evaluation_started
                selected, refit = fit["selection"], fit["refit"]
                cv_training = sum(r["training_seconds"] for r in fit["cv"])
                cv_validation = sum(r["validation_seconds"] for r in fit["cv"])
                cv_serialization = sum(r["serialization_seconds"] for r in fit["cv"])
                generation_seconds = generation["metadata"]["generation_seconds"]
                environments = sorted(set(fit["environment_paths"] + generation["environment_paths"]))
                hardware = [read_json(task_dir / path) for path in environments]
                result = {**metrics, "method": method, "n": task["n"], "d": config["d"],
                    "seed": task["seed"], "fit_seed": task["fit_seed"],
                    "task_id": task["task_id"], "profile": manifest["identity"]["profile"],
                    "study_identity_sha256": manifest["identity_sha256"],
                    "experiment_config": config, "selected_lambda": selected["selected_lambda"],
                    "selected_positive_epochs": selected["selected_positive_epochs"],
                    "selected_fraction": selected["selected_fraction"],
                    "selected_validation_loss": selected["selected_validation_loss"],
                    "actual_updates": refit["actual_updates"],
                    "actual_positive_epochs": refit["actual_positive_epochs"],
                    "parameter_count": refit["config"]["parameter_count"],
                    "cv_training_seconds": cv_training, "cv_validation_seconds": cv_validation,
                    "cv_serialization_seconds": cv_serialization,
                    "refit_training_seconds": refit["training_seconds"],
                    "refit_validation_seconds": refit["validation_seconds"],
                    "refit_serialization_seconds": refit["serialization_seconds"],
                    "refit_wall_seconds": refit["wall_seconds"],
                    "generation_seconds": generation_seconds, "evaluation_seconds": evaluation_seconds,
                    "training_and_generation_seconds": cv_training+cv_validation+refit["training_seconds"]+generation_seconds,
                    "total_compute_seconds": cv_training+cv_validation+cv_serialization+refit["training_seconds"]+refit["validation_seconds"]+refit["serialization_seconds"]+generation_seconds+evaluation_seconds,
                    "timing_scope": "Completed CV training/validation, all-data refit, and generation; queue time and interrupted unfinished stages excluded; serialization and evaluation separate",
                    "device": args.device, "gpu_models": sorted({h["gpu"]["name"] for h in hardware if h.get("gpu")}),
                    "torch_version": environment["torch"], "python_version": environment["python"],
                    "environment_paths": environments,
                    "data_path": relative(data_path, task_dir), "data_sha256": sha256(data_path),
                    "model_path": relative(fit["model_path"], task_dir), "model_sha256": sha256(fit["model_path"]),
                    "selection_path": relative(fit["selection_path"], task_dir), "selection_sha256": sha256(fit["selection_path"]),
                    "generated_path": relative(generation["path"], task_dir), "generated_sha256": sha256(generation["path"]),
                    "reference_sha256": reference_record["reference_sha256"],
                    "training_sampler_stats": data_record["training_sampler_stats"],
                    "method_order": list(method_order)}
                write_json(task_dir / "methods" / method / "metrics.json", result)
                results[method] = result
                announce("method_complete", method=method,
                         **{key: result[key] for key in ("n", "seed", "tv1", "tv2", "llgap", "h_tv", "h_w1")})
            # Recheck frozen study sources and reference before declaring success.
            ensure_study(root, args.profile)
            validate_reference(root, manifest)
            result = {"status": "complete", "completed_utc": now(),
                "task_identity": task, "study_identity_sha256": manifest["identity_sha256"],
                "methods": results, "method_order": list(method_order),
                "environment_paths": sorted({p for row in results.values() for p in row["environment_paths"]}),
                "evaluation_reference_setup_seconds": reference_setup_seconds,
                "current_attempt_wall_seconds": time.perf_counter()-began,
                "artifact_sha256": artifact_map(task_dir)}
            write_json(task_dir / "result.json", result)
            validate_task_record(task_dir, manifest, task)
            attempt_record.update(status="complete", completed_utc=now(),
                                  current_attempt_seconds=time.perf_counter()-began)
            write_json(attempt_path, attempt_record)
            write_json(task_dir / "progress.json", attempt_record)
            announce("paired_replication_complete", n=task["n"], seed=task["seed"],
                     result=str(task_dir / "result.json"))
            return result
        except BaseException as error:
            attempt_record.update(status="failed", failed_utc=now(),
                current_attempt_seconds=time.perf_counter()-began,
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            write_json(attempt_path, attempt_record)
            write_json(task_dir / "progress.json", attempt_record)
            announce("paired_replication_failed", n=task["n"], seed=task["seed"],
                     error=str(error), details=str(attempt_path))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--profile", choices=("n1000", "full", "smoke"), default="n1000")
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SGE_TASK_ID", "1")))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("NSLOTS", "1")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    config, tasks = profile_config(args.profile), task_plan(args.profile)
    if args.task_id < 1 or args.task_id > len(tasks):
        parser.error(f"--task-id must be between 1 and {len(tasks)} for profile {args.profile}")
    root = resolve_root(args.root if args.root is not None else Path("outputs") / f"{args.profile}_v1")
    task = tasks[args.task_id-1]
    if args.dry_run:
        print(json.dumps({"root": str(root), "profile": args.profile, "task": task,
                          "paired_tasks": len(tasks), "methods": ["cindes", "dsm"],
                          "device": args.device, "threads": args.threads,
                          "config": config}, indent=2, allow_nan=False))
        return
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    manifest = ensure_study(root, args.profile)
    run(args, root, task, config, manifest)


if __name__ == "__main__":
    main()
