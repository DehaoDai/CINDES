"""Run one resumable paired replication from the frozen SCC task manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import study

sys.path.insert(0, str(study.PACKAGE_ROOT / "source"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def array_sha(values):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def verify_files(base, hashes):
    for name, expected in hashes.items():
        path = Path(base) / name
        require(path.is_file() and study.sha(path) == expected, f"Saved evidence changed or is missing: {path}")


def stage_hashes(directory):
    return {str(p.relative_to(directory)): study.sha(p)
            for p in sorted(directory.rglob("*")) if p.is_file()
            and not p.name.startswith(".") and ".tmp." not in p.name
            and p.name not in ("stage_complete.json", "model_record.json")}


def archive_incomplete(directory, run_path):
    if not directory.exists() or not any(directory.iterdir()):
        return
    metadata_path = directory / "metadata.json"
    if metadata_path.exists() and study.read_json(metadata_path).get("status") == "complete":
        return
    # Do not move an old fit while a surviving worker still holds its lock.
    with study.locked(directory / ".training.lock"):
        destination = run_path / "interrupted" / (str(directory.relative_to(run_path)).replace("/", "__") + "__" + uuid.uuid4().hex)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(directory, destination)
    with (run_path / "recovery_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": study.utc_now(), "action": "archived_incomplete_fit",
                                 "stage": str(directory.relative_to(run_path)),
                                 "saved_partial_artifacts": str(destination.relative_to(run_path))}) + "\n")


def cv_directory(run_path, method, lam, fold):
    return run_path / "cv" / method / f"lambda_{lam:g}" / f"fold_{fold}"


def validate_reference(root, manifest):
    path = root / "reference" / "reference_complete.json"
    require(path.is_file(), "Shared reference is not ready. Run prepare_reference.py first or wait for the reference job.")
    record = study.read_json(path)
    require(record["status"] == "complete" and record["study_id"] == manifest["study_id"]
            and record["profile"] == manifest["config"]["profile"], "Shared reference belongs to a different study/profile")
    verify_files(root, record["files_sha256"])
    return study.sha(path)


def prepare_data(run_path, manifest, spec):
    import numpy as np
    from compositional_dgp import BoundedCompositionalDGP
    completion = run_path / "data_complete.json"
    path = run_path / "observed_data.npz"
    identity = {"study_id": manifest["study_id"], **spec}
    if completion.exists():
        record = study.read_json(completion)
        require(record["identity"] == identity, "Observed data belong to a different task")
        verify_files(run_path, record["files_sha256"])
        return record
    if path.exists():
        archive = run_path / "interrupted" / f"uncommitted_observed_data_{uuid.uuid4().hex}.npz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, archive)
    dgp = BoundedCompositionalDGP(**manifest["config"]["dgp"])
    raw = dgp.sample(spec["n"], np.random.default_rng(spec["seed"]))
    pool = raw.astype(np.float32)
    order = np.random.default_rng(121000000+spec["seed"]).permutation(spec["n"])
    fold_id = np.empty(spec["n"], dtype=np.int64)
    for fold, ids in enumerate(np.array_split(order, 3)):
        fold_id[ids] = fold
    tmp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, pool=pool, pool_float64=raw, fold_id=fold_id)
    os.replace(tmp, path)
    record = {"status": "complete", "identity": identity, "sampler_stats": dgp.last_sample_stats,
              "dgp_config": dgp.config, "pool_sha256": array_sha(pool),
              "pool_float64_sha256": array_sha(raw), "fold_id_sha256": array_sha(fold_id),
              "fold_validation_sizes": [int((fold_id == f).sum()) for f in range(3)],
              "shape": list(pool.shape), "files_sha256": {path.name: study.sha(path)},
              "completed_at_utc": study.utc_now()}
    study.atomic_json(completion, record)
    return record


def cv_identity(manifest, spec, method, lam, fold, run_path):
    config = manifest["config"]
    return {"study_id": manifest["study_id"], "task": spec, "method": method,
            "lambda": lam, "fold": fold, "data_complete_sha256": study.sha(run_path / "data_complete.json"),
            "max_updates": study.horizons(spec["n"], config)["cv_updates"][fold],
            "validation_checks": config["validation_checks"]}


def verified_stage(directory, identity, torch_version=None):
    path = directory / "stage_complete.json"
    if not path.exists():
        return False
    record = study.read_json(path)
    require(record["status"] == "complete" and record["identity"] == identity, f"Stage identity mismatch: {directory}")
    if torch_version is not None:
        require(record["torch_version"] == torch_version, "Do not resume a training study with a different PyTorch build")
    verify_files(directory, record["files_sha256"])
    return True


def history(directory):
    return [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]


def expected_presentations(n, updates, batch):
    epochs, batches = divmod(updates, math.ceil(n/batch))
    return epochs*n + min(n, batches*batch)


def check_cv(directory, metadata, config, spec, method, fold):
    import torch
    horizon = study.horizons(spec["n"], config)
    steps = horizon["cv_updates"][fold]
    val_size = horizon["validation_sizes"][fold]
    require(metadata["status"] == "complete" and metadata["actual_updates"] == steps, "CV horizon incomplete")
    require(metadata["config"]["n_train"] == spec["n"]-val_size and metadata["config"]["n_val"] == val_size,
            "CV observation counts differ")
    positives = config["max_positive_epochs"]*(spec["n"]-val_size)
    require(metadata["positive_occurrences"] == positives, "CV positive-data exposure differs")
    require(metadata["negative_occurrences"] == (config["fake_ratio"]*positives if method == "cindes" else 0),
            "CV negative-data exposure differs")
    schedule = [math.ceil(j*steps/config["validation_checks"]) for j in range(config["validation_checks"]+1)]
    rows = history(directory)
    require([r["check"] for r in rows] == list(range(config["validation_checks"]+1)), "CV checks missing")
    require([r["update"] for r in rows] == schedule, "CV update schedule differs")
    require(all(math.isfinite(r["validation_loss"]) for r in rows), "Nonfinite CV loss")
    for row in rows:
        expected = expected_presentations(spec["n"]-val_size, row["update"], config["positive_batch"])
        require(row["positive_occurrences"] == expected, "CV checkpoint exposure differs")
    for filename in ("best.pt", "last.pt"):
        payload = torch.load(directory / filename, map_location="cpu", weights_only=True)
        require(payload["config"] == metadata["config"], "CV checkpoint/config mismatch")
        require(payload["update"] in schedule, "CV checkpoint has an unplanned update")
    if method == "cindes" and config["retain_cindes_cv_states"]:
        trajectory = torch.load(directory / "trajectory.pt", map_location="cpu", weights_only=True)
        require(sorted(trajectory["states"]) == schedule, "CINDES CV trajectory missing states")


def cv_worker(root, manifest, spec, method, lam, fold):
    import numpy as np
    import torch
    import balanced_training
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = manifest["config"]
    require(method in study.METHODS and lam in config["lambda_grid"] and fold in range(3), "Unplanned CV worker")
    run_path = study.run_dir(root, spec)
    directory = cv_directory(run_path, method, lam, fold)
    identity = cv_identity(manifest, spec, method, lam, fold, run_path)
    if verified_stage(directory, identity, str(torch.__version__)):
        return
    archive_incomplete(directory, run_path)
    with np.load(run_path / "observed_data.npz", allow_pickle=False) as data:
        pool, folds = data["pool"], data["fold_id"]
    _, metadata = balanced_training.train_cv(
        pool[folds != fold], pool[folds == fold], method=method,
        l1_lambda=lam, seed=100*spec["seed"]+fold, output_dir=directory,
        max_updates=identity["max_updates"], validation_checks=config["validation_checks"],
        batch_size=config["positive_batch"], normalizer_size=config["normalizer_size"],
        retain_validation_states=method == "cindes" and config["retain_cindes_cv_states"],
        device="cpu", verbose=False)
    check_cv(directory, metadata, config, spec, method, fold)
    study.atomic_json(directory / "stage_complete.json", {
        "status": "complete", "identity": identity, "torch_version": str(torch.__version__),
        "files_sha256": stage_hashes(directory), "completed_at_utc": study.utc_now()})
    print(json.dumps({"stage": "cv", "method": method, "lambda": lam, "fold": fold, "status": "complete"}), flush=True)


def select_models(run_path, manifest, spec):
    config = manifest["config"]
    destination = run_path / "selection.json"
    if destination.exists():
        result = study.read_json(destination)
        require(result["study_id"] == manifest["study_id"] and result["task"] == spec, "Selection task mismatch")
        verify_files(run_path, result["inputs_sha256"])
        return result
    sizes = study.horizons(spec["n"], config)["validation_sizes"]
    selected, curves, hashes = {}, {}, {}
    groups = {}
    for method in study.METHODS:
        candidates, curves[method] = [], {}
        for lam in config["lambda_grid"]:
            folds = []
            for fold in range(3):
                directory = cv_directory(run_path, method, lam, fold)
                identity = cv_identity(manifest, spec, method, lam, fold, run_path)
                require(verified_stage(directory, identity), "CV stage is not complete")
                metadata = study.read_json(directory / "metadata.json")
                groups.setdefault(fold, set()).add(metadata["batch_schedule_sha256"])
                rows = history(directory)
                require(len(rows) == config["validation_checks"]+1, "Incomplete CV history")
                folds.append(rows)
                for name in ("stage_complete.json", "metadata.json", "history.jsonl"):
                    path = directory / name
                    hashes[str(path.relative_to(run_path))] = study.sha(path)
            curve = []
            for j in range(config["validation_checks"]+1):
                losses = [folds[f][j]["validation_loss"] for f in range(3)]
                value = sum(sizes[f]*losses[f] for f in range(3))/spec["n"]
                curve.append({"check": j, "weighted_cv_loss": value, "fold_losses": losses})
            curves[method][f"{lam:g}"] = curve
            best = min(curve, key=lambda row: (row["weighted_cv_loss"], row["check"]))
            candidates.append({"lambda": lam, **best})
        ranking = sorted(candidates, key=lambda row: (row["weighted_cv_loss"], row["lambda"], row["check"]))
        chosen = ranking[0]
        full_horizon = study.horizons(spec["n"], config)["full_updates"]
        selected[method] = {**chosen,
                            "full_stop_updates": math.ceil(chosen["check"]*full_horizon/config["validation_checks"]),
                            "full_cosine_horizon": full_horizon,
                            "candidate_ranking": ranking}
    require(all(len(values) == 1 for values in groups.values()), "Positive minibatches differ by method/penalty")
    study.atomic_json(run_path / "cv_curves.json", curves)
    hashes["cv_curves.json"] = study.sha(run_path / "cv_curves.json")
    result = {"status": "frozen_before_refit_and_generation", "study_id": manifest["study_id"],
              "task": spec, "selected": selected, "inputs_sha256": hashes,
              "known_target_information_used": False, "generation_metrics_used": False,
              "frozen_at_utc": study.utc_now()}
    study.atomic_json(destination, result)
    return result


def refit_identity(manifest, spec, method, run_path):
    return {"study_id": manifest["study_id"], "task": spec, "method": method,
            "selection_sha256": study.sha(run_path / "selection.json"),
            "data_complete_sha256": study.sha(run_path / "data_complete.json")}


def refit_worker(root, manifest, spec, method):
    import numpy as np
    import torch
    import balanced_training
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(method in study.METHODS, "Unplanned refit method")
    config, run_path = manifest["config"], study.run_dir(root, spec)
    selection = select_models(run_path, manifest, spec)["selected"][method]
    directory = run_path / "refits" / method
    identity = refit_identity(manifest, spec, method, run_path)
    if verified_stage(directory, identity, str(torch.__version__)):
        require((directory / "model_record.json").exists(), "Completed refit is missing its model record")
        return
    archive_incomplete(directory, run_path)
    with np.load(run_path / "observed_data.npz", allow_pickle=False) as data:
        pool = data["pool"]
    _, metadata = balanced_training.refit_fixed(
        pool, method=method, l1_lambda=selection["lambda"], seed=spec["seed"],
        output_dir=directory, stop_updates=selection["full_stop_updates"],
        scheduler_updates=selection["full_cosine_horizon"], batch_size=config["positive_batch"],
        device="cpu", verbose=False)
    require(metadata["status"] == "complete" and metadata["actual_updates"] == selection["full_stop_updates"],
            "Refit duration differs from frozen selection")
    positives = expected_presentations(spec["n"], selection["full_stop_updates"], config["positive_batch"])
    require(metadata["positive_occurrences"] == positives, "Refit real-data exposure differs")
    require(metadata["negative_occurrences"] == (config["fake_ratio"]*positives if method == "cindes" else 0),
            "Refit negative-data exposure differs")
    checkpoint = directory / "model.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(payload["config"] == metadata["config"] and payload["update"] == selection["full_stop_updates"],
            "Refit checkpoint/config mismatch")
    require(metadata["checkpoint_sha256"] == study.sha(checkpoint), "Refit checkpoint digest differs")
    cv_seconds = sum(study.read_json(cv_directory(run_path, method, lam, f) / "metadata.json")["wall_seconds"]
                     for lam in config["lambda_grid"] for f in range(3))
    record = {"method": method, "checkpoint": str(checkpoint.relative_to(run_path)),
              "checkpoint_sha256": study.sha(checkpoint), "selected_lambda": selection["lambda"],
              "selected_check": selection["check"], "updates": selection["full_stop_updates"],
              "full_cosine_horizon": selection["full_cosine_horizon"],
              "training_metadata": str((directory / "metadata.json").relative_to(run_path)),
              "training_metadata_sha256": study.sha(directory / "metadata.json"),
              "training_seconds": metadata["wall_seconds"], "cv_sum_fit_wall_seconds": cv_seconds,
              "n_params": metadata["config"]["total_parameter_count"],
              "positive_occurrences": metadata["positive_occurrences"],
              "negative_occurrences": metadata["negative_occurrences"],
              "selection_sha256": study.sha(run_path / "selection.json"),
              "training_data_sha256": array_sha(pool), "model_config": payload["model_config"]}
    study.atomic_json(directory / "model_record.json", record)
    hashes = stage_hashes(directory)
    hashes["model_record.json"] = study.sha(directory / "model_record.json")
    study.atomic_json(directory / "stage_complete.json", {
        "status": "complete", "identity": identity, "torch_version": str(torch.__version__),
        "files_sha256": hashes, "completed_at_utc": study.utc_now()})
    print(json.dumps({"stage": "refit", "method": method, "updates": record["updates"], "status": "complete"}), flush=True)


def launch_workers(kind, root, manifest, spec, workers, environment, status_callback):
    config, run_path = manifest["config"], study.run_dir(root, spec)
    if kind == "cv":
        tasks = [(m, lam, fold) for lam in config["lambda_grid"] for fold in range(3) for m in study.METHODS]
    else:
        tasks = [(m,) for m in study.METHODS]
    (run_path / "logs").mkdir(exist_ok=True)
    child_env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                     NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")

    def one(task):
        method = task[0]
        if kind == "cv":
            _, lam, fold = task
            directory = cv_directory(run_path, method, lam, fold)
            identity = cv_identity(manifest, spec, method, lam, fold, run_path)
            extra = ["--lambda-value", str(lam), "--fold", str(fold)]
            label = f"cv_{method}_lambda_{lam:g}_fold_{fold}"
        else:
            directory = run_path / "refits" / method
            identity = refit_identity(manifest, spec, method, run_path)
            extra = []
            label = f"refit_{method}"
        if verified_stage(directory, identity, environment["torch"]):
            return label, True
        command = [sys.executable, str(Path(__file__)), "--worker", kind, "--method", method,
                   "--root", str(root), "--profile", config["profile"], "--task-id", str(spec["task_id"]), *extra]
        with (run_path / "logs" / f"{label}.log").open("a", buffering=1, encoding="utf-8") as handle:
            handle.write(json.dumps({"started_at_utc": study.utc_now(), "command": command}) + "\n")
            completed = subprocess.run(command, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{label} failed with exit {completed.returncode}; see {run_path/'logs'/f'{label}.log'}")
        return label, False

    started = time.perf_counter()
    reused = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        futures = [executor.submit(one, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            label, previous = future.result()
            reused += int(previous)
            status_callback(kind, {"completed": count, "total": len(tasks), "last": label,
                                   "reused_stages": reused, "elapsed_seconds": time.perf_counter()-started})
            print(json.dumps({"stage": kind, "completed": count, "total": len(tasks), "last": label,
                              "reused": previous}), flush=True)
    return {"elapsed_seconds_this_attempt": time.perf_counter()-started,
            "stage_count": len(tasks), "reused_stages": reused, "workers": min(workers, len(tasks))}


def complete_file_hashes(run_path):
    ignored_top = {"logs", "interrupted"}
    ignored_names = {"run_complete.json", "status.json", "generation_progress.json", "run_error.json"}
    return {str(p.relative_to(run_path)): study.sha(p) for p in sorted(run_path.rglob("*"))
            if p.is_file() and p.relative_to(run_path).parts[0] not in ignored_top
            and p.name not in ignored_names and not p.name.startswith(".")
            and ".tmp." not in p.name and p.suffix != ".lock"}


def main_task(root, manifest, spec, workers, device_name):
    reference_hash = validate_reference(root, manifest)
    run_path = study.run_dir(root, spec)
    run_path.mkdir(parents=True, exist_ok=True)
    with study.locked(run_path / ".run.lock"):
        completion_path = run_path / "run_complete.json"
        if completion_path.exists():
            completed = study.read_json(completion_path)
            require(completed["status"] == "complete" and completed["study_id"] == manifest["study_id"]
                    and all(completed[key] == spec[key] for key in spec), "Completed run identity differs")
            require(completed["reference_complete_sha256"] == reference_hash, "Reference changed")
            verify_files(run_path, completed["files_sha256"])
            print(json.dumps({"status": "verified_existing_complete_run", **spec}), flush=True)
            return
        import generation
        device, environment = generation.configure_device(device_name, manifest["config"])
        environment["cv_workers"] = workers
        environment["cpu_affinity_count"] = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        attempt = uuid.uuid4().hex
        study.atomic_json(run_path / "attempts" / f"environment_{attempt}.json", environment)
        started = time.perf_counter()

        def status_callback(stage, details):
            study.atomic_json(run_path / "status.json", {
                "status": "running", "study_id": manifest["study_id"], **spec,
                "stage": stage, "attempt": attempt, "updated_at_utc": study.utc_now(), **details})

        try:
            status_callback("observed_data", {})
            data_record = prepare_data(run_path, manifest, spec)
            cv_timing = launch_workers("cv", root, manifest, spec, workers, environment, status_callback)
            selection = select_models(run_path, manifest, spec)
            status_callback("selection_frozen", {"selected": selection["selected"]})
            refit_timing = launch_workers("refit", root, manifest, spec, workers, environment, status_callback)
            models = {m: study.read_json(run_path / "refits" / m / "model_record.json") for m in study.METHODS}
            study.atomic_json(run_path / "models.json", models)
            generation_records = generation.generate_pair(
                run_path, manifest, spec, models, device, environment, status_callback)
            status_callback("evaluation", {})
            from evaluate import evaluate_run
            result = evaluate_run(root, run_path, manifest, spec, models, generation_records, environment)
            require(result["study_id"] == manifest["study_id"] and set(result["methods"]) == set(study.METHODS),
                    "Evaluation returned an incompatible record")
            result.update({"status": "complete", **spec, "profile": manifest["config"]["profile"],
                           "reference_complete_sha256": reference_hash,
                           "data_record": data_record, "selection": selection["selected"],
                           "timing": {"cv": cv_timing, "refit": refit_timing,
                                      "task_elapsed_seconds_this_attempt": time.perf_counter()-started},
                           "completed_at_utc": study.utc_now(), "final_attempt": attempt})
            # The result and its referenced files are immutable after this marker.
            result["files_sha256"] = complete_file_hashes(run_path)
            study.atomic_json(completion_path, result)
            study.atomic_json(run_path / "status.json", {
                "status": "complete", "study_id": manifest["study_id"], **spec,
                "completed_at_utc": result["completed_at_utc"]})
            print(json.dumps({"status": "ALL_DONE", **spec, "run_complete": str(completion_path)}), flush=True)
        except BaseException as exc:
            error = {"status": "failed", "study_id": manifest["study_id"], **spec, "attempt": attempt,
                     "error_type": type(exc).__name__, "message": str(exc),
                     "traceback": traceback.format_exc(), "recorded_at_utc": study.utc_now()}
            study.atomic_json(run_path / "attempts" / f"error_{attempt}.json", error)
            study.atomic_json(run_path / "status.json", error)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=study.DEFAULT_ROOT)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--cv-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=("cv", "refit"), help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=study.METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-value", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--fold", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = study.make_config(args.profile)
    task_id = args.task_id
    if task_id is None and os.environ.get("SGE_TASK_ID", "").isdigit():
        task_id = int(os.environ["SGE_TASK_ID"])
    if args.dry_run:
        specs = study.task_specs(config) if task_id is None else [study.task_spec(task_id, config)]
        print(json.dumps({
            "profile": args.profile, "tasks": specs,
            "total_paired_replications": len(study.task_specs(config)),
            "total_final_method_fits": 2*len(study.task_specs(config)),
            "CV_fits_per_replication": 2*3*len(config["lambda_grid"]),
            "n_generated_per_method_replication": config["n_gen"],
            "horizons_by_n": {str(n): study.horizons(n, config) for n in config["n_values"]},
            "output_root": str(args.root.resolve()), "config": config}, indent=2))
        return
    if task_id is None:
        parser.error("--task-id or a numeric SGE_TASK_ID is required")
    if args.cv_workers < 1:
        parser.error("--cv-workers must be positive")
    if not args.worker and os.environ.get("NSLOTS", "").isdigit() and args.cv_workers > int(os.environ["NSLOTS"]):
        parser.error("--cv-workers exceeds the allocated NSLOTS")
    root = args.root.resolve()
    manifest = study.load_manifest(root, args.profile)
    spec = study.task_spec(task_id, manifest["config"])
    if args.worker:
        require(args.method is not None, "Worker method is required")
        if args.worker == "cv":
            cv_worker(root, manifest, spec, args.method, args.lambda_value, args.fold)
        else:
            refit_worker(root, manifest, spec, args.method)
    else:
        main_task(root, manifest, spec, args.cv_workers, args.device)


if __name__ == "__main__":
    main()
