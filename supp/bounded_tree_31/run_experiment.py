#!/usr/bin/env python3
"""Run the unchanged bounded-tree driver, with isolated outputs and metadata."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from run_original import cpu_thread_budget, set_thread_environment
from study_config import (METHODS, default_root, driver_settings,
                          expected_experiment_config, expected_pairs,
                          profile_settings)

PACKAGE = Path(__file__).resolve().parent
ORIGINAL = PACKAGE / "experiment"


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def atomic_json(path, value):
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def source_hashes():
    manifest = ORIGINAL / "original_python_sha256.json"
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    required = {"cindes_vs_dsm_tree.py", "tree_mrf.py", "cindes_vs_dsm_mrf.py"}
    if set(expected) != required:
        raise ValueError(f"Unexpected source manifest entries in {manifest}")
    actual = {name: hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest()
              for name in sorted(required)}
    if actual != expected:
        raise ValueError("Original scientific source hashes differ from the packaged manifest. "
                         "Use an unmodified package for this reproduction study.")
    return actual


@contextmanager
def exclusive_lock(path):
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("Task locking requires Linux or macOS; use Linux/WSL on Windows.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError(f"Could not lock {path}; another task may be active.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_manifest(root, profile, hashes):
    settings = profile_settings(profile)
    expected = {
        "schema": "bounded_reproduction_wrapper_v1",
        "profile": profile,
        "n_values": settings["n_values"], "seeds": settings["seeds"],
        "expected_paired_tasks": settings["expected_paired_tasks"],
        "experiment_config": expected_experiment_config(profile),
        "original_source_sha256": hashes,
        "protocol_notes": {
            "final_support_clipping": True,
            "noise_on_last_reverse_step": False,
            "extra_cindes_terminal_step": False,
            "grid_sampler": "midpoint_grid_with_within_cell_uniform_jitter",
            "reported_time": "training_plus_generation_excluding_metrics_and_data",
            "source_identity": "packaged_recovered_files_not_a_historical_run_hash_attestation",
            "execution_bridge": "run_original.py caps PyTorch threads after original helper import",
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    # Different array tasks can start together. The manifest is created under a
    # short blocking lock, distinct from the nonblocking per-replication lock.
    import fcntl
    with (root / ".manifest.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "study_manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != expected:
                raise ValueError(f"Study settings or source hashes differ in {path}. Use a new --root.")
        else:
            atomic_json(path, expected)


def checked_rows(path, profile, n, seed):
    # Execution and aggregation use exactly the same acceptance rules.
    from summarize_results import ValidationError, check_row, read_json
    if not path.exists():
        return {}
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = read_json(line)
            _, _, method = check_row(row, expected_experiment_config(profile), {(n, seed)})
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Invalid result in {path}:{line_number}: {exc}. "
                             "Preserve the file for inspection.") from exc
        if method in rows:
            raise ValueError(f"Duplicate {method} result in {path}:{line_number}")
        rows[method] = row
    return rows


def driver_command(profile, n, seed, out):
    s = driver_settings(profile)
    command = [sys.executable, "-u", str(PACKAGE / "run_original.py"),
               "--single-run", "--n", str(n), "--seed", str(seed)]
    flags = {"d": "--d", "R": "--R", "K": "--K", "M": "--M",
             "n_gen": "--n-gen", "cindes_steps": "--cindes-steps",
             "dsm_steps": "--dsm-steps", "coupling": "--coupling",
             "unary": "--unary", "generation_batch": "--generation-batch"}
    for key, flag in flags.items():
        command.extend((flag, str(s[key])))
    command.extend(("--out", str(out)))
    return command


def _collect_environment_metadata(require_cuda):
    """Run only in the short-lived metadata worker, never in the controller."""
    import numpy as np
    import torch

    torch.set_num_threads(cpu_thread_budget())
    has_cuda = torch.cuda.is_available()
    if require_cuda and not has_cuda:
        raise RuntimeError("A working CUDA GPU was requested but is unavailable. Run full fits inside a GPU allocation.")
    gpu = None
    if has_cuda:
        x = torch.ones((16, 16), device="cuda")
        y = torch.special.ndtri(torch.special.ndtr((x @ x) / 32))
        torch.cuda.synchronize()
        if not bool(torch.isfinite(y).all()):
            raise RuntimeError("CUDA preflight returned non-finite values.")
        gpu = {"name": torch.cuda.get_device_name(0),
               "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
               "compute_capability": list(torch.cuda.get_device_capability(0))}
    return {
        "_metadata_process_id": os.getpid(),
        "python": sys.version, "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": "cuda" if has_cuda else "cpu", "gpu": gpu,
        "host": platform.node(), "platform": platform.platform(),
        "torch_cpu_threads": torch.get_num_threads(),
        "environment": {key: os.environ.get(key) for key in (
            "JOB_ID", "SGE_TASK_ID", "NSLOTS", "CUDA_VISIBLE_DEVICES",
            "CINDES_CPU_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "LOADEDMODULES", "CONDA_DEFAULT_ENV")},
    }


def environment_metadata(require_cuda):
    """Release the probe's CUDA context before starting the training process.

    SCC GPUs default to exclusive-process mode. Allocating a CUDA tensor in
    this controller would keep its context alive while the trainer starts.
    A synchronous worker exits and releases that context before we return.
    """
    probe = (
        "import json, sys\n"
        "from run_experiment import _collect_environment_metadata\n"
        "print(json.dumps(_collect_environment_metadata(sys.argv[1] == '1')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-u", "-c", probe, "1" if require_cuda else "0"],
        cwd=PACKAGE, env=os.environ.copy(), text=True, capture_output=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"Environment preflight process failed with status {result.returncode}:\n{detail}"
        )
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="", flush=True)
    try:
        metadata = json.loads(result.stdout)
        probe_pid = metadata.pop("_metadata_process_id")
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("Environment preflight returned invalid metadata.") from exc
    metadata["environment_probe"] = {
        "mode": "subprocess_exited_before_training",
        "controller_pid": os.getpid(), "probe_pid": probe_pid,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return metadata


def run_pair(root, profile, n, seed, hashes, require_cuda):
    directory = root / "runs" / f"n_{n:04d}" / f"seed_{seed:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "results.jsonl"
    with exclusive_lock(directory / ".task.lock"):
        before = checked_rows(out, profile, n, seed)
        if set(before) == set(METHODS):
            print(f"Already complete: n={n}, seed={seed}. Validated both saved results.", flush=True)
            return
        set_thread_environment(cpu_thread_budget())
        metadata = environment_metadata(require_cuda)
        run_id = timestamp() + f"_pid{os.getpid()}"
        command = driver_command(profile, n, seed, out)
        metadata.update({"run_id": run_id, "profile": profile, "n": n,
                         "seed": seed, "original_source_sha256": hashes,
                         "experiment_config": expected_experiment_config(profile),
                         "completed_methods_before": sorted(before),
                         "command": command})
        atomic_json(directory / f"environment_{run_id}.json", metadata)
        print(f"Running paired replication n={n}, seed={seed}, device={metadata['device']}.", flush=True)
        start = time.monotonic()
        with (directory / "task.log").open("a", encoding="utf-8") as log:
            log.write("\n" + json.dumps({"started_utc": run_id, "command": command}) + "\n")
            log.flush()
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            with subprocess.Popen(command, cwd=PACKAGE, env=child_env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                try:
                    for line in process.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
        elapsed = time.monotonic() - start
        if code:
            atomic_json(directory / f"failed_{run_id}.json",
                        {"returncode": code, "wrapper_elapsed_seconds": elapsed})
            raise RuntimeError(f"Original driver exited with status {code}; inspect {directory / 'task.log'}.")
        after = checked_rows(out, profile, n, seed)
        if set(after) != set(METHODS):
            raise RuntimeError(f"Driver returned without both results in {out}.")
        atomic_json(directory / "run_complete.json", {
            "profile": profile, "n": n, "seed": seed, "methods": list(METHODS),
            "completed_utc": timestamp(), "final_attempt_run_id": run_id,
            "wrapper_elapsed_seconds_final_attempt": elapsed,
            "resumed": bool(before),
            "result_file": "results.jsonl",
            "saved_sample_files": [p.name for p in sorted(directory.glob("results_qual_*.npz"))],
            "original_source_sha256": hashes,
        })
        print(f"Complete: n={n}, seed={seed}; results in {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--root", type=Path, help="Output directory; profile-specific default.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Run all profile tasks sequentially.")
    selection.add_argument("--task-id", type=int, help="One-based task number in sample-size/seed order.")
    selection.add_argument("--n", type=int, help="One sample size; also supply --seed.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without importing NumPy/PyTorch.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail before fitting if CUDA is unavailable.")
    args = parser.parse_args()
    if (args.n is None) != (args.seed is None):
        parser.error("Use --n and --seed together.")
    pairs = expected_pairs(args.profile)
    if args.task_id is not None:
        if not 1 <= args.task_id <= len(pairs):
            parser.error(f"--task-id must be between 1 and {len(pairs)} for {args.profile}.")
        selected = [pairs[args.task_id - 1]]
    elif args.n is not None:
        if (args.n, args.seed) not in pairs:
            parser.error("The requested (n, seed) is outside the fixed profile.")
        selected = [(args.n, args.seed)]
    elif args.all or args.dry_run:
        selected = pairs
    else:
        parser.error("Choose --all, --task-id, --n with --seed, or --dry-run.")
    root = (args.root or Path(default_root(args.profile))).expanduser().resolve()
    hashes = source_hashes()
    plan = profile_settings(args.profile)
    plan.update({"output_root": str(root), "selected_paired_tasks": len(selected),
                 "tasks": [{"task_id": pairs.index(pair) + 1, "n": pair[0], "seed": pair[1]}
                           for pair in selected],
                 "require_cuda": args.require_cuda, "original_source_sha256": hashes})
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    ensure_manifest(root, args.profile, hashes)
    for n, seed in selected:
        run_pair(root, args.profile, n, seed, hashes, args.require_cuda)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
