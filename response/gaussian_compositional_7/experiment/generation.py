"""Resumable paired Gaussian-reference sampling, without target information."""
import hashlib
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import study

sys.path.insert(0, str(study.PACKAGE_ROOT / "source"))


def configure_device(requested, config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import scipy
    import torch
    import subgaussian_methods  # Apply its legacy thread initialization before our explicit setting.
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if config["profile"] == "full" and requested != "cuda":
        raise ValueError("The full SCC protocol requires CUDA; CPU is only allowed for smoke tests")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No usable CUDA GPU. Submit the GPU array through qsub; no CPU fallback is used.")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(requested)
    x = torch.ones((16, 16), dtype=torch.float32, device=device)
    y = torch.nn.functional.softplus(x @ x, beta=5)
    if requested == "cuda":
        torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all()):
        raise RuntimeError("Device preflight produced nonfinite values")
    environment = {
        "recorded_at_utc": study.utc_now(), "host": platform.node(),
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "generation_device": str(device), "torch_cpu_threads": torch.get_num_threads(),
        "cpu_slots": int(os.environ.get("NSLOTS", "1")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "job_id": os.environ.get("JOB_ID"), "sge_task_id": os.environ.get("SGE_TASK_ID"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": False, "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if requested == "cuda":
        props = torch.cuda.get_device_properties(0)
        environment.update(gpu=props.name, gpu_memory_bytes=props.total_memory,
                           gpu_compute_capability=list(torch.cuda.get_device_capability(0)))
    return device, environment


def runtime_signature(environment, config):
    return {key: environment.get(key) for key in
            ("torch", "cuda_runtime", "generation_device", "gpu", "gpu_compute_capability",
             "allow_tf32", "deterministic_algorithms", "float32_matmul_precision",
             "cublas_workspace_config")} | {
                "generation_batch": config["generation_batch"], "dtype": "float32"}


_GENERATION_ARTIFACTS = (
    "shards", "samples", "sampler_verification.json", "generation_complete.json",
    "generation_progress.json", "generation_runtime.json",
)


def _finish_generation_archive(run_path, journal, expected_inputs):
    """Finish a previously checked move even if the first attempt was interrupted."""
    if journal.get("inputs") != expected_inputs:
        raise RuntimeError("Generation restart journal belongs to different inputs")
    relative = Path(journal["archive_directory"])
    if (len(relative.parts) != 2 or relative.parts[0] != "interrupted"
            or not relative.parts[1].startswith("generation_runtime_")):
        raise RuntimeError("Invalid generation archive location")
    directory = run_path / relative
    directory.mkdir(parents=True, exist_ok=True)
    for name in journal["artifacts"]:
        if name not in _GENERATION_ARTIFACTS:
            raise RuntimeError("Unexpected generation artifact in restart journal")
        source, destination = run_path / name, directory / name
        if source.exists():
            if destination.exists():
                raise RuntimeError("Generation restart archive has conflicting artifacts")
            os.replace(source, destination)
        elif not destination.exists():
            raise RuntimeError("Generation restart artifact disappeared")
    for name, expected_hash in journal["files_sha256"].items():
        path = directory / name
        if not path.is_file() or study.sha(path) != expected_hash:
            raise RuntimeError("Archived generation evidence changed")
    # The moved journal is itself the durable explanation of the archived run.
    os.replace(run_path / "generation_restart_pending.json", directory / "restart_reason.json")
    with (run_path / "recovery_events.jsonl").open("a", encoding="utf-8") as handle:
        import json
        handle.write(json.dumps({"time": study.utc_now(), "action": "archived_generation_runtime_change",
                                 "saved_partial_artifacts": str(relative),
                                 "old_runtime": journal["old_runtime"],
                                 "requested_runtime": journal["requested_runtime"]}) + "\n")


def reconcile_generation_runtime(run_path, manifest, spec, model_records, signature):
    """Keep a pair on one numerical runtime while preserving earlier GPU work."""
    if (run_path / "run_complete.json").exists():
        raise RuntimeError("A completed paired run is immutable")
    config = manifest["config"]
    inputs = {"study_id": manifest["study_id"], "task": spec,
              "checkpoints": {m: r["checkpoint_sha256"] for m, r in model_records.items()}}
    pending = run_path / "generation_restart_pending.json"
    if pending.exists():
        _finish_generation_archive(run_path, study.read_json(pending), inputs)
    signatures = []
    marker = run_path / "generation_runtime.json"
    if marker.exists():
        record = study.read_json(marker)
        if record.get("inputs") != inputs:
            raise RuntimeError("Generation input identity changed")
        signatures.append(record["runtime_signature"])
    verification_path = run_path / "sampler_verification.json"
    if verification_path.exists():
        record = study.read_json(verification_path)
        identity = record.get("identity", {})
        if (record.get("status") != "passed" or identity.get("study_id") != inputs["study_id"]
                or identity.get("checkpoints") != inputs["checkpoints"]):
            raise RuntimeError("Sampler verification input identity changed")
        signatures.append(identity["signature"])
    n_shards = math.ceil(config["n_gen"]/config["shard_size"])
    for method in study.METHODS:
        for metadata_path in sorted((run_path / "shards" / method).glob("shard_*.json")):
            row = study.read_json(metadata_path)
            identity = dict(row.get("identity", {}))
            old_signature = identity.pop("runtime_signature", None)
            shard = identity.get("shard")
            if not isinstance(shard, int) or not 0 <= shard < n_shards:
                raise RuntimeError("Saved shard index is outside this experiment")
            expected = {"study_id": manifest["study_id"], "task_id": spec["task_id"],
                        "method": method, "shard": shard,
                        "n_samples": min(config["shard_size"], config["n_gen"]-shard*config["shard_size"]),
                        "checkpoint_sha256": model_records[method]["checkpoint_sha256"]}
            if row.get("status") != "complete" or identity != expected or old_signature is None:
                raise RuntimeError("Saved generation shard input identity changed")
            if metadata_path.name != f"shard_{shard:02d}.json":
                raise RuntimeError("Saved generation shard filename changed")
            array_path = metadata_path.with_suffix(".npy")
            if not array_path.is_file() or study.sha(array_path) != row["sample_sha256"]:
                raise RuntimeError("Saved generation shard is missing or changed")
            signatures.append(old_signature)
    completion = run_path / "generation_complete.json"
    if completion.exists():
        record = study.read_json(completion)
        if (record.get("status") != "complete" or record.get("study_id") != manifest["study_id"]
                or record.get("task_id") != spec["task_id"] or set(record["methods"]) != set(study.METHODS)):
            raise RuntimeError("Generation completion input identity changed")
        if not verification_path.is_file() or study.sha(verification_path) != record["sampler_verification_sha256"]:
            raise RuntimeError("Generation sampler verification changed")
        for method, value in record["methods"].items():
            if value["samples"] != f"samples/{method}.npy":
                raise RuntimeError("Combined generation sample path changed")
            path = run_path / value["samples"]
            if not path.is_file() or study.sha(path) != value["samples_sha256"]:
                raise RuntimeError("Combined generation samples changed")
            signatures.append(value["runtime_signature"])
    if signatures and any(item != signatures[0] for item in signatures):
        raise RuntimeError("Saved generation already mixes numerical runtimes")
    if signatures and signatures[0] != signature:
        present = [name for name in _GENERATION_ARTIFACTS if (run_path / name).exists()]
        hashes = {}
        for name in present:
            path = run_path / name
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            hashes.update({str(p.relative_to(run_path)): study.sha(p) for p in paths if p.is_file()})
        journal = {"inputs": inputs, "old_runtime": signatures[0], "requested_runtime": signature,
                   "reason": "Regenerate both methods together after a numerical-runtime or GPU change; reuse frozen fits.",
                   "archive_directory": "interrupted/generation_runtime_"+uuid.uuid4().hex,
                   "artifacts": present, "files_sha256": hashes, "recorded_at_utc": study.utc_now()}
        study.atomic_json(pending, journal)
        _finish_generation_archive(run_path, journal, inputs)
        print("ARCHIVED_GENERATION_RUNTIME_CHANGE; reusing fitted models and restarting both samplers", flush=True)
    if not marker.exists():
        study.atomic_json(marker, {"inputs": inputs, "runtime_signature": signature,
                                   "recorded_at_utc": study.utc_now()})


def make_score(model, method, device, particle_seed, K, R):
    import torch
    from subgaussian_methods import cindes_score
    if method == "cindes":
        particles = torch.Generator(device=device).manual_seed(particle_seed)
        return lambda y, t: cindes_score(model, R, y, t, K, particles), particles
    if method != "dsm":
        raise ValueError("Unknown method")
    return lambda y, t: model(y, torch.full((len(y), 1), t, device=y.device, dtype=y.dtype)), None


def tracked_backward_sample(score_fn, d, n_samples, *, T, delta, M, seed,
                            batch, device, particle_generator=None, progress=None):
    """The unchanged original update, with explicit device and RNG hashes."""
    import torch
    from subgaussian_methods import make_timesteps
    if n_samples < 1 or batch < 1:
        raise ValueError("n_samples and batch must be positive")
    reverse = torch.Generator(device=device).manual_seed(seed + 999)
    times = make_timesteps(T, delta, M)
    pieces, hashes = [], []
    with torch.no_grad():
        for start in range(0, n_samples, batch):
            size = min(batch, n_samples-start)
            y = torch.randn((size, d), generator=reverse, device=device)
            initial_hash = hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest()
            for j in range(M):
                tau, tau_next = T-times[j], T-times[j+1]
                score = score_fn(y, tau)
                h = tau-tau_next
                decay = math.exp(-h)
                one_minus_decay = -math.expm1(-h)
                mean = decay*y + 2.0*one_minus_decay*(score+y)
                variance = -math.expm1(-2.0*h)
                noise = torch.randn(y.shape, generator=reverse, device=device)
                y = mean + math.sqrt(variance)*noise
            if not bool(torch.isfinite(y).all()):
                raise FloatingPointError("Nonfinite generated observations")
            pieces.append(y.cpu())
            hashes.append({
                "initial_gaussian_sha256": initial_hash,
                "ending_reverse_rng_sha256": hashlib.sha256(reverse.get_state().cpu().numpy().tobytes()).hexdigest(),
                "ending_particle_rng_sha256": (
                    hashlib.sha256(particle_generator.get_state().cpu().numpy().tobytes()).hexdigest()
                    if particle_generator is not None else None),
            })
            if progress:
                progress({"start": start, "stop": start+size, "n_samples": n_samples})
    return torch.cat(pieces).numpy(), {"batches": len(hashes), "batch_rng_hashes": hashes,
                                        "score_calls": len(hashes)*M}


def _load_model(run_path, record, method, device):
    import torch
    import l1_training
    path = run_path / record["checkpoint"]
    if study.sha(path) != record["checkpoint_sha256"]:
        raise RuntimeError("Refit checkpoint changed before generation")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["config"]["l1_lambda"] != record["selected_lambda"]
            or payload["update"] != record["updates"]
            or payload["model_config"]["method"] != method
            or payload["model_config"]["input_features"] != "raw"):
        raise RuntimeError("Refit checkpoint does not match frozen selection")
    model = l1_training.build_model(payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def verify_sampler(run_path, manifest, spec, model_records, device, environment):
    import numpy as np
    import subgaussian_methods as base
    path = run_path / "sampler_verification.json"
    signature = runtime_signature(environment, manifest["config"])
    expected = {"study_id": manifest["study_id"], "signature": signature,
                "checkpoints": {m: r["checkpoint_sha256"] for m, r in model_records.items()}}
    if path.exists():
        old = study.read_json(path)
        if old.get("identity") != expected or old.get("status") != "passed":
            raise RuntimeError("Existing sampler verification has incompatible inputs/runtime")
        return old
    traces, checks = {}, []
    original_device = base.DEVICE
    try:
        base.DEVICE = device
        for method in study.METHODS:
            model = _load_model(run_path, model_records[method], method, device)
            score, particle_gen = make_score(model, method, device, 96000003, 20, 25.)
            values, trace = tracked_backward_sample(
                score, 7, 7, T=3., delta=.001, M=manifest["config"]["M"],
                seed=96000001, batch=3, device=device, particle_generator=particle_gen)
            score, _ = make_score(model, method, device, 96000003, 20, 25.)
            original = base.backward_sample(
                score, 7, 7, T=3., delta=.001, M=manifest["config"]["M"],
                seed=96000001, batch=3)
            if not np.array_equal(values, original):
                raise RuntimeError(f"Sampler arithmetic changed for {method}")
            checks.append({"method": method, "bitwise_equal_original": True,
                           "max_absolute_difference": float(np.abs(values-original).max())})
            traces[method] = trace["batch_rng_hashes"]
            del model
    finally:
        base.DEVICE = original_device
    for c, d in zip(traces["cindes"], traces["dsm"]):
        for key in ("initial_gaussian_sha256", "ending_reverse_rng_sha256"):
            if c[key] != d[key]:
                raise RuntimeError("Verification reverse randomness differs by method")
    record = {"status": "passed", "identity": expected, "checks": checks,
              "includes_partial_final_batch": True, "K": 20, "n": 7, "batch": 3,
              "M": manifest["config"]["M"], "recorded_at_utc": study.utc_now()}
    study.atomic_json(path, record)
    return record


def _atomic_npy(path, values):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with tmp.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(tmp, path)


def generate_pair(run_path, manifest, spec, model_records, device, environment, status_callback=None):
    import numpy as np
    import torch
    config = manifest["config"]
    signature = runtime_signature(environment, config)
    reconcile_generation_runtime(run_path, manifest, spec, model_records, signature)
    verification = verify_sampler(run_path, manifest, spec, model_records, device, environment)
    records = {}
    n_shards = math.ceil(config["n_gen"]/config["shard_size"])
    for method in study.METHODS:
        model = _load_model(run_path, model_records[method], method, device)
        for shard in range(n_shards):
            directory = run_path / "shards" / method
            array_path = directory / f"shard_{shard:02d}.npy"
            metadata_path = directory / f"shard_{shard:02d}.json"
            size = min(config["shard_size"], config["n_gen"]-shard*config["shard_size"])
            identity = {"study_id": manifest["study_id"], "task_id": spec["task_id"],
                        "method": method, "shard": shard, "n_samples": size,
                        "checkpoint_sha256": model_records[method]["checkpoint_sha256"],
                        "runtime_signature": signature}
            if metadata_path.exists():
                row = study.read_json(metadata_path)
                if (row.get("status") != "complete" or row.get("identity") != identity
                        or not array_path.is_file() or study.sha(array_path) != row["sample_sha256"]):
                    raise RuntimeError(f"Completed generation shard differs: {metadata_path}")
                records[(method, shard)] = row
                print(f"VERIFIED_SHARD method={method} shard={shard}", flush=True)
                continue
            if array_path.exists():
                archived = run_path / "interrupted" / f"uncommitted_{method}_{shard}_{uuid.uuid4().hex}.npy"
                archived.parent.mkdir(parents=True, exist_ok=True)
                os.replace(array_path, archived)
            reverse_seed = 98000000 + 10000*spec["seed"] + shard
            particle_seed = 99000000 + 10000*spec["seed"] + shard
            score, particles = make_score(model, method, device, particle_seed, config["K"], config["R"])
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            def progress(row):
                if row["stop"] == row["n_samples"] or row["stop"] % config["generation_batch"] == 0:
                    update = {"method": method, "shard": shard, **row,
                              "elapsed_seconds": time.perf_counter()-started}
                    study.atomic_json(run_path / "generation_progress.json", update)
                    if status_callback:
                        status_callback("generation", update)

            values, diagnostics = tracked_backward_sample(
                score, config["d"], size, T=config["T"], delta=config["delta"],
                M=config["M"], seed=reverse_seed, batch=config["generation_batch"],
                device=device, particle_generator=particles, progress=progress)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if values.shape != (size, config["d"]) or values.dtype != np.float32:
                raise RuntimeError("Unexpected generated sample schema")
            _atomic_npy(array_path, values)
            row = {"status": "complete", "identity": identity,
                   "sample_sha256": study.sha(array_path),
                   "elapsed_seconds": time.perf_counter()-started,
                   "reverse_seed_argument": reverse_seed, "reverse_rng_seed": reverse_seed+999,
                   "particle_rng_seed": particle_seed if method == "cindes" else None,
                   "diagnostics": diagnostics, "environment": environment,
                   "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                   "completed_at_utc": study.utc_now()}
            study.atomic_json(metadata_path, row)
            records[(method, shard)] = row
            print(f"COMPLETED_SHARD method={method} shard={shard} rows={size}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    paired_checks = 0
    for shard in range(n_shards):
        c, d = (records[(m, shard)]["diagnostics"]["batch_rng_hashes"] for m in study.METHODS)
        if len(c) != len(d):
            raise RuntimeError("Paired generation batch counts differ")
        for first, second in zip(c, d):
            if (first["initial_gaussian_sha256"] != second["initial_gaussian_sha256"]
                    or first["ending_reverse_rng_sha256"] != second["ending_reverse_rng_sha256"]
                    or first["ending_particle_rng_sha256"] is None
                    or second["ending_particle_rng_sha256"] is not None):
                raise RuntimeError("Paired reverse randomness failed verification")
            paired_checks += 1
    output = {}
    for method in study.METHODS:
        pieces = [np.load(run_path / "shards" / method / f"shard_{s:02d}.npy", allow_pickle=False)
                  for s in range(n_shards)]
        values = np.concatenate(pieces)
        if values.shape != (config["n_gen"], config["d"]) or not np.isfinite(values).all():
            raise RuntimeError("Incomplete combined generated array")
        path = run_path / "samples" / f"{method}.npy"
        if path.exists():
            old = np.load(path, allow_pickle=False)
            if not np.array_equal(old, values):
                raise RuntimeError("Existing combined samples differ from their shards")
        else:
            _atomic_npy(path, values)
        output[method] = {
            "samples": str(path.relative_to(run_path)), "samples_sha256": study.sha(path),
            "generation_seconds": sum(records[(method, s)]["elapsed_seconds"] for s in range(n_shards)),
            "shard_metadata": [f"shards/{method}/shard_{s:02d}.json" for s in range(n_shards)],
            "paired_reverse_rng_batch_checks": paired_checks,
            "n_samples": len(values), "n_shards": n_shards, "dtype": str(values.dtype),
            "runtime_signature": signature, "sampler_verification": "sampler_verification.json",
            "timing_scope": "sum of complete shard elapsed times; includes resumed completed shards",
        }
    study.atomic_json(run_path / "generation_complete.json", {
        "status": "complete", "study_id": manifest["study_id"], "task_id": spec["task_id"],
        "methods": output, "paired_reverse_rng_batch_checks": paired_checks,
        "sampler_verification_sha256": study.sha(run_path / "sampler_verification.json"),
        "completed_at_utc": study.utc_now()})
    return output
