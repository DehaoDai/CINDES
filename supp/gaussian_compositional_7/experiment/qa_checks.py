"""Small implementation checks; never submit a real SCC job or fit the full study.

Use --static for configuration/shell checks, --sampler for synthetic-network
arithmetic checks, --runtime for synthetic recovery checks, and --submission
for the isolated mocked-qsub check.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
import study


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def static_checks():
    full = study.make_config("full")
    smoke = study.make_config("smoke")
    tasks = study.task_specs(full)
    require(len(tasks) == 80, "Expected 80 paired tasks")
    require([r["task_id"] for r in tasks] == list(range(1, 81)), "Task IDs are not contiguous")
    require(len({r["seed"] for r in tasks}) == 80, "Repeated fitted-data seed")
    require(set(r["seed"] for r in tasks).isdisjoint({3, 4, 5}), "Pilot seeds reused")
    expected = {
        500: {"validation_sizes": [167, 167, 166], "cv_updates": [4000]*3, "full_updates": 4000},
        1000: {"validation_sizes": [334, 333, 333], "cv_updates": [6000]*3, "full_updates": 8000},
        1500: {"validation_sizes": [500, 500, 500], "cv_updates": [8000]*3, "full_updates": 12000},
        2000: {"validation_sizes": [667, 667, 666], "cv_updates": [12000]*3, "full_updates": 16000},
    }
    for n, values in expected.items():
        require(study.horizons(n, full) == values, f"Unexpected horizon or fold sizes for n={n}")
        selected = [r for r in tasks if r["n"] == n]
        require([r["replication"] for r in selected] == list(range(1, 21)), "Missing replication")
        require(sum(values["validation_sizes"]) == n, "Fold count differs from n")
        full_batches = (n + 255)//256
        for check in range(101):
            updates = values["full_updates"] * check // 100
            require(updates % full_batches == 0, "Selected full stop is not a complete positive epoch")
            require(updates // full_batches == 20*check, "Selected epoch fraction changed")
    require(len(tasks)*2*len(full["lambda_grid"])*full["cv_folds"] == 2400, "Wrong number of CV fits")
    require(len(tasks)*2 == 160, "Wrong number of full-data refits")
    require(len(tasks)*2*full["n_gen"] == 1600000, "Wrong number of generated samples")
    require(smoke["profile"] == "smoke" and len(study.task_specs(smoke)) == 1, "Smoke task mapping")
    require(set(r["seed"] for r in study.task_specs(smoke)).isdisjoint(r["seed"] for r in tasks),
            "Smoke/full seed overlap")
    require(full["K"] % 2 == 0 and full["M"] % 2 == 0, "Antithetic/grid settings")
    require(full["generation_device"] == "cuda", "Full study must fail rather than silently use CPU")
    require(full["terminal_correction"] == "none" and not full["support_clipping"]
            and full["last_reverse_step_noise"], "Endpoint or support policy changed")
    scripts = sorted(ROOT.glob("*.sh")) + sorted(ROOT.glob("*.qsub"))
    checked = []
    for path in scripts:
        data = path.read_bytes()
        data.decode("ascii")
        require(b"\r" not in data, f"CRLF line endings in {path.name}")
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)
        checked.append(path.name)
    return {"status": "passed", "full_tasks": 80, "CV_fits": 2400, "full_refits": 160,
            "generated_samples": 1600000, "horizons": expected,
            "shell_syntax_ASCII_LF_checked": checked}


def sampler_checks():
    import numpy as np
    import torch
    import generation
    import l1_training
    import subgaussian_methods as original
    torch.set_num_threads(1)
    device = torch.device("cpu")
    original.DEVICE = device
    traces = {}
    checks = []
    # Initialized synthetic networks only. No target data or trained models.
    for method in ("cindes", "dsm"):
        torch.manual_seed(401 if method == "cindes" else 402)
        spec = l1_training.model_spec(
            method, 7, "classifier" if method == "cindes" else "residual",
            activation="softplus", softplus_beta=5.0)
        model = l1_training.build_model(spec).to(device).eval()
        score, particles = generation.make_score(model, method, device, 43001, 20, 25.0)
        tracked, diagnostics = generation.tracked_backward_sample(
            score, 7, 7, T=3.0, delta=0.001, M=200, seed=43002, batch=3,
            device=device, particle_generator=particles)
        score, _ = generation.make_score(model, method, device, 43001, 20, 25.0)
        expected = original.backward_sample(
            score, 7, 7, T=3.0, delta=0.001, M=200, seed=43002, batch=3)
        require(np.array_equal(tracked, expected), f"Production sampler arithmetic changed for {method}")
        require(tracked.shape == (7, 7) and np.isfinite(tracked).all(), "Sampler output schema")
        traces[method] = diagnostics["batch_rng_hashes"]
        checks.append({"method": method, "bitwise_equal_original": True,
                       "maximum_absolute_difference": float(np.max(np.abs(tracked-expected)))})
    require(len(traces["cindes"]) == len(traces["dsm"]) == 3, "Final partial batch omitted")
    for c, d in zip(traces["cindes"], traces["dsm"]):
        for key in ("initial_gaussian_sha256", "ending_reverse_rng_sha256"):
            require(c[key] == d[key], "Reverse randomness differs across methods")
        require(c["ending_particle_rng_sha256"] is not None and d["ending_particle_rng_sha256"] is None,
                "Particle RNG is not separate")

    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((len(x), 1), device=x.device, dtype=x.dtype)

    y = torch.tensor([[0., 0., 0., 0., 0., 0., 0.], [.1, -.2, .3, -.4, .5, -.6, .7]])
    for t in (3.0, .1, .001):
        got = original.cindes_score(Constant(), 25., y, t, 20, torch.Generator().manual_seed(99))
        require(torch.equal(got, -y), "Constant classifier does not give exact Gaussian score")
    return {"status": "passed", "synthetic_only": True, "device": "cpu",
            "sampler_checks": checks, "partial_batch_size": 1,
            "paired_reverse_rng_checks": 3, "constant_classifier_exact_gaussian_score": True,
            "CUDA_execution_tested": False}


def submission_checks():
    """Run the driver in an isolated copy with a local qsub stub on PATH."""
    names = ("submit_scc.sh", "prepare_reference.qsub", "full_array.qsub",
             "summarize.qsub", "scc_env.sh")
    for name in names:
        require((ROOT/name).is_file(), f"Missing submission file: {name}")
    driver = (ROOT/"submit_scc.sh").read_text()
    require("/usr/local/bin/qsub" not in driver and "/usr/bin/qsub" not in driver,
            "Refuse to mock a driver with an absolute scheduler path")
    array = (ROOT/"full_array.qsub").read_text()
    require("#$ -t 1-80" in array and "#$ -tc 2" in array, "Array size/concurrency changed")
    require("#$ -l gpus=1" in array and "#$ -pe omp 4" in array, "GPU/CPU allocation changed")
    require("#$ -l gpu_c=8.0" in array and "#$ -l gpu_memory=16G" in array,
            "GPU compatibility allocation changed")
    with tempfile.TemporaryDirectory(prefix="cindes_mock_qsub_") as temporary:
        base = Path(temporary)
        copy = base/"package"
        copy.mkdir()
        for name in names:
            shutil.copy2(ROOT/name, copy/name)
        fakebin = base/"bin"
        fakebin.mkdir()
        stub = fakebin/"qsub"
        stub.write_text(r"""#!/usr/bin/env python3
import json, os, pathlib, sys
path = pathlib.Path(os.environ["CINDES_QA_QSUB_LOG"])
lines = path.read_text().splitlines() if path.exists() else []
job = 900001 + len(lines)
with path.open("a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\n")
suffix = ".1-80:1" if sys.argv[-1] == "full_array.qsub" else ""
print(str(job) + suffix)
""")
        stub.chmod(0o755)
        log = base/"qsub_calls.jsonl"
        environment = dict(os.environ)
        environment["PATH"] = str(fakebin)+os.pathsep+environment.get("PATH", "")
        environment["CINDES_QA_QSUB_LOG"] = str(log)
        # Dry-run is required to be read-only and independent of scheduler state.
        dry = subprocess.run(["bash", str(copy/"submit_scc.sh"), "--dry-run"],
                             env=environment, capture_output=True, text=True, check=True)
        require(not log.exists(), "Dry-run invoked qsub")
        actual = subprocess.run(["bash", str(copy/"submit_scc.sh")], env=environment,
                                capture_output=True, text=True, check=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        expected = [
            ["-terse", "-P", "dgmgrp", "prepare_reference.qsub"],
            ["-terse", "-P", "dgmgrp", "-hold_jid", "900001", "full_array.qsub"],
            ["-terse", "-P", "dgmgrp", "-hold_jid", "900002", "summarize.qsub"],
        ]
        require(calls == expected, "Submission arguments or dependency IDs differ")
        records = list((copy/"submissions").glob("*.txt"))
        require(len(records) == 1, "Submission record missing or duplicated")
        record = records[0].read_text()
        for key, value in (("reference_job",900001), ("array_job",900002), ("summary_job",900003)):
            require(f"{key}={value}" in record, f"Missing parsed scheduler ID: {key}")
        require("ARRAY_JOB_ID" in dry.stdout and "900002" in actual.stdout, "Submission output incomplete")
    return {"status": "passed", "mocked": True, "real_scheduler_invoked": False,
            "dry_run_submits_nothing": True, "qsub_calls": calls,
            "array_terse_reply_parsed": "900002.1-80:1",
            "dependency_chain": "reference -> array80 -> strict_summary"}


def runtime_checks():
    """Use synthetic files to check GPU-change restart and interrupted moves."""
    import numpy as np
    import generation
    from unittest.mock import patch
    config = study.make_config("smoke")
    spec = study.task_specs(config)[0]
    manifest = {"study_id": "synthetic_runtime_qa_only", "config": config}
    models = {m: {"checkpoint_sha256": hashlib.sha256(m.encode()).hexdigest()}
              for m in study.METHODS}
    environment = {
        "torch": "synthetic", "cuda_runtime": "synthetic", "generation_device": "cuda",
        "gpu": "synthetic_gpu_A", "gpu_compute_capability": [8, 0],
        "allow_tf32": False, "deterministic_algorithms": True,
        "float32_matmul_precision": "highest", "cublas_workspace_config": ":4096:8",
    }
    old_signature = generation.runtime_signature(environment, config)
    new_signature = generation.runtime_signature(
        dict(environment, gpu="synthetic_gpu_B", gpu_compute_capability=[8, 6]), config)
    require(old_signature != new_signature, "GPU hardware missing from numerical runtime signature")

    def fixture(base):
        base.mkdir()
        (base/"refits").mkdir()
        (base/"refits"/"unchanged_model_fixture.bin").write_bytes(b"frozen fitted model fixture")
        generation.reconcile_generation_runtime(base, manifest, spec, models, old_signature)
        verification = base/"sampler_verification.json"
        study.atomic_json(verification, {
            "status": "passed", "identity": {"study_id": manifest["study_id"],
                "signature": old_signature, "checkpoints": {m: r["checkpoint_sha256"] for m, r in models.items()}}})
        methods = {}
        for method in study.METHODS:
            arrays = []
            for shard in range(2):
                path = base/"shards"/method/f"shard_{shard:02d}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                values = np.full((32, 7), float(shard)+(0.1 if method == "cindes" else 0.2), dtype=np.float32)
                np.save(path, values, allow_pickle=False)
                arrays.append(values)
                study.atomic_json(path.with_suffix(".json"), {
                    "status": "complete", "sample_sha256": study.sha(path),
                    "identity": {"study_id": manifest["study_id"], "task_id": spec["task_id"],
                        "method": method, "shard": shard, "n_samples": 32,
                        "checkpoint_sha256": models[method]["checkpoint_sha256"],
                        "runtime_signature": old_signature}})
            path = base/"samples"/f"{method}.npy"
            path.parent.mkdir(exist_ok=True)
            np.save(path, np.concatenate(arrays), allow_pickle=False)
            methods[method] = {"samples": f"samples/{method}.npy", "samples_sha256": study.sha(path),
                               "runtime_signature": old_signature}
        study.atomic_json(base/"generation_complete.json", {
            "status": "complete", "study_id": manifest["study_id"], "task_id": spec["task_id"],
            "sampler_verification_sha256": study.sha(verification), "methods": methods})
        return {str(p.relative_to(base)): study.sha(p) for p in base.rglob("*") if p.is_file()}

    def expect_rejection(callback, explanation):
        try:
            callback()
        except RuntimeError:
            return
        raise AssertionError(explanation)

    def check_archive(base, before):
        archives = list((base/"interrupted").glob("generation_runtime_*"))
        require(len(archives) == 1, "Expected exactly one preserved generation archive")
        archive = archives[0]
        for name, expected_hash in before.items():
            if name.startswith("refits/"):
                require(study.sha(base/name) == expected_hash, "Runtime restart changed fitted model bytes")
            else:
                require(study.sha(archive/name) == expected_hash, "Archived generation evidence changed")
        require(not (base/"shards").exists() and not (base/"samples").exists(),
                "Old method samples remained active after runtime restart")
        require(not (base/"generation_restart_pending.json").exists(), "Archive journal was not completed")
        require((archive/"restart_reason.json").is_file(), "Archive lost its restart reason")
        require(study.read_json(base/"generation_runtime.json")["runtime_signature"] == new_signature,
                "New runtime marker is incorrect")

    with tempfile.TemporaryDirectory(prefix="cindes_runtime_qa_") as temporary:
        root = Path(temporary)
        base = root/"ordinary"
        before = fixture(base)
        generation.reconcile_generation_runtime(base, manifest, spec, models, old_signature)
        require(all(study.sha(base/name) == value for name, value in before.items()),
                "Same-runtime resume changed committed bytes")
        generation.reconcile_generation_runtime(base, manifest, spec, models, new_signature)
        check_archive(base, before)

        interrupted = root/"interrupted_move"
        before = fixture(interrupted)
        original_replace = generation.os.replace
        injected = {"raised": False}

        def replace_with_interruption(source, destination):
            if Path(source).name == "samples" and not injected["raised"]:
                injected["raised"] = True
                raise RuntimeError("Synthetic interruption after the shards move")
            return original_replace(source, destination)

        with patch.object(generation.os, "replace", side_effect=replace_with_interruption):
            expect_rejection(lambda: generation.reconcile_generation_runtime(
                interrupted, manifest, spec, models, new_signature), "Injected archive interruption did not fire")
        require(injected["raised"] and (interrupted/"generation_restart_pending.json").is_file(),
                "Interrupted archive lost its journal")
        generation.reconcile_generation_runtime(interrupted, manifest, spec, models, new_signature)
        check_archive(interrupted, before)

        corrupt = root/"corrupt"
        fixture(corrupt)
        with (corrupt/"shards"/"cindes"/"shard_00.npy").open("ab") as handle:
            handle.write(b"corruption")
        expect_rejection(lambda: generation.reconcile_generation_runtime(
            corrupt, manifest, spec, models, new_signature), "Runtime restart silently accepted a corrupt shard")
        require(not (corrupt/"interrupted").exists(), "Corruption was incorrectly treated as a routine restart")

        mismatch = root/"model_mismatch"
        fixture(mismatch)
        changed_models = {m: dict(value) for m, value in models.items()}
        changed_models["cindes"]["checkpoint_sha256"] = "changed_checkpoint"
        expect_rejection(lambda: generation.reconcile_generation_runtime(
            mismatch, manifest, spec, changed_models, new_signature), "Changed models were silently accepted")
        require(not (mismatch/"interrupted").exists(), "Changed checkpoint was incorrectly archived as a runtime change")
    return {"status": "passed", "synthetic_metadata_only": True, "CUDA_execution_tested": False,
            "same_runtime_preserves_committed_bytes": True, "hardware_change_archives_both_methods": True,
            "fitted_models_preserved": True, "interrupted_archive_journal_recovers": True,
            "corrupt_shard_rejected": True, "checkpoint_identity_change_rejected": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--sampler", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not any((args.static, args.sampler, args.runtime, args.submission)):
        args.static = args.sampler = args.runtime = args.submission = True
    result = {"status": "passed", "real_SCC_jobs_submitted": False, "full_experiment_executed": False,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.static:
        result["static"] = static_checks()
    if args.sampler:
        result["sampler"] = sampler_checks()
    if args.runtime:
        result["runtime"] = runtime_checks()
    if args.submission:
        result["submission"] = submission_checks()
    if args.out:
        study.atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
