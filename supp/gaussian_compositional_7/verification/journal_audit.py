#!/usr/bin/env python3
"""Independently check the supplied records, scientific sources, and tables.

This checker uses only the Python standard library and does not fit a model.
It does not import reproduce_tables.py or the experimental Python modules.
Use --mock-scheduler to additionally test the submission wrapper with a local
qsub replacement; this option requires Bash but never submits a real job.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


STUDY_ID = "1832ac1cbf9358b8642b2996b73f513e92d088dc5cada686f03e74c7edabee83"
METHODS = ("cindes", "dsm")
SAMPLE_SIZES = (500, 1000, 1500, 2000)
METRICS = ("tv1", "tv2", "llgap", "h_tv", "h_w1")
TIMES = ("final_fit_seconds", "generation_seconds",
         "fit_plus_generation_seconds", "summed_cv_fit_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, message):
    require(math.isfinite(a) and math.isfinite(b), message + ": nonfinite value")
    require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), message)


def check_subset(projected, original, name):
    """A published projection may remove fields, but cannot change a value."""
    if isinstance(projected, dict):
        require(isinstance(original, dict), name + ": changed object type")
        for key, value in projected.items():
            require(key in original, name + ": added field " + key)
            check_subset(value, original[key], name + "/" + key)
    else:
        require(projected == original, name + ": retained value changed")


def independent_describe(values):
    require(len(values) == 20, "A summary does not have 20 replications")
    mean = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in values) / (len(values) - 1))
    return {"n_rep": len(values), "mean": mean, "sd": sd,
            "se": sd / math.sqrt(len(values))}


def verify_sources(package, manifest):
    expected = manifest["source_sha256"]
    recorded = read_json(package / "experiment/original_python_sha256.json")
    require(expected == recorded and len(expected) == 14,
            "The two original Python source inventories differ")
    experiment = package / "experiment"
    actual_names = {p.relative_to(experiment).as_posix()
                    for pattern in ("*.py", "source/*.py")
                    for p in experiment.glob(pattern)}
    require(actual_names == set(expected), "Scientific Python file inventory changed")
    for name, expected_hash in expected.items():
        require(sha(experiment / name) == expected_hash,
                "Scientific Python source changed: " + name)
    core = {key: manifest[key] for key in ("config", "tasks", "source_sha256")}
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    require(digest == manifest["study_id"] == STUDY_ID, "Frozen study identity changed")
    return {"status": "passed", "original_python_files": 14,
            "all_original_source_hashes_match": True,
            "frozen_study_identity_matches": True}


def verify_records(package, manifest, provenance):
    root = package / "reported_results"
    for name, expected_hash in provenance["projection_sha256"].items():
        require(sha(root / name) == expected_hash,
                "Published record hash differs: " + name)
    tasks = {row["task_id"]: row for row in manifest["tasks"]}
    require(set(tasks) == set(range(1, 81)), "Expected task IDs 1 through 80")
    files = sorted((root / "runs").glob("n_*/rep_*/run_complete.json"))
    require(len(files) == 80, "Expected 80 complete paired run records")
    records, rows, seen, reference_hashes, hardware = [], [], set(), set(), {}
    for path in files:
        run = read_json(path)
        task = run["task_id"]
        require(task not in seen, "Repeated paired task ID")
        seen.add(task)
        require(run["status"] == "complete" and run["profile"] == "full",
                "A run is partial or uses a different profile")
        require(run["study_id"] == STUDY_ID and run["reference_n"] == 1048576,
                "A run uses a different study or evaluation reference size")
        spec = tasks[task]
        require(all(run[k] == spec[k] for k in ("n", "replication", "seed")),
                "A run differs from its frozen task specification")
        require(run["seed"] == 999 + task, "Unexpected fit seed")
        require(set(run["methods"]) == set(METHODS), "Unpaired method result")
        reference_hashes.add(run["reference_bank_sha256"])
        gpu = run["environment"]["gpu"]
        hardware[gpu] = hardware.get(gpu, 0) + 1
        common_data = run["data_record"]["pool_sha256"]
        signatures = []
        for method in METHODS:
            row = run["methods"][method]
            require(row["method"] == method and row["study_id"] == STUDY_ID,
                    "Method record identity differs")
            require(all(row[k] == run[k] for k in ("task_id", "n", "replication", "seed")),
                    "Method record and paired task differ")
            require(row["n_gen"] == 10000 and row["d"] == 7,
                    "Wrong generated-sample count or dimension")
            model = row["model_record"]
            generation = row["generation_record"]
            require(model["training_data_sha256"] == common_data,
                    "Methods do not share their recorded training data")
            signatures.append(generation["runtime_signature"])
            require(generation["runtime_signature"]["gpu"] == gpu,
                    "GPU type differs between run and generation record")
            close(row["training_seconds"], model["training_seconds"], "Refit timer mismatch")
            close(row["generation_seconds"], generation["generation_seconds"],
                  "Generation timer mismatch")
            value = {"n": row["n"], "method": method, "task_id": task,
                     **{key: float(row[key]) for key in METRICS},
                     "final_fit_seconds": float(row["training_seconds"]),
                     "generation_seconds": float(row["generation_seconds"]),
                     "fit_plus_generation_seconds": float(row["training_seconds"] + row["generation_seconds"]),
                     "summed_cv_fit_seconds": float(model["cv_sum_fit_wall_seconds"])}
            require(all(math.isfinite(value[key]) for key in METRICS + TIMES),
                    "Nonfinite recorded metric or timer")
            require(all(value[key] >= 0 for key in TIMES), "Negative recorded timer")
            rows.append(value)
        require(signatures[0] == signatures[1], "Paired generation environments differ")
        records.append((path.relative_to(root).as_posix(), run))
    require(seen == set(tasks), "Missing or unexpected paired run")
    require(len(reference_hashes) == 1, "Evaluation references differ across replications")
    summaries = []
    for n in SAMPLE_SIZES:
        for method in METHODS:
            selected = [r for r in rows if (r["n"], r["method"]) == (n, method)]
            require(len(selected) == 20, "Wrong replication count in a sample-size/method cell")
            for metric in METRICS + TIMES:
                summaries.append({"n": n, "method": method, "metric": metric,
                                  **independent_describe([r[metric] for r in selected])})
    return records, summaries, {"status": "passed", "paired_runs": len(records),
                               "method_results": len(rows), "shared_reference": True,
                               "paired_gpu_environments_match": True,
                               "gpu_allocations": hardware}


def check_statistics(actual, expected):
    mapping = {(r["n"], r["method"], r["metric"]): r for r in actual}
    require(len(mapping) == len(actual) == len(expected) == 72,
            "Incorrect or duplicated summary inventory")
    maximum = 0.0
    for row in expected:
        key = (row["n"], row["method"], row["metric"])
        require(key in mapping, "Expected summary is absent")
        value = mapping[key]
        require(value["n_rep"] == row["n_rep"] == 20, "Wrong summary sample size")
        for stat in ("mean", "sd", "se"):
            close(value[stat], row[stat], "Summary differs: " + str(key) + "/" + stat)
            maximum = max(maximum, abs(value[stat] - row[stat]))
        precision = 4 if row["metric"] in METRICS else 2
        for stat in ("mean", "sd"):
            require(format(value[stat], f".{precision}f") == format(row[stat], f".{precision}f"),
                    "Displayed table value differs: " + str(key) + "/" + stat)
    return maximum


def verify_table_builder(package, summaries):
    fixture = read_json(package / "expected_tables/reported_statistics.json")["statistics"]
    max_fixture_difference = check_statistics(summaries, fixture)
    with tempfile.TemporaryDirectory(prefix="cindes_table_audit_") as temporary:
        output = Path(temporary) / "tables"
        result = subprocess.run([sys.executable, str(package / "reproduce_tables.py"),
                                 "--output", str(output), "--check"],
                                capture_output=True, text=True)
        require(result.returncode == 0, "Table builder failed: " + result.stderr.strip())
        produced = read_json(output / "statistics.json")["statistics"]
        max_builder_difference = check_statistics(produced, summaries)
        for filename, label in (("accuracy_table.tex", "tab:cindes_accuracy_full"),
                                ("runtime_table.tex", "tab:cindes_runtime_full")):
            text = (output / filename).read_text(encoding="utf-8")
            require(text.count("\\label{" + label + "}") == 1, "Missing or duplicate table label")
            require(text.index("\\caption{") < text.index("\\label{" + label + "}"),
                    "Table label appears before its caption")
            require("\\resizebox" not in text and "\\small" in text,
                    "Table font is not the agreed fixed small size")
    return {"status": "passed", "summary_rows": 72,
            "mean_sd_se_values_checked": 216, "formatted_mean_sd_values_checked": 144,
            "maximum_absolute_difference_from_expected": max_fixture_difference,
            "maximum_absolute_difference_from_table_builder": max_builder_difference,
            "both_table_labels_and_font_settings_checked": True,
            "scope": "Reaggregation of recorded outcomes and timers; no raw-sample metric recomputation."}


def verify_original_archive(archive, package, records, provenance, summaries):
    require(sha(archive) == provenance["source_archive_sha256"], "Original archive SHA256 differs")
    with zipfile.ZipFile(archive) as handle:
        bundle = json.loads(handle.read("bundle_manifest.json"))
        for name, digest in bundle["files_sha256"].items():
            require(hashlib.sha256(handle.read(name)).hexdigest() == digest,
                    "Archive member hash differs: " + name)
        for name, digest in provenance["original_member_sha256"].items():
            require(hashlib.sha256(handle.read(name)).hexdigest() == digest,
                    "Original provenance hash differs: " + name)
        for name, record in records:
            check_subset(record, json.loads(handle.read(name)), name)
        original_summary = list(csv.DictReader(io.StringIO(handle.read("summary/summary.csv").decode())))
        lookup = {(r["n"], r["method"], r["metric"]): r for r in summaries}
        for row in original_summary:
            for metric in METRICS:
                got = lookup[(int(row["n"]), row["method"], metric)]
                for stat in ("mean", "sd", "se"):
                    close(got[stat], float(row[metric + "_" + stat]), "Original SCC summary differs")
    return {"status": "passed", "archive_sha256": provenance["source_archive_sha256"],
            "manifested_members_verified": len(bundle["files_sha256"]),
            "all_retained_run_record_values_match_original": True,
            "original_accuracy_summary_matches": True,
            "scope": "Original compact return archive, which does not contain the raw arrays/checkpoints."}


def verify_scheduler(package):
    require(shutil.which("bash") is not None, "--mock-scheduler requires Bash")
    names = ("submit_scc.sh", "prepare_reference.qsub", "full_array.qsub", "summarize.qsub", "scc_env.sh")
    with tempfile.TemporaryDirectory(prefix="cindes_scheduler_audit_") as temporary:
        base = Path(temporary)
        scripts, fakebin, caller = base / "scripts", base / "bin", base / "caller"
        for path in (scripts, fakebin, caller):
            path.mkdir()
        for name in names:
            shutil.copy2(package / "experiment" / name, scripts / name)
            subprocess.run(["bash", "-n", str(scripts / name)], check=True, capture_output=True)
        driver = (scripts / "submit_scc.sh").read_text()
        require("/usr/local/bin/qsub" not in driver and "/usr/bin/qsub" not in driver,
                "The scheduler wrapper contains a fixed qsub path")
        stub = fakebin / "qsub"
        stub.write_text("#!/usr/bin/env python3\n"
                        "import json,os,pathlib,sys\n"
                        "p=pathlib.Path(os.environ['CINDES_AUDIT_QSUB_LOG'])\n"
                        "n=len(p.read_text().splitlines()) if p.exists() else 0\n"
                        "with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                        "suffix='.1-80:1' if sys.argv[-1]=='full_array.qsub' else ''\n"
                        "print(str(900001+n)+suffix)\n", encoding="utf-8")
        stub.chmod(0o755)
        log = base / "calls.jsonl"
        env = dict(os.environ)
        for key in ("SCC_PROJECT", "CINDES_OUTPUT_ROOT", "CINDES_ENV_SCRIPT", "PYTHON_BIN", "BASH_ENV", "ENV"):
            env.pop(key, None)
        env.update(PATH=str(fakebin) + os.pathsep + env.get("PATH", ""),
                   CINDES_AUDIT_QSUB_LOG=str(log), PYTHON_BIN=sys.executable)
        failed = subprocess.run(["bash", str(scripts / "submit_scc.sh"), "--dry-run"],
                                cwd=caller, env=env, capture_output=True, text=True)
        require(failed.returncode != 0 and not log.exists(), "Missing project did not fail before submission")
        activation = caller / "activation script.sh"
        activation.write_text("# Mock activation script, never executed.\n", encoding="utf-8")
        env.update(SCC_PROJECT="audit_project", CINDES_OUTPUT_ROOT="new full study",
                   CINDES_ENV_SCRIPT="activation script.sh")
        dry = subprocess.run(["bash", str(scripts / "submit_scc.sh"), "--dry-run"],
                             cwd=caller, env=env, capture_output=True, text=True)
        require(dry.returncode == 0 and not log.exists() and not (scripts / "submissions").exists(),
                "Dry run failed or made a submission")
        submitted = subprocess.run(["bash", str(scripts / "submit_scc.sh")],
                                   cwd=caller, env=env, capture_output=True, text=True)
        require(submitted.returncode == 0, "Mock submission failed: " + submitted.stderr.strip())
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assignments = "CINDES_OUTPUT_ROOT=" + str((caller / "new full study").resolve())
        assignments += ",PYTHON_BIN=" + sys.executable
        assignments += ",CINDES_ENV_SCRIPT=" + str(activation.resolve())
        prefix = ["-terse", "-P", "audit_project", "-v", assignments]
        expected = [prefix + ["prepare_reference.qsub"],
                    prefix + ["-hold_jid", "900001", "full_array.qsub"],
                    prefix + ["-hold_jid", "900002", "summarize.qsub"]]
        require(calls == expected, "Mock scheduler arguments or dependencies differ")
        require(not (caller / "new full study").exists(), "Submission wrapper created experiment outputs")
        logs = list((scripts / "submissions").glob("*.txt"))
        require(len(logs) == 1, "Submission record missing or duplicated")
        record = logs[0].read_text()
        require(all(f"{name}={number}" in record for name, number in
                    (("reference_job", 900001), ("array_job", 900002), ("summary_job", 900003))),
                "Parsed job identifiers are absent from the submission record")
    return {"status": "passed", "real_scheduler_invoked": False,
            "mock_qsub_calls": 3, "dependency_chain": "reference -> array80 -> summary",
            "missing_project_rejected": True, "dry_run_submits_nothing": True,
            "caller_relative_paths_with_spaces_preserved": True,
            "custom_activation_script_transmitted": True,
            "array_job_reply_parsed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--original-archive", type=Path,
                        help="Optional original results_for_tables.zip for a deeper provenance check.")
    parser.add_argument("--mock-scheduler", action="store_true",
                        help="Also check the SCC wrapper using a temporary local qsub stub.")
    parser.add_argument("--output", type=Path, help="Optionally write a JSON audit report.")
    args = parser.parse_args()
    package = args.package_root.resolve()
    manifest = read_json(package / "reported_results/study_manifest.json")
    provenance = read_json(package / "reported_results/provenance.json")
    report = {"status": "passed", "study_id": STUDY_ID,
              "scope": "Scientific source identity, saved-run aggregation and table generation; no model fitting or GPU execution.",
              "sources": verify_sources(package, manifest)}
    records, summaries, record_report = verify_records(package, manifest, provenance)
    report["records"] = record_report
    report["tables"] = verify_table_builder(package, summaries)
    if args.original_archive:
        report["original_archive"] = verify_original_archive(args.original_archive, package, records, provenance, summaries)
    if args.mock_scheduler:
        report["mock_scheduler"] = verify_scheduler(package)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PASS: 14 scientific source files, 80 paired runs, 216 summary statistics and 144 displayed table values.")
    print("Table labels and common small font verified; no model was trained.")
    if args.original_archive:
        print("Original archive: all manifested hashes and retained numerical records verified.")
    if args.mock_scheduler:
        print("SCC wrapper: three dependent submissions verified using a local mock; no real jobs submitted.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print("AUDIT FAILED: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
