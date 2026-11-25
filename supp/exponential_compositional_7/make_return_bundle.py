"""Create a verified results archive for later tables; raw files stay on SCC.

The default bundle contains metric records, CV histories, logs, environment
records, source snapshots, and the tables. Add --include-samples and/or
--include-checkpoints for a larger archive containing those raw artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import common
from summarize_results import (METHODS, locked, resolve_root, sha256, read_json,
                               require, validate_record, validate_reference_binding)

HERE = Path(__file__).resolve().parent
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".tex", ".txt", ".log", ".md", ".py", ".sh", ".qsub", ".yaml", ".yml"}
ARRAY_SUFFIXES = {".npz", ".npy"}
CHECKPOINT_SUFFIXES = {".pt", ".pth"}
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def selected_files(root, include_samples, include_checkpoints):
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_DIRS)
        for name in dirs:
            require(not (Path(directory) / name).is_symlink(), "A linked directory cannot be archived")
        for name in sorted(names):
            path = Path(directory) / name
            relative = path.relative_to(root)
            if path.suffix in (".lock", ".tmp", ".partial", ".zip", ".pyc") or name.endswith("~"):
                continue
            if path.name in ("bundle_receipt.json",):
                continue
            include = (path.suffix in TEXT_SUFFIXES or relative.parts[0] in ("logs", "submissions", "source_snapshot"))
            include = include or (include_samples and path.suffix in ARRAY_SUFFIXES)
            include = include or (include_checkpoints and path.suffix in CHECKPOINT_SUFFIXES)
            if include and path.is_file():
                require(not path.is_symlink(), "A linked file cannot be archived: " + str(relative))
                files[relative.as_posix()] = path
    return files


def verify_summary(root, profile, allow_partial):
    manifest = common.load_study(root, profile)
    summary_path = root / "summary" / "summary.json"
    require(summary_path.is_file(), "Run summarize_results.py successfully before creating the archive")
    summary = read_json(summary_path)
    require(summary.get("status") in ("complete", "partial"), "Invalid saved summary status")
    require(allow_partial or summary["status"] == "complete", "Use --allow-partial to archive an incomplete study")
    require(summary.get("profile") == profile and summary.get("study_identity_sha256") == manifest["identity_sha256"],
            "Summary belongs to another study")
    require(summary.get("raw_artifact_hashes_checked") is True,
            "A rebuilt export does not replace strict validation of the original SCC outputs")
    require(summary.get("analysis_source_sha256") == sha256(HERE / "summarize_results.py"),
            "The summarizer changed; rerun it before creating the archive")
    tasks = manifest["identity"]["tasks"]
    expected = {f"runs/n_{task['n']}/seed_{task['seed']}/result.json": task for task in tasks}
    recorded = summary.get("run_result_sha256", {})
    require(set(recorded).issubset(expected), "Unknown run in the summary")
    require(summary.get("completed_pairs") == len(recorded) and summary.get("expected_pairs") == len(tasks),
            "Summary counts differ from its task records")
    if summary["status"] == "complete":
        require(set(recorded) == set(expected), "Complete summary is missing paired runs")
    reference = common.validate_reference(root, manifest) if recorded else None
    for name, digest in recorded.items():
        path = common.safe_child(root, name)
        require(path.is_file() and sha256(path) == digest, "A completed run changed after summarization: " + name)
        row = read_json(path)
        validate_record(row, expected[name], manifest)
        validate_reference_binding(row, reference["reference_sha256"])
    # Refuse an archive of an outdated partial summary after another task finishes.
    completed_now = set()
    for path in (root / "runs").glob("n_*/seed_*/result.json"):
        row = read_json(path)
        if row.get("status") == "complete":
            completed_now.add(path.relative_to(root).as_posix())
    require(completed_now == set(recorded), "Completed tasks changed; rerun the summarizer before bundling")
    reference_meta = root / "reference" / "metadata.json"
    if recorded:
        require(sha256(reference_meta) == summary.get("reference_metadata_sha256"),
                "Reference metadata changed after summarization")
    for name, digest in summary.get("output_sha256", {}).items():
        path = common.safe_child(root / "summary", name)
        require(path.is_file() and sha256(path) == digest, "A summary output changed: " + name)
    require(summary.get("output_sha256"), "The summary lacks output hashes")
    return manifest, summary


def make_bundle(root, profile="n1000", allow_partial=False, include_samples=False, include_checkpoints=False):
    root = Path(root).resolve()
    manifest, summary = verify_summary(root, profile, allow_partial)
    files = selected_files(root, include_samples, include_checkpoints)
    # Preserve the exact reporting code used for this export even if its formatting
    # has been updated since the study's original source snapshot was made.
    for name in ("summarize_results.py", "make_return_bundle.py", "common.py", "study_config.py"):
        files["analysis_code/" + name] = HERE / name
    required = {"manifest.json", "study_config.json", "task_plan.json", "summary/summary.json",
                "summary/summary.csv", "summary/runtime.csv", "summary/all_runs.jsonl", "summary/export.json",
                "summary/accuracy_table.tex", "summary/runtime_table.tex"}
    require(required.issubset(files), "Required export is missing: " + ", ".join(sorted(required - files.keys())))
    identities = {name: {"sha256": sha256(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}
    if include_samples or include_checkpoints:
        for relative in summary["run_result_sha256"]:
            path = root / relative
            row = read_json(path)
            for name, expected in row["artifact_sha256"].items():
                artifact = common.safe_child(path.parent, name)
                key = artifact.relative_to(root).as_posix()
                if key in identities:
                    require(identities[key]["sha256"] == expected, "Requested raw artifact changed: " + key)
    notes = """CINDES versus DSM: exponential-tail SCC results

summary/summary.csv contains accuracy means, sample SDs (ddof=1), and SEs.
summary/runtime.csv contains the computation-time components and totals.
summary/all_runs.jsonl has one full method record per completed paired dataset.
summary/run_records.jsonl retains the paired completion and artifact records.
summary/paired_differences.csv retains CINDES-minus-DSM differences by seed.
The two LaTeX tables use the same small font and show the actual replication count.

The default archive includes data/source fingerprints, model-selection metadata,
CV histories, environment records, logs, and source snapshots. Training arrays,
generated samples, and fitted model checkpoints remain in the SCC output folder
unless the optional raw-artifact switches were used. Keep the complete output
folder for later diagnostic analyses. The bundle manifest records those switches.

To regenerate all numeric tables from this archive, after extracting it run:

python3 analysis_code/summarize_results.py --from-export summary/export.json --output regenerated_tables

This regeneration uses saved metric records. It does not recheck omitted raw
samples or refit a model; the original summary records its validation scope.

Generation shards are not independent training replications. LLgap is signed,
not a divergence. Timings sum the explicitly recorded components and are not
queue waiting times or the elapsed time of the whole parallel array. Hardware
and attempt records identify the devices used in each replication.

If the status is partial, this archive includes only completed pairs and the
tables visibly identify the incomplete study. A smoke-profile archive contains
workflow checks only and must not be used for scientific conclusions.
"""
    receipt = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "profile": profile, "status": summary["status"], "study_identity_sha256": manifest["identity_sha256"],
               "completed_pairs": summary["completed_pairs"], "expected_pairs": summary["expected_pairs"],
               "include_samples": include_samples, "include_checkpoints": include_checkpoints,
               "summary_sha256": sha256(root / "summary" / "summary.json"),
               "bundle_source_sha256": sha256(Path(__file__)), "files": identities,
               "verification_scope": "Archive contents, recorded summary outputs, completed-run identities, and source snapshot checked. Raw artifacts not included retain their original strict-summary verification record."}
    generated = {"READ_ME.txt": notes.encode(), "bundle_manifest.json": (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()}
    destination = root / "summary" / "results_for_tables.zip"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="results_for_tables_", suffix=".partial", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for name, path in sorted(files.items()):
                kind = zipfile.ZIP_STORED if path.suffix in (".npz", ".zip", ".gz") else zipfile.ZIP_DEFLATED
                archive.write(path, arcname=name, compress_type=kind)
            for name, data in generated.items():
                archive.writestr(name, data)
        with zipfile.ZipFile(temporary) as archive:
            require(archive.testzip() is None, "ZIP CRC verification failed")
            require(len(archive.namelist()) == len(set(archive.namelist())), "Duplicate archive paths")
            for name, item in identities.items():
                value = hashlib.sha256()
                with archive.open(name) as handle:
                    for part in iter(lambda: handle.read(1024 * 1024), b""):
                        value.update(part)
                require(value.hexdigest() == item["sha256"] and archive.getinfo(name).file_size == item["bytes"],
                        "Archived content differs from the recorded source: " + name)
        for name, path in files.items():
            require(sha256(path) == identities[name]["sha256"], "Source changed during packaging: " + name)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    result = {"status": summary["status"], "profile": profile, "completed_pairs": summary["completed_pairs"],
              "expected_pairs": summary["expected_pairs"], "file": str(destination), "bytes": destination.stat().st_size,
              "sha256": sha256(destination), "crc_verified": True, "archived_hashes_verified": True,
              "include_samples": include_samples, "include_checkpoints": include_checkpoints}
    common.write_json(root / "summary" / "bundle_receipt.json", result)
    print(json.dumps(result, indent=2))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--profile", choices=("n1000", "full", "smoke"), default="n1000")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--include-samples", action="store_true")
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()
    root = resolve_root(args.root, args.profile)
    try:
        with locked(root / ".return_bundle.lock"):
            make_bundle(root, args.profile, args.allow_partial, args.include_samples, args.include_checkpoints)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        import sys
        print("Bundle failed: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
