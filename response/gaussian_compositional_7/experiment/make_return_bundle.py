"""Create a compact, verified archive for later tables; keep raw data on SCC."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid
import zipfile

import study


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_bundle(root, profile):
    root = Path(root).resolve()
    manifest = study.load_manifest(root, profile)
    summary_path = root / "summary" / "summary.json"
    require(summary_path.is_file(), "Run the strict summarizer successfully first")
    summary = study.read_json(summary_path)
    expected_paths = {
        str((study.run_dir(root, spec) / "run_complete.json").relative_to(root))
        for spec in manifest["tasks"]
    }
    require(summary["status"] == "complete" and summary["profile"] == profile
            and summary["study_id"] == manifest["study_id"], "Summary identity or completion mismatch")
    require(summary["n_complete_pairs"] == summary["n_expected_pairs"] == len(expected_paths),
            "Summary is not a complete study")
    records = summary["run_record_sha256"]
    require(set(records) == expected_paths, "Summary has missing or unexpected run records")
    for name, expected_hash in records.items():
        require((root / name).is_file() and study.sha(root / name) == expected_hash,
                f"Run record changed after strict summarization: {name}")
    reference_path = root / "reference" / "reference_complete.json"
    require(study.sha(reference_path) == summary["reference_complete_sha256"],
            "Reference record changed after strict summarization")
    reference = study.read_json(reference_path)
    reference_summary = root / "reference" / "reference_summary.json"
    require(study.sha(reference_summary) == reference["files_sha256"]["reference/reference_summary.json"],
            "Reference summary hash mismatch")

    required = [root / "study_manifest.json", root / "task_manifest.csv",
                reference_path, reference_summary,
                root / "source_snapshot" / "snapshot_manifest.json"]
    required.extend(root / "summary" / name for name in (
        "per_run.csv", "summary.csv", "paired_differences.csv", "timing.csv",
        "summary_all_runs.jsonl", "summary.json", "primary_table.tex", "H_table.tex"))
    for path in required:
        require(path.is_file(), f"Missing expected export: {path.relative_to(root)}")
    files = {path.relative_to(root): path for path in required}
    for path in (root / "summary").iterdir():
        if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".tex", ".txt"):
            files[path.relative_to(root)] = path
    for spec in manifest["tasks"]:
        for path in study.run_dir(root, spec).glob("*.json"):
            files[path.relative_to(root)] = path

    receipt = {
        "profile": profile, "study_id": manifest["study_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_paired_replications": len(expected_paths),
        "strict_summary_verified": True,
        "verification_scope": "Complete-run and reference records match the preceding strict summary; raw arrays were verified by the summarizer and remain on SCC.",
        "files_sha256": {str(name): study.sha(path) for name, path in sorted(files.items())},
    }
    notes = """CINDES versus DSM: compact results for later tables

This archive contains the complete validated summary, per-run metrics, paired
comparisons, timings, selected models' metadata, study configuration, seed map,
reference calibration summary, and source fingerprints. The primary table is
summary/primary_table.tex; its entries are mean +/- sample SD, not standard error.

Raw training data, fold assignments, model checkpoints, CV histories/trajectories,
generated arrays and shards, reference arrays, detailed logs, and interrupted
attempts remain in the original SCC output directory. Keep that directory for
subsequent analyses. This compact archive is sufficient to recreate the reported
tables but is not a replacement for all raw outputs.

summary/summary_all_runs.jsonl has one record per paired replication, with both
methods and the hashes of their evidence files. generation shards are not extra
replications. LLgap is signed; it is not a KL divergence. Target-reference sampling
uncertainty and target-versus-target baselines are reported separately and are
never subtracted from the reported method metrics.

If profile is 'smoke', these are workflow checks only, not scientific results.
"""
    destination = root / "summary" / "results_for_tables.zip"
    temporary = destination.with_name(destination.name + ".tmp." + uuid.uuid4().hex)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in sorted(files.items()):
                archive.write(path, arcname=str(name))
            archive.writestr("READ_ME.txt", notes)
            archive.writestr("bundle_manifest.json", json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            require(archive.testzip() is None, "Archive CRC verification failed")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"status": "complete", "profile": profile,
                      "paired_replications": len(expected_paths), "file": str(destination),
                      "bytes": destination.stat().st_size, "sha256": study.sha(destination)}, indent=2))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=study.DEFAULT_ROOT)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    args = parser.parse_args()
    with study.locked(args.root / ".return_bundle.lock"):
        make_bundle(args.root, args.profile)


if __name__ == "__main__":
    main()
