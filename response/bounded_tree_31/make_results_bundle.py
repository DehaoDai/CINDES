#!/usr/bin/env python3
"""Export validated bounded-study records, metadata, and tables to a ZIP.

Generated sample arrays are optional; fitted models are not saved by the
original experiment. By default this creates summary/results_for_tables.zip.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

from study_config import default_root
from summarize_results import ValidationError, load_results, print_status, write_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, help="destination ZIP; default ROOT/summary/results_for_tables.zip")
    parser.add_argument("--allow-incomplete", action="store_true", help="allow an explicitly partial result bundle")
    parser.add_argument("--include-samples", action="store_true", help="also include saved generated/reference NPZ arrays")
    args = parser.parse_args(argv)
    root = (args.root or Path(default_root(args.profile))).resolve()
    destination = (args.out or root / "summary" / "results_for_tables.zip").resolve()
    temp_path = None
    try:
        rows, report = load_results(root, args.profile)
        print_status(report)
        if report["status"] != "complete" and not args.allow_incomplete:
            raise ValidationError("the study is incomplete; use --allow-incomplete only for an explicitly partial bundle")
        if not rows:
            raise ValidationError("there are no complete pairs to export")
        write_outputs(root, rows, report)
        files = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or any(part.startswith(".") for part in relative.parts):
                continue
            if path.resolve() == destination:
                continue
            in_runs = relative.parts[0] == "runs"
            in_summary = relative.parts[0] == "summary"
            execution_record = relative.parts[0] in {"logs", "submissions"}
            top_metadata = len(relative.parts) == 1 and path.suffix == ".json"
            standard = path.suffix.lower() in {".json", ".jsonl", ".csv", ".tex", ".txt", ".md"}
            samples = args.include_samples and path.suffix.lower() == ".npz"
            if ((in_runs or in_summary or top_metadata) and (standard or samples)) or execution_record:
                files.append(path)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "profile": args.profile,
            "study_status": report["status"],
            "complete_pairs": report["complete_pairs"],
            "expected_pairs": report["expected_pairs"],
            "sample_arrays_included": args.include_samples,
            "scope": "Raw method results, run/study metadata, validated summaries, LaTeX tables, scheduler logs, and saved submission scripts.",
            "files": [],
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".bounded_bundle_", suffix=".zip", dir=destination.parent, delete=False) as temporary:
            temp_path = Path(temporary.name)
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                name = path.relative_to(root).as_posix()
                data = path.read_bytes()
                archive.writestr("bounded_results/" + name, data)
                manifest["files"].append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            archive.writestr("bounded_results/BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
            archive.writestr("bounded_results/README.txt", (
                "Bounded CINDES versus DSM results\n\n"
                f"Profile: {args.profile}\nStatus: {report['status']}\n"
                f"Complete paired replications: {report['complete_pairs']} / {report['expected_pairs']}\n\n"
                "The summary directory contains validated CSV tables and LaTeX tables.\n"
                "Raw method results and run metadata are in runs/. Copy this directory\n"
                "beside the reproducibility code and rerun:\n"
                f"  python summarize_results.py --root bounded_results --profile {args.profile}\n\n"
                "For a partial bundle, add --allow-incomplete. Only complete paired\n"
                "replications enter partial tables. Summary SD is the sample SD\n"
                "across replications, not the standard error. Runtime includes\n"
                "training and generation and excludes data sampling and evaluation.\n"
                "The log-likelihood gap is signed and is not a KL divergence.\n"
                "BUNDLE_MANIFEST.json lists each included file and its SHA-256 digest.\n"
            ))
        temp_path.replace(destination)
        print(f"Wrote {destination} ({destination.stat().st_size:,} bytes; {len(files)} data/metadata files).")
        return 0
    except (ValidationError, OSError, zipfile.BadZipFile) as exc:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
