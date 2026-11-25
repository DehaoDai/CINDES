"""Verify the retained protocol and reporting with explicitly synthetic rows.

This check never trains models, imports PyTorch, or contacts SCC. Temporary
metric values exercise aggregation only and are not experimental results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import run_experiment
import study_config
import summarize_results


def main():
    baseline = json.loads((PACKAGE / "verification/historical_protocol.json").read_text())
    configuration = study_config.expected_experiment_config("full")
    assert configuration == baseline["experiment_config"]
    hashes = run_experiment.source_hashes()
    assert hashes == baseline["original_scc_source_sha256"]
    pairs = [(n, seed) for n in (500, 1000, 1500, 2000) for seed in range(20)]
    assert study_config.expected_pairs("full") == pairs
    assert study_config.default_root("full") == "outputs/bounded_n500_2000_v1"
    assert study_config.default_root("smoke") == "outputs/bounded_n500_2000_smoke_v1"
    checks = ["Historical scientific configuration and original SCC source hashes match",
              "All 80 sample-size/seed assignments and separate output roots match"]
    with tempfile.TemporaryDirectory(prefix="bounded_reporting_check_") as directory:
        root = Path(directory)
        updates = {500: 19998, 1000: 20004, 1500: 19998, 2000: 19992}
        for n, seed in pairs:
            folder = root / "runs" / f"n_{n:04d}" / f"seed_{seed:02d}"
            folder.mkdir(parents=True)
            records = []
            for method, offset in (("cindes", 0.1), ("dsm", 0.2)):
                row = {"n": n, "seed": seed, "d": 31, "graph": "btree",
                       "method": method, "experiment_config": configuration,
                       f"{method}_params": 37249 if method == "cindes" else 43167,
                       f"{method}_updates": updates[n] if method == "cindes" else 60000}
                for metric in summarize_results.METRICS:
                    row[f"{method}_{metric}"] = offset + seed / 10000
                records.append(row)
            (folder / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        rows, report = summarize_results.load_results(root, "full")
        assert report["status"] == "complete" and report["complete_pairs"] == 80
        assert len(rows) == 160
        statistics, destination = summarize_results.write_outputs(root, rows, report)
        result = next(row for row in statistics if row["n"] == 500 and row["method"] == "cindes" and row["metric"] == "tv1")
        assert abs(result["mean"] - 0.10095) < 1e-12
        assert abs(result["sd"] - math.sqrt(35) / 10000) < 1e-12
        assert result["n_rep"] == 20
        with (destination / "summary.csv").open() as handle:
            summary_rows = list(csv.DictReader(handle))
        assert len(summary_rows) == 56
        assert {int(row["n"]) for row in summary_rows} == {500, 1000, 1500, 2000}
        assert "tab:cindes_bounded_tree" in (destination / "accuracy_table.tex").read_text()
        assert "tab:cindes_bounded_runtime" in (destination / "runtime_table.tex").read_text()
        checks.append("160 synthetic records aggregate to four sizes with correct mean and sample SD")

        complete = subprocess.run([sys.executable, str(PACKAGE / "run_experiment.py"),
                                   "--profile", "full", "--root", str(root),
                                   "--n", "500", "--seed", "0", "--require-cuda"],
                                  text=True, capture_output=True)
        assert complete.returncode == 0, complete.stderr
        assert "Already complete: n=500, seed=0" in complete.stdout
        checks.append("A validated completed pair is reused before importing the GPU environment")

        (root / "submissions").mkdir()
        (root / "logs").mkdir()
        (root / "submissions/submission_fixture.array.qsub").write_text("# Synthetic reporting fixture\n#$ -t 1-80\n#$ -tc 2\n")
        (root / "logs/cindes_bnd.o123.1").write_text("Synthetic scheduler log\n")
        bundle = subprocess.run([sys.executable, str(PACKAGE / "make_results_bundle.py"),
                                 "--root", str(root), "--profile", "full"], text=True, capture_output=True)
        assert bundle.returncode == 0, bundle.stderr
        with zipfile.ZipFile(destination / "results_for_tables.zip") as archive:
            assert archive.testzip() is None
            assert "bounded_results/submissions/submission_fixture.array.qsub" in archive.namelist()
            assert "bounded_results/logs/cindes_bnd.o123.1" in archive.namelist()
            manifest = json.loads(archive.read("bounded_results/BUNDLE_MANIFEST.json"))
            assert manifest["complete_pairs"] == manifest["expected_pairs"] == 80
        checks.append("The strict results bundle includes validated records, tables, and scheduler provenance")

        target = root / "runs/n_2000/seed_19/results.jsonl"
        original = target.read_text()
        target.write_text(original.splitlines()[0] + "\n")
        missing = subprocess.run([sys.executable, str(PACKAGE / "summarize_results.py"),
                                  "--root", str(root), "--profile", "full"], text=True, capture_output=True)
        assert missing.returncode != 0
        target.write_text(original)
        target.write_text(original + original.splitlines()[0] + "\n")
        duplicate = subprocess.run([sys.executable, str(PACKAGE / "summarize_results.py"),
                                    "--root", str(root), "--profile", "full"], text=True, capture_output=True)
        assert duplicate.returncode != 0
        checks.append("Strict aggregation rejects missing method results and duplicate records")

    report = {"status": "passed", "scope": "Protocol comparison and synthetic reporting checks; no model training or SCC submission",
              "checks": checks, "original_scc_source_sha256": hashes,
              "reporting_source_sha256": {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
                                          for name in ("study_config.py", "run_experiment.py", "run_original.py",
                                                       "summarize_results.py", "make_results_bundle.py")}}
    (PACKAGE / "verification/protocol_and_reporting_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
