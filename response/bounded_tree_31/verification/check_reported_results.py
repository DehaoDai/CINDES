#!/usr/bin/env python3
"""Check the included archived records and reproduce all reported statistics."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from run_experiment import source_hashes
from study_config import expected_experiment_config, expected_pairs
from summarize_results import load_results, summarize


def main():
    root = PACKAGE / "reported_results"
    provenance = json.loads((root / "provenance.json").read_text())
    pairs = expected_pairs("full")
    assert pairs == [(n, seed) for n in (500, 1000, 1500, 2000) for seed in range(20)]
    assert len(provenance["records"]) == len(pairs) == 80
    assert provenance["experiment_config"] == expected_experiment_config("full")
    assert provenance["packaged_scientific_source_sha256"] == source_hashes()
    for item in provenance["records"]:
        data = (root / item["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]
    rows, validation = load_results(root, "full")
    assert validation["complete_pairs"] == validation["expected_pairs"] == 80
    assert validation["observed_method_results"] == validation["included_method_results"] == 160
    assert validation["status"] == "complete"
    stats = summarize(rows)
    computed = {(r["n"], r["method"], r["metric"]): r for r in stats}
    with (root / "reference_summary.csv").open(newline="") as stream:
        reference = list(csv.DictReader(stream))
    assert len(reference) == len(computed) == 56
    discrepancies = []
    for saved in reference:
        actual = computed[int(saved["n"]), saved["method"], saved["metric"]]
        assert int(saved["n_rep"]) == actual["n_rep"] == 20
        for field in ("mean", "sd", "se"):
            difference = abs(actual[field] - float(saved[field]))
            assert math.isclose(actual[field], float(saved[field]), rel_tol=1e-12, abs_tol=1e-12)
            discrepancies.append(difference)
    report = {
        "status": "passed",
        "scope": "Included real archived result records and aggregation; no fitting or sample generation",
        "sample_sizes": [500, 1000, 1500, 2000],
        "complete_pairs": 80, "method_records": 160,
        "reproduced_summary_rows": 56, "sample_sd_denominator": 19,
        "maximum_absolute_difference_from_reference": max(discrepancies),
        "records_excluded": 0,
        "checks": [
            "All 80 archived per-pair file SHA256 digests agree with provenance",
            "Exact profile, source hashes, parameter counts and update counts match",
            "Every sample size has 20 paired seeds 0-19 and both methods",
            "All 56 means, sample SDs and SEs match the verified summaries",
        ],
        "limitations": "Archived results do not include generated arrays or trained models. These checks recompute statistics from saved metrics, not individual metrics from observations.",
    }
    (PACKAGE / "verification/reported_results_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
