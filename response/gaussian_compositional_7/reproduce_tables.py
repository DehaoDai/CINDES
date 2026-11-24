#!/usr/bin/env python3
"""Rebuild the two reported tables using only the Python standard library.

By default this reads the supplied per-replication results. Use --root to read a
new complete full-study output tree. No model fitting or sample generation is
performed here. The table metrics were computed by experiment/evaluate.py;
this script aggregates those saved measurements across independent datasets.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from collections import Counter


HERE = Path(__file__).resolve().parent
METHODS = ("cindes", "dsm")
SAMPLE_SIZES = (500, 1000, 1500, 2000)
METRICS = ("tv1", "tv2", "llgap", "h_tv", "h_w1")
TIMES = ("final_fit_seconds", "generation_seconds",
         "fit_plus_generation_seconds", "summed_cv_fit_seconds")
FIELDS = METRICS + TIMES
TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reject_constant(value):
    raise ValueError("Non-finite JSON constant: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject_constant)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def number(value, description, nonnegative=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), "Invalid number: " + description)
    require(not nonnegative or value >= 0, "Negative value: " + description)
    return float(value)


def load_results(root):
    """Validate the frozen full-study design and return all paired outcomes."""
    manifest_path = root / "study_manifest.json"
    manifest = read_json(manifest_path)
    reported_manifest = read_json(HERE / "reported_results/study_manifest.json")
    require(manifest["config"] == reported_manifest["config"],
            "This command requires the complete, unchanged full-study protocol. "
            "Use experiment/summarize_results.py for a different profile.")
    require(manifest["source_sha256"] == reported_manifest["source_sha256"],
            "Experimental source hashes differ from the source used for the tables.")
    expected = [{"task_id": i * 20 + r, "n": n, "replication": r,
                 "seed": 1000 + i * 20 + r - 1}
                for i, n in enumerate(SAMPLE_SIZES) for r in range(1, 21)]
    require(manifest["tasks"] == expected, "Unexpected task roster or seeds.")
    core = {k: manifest[k] for k in ("config", "tasks", "source_sha256")}
    require(digest(core) == manifest["study_id"], "Manifest identity is invalid.")

    provenance_path = root / "provenance.json"
    if provenance_path.is_file():
        provenance = read_json(provenance_path)
        for name, expected_hash in provenance.get("projection_sha256", {}).items():
            path = root / name
            require(path.is_file() and sha256(path) == expected_hash,
                    "Supplied-result integrity check failed: " + name)

    paths = sorted(root.glob("runs/n_*/rep_*/run_complete.json"))
    require(len(paths) == 80,
            f"Expected 80 completed paired replications; found {len(paths)}. "
            "Finish the full study before constructing these tables.")
    expected_by_id = {x["task_id"]: x for x in expected}
    seen, paired, rows, input_hashes = set(), [], [], {}
    reference_hashes, dgp_ids = set(), set()
    input_hashes["study_manifest.json"] = sha256(manifest_path)
    for path in paths:
        run = read_json(path)
        task = run["task_id"]
        require(task in expected_by_id and task not in seen,
                "Unexpected or duplicate task: " + str(task))
        seen.add(task)
        spec = expected_by_id[task]
        require(all(run[k] == v for k, v in spec.items()),
                "Task identity mismatch: " + str(task))
        require(run["status"] == "complete" and run["profile"] == "full",
                "Incomplete or wrong-profile task: " + str(task))
        require(run["study_id"] == manifest["study_id"], "Mixed study identities.")
        require(set(run["methods"]) == set(METHODS), "A paired method is missing.")
        require(run["reference_n"] == 1048576, "Incorrect reference sample size.")
        reference_hashes.add(run["reference_bank_sha256"])
        require(run["data_record"]["shape"] == [spec["n"], 7],
                "Observed data shape differs from its task.")
        require(run["data_record"]["status"] == "complete", "Incomplete data record.")
        require(all(run["data_record"]["dgp_config"][key] == value
                    for key, value in manifest["config"]["dgp"].items()),
                "The recorded DGP parameters disagree with the frozen design.")
        dgp_ids.add(digest(run["data_record"]["dgp_config"]))
        pair_signatures = []
        for method in METHODS:
            result = run["methods"][method]
            require(all(result[k] == v for k, v in spec.items()),
                    "Method and task identities disagree.")
            require(result["method"] == method and result["profile"] == "full"
                    and result["study_id"] == manifest["study_id"],
                    "Mixed method, profile, or study identity.")
            require(result["d"] == 7 and result["n_gen"] == 10000,
                    "Incorrect dimension or generation sample size.")
            model = result["model_record"]
            generated = result["generation_record"]
            require(model["training_data_sha256"] == run["data_record"]["pool_sha256"],
                    "The methods were not fitted to the recorded common dataset.")
            require(generated["n_samples"] == 10000 and generated["n_shards"] == 10,
                    "Incomplete generation record.")
            require(generated["runtime_signature"]["generation_device"] == "cuda",
                    "The full-study generation protocol requires CUDA.")
            require(result["selected_lambda"] in manifest["config"]["lambda_grid"],
                    "Penalty not in the frozen cross-validation grid.")
            require(result["updates"] == model["updates"]
                    == run["selection"][method]["full_stop_updates"],
                    "Selected and refitted training durations disagree.")
            require(result["selected_lambda"] == model["selected_lambda"]
                    == run["selection"][method]["lambda"],
                    "Selected and refitted penalties disagree.")
            row = {**spec, "method": method}
            for key in METRICS:
                row[key] = number(result[key], f"{task}/{method}/{key}", key != "llgap")
                if key in ("tv1", "tv2", "h_tv"):
                    require(row[key] <= 1, "Histogram TV exceeds one.")
            row["final_fit_seconds"] = number(result["training_seconds"], "refit time", True)
            row["generation_seconds"] = number(result["generation_seconds"], "generation time", True)
            row["fit_plus_generation_seconds"] = row["final_fit_seconds"] + row["generation_seconds"]
            row["summed_cv_fit_seconds"] = number(model["cv_sum_fit_wall_seconds"], "CV sum", True)
            require(math.isclose(row["final_fit_seconds"], model["training_seconds"],
                                 rel_tol=TOLERANCE, abs_tol=TOLERANCE), "Refit timers disagree.")
            require(math.isclose(row["generation_seconds"], generated["generation_seconds"],
                                 rel_tol=TOLERANCE, abs_tol=TOLERANCE), "Generation timers disagree.")
            row.update({"selected_lambda": result["selected_lambda"],
                        "selected_check": model["selected_check"],
                        "updates": result["updates"], "n_params": result["n_params"],
                        "positive_occurrences": result["positive_occurrences"],
                        "selected_positive_epochs": result["positive_occurrences"] / spec["n"],
                        "gpu": generated["runtime_signature"].get("gpu", "CPU"),
                        "generation_device": generated["runtime_signature"]["generation_device"]})
            pair_signatures.append(generated["runtime_signature"])
            rows.append(row)
        require(pair_signatures[0] == pair_signatures[1],
                "The two methods have different generation hardware or settings within a pair.")
        paired.append(run)
        input_hashes[path.relative_to(root).as_posix()] = sha256(path)
    require(seen == set(expected_by_id), "Missing paired replications.")
    require(len(reference_hashes) == 1, "The reference bank differs across replications.")
    require(len(dgp_ids) == 1, "The DGP differs across replications.")
    rows.sort(key=lambda r: (r["n"], r["replication"], METHODS.index(r["method"])))
    paired.sort(key=lambda r: r["task_id"])
    return manifest, paired, rows, input_hashes


def describe(values):
    require(len(values) == 20, "Each summary must contain exactly 20 replications.")
    sd = statistics.stdev(values)
    return {"n_rep": len(values), "mean": statistics.fmean(values), "sd": sd,
            "se": sd / math.sqrt(len(values))}


def aggregate(rows):
    summaries = []
    for n in SAMPLE_SIZES:
        for method in METHODS:
            group = [r for r in rows if r["n"] == n and r["method"] == method]
            for metric in FIELDS:
                summaries.append({"n": n, "method": method, "metric": metric,
                                  **describe([r[metric] for r in group])})
    return summaries


def compare_expected(summaries):
    reference = read_json(HERE / "expected_tables/reported_statistics.json")
    lookup = {(r["n"], r["method"], r["metric"]): r for r in summaries}
    checked = 0
    for expected in reference["statistics"]:
        key = (expected["n"], expected["method"], expected["metric"])
        actual = lookup[key]
        require(actual["n_rep"] == expected["n_rep"], "Replication count changed.")
        for field in ("mean", "sd", "se"):
            require(math.isclose(actual[field], expected[field], rel_tol=TOLERANCE,
                                 abs_tol=TOLERANCE),
                    f"Published-result check failed for {key}/{field}: "
                    f"{actual[field]} versus {expected[field]}")
            checked += 1
        digits = 4 if expected["metric"] in METRICS else 2
        for field in ("mean", "sd"):
            require(f"{actual[field]:.{digits}f}" == f"{expected[field]:.{digits}f}",
                    "A displayed table value differs from the reported table.")
    require(len(reference["statistics"]) == len(summaries), "Expected-statistic count differs.")
    return {"passed": True, "scalars_checked": checked,
            "relative_tolerance": TOLERANCE, "absolute_tolerance": TOLERANCE,
            "displayed_mean_and_sd_cells_checked": 2 * len(summaries)}


def write_csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summary_lookup(summaries):
    return {(r["n"], r["method"], r["metric"]): r for r in summaries}


def is_better(metric, mean_a, mean_b):
    # LLgap is signed. A more negative value is not automatically better.
    return abs(mean_a) < abs(mean_b) if metric == "llgap" else mean_a < mean_b


def latex_cell(entry, digits, bold):
    value = f"{entry['mean']:.{digits}f} \\pm {entry['sd']:.{digits}f}"
    return "$\\boldsymbol{" + value + "}$" if bold else "$" + value + "$"


def latex_table(summaries, runtime=False, cv_workers=4, gpu_types_varied=True):
    lookup = summary_lookup(summaries)
    metrics = TIMES if runtime else METRICS
    label = "tab:cindes_runtime_full" if runtime else "tab:cindes_accuracy_full"
    caption = (
        "Computation times in seconds for the unbounded compositional experiment. "
        "Entries are the mean $\\pm$ sample standard deviation over 20 paired replications. "
        "Boldface marks a lower mean computation time for CINDES than for DSM at the same sample size."
        if runtime else
        "Comparison of CINDES and DSM for the unbounded compositional distribution in dimension $d=7$. "
        "Entries are the mean $\\pm$ sample standard deviation over 20 paired replications, "
        "with 10,000 generated observations per fitted model. Boldface marks a lower mean "
        "discrepancy for CINDES (a mean closer to zero for the signed LLgap). "
        "Boldface does not indicate statistical significance."
    )
    headings = (
        r"$n$ & Method & \shortstack{Final refit\\(CPU)} & \shortstack{Generation\\(GPU)} & \shortstack{Refit $+$\\generation} & \shortstack{Cross-validation\\(summed fit times)} \\"
        if runtime else r"$n$ & Method & TV1 & TV2 & LLgap & $H$-TV & $H$-$W_1$ \\"
    )
    notes = (
        "Final refit is the recorded elapsed time for fitting the selected model on all $n$ observations, "
        "excluding cross-validation. Generation is the recorded elapsed time for producing 10,000 observations. "
        "Refit plus generation is calculated within each replication before computing its mean and standard deviation. "
        "The cross-validation column summarizes the recorded sum of the 15 individual fit times for each method, "
        "including validation. Up to four fits were run simultaneously on CPUs, so this is cumulative fitting time, "
        "rather than the elapsed duration of the parallel cross-validation stage. Both methods generated samples "
        "using the same GPU within each replication; GPU hardware varied across replications."
        if runtime else
        "TV1 and TV2 are histogram total variation discrepancies averaged over the seven coordinates and the "
        "six generating pairs, respectively. $H$-TV is the histogram total variation discrepancy between the "
        "generated and target-reference distributions of $H(X)$, and $H$-$W_1$ is their one-dimensional "
        "Wasserstein-1 distance computed from the unbinned $H$-values. LLgap is the signed reference-minus-generated "
        "mean log-potential gap, not a divergence. "
        "The common independent target-reference sample contains 1,048,576 observations."
    )
    if runtime:
        if cv_workers != 4:
            notes = notes.replace("Up to four fits", f"Up to {cv_workers} fits")
        if not gpu_types_varied:
            notes = notes.replace("GPU hardware varied across replications.",
                                  "the same GPU type was used across replications.")
    out = [r"% Requires \usepackage{amsmath,booktabs}.", r"\begin{table}[htbp]",
           r"\centering", "\\caption{" + caption + "}", "\\label{" + label + "}",
           r"\small", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.15}",
           "\\begin{tabular}{rl" + "c" * len(metrics) + "}", r"\toprule", headings, r"\midrule"]
    for i, n in enumerate(SAMPLE_SIZES):
        if i:
            out.append(r"\midrule")
        for method in METHODS:
            cells = [str(n) if method == "cindes" else "", method.upper()]
            for metric in metrics:
                entry = lookup[n, method, metric]
                bold = method == "cindes" and is_better(metric, entry["mean"], lookup[n, "dsm", metric]["mean"])
                cells.append(latex_cell(entry, 2 if runtime else 4, bold))
            out.append(" & ".join(cells) + r" \\")
    out.extend([r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
                r"\begin{minipage}{\textwidth}", r"\footnotesize", r"\textit{Notes.} " + notes,
                r"\end{minipage}", r"\end{table}", ""])
    return "\n".join(out)


def plain_tables(summaries):
    lookup = summary_lookup(summaries)
    output = []
    for title, metrics, digits in (("Accuracy (mean +/- sample SD)", METRICS, 4),
                                   ("Runtime in seconds (mean +/- sample SD)", TIMES, 2)):
        output.extend([title, "n,method," + ",".join(metrics)])
        for n in SAMPLE_SIZES:
            for method in METHODS:
                cells = [str(n), method]
                for metric in metrics:
                    entry = lookup[n, method, metric]
                    cells.append(f"{entry['mean']:.{digits}f} +/- {entry['sd']:.{digits}f}")
                output.append(",".join(cells))
        output.append("")
    return "\n".join(output)


def paired_summaries(rows):
    lookup = {(r["n"], r["replication"], r["method"]): r for r in rows}
    output = []
    for n in SAMPLE_SIZES:
        for metric in METRICS:
            values, preferred, ties = [], 0, 0
            for replication in range(1, 21):
                a = lookup[n, replication, "cindes"][metric]
                b = lookup[n, replication, "dsm"][metric]
                values.append(a - b)
                preferred += int(is_better(metric, a, b))
                ties += int(abs(a) == abs(b) if metric == "llgap" else a == b)
            output.append({"n": n, "metric": metric, "difference": "cindes_minus_dsm",
                           **describe(values), "cindes_preferred_count": preferred,
                           "ties": ties,
                           "preference_rule": "closer_to_zero" if metric == "llgap" else "lower"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "reported_results",
                        help="Complete study result directory (default: bundled reported_results).")
    parser.add_argument("--output", type=Path, default=HERE / "tables",
                        help="Output directory (default: tables beside this script).")
    parser.add_argument("--check", action="store_true",
                        help="Also compare with the reported mean/SD/SE and displayed table values.")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    manifest, paired, rows, hashes = load_results(root)
    summaries = aggregate(rows)
    check_result = compare_expected(summaries) if args.check else {"requested": False}
    output.mkdir(parents=True, exist_ok=True)
    columns = ["n", "method", "metric", "n_rep", "mean", "sd", "se"]
    write_csv(output / "accuracy_summary.csv", [r for r in summaries if r["metric"] in METRICS], columns)
    write_csv(output / "runtime_summary.csv", [r for r in summaries if r["metric"] in TIMES], columns)
    write_csv(output / "per_run.csv", rows, list(rows[0]))
    differences = paired_summaries(rows)
    write_csv(output / "paired_differences.csv", differences, list(differences[0]))
    cv_workers = max(r["timing"]["cv"]["workers"] for r in paired)
    gpu_types_varied = len({r["gpu"] for r in rows}) > 1
    for name, runtime in (("accuracy_table.tex", False), ("runtime_table.tex", True)):
        (output / name).write_text(latex_table(summaries, runtime, cv_workers, gpu_types_varied),
                                  encoding="utf-8")
    (output / "tables.tex").write_text(
        "\\documentclass[10pt]{article}\n\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{amsmath,booktabs}\n\\begin{document}\n"
        "\\input{accuracy_table.tex}\n\\clearpage\n"
        "\\input{runtime_table.tex}\n\\end{document}\n", encoding="utf-8")
    readable = plain_tables(summaries)
    (output / "tables.txt").write_text(readable, encoding="utf-8")
    write_json(output / "statistics.json", {"study_id": manifest["study_id"], "statistics": summaries})
    report = {"status": "passed", "paired_replications": len(paired), "method_results": len(rows),
              "sample_standard_deviation_ddof": 1,
              "all_individual_llgaps_positive": all(r["llgap"] > 0 for r in rows),
              "gpu_allocations_per_pair": dict(Counter(r["methods"]["cindes"]["generation_record"]
                                                        ["runtime_signature"].get("gpu", "CPU") for r in paired)),
              "reported_result_check": check_result, "input_sha256": hashes,
              "scope": "Summary statistics recomputed from recorded per-replication metrics and timers. "
                       "This does not rerun metric evaluation from sample arrays or retrain models.",
              "cv_timing_scope": "Each per-run CV total was recorded by the original experiment as the sum "
                                 "of 15 fit elapsed times. The compact reported results retain these totals; "
                                 "individual CV worker timers are not included."}
    write_json(output / "validation_report.json", report)
    print("Validated 80 complete paired replications (160 method outcomes).")
    if args.check:
        print(f"Reported-table check passed: {check_result['scalars_checked']} mean/SD/SE values "
              f"at relative and absolute tolerance {TOLERANCE:g}; all 144 displayed mean/SD values match.")
    print(readable)
    print("Wrote LaTeX tables, CSV summaries, and validation report to " + str(output))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print("Table reproduction failed: " + str(exc), file=sys.stderr)
        sys.exit(1)
