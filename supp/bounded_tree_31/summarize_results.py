#!/usr/bin/env python3
"""Validate paired bounded-tree results and rebuild CSV and LaTeX tables.

This script uses only the Python standard library. It never fits a model.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

from study_config import default_root, expected_experiment_config, expected_pairs


METHODS = ("cindes", "dsm")
METRICS = ("tv1", "tv2", "llgap", "energy", "sw", "clip", "time")


class ValidationError(ValueError):
    """An input is malformed or belongs to a different experiment."""


def reject_constant(value):
    raise ValidationError(f"non-finite JSON constant {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(text):
    value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
    def check_finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValidationError("non-finite number in JSON record")
        if isinstance(item, dict):
            for member in item.values():
                check_finite(member)
        elif isinstance(item, list):
            for member in item:
                check_finite(member)
    check_finite(value)
    return value


def numeric(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite number")
    return float(value)


def check_row(row, config, pairs):
    if not isinstance(row, dict):
        raise ValidationError("each JSONL record must be an object")
    for field in ("n", "seed", "d"):
        if type(row.get(field)) is not int:
            raise ValidationError(f"{field} must be an integer")
    pair = (row["n"], row["seed"])
    if pair not in pairs:
        raise ValidationError(f"unexpected (n, seed)={pair}")
    method = row.get("method")
    if method not in METHODS:
        raise ValidationError(f"unknown method {method!r}")
    if row["d"] != config["d"] or row.get("graph") != "btree":
        raise ValidationError("dimension or graph does not match the selected profile")
    saved = row.get("experiment_config")
    if saved != config:
        raise ValidationError("experiment_config does not match the selected profile")
    # Python considers False and 0 equal; reject that ambiguity for this flag.
    if type(saved.get("denoise_final")) is not bool:
        raise ValidationError("denoise_final must be a JSON boolean")
    for metric in METRICS:
        value = numeric(row.get(f"{method}_{metric}"), f"{method}_{metric}")
        if metric in ("tv1", "tv2", "clip") and not 0 <= value <= 1:
            raise ValidationError(f"{method}_{metric} must lie in [0, 1]")
        if metric in ("time", "sw") and value < 0:
            raise ValidationError(f"{method}_{metric} must be nonnegative")
    # The original training code rounds requested updates to complete epochs.
    batches = max(1, math.ceil((3 if method == "cindes" else 1) * row["n"] / 256))
    requested = config[f"{method}_steps_requested"]
    updates = max(1, round(requested / batches)) * batches
    params = 128 * config["d"] + 33281 if method == "cindes" else 257 * config["d"] + 35200
    for field, expected in (("updates", updates), ("params", params)):
        value = row.get(f"{method}_{field}")
        if type(value) is not int or value != expected:
            raise ValidationError(f"{method}_{field} must be {expected}, got {value!r}")
    return (*pair, method)


def load_results(root: Path, profile: str, inputs=None):
    """Read and validate all rows; missing rows are reported, not invented."""
    config = expected_experiment_config(profile)
    pairs = set(expected_pairs(profile))
    supplied_inputs = inputs is not None
    if inputs is None:
        inputs = sorted((root / "runs").glob("**/results.jsonl"))
    paths = [Path(path).resolve() for path in inputs]
    if len(paths) != len(set(paths)):
        raise ValidationError("the same input file was supplied more than once")
    rows, input_records = {}, []
    for path in paths:
        if not path.is_file():
            raise ValidationError(f"input file does not exist: {path}")
        raw = path.read_bytes()
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValidationError(f"{path}: not valid UTF-8") from exc
        count = 0
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = read_json(line)
                key = check_row(row, config, pairs)
                if key in rows:
                    raise ValidationError(f"duplicate result for n={key[0]}, seed={key[1]}, method={key[2]}")
                if not supplied_inputs:
                    expected_path = (root / "runs" / f"n_{key[0]:04d}" / f"seed_{key[1]:02d}" / "results.jsonl").resolve()
                    if path != expected_path:
                        raise ValidationError(f"result is stored under the wrong n/seed directory; expected {expected_path}")
                rows[key] = row
                count += 1
            except (ValueError, TypeError, KeyError) as exc:
                raise ValidationError(f"{path}, line {number}: {exc}") from exc
        try:
            label = str(path.relative_to(root.resolve()))
        except ValueError:
            label = str(path)
        input_records.append({"path": label, "sha256": hashlib.sha256(raw).hexdigest(), "records": count})
    complete_pairs = sorted(pair for pair in pairs if all((*pair, method) in rows for method in METHODS))
    missing = [{"n": n, "seed": seed, "method": method} for n, seed in sorted(pairs)
               for method in METHODS if (n, seed, method) not in rows]
    eligible = {key: row for key, row in rows.items() if key[:2] in set(complete_pairs)}
    excluded = [{"n": key[0], "seed": key[1], "method": key[2]} for key in sorted(rows) if key not in eligible]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "status": "complete" if not missing else "incomplete",
        "expected_pairs": len(pairs),
        "complete_pairs": len(complete_pairs),
        "expected_method_results": 2 * len(pairs),
        "observed_method_results": len(rows),
        "included_method_results": len(eligible),
        "missing_results": missing,
        "excluded_unpaired_results": excluded,
        "complete_pairs_by_n": {str(n): sum(pair[0] == n for pair in complete_pairs) for n in sorted({p[0] for p in pairs})},
        "experiment_config": config,
        "inputs": input_records,
        "aggregation": "paired replications only; sample SD uses ddof=1; SE = SD / sqrt(n_rep)",
        "runtime_definition": "training plus generation, excluding data sampling and metric evaluation",
        "llgap_definition": "reference minus generated mean analytic target log potential; signed, not a KL divergence",
    }
    return eligible, report


def summarize(rows):
    grouped = {}
    for (n, seed, method), row in sorted(rows.items()):
        for metric in METRICS:
            grouped.setdefault((n, method, metric), []).append(row[f"{method}_{metric}"])
    result = []
    for (n, method, metric), values in sorted(grouped.items(), key=lambda item: (item[0][0], METHODS.index(item[0][1]), METRICS.index(item[0][2]))):
        sd = statistics.stdev(values) if len(values) > 1 else None
        result.append({"n": n, "method": method, "metric": metric, "n_rep": len(values),
                       "mean": statistics.fmean(values), "sd": sd,
                       "se": sd / math.sqrt(len(values)) if sd is not None else None})
    return result


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def cell(stat, digits, bold=False):
    if stat is None:
        return r"$\text{--}$"
    value = f"{stat['mean']:.{digits}f}"
    spread = f"{stat['sd']:.{digits}f}" if stat["sd"] is not None else r"\text{--}"
    text = value + r" \pm " + spread
    return "$" + (r"\boldsymbol{" + text + "}" if bold else text) + "$"


def latex_tables(stats, report):
    lookup = {(s["n"], s["method"], s["metric"]): s for s in stats}
    sizes = sorted(int(n) for n in report["complete_pairs_by_n"])
    complete = report["status"] == "complete"
    if complete and report["profile"] == "full":
        description = "Mean $\\pm$ sample standard deviation over 20 paired replications at each sample size."
    elif report["profile"] == "smoke":
        description = "Reduced smoke test only; these values are not scientific results. A single replication does not define a sample standard deviation."
    else:
        description = "Partial results; only complete paired replications are included. Replication counts appear in the accompanying validation report; missing cells are shown as dashes."
    accuracy = [r"% Requires \usepackage{booktabs,amsmath}. Generated from validated per-replication results.",
                r"\begin{table}[htbp]", r"\centering", r"\small",
                r"\caption{Bounded binary-tree experiment. " + description + r" TV$_1$ and TV$_2$ are average histogram total variation errors for coordinate and parent--child marginals. The log-likelihood gap is signed. Boldface marks a smaller CINDES mean within a sample size and metric; it does not indicate a significance test.}",
                r"\label{tab:cindes_bounded_tree}", r"\begin{tabular}{rlccc}", r"\toprule",
                r"$n$ & Method & TV$_1$ & TV$_2$ & Log-likelihood gap \\", r"\midrule"]
    runtime = [r"% Requires \usepackage{booktabs,amsmath}. The font matches the accuracy table.",
               r"\begin{table}[htbp]", r"\centering", r"\small",
               r"\caption{Computation time for the bounded binary-tree experiment, in seconds. " + description + r" Each time includes model training and generation of the samples used for evaluation. Data sampling and metric evaluation are excluded.}",
               r"\label{tab:cindes_bounded_runtime}", r"\begin{tabular}{rlc}", r"\toprule",
               r"$n$ & Method & Training and generation time (s) \\", r"\midrule"]
    for index, n in enumerate(sizes):
        if index:
            accuracy.append(r"\addlinespace")
            runtime.append(r"\addlinespace")
        for method in METHODS:
            label = "CINDES" if method == "cindes" else "DSM"
            prefix = str(n) if method == "cindes" else ""
            values = []
            for metric in ("tv1", "tv2", "llgap"):
                stat = lookup.get((n, method, metric))
                other = lookup.get((n, "dsm", metric))
                bold = method == "cindes" and stat is not None and other is not None and stat["mean"] < other["mean"]
                values.append(cell(stat, 4, bold))
            accuracy.append(f"{prefix} & {label} & " + " & ".join(values) + r" \\")
            runtime.append(f"{prefix} & {label} & " + cell(lookup.get((n, method, "time")), 2) + r" \\")
    for table in (accuracy, runtime):
        table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(accuracy), "\n".join(runtime)


def atomic_text(path, text):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def write_outputs(root, rows, report):
    stats = summarize(rows)
    destination = root / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_text(destination / "summary.csv", csv_text(stats, ("n", "method", "metric", "n_rep", "mean", "sd", "se")))
    ordered_rows = [row for key, row in sorted(rows.items(), key=lambda item: (item[0][0], item[0][1], METHODS.index(item[0][2])))]
    atomic_text(destination / "summary_all_runs.jsonl", "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in ordered_rows))
    normalized = [{"n": row["n"], "seed": row["seed"], "method": row["method"],
                   **{metric: row[f"{row['method']}_{metric}"] for metric in (*METRICS, "params", "updates")}}
                  for row in ordered_rows]
    atomic_text(destination / "per_run.csv", csv_text(normalized, ("n", "seed", "method", *METRICS, "params", "updates")))
    differences = []
    for n, seed in sorted({key[:2] for key in rows}):
        for metric in METRICS:
            c = rows[n, seed, "cindes"][f"cindes_{metric}"]
            d = rows[n, seed, "dsm"][f"dsm_{metric}"]
            differences.append({"n": n, "seed": seed, "metric": metric, "cindes": c, "dsm": d, "cindes_minus_dsm": c - d})
    atomic_text(destination / "paired_differences.csv", csv_text(differences, ("n", "seed", "metric", "cindes", "dsm", "cindes_minus_dsm")))
    accuracy, runtime = latex_tables(stats, report)
    atomic_text(destination / "accuracy_table.tex", accuracy)
    atomic_text(destination / "runtime_table.tex", runtime)
    document = [
        r"\documentclass{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs,amsmath}",
        r"\begin{document}",
        r"\section*{Bounded CINDES versus DSM experiment}",
        f"These tables summarize {report['complete_pairs']} complete paired replications "
        f"({report['included_method_results']} method results) for the {report['profile']} profile.",
    ]
    if report["profile"] == "smoke":
        document.append("This is a reduced implementation check; its values must not be reported as scientific experimental results.")
    elif report["status"] != "complete":
        document.append("The study is incomplete. Only complete pairs are summarized, and unobserved cells are shown as dashes.")
    document.extend(["", accuracy, runtime, r"\end{document}", ""])
    atomic_text(destination / "tables.tex", "\n".join(document))
    atomic_text(destination / "validation_report.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    return stats, destination


def print_status(report):
    print(f"Validated {report['complete_pairs']} of {report['expected_pairs']} complete paired replications ({report['observed_method_results']} of {report['expected_method_results']} method results).")
    for n, count in report["complete_pairs_by_n"].items():
        print(f"  n={n}: {count} complete pairs")
    if report["excluded_unpaired_results"]:
        print(f"  {len(report['excluded_unpaired_results'])} individual method results await their paired method.")
    if report["missing_results"]:
        preview = ", ".join(f"n={x['n']}/seed={x['seed']}/{x['method']}" for x in report["missing_results"][:8])
        print("Missing: " + preview + (" ..." if len(report["missing_results"]) > 8 else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--root", type=Path, help="study output root; defaults to the selected profile's root")
    parser.add_argument("--input", type=Path, nargs="+", help="optional explicit original result JSONL files instead of root/runs")
    parser.add_argument("--status", action="store_true", help="validate existing results and show progress without writing tables")
    parser.add_argument("--allow-incomplete", action="store_true", help="explicitly generate partial tables using complete pairs only")
    args = parser.parse_args(argv)
    root = args.root if args.root is not None else Path(default_root(args.profile))
    try:
        rows, report = load_results(root, args.profile, args.input)
        print_status(report)
        if args.status:
            return 0
        if report["status"] != "complete" and not args.allow_incomplete:
            raise ValidationError("the study is incomplete; no tables were written. Use --status to inspect progress or --allow-incomplete for explicitly partial tables.")
        if not rows:
            raise ValidationError("there are no complete pairs to summarize")
        stats, destination = write_outputs(root, rows, report)
        print("n     method   metric  n_rep           mean             sd             se")
        for row in stats:
            sd = f"{row['sd']:.8g}" if row["sd"] is not None else "--"
            se = f"{row['se']:.8g}" if row["se"] is not None else "--"
            print(f"{row['n']:<5} {row['method']:<8} {row['metric']:<7} {row['n_rep']:>5} {row['mean']:>14.8g} {sd:>14} {se:>14}")
        print("SD uses ddof=1; SE = SD / sqrt(n_rep). A single pair has no sample SD or SE.")
        print(f"Wrote CSV, JSONL, LaTeX, and validation files to {destination}")
        return 0
    except (ValidationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
