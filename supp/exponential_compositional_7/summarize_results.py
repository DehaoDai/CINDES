"""Validate the paired SCC study and write accuracy, runtime, and LaTeX tables.

This module uses only the Python standard library.  The ordinary command
validates every completed run and its saved artifacts.  --status reads only
small metadata files.  --from-export rebuilds tables from an archived,
previously validated metric export and does not claim to recheck raw samples.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import common
import study_config

HERE = Path(__file__).resolve().parent
METHODS = ("cindes", "dsm")
METRICS = ("tv1", "tv2", "llgap", "h_tv", "h_w1")
TIMINGS = ("cv_training_seconds", "cv_validation_seconds", "cv_serialization_seconds",
           "refit_training_seconds", "refit_validation_seconds", "refit_serialization_seconds",
           "generation_seconds", "evaluation_seconds", "training_and_generation_seconds",
           "total_compute_seconds")
ACCURACY_FIELDS = ("n", "method", "metric", "n_rep", "mean", "sd", "se",
                   "expected_replications", "profile", "partial_study")


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


@contextmanager
def locked(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_root(value, profile):
    path = (Path(value) if value is not None else Path("outputs") / f"exponential_{profile}_v1").expanduser()
    return path.resolve() if path.is_absolute() else (HERE / path).resolve()


def task_directory(root, task):
    return root / "runs" / f"n_{task['n']}" / f"seed_{task['seed']}"


def descriptive(values):
    values = [float(value) for value in values]
    require(all(math.isfinite(value) for value in values), "Nonfinite number in a summary")
    if not values:
        return {"n_rep": 0, "mean": None, "sd": None, "se": None}
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {"n_rep": len(values), "mean": statistics.mean(values), "sd": sd,
            "se": None if sd is None else sd / math.sqrt(len(values))}


def finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_record(row, task, manifest):
    cfg = manifest["identity"]["config"]
    require(row.get("status") == "complete", "Run is not complete")
    require(row.get("task_identity") == task, "Task identity differs from the frozen task plan")
    require(row.get("study_identity_sha256") == manifest["identity_sha256"], "Study identity differs")
    require(set(row.get("methods", {})) == set(METHODS), "A complete run must contain exactly both methods")
    require(isinstance(row.get("artifact_sha256"), dict) and row["artifact_sha256"], "Missing artifact hashes")
    for method in METHODS:
        value = row["methods"][method]
        require(value.get("method") == method, "Method name differs from its dictionary key")
        for key in ("n", "seed"):
            require(value.get(key) == task[key], f"Method {key} differs from the task")
        require(value.get("n_gen") == cfg["n_gen"], "Generated sample count differs from the protocol")
        for key in METRICS:
            require(finite_number(value.get(key)), f"Invalid {method} metric: {key}")
        for key in ("tv1", "tv2", "h_tv"):
            require(-1e-12 <= value[key] <= 1 + 1e-12, f"Histogram TV outside [0,1]: {key}")
        require(value["h_w1"] >= 0, "Negative Wasserstein distance")
        require(value.get("selected_lambda") in cfg["lambdas"], "Selected penalty is outside the frozen grid")
        require(finite_number(value.get("selected_positive_epochs"))
                and 0 <= value["selected_positive_epochs"] <= cfg["epochs"], "Selected duration is invalid")
        require(isinstance(value.get("actual_updates"), int) and value["actual_updates"] >= 0,
                "Invalid number of refit updates")
        require(isinstance(value.get("parameter_count"), int) and value["parameter_count"] > 0,
                "Invalid parameter count")
        for key in TIMINGS:
            require(finite_number(value.get(key)) and value[key] >= 0, f"Invalid {method} timing: {key}")
        expected = sum(value[key] for key in ("cv_training_seconds", "cv_validation_seconds",
                                              "refit_training_seconds", "generation_seconds"))
        require(math.isclose(value["training_and_generation_seconds"], expected, rel_tol=1e-10, abs_tol=1e-7),
                "Training-and-generation timing is not the sum of its declared components")
        total = sum(value[key] for key in TIMINGS[:8])
        require(math.isclose(value["total_compute_seconds"], total, rel_tol=1e-10, abs_tol=1e-7),
                "Total recorded compute time is not the sum of its declared components")
        for name in (f"methods/{method}/metrics.json", f"methods/{method}/selection.json",
                     f"methods/{method}/refit/model.pt", f"methods/{method}/generated.npz"):
            require(name in row["artifact_sha256"], "Essential artifact absent from run hashes: " + name)


def validate_reference_binding(row, expected_sha256):
    for method in METHODS:
        require(row["methods"][method].get("reference_sha256") == expected_sha256,
                f"The {method} metrics use a different evaluation reference sample")


def load_manifest(root, profile, verify_sources=True):
    path = root / "manifest.json"
    if not path.exists():
        return None
    if verify_sources:
        return common.load_study(root, profile)
    manifest = read_json(path)
    identity = manifest.get("identity", {})
    require(identity.get("profile") == profile and identity.get("config") == study_config.profile_config(profile)
            and identity.get("tasks") == study_config.task_plan(profile),
            "The stored study profile, configuration, or task plan differs")
    require(manifest.get("identity_sha256") == common.canonical_hash(identity),
            "Manifest lacks a valid study identity digest")
    return manifest


def inventory(root, profile, manifest):
    tasks = study_config.task_plan(profile)
    expected = {task_directory(root, task) / "result.json": task for task in tasks}
    records, problems = [], []
    for path, task in expected.items():
        if not path.exists():
            category = "in_progress" if path.parent.exists() else "missing"
            reason = "No complete paired result yet"
            for diagnostic in (path.parent / "failure.json", path.parent / "progress.json"):
                if diagnostic.is_file():
                    try:
                        saved = read_json(diagnostic)
                        if saved.get("status") in ("failed", "error") or diagnostic.name == "failure.json":
                            category = "failed"
                            reason = saved.get("error", saved.get("message", "A task attempt failed"))
                    except (OSError, ValueError):
                        pass
            problems.append({**task, "category": category, "reason": str(reason)})
            continue
        try:
            row = read_json(path)
            if row.get("status") != "complete":
                status = row.get("status", "unknown")
                category = "failed" if status in ("failed", "error") else "in_progress"
                problems.append({**task, "category": category,
                                 "reason": str(row.get("error", "Task status: " + str(status)))})
                continue
            require(manifest is not None, "Completed result exists without a study manifest")
            validate_record(row, task, manifest)
            records.append((path, row))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems.append({**task, "category": "invalid", "reason": str(exc)})
    for path in sorted((root / "runs").glob("n_*/seed_*/result.json")):
        if path not in expected:
            problems.append({"task_id": None, "n": None, "seed": None, "category": "invalid",
                             "reason": "Unexpected result outside the selected profile: " + str(path.relative_to(root))})
    return records, problems


def make_status(profile, manifest, records, problems, hashes_checked=False):
    tasks = study_config.task_plan(profile)
    cfg = study_config.profile_config(profile)
    per_n = []
    for n in cfg["sample_sizes"]:
        per_n.append({"n": n, "complete_pairs": sum(row["task_identity"]["n"] == n for _, row in records),
                      "expected_pairs": sum(task["n"] == n for task in tasks)})
    return {"status": "complete" if len(records) == len(tasks) and not problems else "incomplete",
            "profile": profile, "study_identity_sha256": None if manifest is None else manifest["identity_sha256"],
            "completed_pairs": len(records), "completed_method_results": 2 * len(records),
            "expected_pairs": len(tasks), "per_n": per_n,
            "missing_pairs": sum(row["category"] == "missing" for row in problems),
            "in_progress_pairs": sum(row["category"] == "in_progress" for row in problems),
            "failed_pairs": sum(row["category"] == "failed" for row in problems),
            "invalid_results": sum(row["category"] == "invalid" for row in problems),
            "problems": problems, "checked_file_hashes": hashes_checked, "checked_at_utc": now(),
            "note": "Status checks completion metadata only. Ordinary summarization verifies all recorded artifact hashes."}


def print_status(status):
    print(f"Profile: {status['profile']}")
    print(f"Complete paired replications: {status['completed_pairs']}/{status['expected_pairs']}")
    for row in status["per_n"]:
        print(f"  n={row['n']}: {row['complete_pairs']}/{row['expected_pairs']} complete pairs")
    print(f"Missing: {status['missing_pairs']}; in progress: {status['in_progress_pairs']}; "
          f"failed: {status['failed_pairs']}; invalid: {status['invalid_results']}")
    for item in status["problems"][:10]:
        print(f"  task={item.get('task_id')} n={item.get('n')} seed={item.get('seed')}: "
              f"{item['category']}: {item['reason']}")
    if len(status["problems"]) > 10:
        print(f"  ... {len(status['problems'])-10} further incomplete/invalid tasks")


def statistics_tables(manifest, records):
    cfg, profile = manifest["identity"]["config"], manifest["identity"]["profile"]
    expected_tasks = manifest["identity"]["tasks"]
    partial = len(records) != len(expected_tasks)
    accuracy, timing, pairs, paired = [], [], [], []
    for n in cfg["sample_sizes"]:
        runs = [row for _, row in records if row["task_identity"]["n"] == n]
        expected = sum(task["n"] == n for task in expected_tasks)
        for method in METHODS:
            for metric, output in [(m, accuracy) for m in METRICS] + [(m, timing) for m in TIMINGS]:
                output.append({"n": n, "method": method, "metric": metric,
                               **descriptive([row["methods"][method][metric] for row in runs]),
                               "expected_replications": expected, "profile": profile, "partial_study": partial})
        for row in runs:
            task = row["task_identity"]
            item = {"n": n, "seed": task["seed"], "task_id": task["task_id"], "profile": profile,
                    "definition": "CINDES_minus_DSM"}
            for metric in METRICS + TIMINGS:
                item[metric] = row["methods"]["cindes"][metric] - row["methods"]["dsm"][metric]
            item["llgap_absolute_difference"] = abs(row["methods"]["cindes"]["llgap"]) - abs(row["methods"]["dsm"]["llgap"])
            pairs.append(item)
        for metric in METRICS + TIMINGS:
            values = [row["methods"]["cindes"][metric] - row["methods"]["dsm"][metric] for row in runs]
            paired.append({"n": n, "metric": metric, "definition": "CINDES_minus_DSM", **descriptive(values),
                           "cindes_numerically_smaller": sum(x < 0 for x in values),
                           "dsm_numerically_smaller": sum(x > 0 for x in values), "ties": sum(x == 0 for x in values),
                           "cindes_closer_to_zero": sum(abs(r["methods"]["cindes"][metric]) < abs(r["methods"]["dsm"][metric]) for r in runs) if metric == "llgap" else None,
                           "dsm_closer_to_zero": sum(abs(r["methods"]["dsm"][metric]) < abs(r["methods"]["cindes"][metric]) for r in runs) if metric == "llgap" else None,
                           "profile": profile, "partial_study": partial})
    return accuracy, timing, pairs, paired


def tex_cell(mean, sd, bold=False, digits=4):
    if mean is None:
        return r"$\mathrm{--}$"
    value = f"{mean:.{digits}f}" + ((r" \pm " + f"{sd:.{digits}f}") if sd is not None else r"\; (\mathrm{SD:\ --})")
    return "$" + (r"\mathbf{" + value + "}" if bold else value) + "$"


def accuracy_latex(manifest, accuracy, partial):
    cfg, profile = manifest["identity"]["config"], manifest["identity"]["profile"]
    lookup = {(r["n"], r["method"], r["metric"]): r for r in accuracy}
    prefix = "SMOKE TEST ONLY. " if profile == "smoke" else ("PARTIAL STUDY. " if partial else "")
    caption = (prefix + "CINDES and DSM for the exponential-tail compositional target with $d=7$. "
               "Entries are mean $\\pm$ sample standard deviation across the completed paired datasets; "
               "the actual replication counts are shown.")
    lines = [r"% Requires \usepackage{booktabs}.", r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}", r"\caption{" + caption + "}",
             r"\label{tab:cindes_exponential_accuracy}", r"\begin{tabular}{rrlccccc}", r"\toprule",
             r"$n$ & Reps & Method & TV1 & TV2 & LLgap & $H$-TV & $H$-$W_1$ \\", r"\midrule"]
    for index, n in enumerate(cfg["sample_sizes"]):
        if index:
            lines.append(r"\addlinespace")
        for method in METHODS:
            row = lookup[n, method, "tv1"]
            other = "dsm" if method == "cindes" else "cindes"
            cells = []
            for metric in METRICS:
                this, that = lookup[n, method, metric], lookup[n, other, metric]
                better = this["mean"] is not None and that["mean"] is not None and metric != "llgap" and this["mean"] < that["mean"]
                cells.append(tex_cell(this["mean"], this["sd"], better))
            lines.append(f"{n} & {row['n_rep']} & {method.upper()} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{0.98\linewidth}\footnotesize",
              r"TV2 averages over the six generating pairs. Bold entries identify the smaller mean distance. "
              r"LLgap is signed and is not a divergence, so its entries are not ranked by numerical size. "
              r"SD is undefined when only one paired replication is complete.",
              r"\end{minipage}", r"\end{table}", ""]
    return "\n".join(lines)


def runtime_latex(manifest, timing, partial):
    cfg, profile = manifest["identity"]["config"], manifest["identity"]["profile"]
    shown = ("cv_training_seconds", "cv_validation_seconds", "refit_training_seconds",
             "generation_seconds", "training_and_generation_seconds")
    lookup = {(r["n"], r["method"], r["metric"]): r for r in timing}
    prefix = "SMOKE TEST ONLY. " if profile == "smoke" else ("PARTIAL STUDY. " if partial else "")
    lines = [r"% Requires \usepackage{booktabs}.", r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\caption{" + prefix + r"Recorded computation times in seconds for the exponential-tail experiment. "
             r"Entries are mean $\pm$ sample standard deviation across the completed paired datasets.}",
             r"\label{tab:cindes_exponential_runtime}", r"\begin{tabular}{rrlccccc}", r"\toprule",
             r"$n$ & Reps & Method & CV fitting & CV validation & Refit & Generation & Total \\", r"\midrule"]
    for index, n in enumerate(cfg["sample_sizes"]):
        if index:
            lines.append(r"\addlinespace")
        for method in METHODS:
            row = lookup[n, method, shown[0]]
            cells = [tex_cell(lookup[n, method, m]["mean"], lookup[n, method, m]["sd"], digits=2) for m in shown]
            lines.append(f"{n} & {row['n_rep']} & {method.upper()} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip", r"\begin{minipage}{0.98\linewidth}\footnotesize",
              r"CV fitting and validation sum the durations of all penalty candidates and folds. "
              r"Total is the sum of the four displayed components; it excludes evaluation, serialization, "
              r"queue time, and process initialization. Tasks can run concurrently, so these sums do not "
              r"give the elapsed time of the whole study. Hardware details are recorded for each attempt.",
              r"\end{minipage}", r"\end{table}", ""]
    return "\n".join(lines)


def export_tables(folder, manifest, records, status, validation_scope, source_root=None):
    folder.mkdir(parents=True, exist_ok=True)
    accuracy, timing, pairs, paired = statistics_tables(manifest, records)
    partial = status["status"] != "complete"
    write_csv(folder / "summary.csv", accuracy, ACCURACY_FIELDS)
    write_csv(folder / "runtime.csv", timing, ACCURACY_FIELDS)
    pair_fields = ("n", "seed", "task_id", "profile", "definition") + METRICS + TIMINGS + ("llgap_absolute_difference",)
    write_csv(folder / "paired_differences.csv", pairs, pair_fields)
    paired_fields = ("n", "metric", "definition", "n_rep", "mean", "sd", "se", "cindes_numerically_smaller",
                     "dsm_numerically_smaller", "ties", "cindes_closer_to_zero", "dsm_closer_to_zero", "profile", "partial_study")
    write_csv(folder / "paired_summary.csv", paired, paired_fields)
    method_records, flat_rows = [], []
    for path, row in records:
        for method in METHODS:
            item = {**row["methods"][method], "profile": status["profile"],
                    "study_identity_sha256": manifest["identity_sha256"], "task_id": row["task_identity"]["task_id"],
                    "run_result": str(path.relative_to(source_root)) if source_root is not None else str(path)}
            method_records.append(item)
            flat_rows.append({k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v) for k, v in item.items()})
    write_text(folder / "all_runs.jsonl", "".join(json.dumps(r, allow_nan=False) + "\n" for r in method_records))
    write_text(folder / "run_records.jsonl", "".join(json.dumps(r, allow_nan=False) + "\n" for _, r in records))
    flat_fields = list(dict.fromkeys(k for r in flat_rows for k in r)) or ["method", "n", "seed", *METRICS, *TIMINGS]
    write_csv(folder / "per_run.csv", flat_rows, flat_fields)
    write_text(folder / "accuracy_table.tex", accuracy_latex(manifest, accuracy, partial))
    write_text(folder / "runtime_table.tex", runtime_latex(manifest, timing, partial))
    notes = ("This directory contains " + ("a partial study" if partial else "a complete study") + ".\n"
             "Replications are independently fitted paired datasets, not generation shards.\n"
             "SD uses ddof=1 and SE=SD/sqrt(n_rep); SD/SE are blank with fewer than two replications.\n"
             "LLgap is signed and is not a divergence. All target metrics use one common independent reference sample.\n"
             "training_and_generation_seconds = cv_training_seconds + cv_validation_seconds + refit_training_seconds + generation_seconds.\n"
             "total_compute_seconds additionally includes saved serialization and evaluation durations.\n"
             "Neither timing is the parallel elapsed duration, queue waiting time, or an end-to-end process benchmark.\n"
             "Device and GPU models used by each method and attempt are preserved in all_runs.jsonl and the run metadata.\n"
             "Validation scope: " + validation_scope + "\n")
    if status["profile"] == "smoke":
        notes = "SMOKE TEST ONLY: these values verify execution and are not scientific results.\n\n" + notes
    write_text(folder / "README.txt", notes)
    exported = {"format_version": 1, "manifest": manifest, "status": status, "validation_scope": validation_scope,
                "records": [{"path": str(path.relative_to(source_root)) if source_root is not None else str(path), "record": row}
                            for path, row in records], "created_utc": now()}
    write_json(folder / "export.json", exported)
    output_names = ["summary.csv", "runtime.csv", "paired_differences.csv", "paired_summary.csv", "all_runs.jsonl",
                    "run_records.jsonl", "per_run.csv", "accuracy_table.tex", "runtime_table.tex", "README.txt", "export.json"]
    summary = {"status": "partial" if partial else "complete", "profile": status["profile"],
               "study_identity_sha256": manifest["identity_sha256"], "completed_pairs": len(records),
               "expected_pairs": len(manifest["identity"]["tasks"]), "validation_scope": validation_scope,
               "raw_artifact_hashes_checked": status["checked_file_hashes"],
               "reference_metadata_sha256": sha256(source_root / "reference" / "metadata.json") if source_root is not None and (source_root / "reference" / "metadata.json").exists() else None,
               "run_result_sha256": {str(path.relative_to(source_root)): sha256(path) for path, _ in records} if source_root is not None else {},
               "analysis_source_sha256": sha256(Path(__file__)), "created_utc": now(), "sd_ddof": 1,
               "output_sha256": {name: sha256(folder / name) for name in output_names},
               "problem_tasks": status["problems"], "accuracy": accuracy, "timing": timing, "paired_summary": paired}
    write_json(folder / "summary.json", summary)
    print("n     method   metric             n_rep            mean              sd              se")
    for row in accuracy:
        def number(value):
            return "--" if value is None else f"{value:.9g}"
        print(f"{row['n']:<5} {row['method']:<8} {row['metric']:<18} {row['n_rep']:>5} "
              f"{number(row['mean']):>15} {number(row['sd']):>15} {number(row['se']):>15}")
    print("\nComputation time in seconds (mean +/- sample SD):")
    for row in timing:
        if row["metric"] in ("cv_training_seconds", "cv_validation_seconds", "refit_training_seconds", "generation_seconds", "training_and_generation_seconds"):
            mean = "--" if row["mean"] is None else f"{row['mean']:.3f}"
            sd = "--" if row["sd"] is None else f"{row['sd']:.3f}"
            print(f"n={row['n']} {row['method']:<6} {row['metric']:<34} {mean} +/- {sd}")
    print(f"\nWrote {folder / 'summary.csv'}")
    print(f"Wrote {folder / 'runtime.csv'}")
    print(f"Wrote {folder / 'accuracy_table.tex'}")
    print(f"Wrote {folder / 'runtime_table.tex'}")
    return summary


def summarize(root, profile="n1000", allow_partial=False, status_only=False):
    root = Path(root).resolve()
    manifest = load_manifest(root, profile, verify_sources=not status_only)
    records, problems = inventory(root, profile, manifest)
    status = make_status(profile, manifest, records, problems)
    print_status(status)
    if status_only:
        return 0
    folder = root / "summary"
    with locked(root / ".summary.lock"):
        write_json(folder / "status.json", status)
        write_csv(folder / "incomplete_tasks.csv", problems,
                  ("task_id", "n", "seed", "fit_seed", "data_seed", "fold_seed", "category", "reason"))
        require(not any(r["category"] == "invalid" for r in problems),
                "Invalid or mixed results found; inspect summary/status.json. No new tables were written.")
        require(allow_partial or not problems,
                "The study is incomplete. Use --status, resume the missing tasks, or explicitly request --allow-partial.")
        require(manifest is not None, "The reference preparation job has not created a study manifest yet")
        if records:
            reference = common.validate_reference(root, manifest)
        elif not allow_partial:
            raise ValueError("No complete paired replications to summarize")
        for path, row in records:
            verified = common.validate_task_record(path.parent, manifest, row["task_identity"])
            require(verified is not None and verified == row, "Task artifacts changed during summarization")
            validate_reference_binding(row, reference["reference_sha256"])
            for method in METHODS:
                require(read_json(path.parent / "methods" / method / "metrics.json") == row["methods"][method],
                        "Method metric record differs from the paired completion record")
        status["checked_file_hashes"] = True
        write_json(folder / "status.json", status)
        summary = export_tables(folder, manifest, records, status,
                                "Complete paired result identities and every recorded artifact hash were checked; reference hash checked when results exist.", root)
        print(f"Validated {len(records)} complete paired replications ({2*len(records)} method results).")
        if summary["status"] != "complete":
            print("PARTIAL STUDY: the tables report only the completed pairs and show their actual counts.")
    return 0


def from_export(export_path, destination):
    exported = read_json(export_path)
    require(exported.get("format_version") == 1, "Unsupported metric-export format")
    manifest, saved_status = exported["manifest"], exported["status"]
    expected = {task["task_id"]: task for task in manifest["identity"]["tasks"]}
    records, seen = [], set()
    for item in exported["records"]:
        row = item["record"]
        task = row["task_identity"]
        require(task["task_id"] in expected and task["task_id"] not in seen, "Unknown or duplicated exported task")
        validate_record(row, expected[task["task_id"]], manifest)
        seen.add(task["task_id"])
        records.append((Path(item["path"]), row))
    require(saved_status["completed_pairs"] == len(records), "Exported count differs from its records")
    status = {**saved_status, "checked_file_hashes": False, "checked_at_utc": now()}
    destination = Path(destination).expanduser().resolve()
    require(destination != Path(export_path).resolve().parent, "Choose a fresh --output directory for rebuilt tables")
    with locked(destination / ".summary.lock"):
        export_tables(destination, manifest, records, status,
                      "Rebuilt from saved metric records. Raw samples, training artifacts, and source-run hashes were not rechecked.")
    print("Rebuilt tables from the archived metric export; raw artifacts were not rechecked.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--profile", choices=("n1000", "full", "smoke"), default="n1000")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--from-export", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.from_export is not None:
            if args.output is None or args.root is not None or args.status or args.allow_partial:
                parser.error("--from-export requires --output and cannot be combined with --root, --status, or --allow-partial")
            return from_export(args.from_export, args.output)
        if args.output is not None:
            parser.error("--output is used only with --from-export")
        return summarize(resolve_root(args.root, args.profile), args.profile, args.allow_partial, args.status)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("Summary failed: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
