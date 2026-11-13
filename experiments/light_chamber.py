"""Aggregate the Light Chamber negative log-likelihood results."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from cindes.real_data import (
    LIGHT_CHAMBER_TARGETS,
    FilteredSummary,
    load_result_matrix,
    summarize_after_outlier_removal,
)

METHOD_FILES = (
    ("Baseline", "result_BaseLine.csv"),
    ("CINDES", "result_NCDE.csv"),
    ("RFCDE", "result_RFCDE.csv"),
    ("MAF", "result_MAF.csv"),
    ("LinCDE", "result_LinCDE.csv"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/light_chamber/raw"),
        help="Directory containing the five target-by-replication CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/light_chamber"),
    )
    return parser.parse_args()


def aggregate_results(
    input_directory: Path,
) -> dict[str, dict[str, FilteredSummary]]:
    summaries = {target: {} for target in LIGHT_CHAMBER_TARGETS}
    replication_count = None
    for method, filename in METHOD_FILES:
        matrix = load_result_matrix(input_directory / filename)
        if replication_count is None:
            replication_count = matrix.shape[1]
        elif matrix.shape[1] != replication_count:
            raise ValueError("all methods must contain the same number of replications")
        for target_index, target in enumerate(LIGHT_CHAMBER_TARGETS):
            summaries[target][method] = summarize_after_outlier_removal(
                matrix[target_index]
            )
    return summaries


def save_summary(
    summaries: dict[str, dict[str, FilteredSummary]],
    output_directory: Path,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "light_chamber_summary.csv"
    fieldnames = (
        "target",
        "method",
        "mean",
        "standard_deviation",
        "first_quartile",
        "third_quartile",
        "lower_bound",
        "upper_bound",
        "total_count",
        "retained_count",
        "removed_count",
        "retained_replications",
        "removed_replications",
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for target in LIGHT_CHAMBER_TARGETS:
            for method, _ in METHOD_FILES:
                summary = summaries[target][method]
                row = asdict(summary)
                row.update(
                    {
                        "target": target,
                        "method": method,
                        "retained_count": summary.retained_count,
                        "removed_count": summary.removed_count,
                        "retained_replications": " ".join(
                            str(index) for index in summary.retained_indices
                        ),
                        "removed_replications": " ".join(
                            str(index) for index in summary.removed_indices
                        ),
                    }
                )
                row.pop("retained_indices")
                row.pop("removed_indices")
                writer.writerow(row)

    latex_path = output_directory / "light_chamber_table.tex"
    latex_path.write_text(_latex_table(summaries))
    return csv_path, latex_path


def _latex_table(
    summaries: dict[str, dict[str, FilteredSummary]],
) -> str:
    methods = [method for method, _ in METHOD_FILES]
    lines = [
        r"\begin{tabular}{l" + "c" * len(methods) + "}",
        r"\hline",
        "Target & " + " & ".join(methods) + r" \\",
        r"\hline",
    ]
    for target in LIGHT_CHAMBER_TARGETS:
        best_method = min(methods, key=lambda method: summaries[target][method].mean)
        entries = []
        for method in methods:
            summary = summaries[target][method]
            mean_text = f"{summary.mean:.3f}"
            if method == best_method:
                mean_text = rf"\textbf{{{mean_text}}}"
            entries.append(
                rf"{mean_text} $\pm$ {summary.standard_deviation:.3f}"
            )
        lines.append(target + " & " + " & ".join(entries) + r" \\")
    lines.extend((r"\hline", r"\end{tabular}", ""))
    return "\n".join(lines)


def print_summary(
    summaries: dict[str, dict[str, FilteredSummary]],
) -> None:
    methods = [method for method, _ in METHOD_FILES]
    header = ["target", *methods]
    rows = [header]
    for target in LIGHT_CHAMBER_TARGETS:
        rows.append(
            [
                target,
                *[
                    (
                        f"{summaries[target][method].mean:.3f} ± "
                        f"{summaries[target][method].standard_deviation:.3f}"
                    )
                    for method in methods
                ],
            ]
        )
    widths = [
        max(len(row[column]) for row in rows)
        for column in range(len(header))
    ]
    for row in rows:
        print(
            "  ".join(
                value.ljust(widths[column])
                for column, value in enumerate(row)
            )
        )


def main() -> None:
    arguments = parse_arguments()
    summaries = aggregate_results(arguments.input_dir)
    print_summary(summaries)
    csv_path, latex_path = save_summary(summaries, arguments.output_dir)
    print(f"[light_chamber] summary={csv_path}")
    print(f"[light_chamber] latex={latex_path}")


if __name__ == "__main__":
    main()
