"""Aggregate four Light Chamber NLL matrices into main-paper Table 5."""

import argparse
import csv
from pathlib import Path

from cindes.real_data import load_result_matrix

TARGETS = (
    "red, green, blue, current",
    "pol_1, pol_2",
    "ir_1, vis_1, ir_2, vis_2",
    "ir_3, vis_3",
)
METHOD_FILES = (
    ("LocScale-NN", "result_BaseLine.csv"),
    ("CINDES", "result_CINDES.csv"),
    ("RFCDE", "result_RFCDE.csv"),
    ("MAF", "result_MAF.csv"),
)


def aggregate(input_dir: Path) -> list[dict]:
    rows = []
    for method, filename in METHOD_FILES:
        matrix = load_result_matrix(input_dir / filename, expected_rows=4)
        if matrix.shape != (4, 100):
            raise ValueError(f"{filename}: expected 4 targets by 100 replications")
        for index, target in enumerate(TARGETS):
            rows.append({
                "target": target,
                "method": method,
                "mean_nll": float(matrix[index].mean()),
                "sample_sd": float(matrix[index].std(ddof=1)),
                "replications": matrix.shape[1],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/light_chamber_multi"))
    args = parser.parse_args()
    rows = aggregate(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "table5.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("target", "method", "mean_nll", "sample_sd", "replications"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for target in TARGETS:
        values = [row for row in rows if row["target"] == target]
        print(target + "  " + "  ".join(
            f"{row['method']}: {row['mean_nll']:.3f} ± {row['sample_sd']:.3f}"
            for row in values
        ))
    print(f"[Table 5] summary={path}")


if __name__ == "__main__":
    main()
