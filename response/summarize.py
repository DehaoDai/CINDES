"""Summarize fresh chain/tree JSONL runs for reviewer-response tables 1, 3, or 4."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

SAMPLE_SIZES = (500, 1000, 2000, 4000, 8000, 16000)
METHODS = ("cindes", "dsm")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize_group(rows: list[dict], key: str, scale: float, expected_seeds: int) -> tuple[float, float, int]:
    seeds = [row["seed"] for row in rows]
    if sorted(seeds) != list(range(expected_seeds)):
        raise ValueError(f"expected seeds 0..{expected_seeds - 1} for {key}, got {sorted(seeds)}")
    values = [float(row[key]) * scale for row in rows]
    return mean(values), stdev(values), len(values)


def collect(table: int, input_path: Path, joint_path: Path | None) -> list[dict]:
    records = read_jsonl(input_path)
    groups = defaultdict(list)
    expected_dimension = {1: 16, 3: 32, 4: 31}[table]
    for record in records:
        if record.get("d") != expected_dimension:
            raise ValueError(f"unexpected dimension in {input_path}")
        if table == 4 and record.get("graph") != "btree":
            raise ValueError(f"unexpected graph in {input_path}")
        groups[record["n"], record["method"]].append(record)
    joint_groups = defaultdict(list)
    if joint_path is not None:
        if table != 3:
            raise ValueError("--joint is supported only for response Table 3")
        for record in read_jsonl(joint_path):
            joint_groups[record["n"], record["method"]].append(record)

    metrics = ("tv1", "tv2", "llgap", "energy", "sw") if table == 1 else ("tv1", "tv2", "llgap")
    output = []
    for n in SAMPLE_SIZES:
        for method in METHODS:
            group = groups[n, method]
            if not group:
                raise ValueError(f"missing {method} n={n} in {input_path}")
            for metric in metrics:
                scale = 1000.0 if metric == "energy" else 100.0 if metric == "sw" else 1.0
                avg, sd, count = summarize_group(group, f"{method}_{metric}", scale, 10)
                output.append(dict(table=table, n=n, method=method, metric=metric,
                                   mean=avg, sample_sd=sd, seeds=count, status="complete"))
            if table == 3 and joint_path is not None:
                joint = joint_groups[n, method]
                if not joint:
                    raise ValueError(f"missing joint metric for {method} n={n}")
                metric = "joint_tv" if method == "cindes" else "joint_tv_lower_bound"
                key = "joint_tv" if method == "cindes" else "atom_lb"
                avg, sd, count = summarize_group(joint, key, 1.0, 3)
                output.append(dict(table=table, n=n, method=method, metric=metric,
                                   mean=avg, sample_sd=sd, seeds=count,
                                   status="3_seeds_vs_10_in_caption"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=int, choices=(1, 3, 4), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--joint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.table, args.input, args.joint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} response Table {args.table} metric cells to {args.output}")


if __name__ == "__main__":
    main()
