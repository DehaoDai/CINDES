"""Combine fresh main-paper simulation JSON files into a table CSV."""

import argparse
import csv
import json
from pathlib import Path

EXPECTED_CELLS = {1: 6, 2: 36, 3: 36}


def collect(table: int, input_dir: Path, expected_repetitions: int, allow_quick: bool) -> list[dict]:
    rows = []
    for path in sorted(input_dir.rglob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("experiment") != f"table{table}":
            continue
        args = payload["arguments"]
        if args["quick"] and not allow_quick:
            raise ValueError(f"quick result cannot enter a paper table: {path}")
        if table == 1:
            for method, stats in payload["summary"].items():
                repetitions = sum(r["method"] == method for r in payload["results"])
                rows.append(dict(table=table, scenario=args["scenario"], dimension="",
                                 sample_size="", method=method,
                                 mean=stats["empirical_tv_mean"],
                                 sample_sd=stats["empirical_tv_standard_deviation"],
                                 repetitions=repetitions, protocol=args["protocol"],
                                 data_layout="", source=str(path)))
        elif table == 2:
            stats = payload["summary"]
            rows.append(dict(table=table, scenario=args["scenario"], dimension="",
                             sample_size=args["sample_size"], method=args["method"],
                             mean=stats["empirical_tv_mean"],
                             sample_sd=stats["empirical_tv_standard_deviation"],
                             repetitions=len(payload["results"]),
                             protocol=args["protocol"] if args["method"] == "cindes" else "",
                             data_layout=args["data_layout"], source=str(path)))
        else:
            for method, stats in payload["summary"].items():
                rows.append(dict(table=table, scenario="", dimension=args["response_dimension"],
                                 sample_size=args["sample_size"], method=method,
                                 mean=stats["reported_empirical_tv_mean"],
                                 sample_sd=stats["reported_empirical_tv_standard_deviation"],
                                 repetitions=len(payload["results"]), protocol="",
                                 data_layout=args["data_layout"], source=str(path)))
    if not rows:
        raise ValueError(f"no Table {table} JSON results in {input_dir}")
    for row in rows:
        if row["repetitions"] != expected_repetitions:
            raise ValueError(f"{row['source']}: expected {expected_repetitions} replications for {row['method']}")
    keys = [(r["scenario"], r["dimension"], r["sample_size"], r["method"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate table cells; separate alternative protocols or seeds into another input directory")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--allow-quick", action="store_true", help="Only for smoke checks")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows = collect(args.table, args.input_dir, args.replications, args.allow_quick)
    if len(rows) != EXPECTED_CELLS[args.table] and not args.allow_incomplete:
        raise ValueError(f"Table {args.table} has {len(rows)}/{EXPECTED_CELLS[args.table]} cells; rerun or use --allow-incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Table {args.table} cells to {args.output}")


if __name__ == "__main__":
    main()
