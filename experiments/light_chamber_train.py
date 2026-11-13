"""Train Light Chamber methods and write the NLL matrices for Tables 4 and 5."""

import argparse
import csv
from pathlib import Path

import numpy as np

from cindes.config import OptimizationConfig, ValidationConfig
from cindes.estimators.explicit import ExplicitCINDES
from cindes.estimators.lincde import LinCDEEstimator
from cindes.estimators.locscale import LocScaleNN
from cindes.estimators.maf import ConditionalMAF
from cindes.estimators.rfcde import RandomForestDensityEstimator
from cindes.real_data import LIGHT_CHAMBER_TARGETS
from cindes.reproducibility import seed_everything

GROUPS = (
    ("red", "green", "blue", "current"),
    ("pol_1", "pol_2"),
    ("ir_1", "vis_1", "ir_2", "vis_2"),
    ("ir_3", "vis_3"),
)
METHOD_FILES = {
    "baseline": "result_BaseLine.csv",
    "cindes": "result_NCDE.csv",
    "rfcde": "result_RFCDE.csv",
    "maf": "result_MAF.csv",
    "lincde": "result_LinCDE.csv",
}


def load_sensor_data(path: Path) -> np.ndarray:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(LIGHT_CHAMBER_TARGETS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing Light Chamber columns: {sorted(missing)}")
        data = np.array([[float(row[name]) for name in LIGHT_CHAMBER_TARGETS]
                         for row in reader], dtype=np.float32)
    if len(data) < 501 or not np.isfinite(data).all():
        raise ValueError("sensor CSV needs at least 501 finite records")
    return data


def split_replication(data: np.ndarray, targets: tuple[int, ...], sample_size: int,
                      seed: int) -> tuple[np.ndarray, ...]:
    """Follow the original 40% train, 10% validation, last-500 test split."""
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(data), size=sample_size, replace=False)
    sampled = data[chosen].astype(np.float64)
    means, scales = sampled.mean(axis=0), sampled.std(axis=0)
    if (scales <= 0).any():
        raise ValueError("a sampled sensor column has zero variance")
    sampled = ((sampled - means) / scales).astype(np.float32)
    y = sampled[:, targets]
    x = np.delete(sampled, targets, axis=1)
    pool = sample_size - 500
    train_end = int(pool * 0.4)
    valid_end = train_end + int(pool * 0.1)
    if train_end < 2 or valid_end <= train_end:
        raise ValueError("sample size leaves too few training or validation records")
    return x[:train_end], y[:train_end], x[train_end:valid_end], y[train_end:valid_end], x[-500:], y[-500:]


def train_nll(method: str, split: tuple[np.ndarray, ...], *, seed: int,
              device: str, quick: bool, multivariate: bool) -> float:
    x_train, y_train, x_valid, y_valid, x_test, y_test = split
    seed_everything(seed)
    epochs = 3 if quick else (1000 if multivariate else 2000)
    patience = 2 if quick else (100 if multivariate else 200)
    if method == "baseline":
        estimator = LocScaleNN(max_epochs=epochs if quick else 1000,
                               patience=patience if quick else 100, seed=seed, device=device)
        estimator.fit(x_train, y_train, x_valid, y_valid)
        return estimator.negative_log_likelihood(x_test, y_test)
    if method == "cindes":
        estimator = ExplicitCINDES(
            optimization=OptimizationConfig(
                max_epochs=epochs, patience=patience, batch_size=len(x_train),
                fake_response_strategy="batch_resampled",
                weight_decay_candidates=(0.0,) if quick else (0.001, 0.0005, 0.0002, 0.0001, 0.0),
            ),
            validation=ValidationConfig(reference_samples=32 if quick else 1000,
                                        batch_size=16), seed=seed, device=device,
        )
        estimator.fit(x=x_train, y=y_train, x_valid=x_valid, y_valid=y_valid)
        log_density = estimator.normalized_log_density(
            x=x_test, y=y_test, reference_samples=32 if quick else 1000,
            seed=seed + 20_000)
        return float(-np.mean(log_density))
    if method == "rfcde":
        estimator = RandomForestDensityEstimator(
            n_estimators=12 if quick else 200, max_depth=4 if quick else 12,
            reference="gaussian", seed=42)
        estimator.fit(x=x_train, y=y_train, rng=np.random.RandomState(seed))
        log_density = estimator.paired_log_density(
            x_test, y_test, normalization_samples=32 if quick else 1000,
            seed=seed + 30_000)
        return float(-np.mean(log_density))
    if method == "maf":
        estimator = ConditionalMAF(
            max_epochs=epochs, patience=patience, batch_size=len(x_train),
            weight_decay_candidates=(0.0,) if quick else (0.001, 0.0005, 0.0002, 0.0001, 0.0),
            seed=seed, device=device)
        estimator.fit(x=x_train, y=y_train, x_valid=x_valid, y_valid=y_valid)
        return float(-np.mean(estimator.paired_log_density(x_test, y_test)))
    if method == "lincde":
        if multivariate:
            raise ValueError("LinCDE only supports univariate responses")
        estimator = LinCDEEstimator(seed=seed)
        estimator.fit(x=x_train, y=y_train)
        return float(-np.mean(estimator.paired_log_density(x_test, y_test, seed=seed + 40_000)))
    raise ValueError(f"unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=int, choices=(4, 5), required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to uniform_reference.csv; the data stays outside Git.")
    parser.add_argument("--method", choices=tuple(METHOD_FILES), required=True)
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--row", type=int, help="Zero-based target/group index for a smoke run")
    args = parser.parse_args()
    if args.table == 5 and args.method == "lincde":
        parser.error("LinCDE only applies to Table 4")
    data = load_sensor_data(args.data)
    if args.sample_size > len(data) or args.sample_size < 510 or args.replications < 1:
        parser.error("sample-size must be 510..data rows and replications must be positive")
    targets = tuple((name,) for name in LIGHT_CHAMBER_TARGETS) if args.table == 4 else GROUPS
    row_ids = range(len(targets)) if args.row is None else (args.row,)
    if any(index < 0 or index >= len(targets) for index in row_ids):
        parser.error("row index is outside the selected table")
    values = np.empty((len(row_ids), args.replications))
    for position, row_id in enumerate(row_ids):
        indices = tuple(LIGHT_CHAMBER_TARGETS.index(name) for name in targets[row_id])
        for repetition in range(args.replications):
            seed = args.seed + repetition
            split = split_replication(data, indices, args.sample_size, seed)
            values[position, repetition] = train_nll(
                args.method, split, seed=seed, device=args.device,
                quick=args.quick, multivariate=args.table == 5)
            print(f"table={args.table} method={args.method} row={row_id} "
                  f"replication={repetition + 1}/{args.replications} "
                  f"nll={values[position, repetition]:.4f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = METHOD_FILES[args.method]
    if args.table == 5 and args.method == "cindes":
        filename = "result_CINDES.csv"
    if args.row is not None or args.quick:
        filename = f"quick_row{args.row}_{filename}"
    path = args.output_dir / filename
    np.savetxt(path, values, delimiter=",", fmt="%.10g")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
