"""Run the unconditional-density experiment in main-paper Table 1."""

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from cindes.config import NetworkConfig, OptimizationConfig, ValidationConfig
from cindes.data import (
    elliptical_gaussian_mixture,
    evaluation_grid,
    spherical_gaussian_mixture,
)
from cindes.estimators import ConditionalMAF, ExplicitCINDES, RandomForestDensityEstimator
from cindes.metrics import mean_absolute_density_error
from cindes.reproducibility import (
    replication_seeds,
    runtime_metadata,
    seed_contract,
    seed_everything,
)

SOURCE_SAMPLE_SIZE = 15_000
TRAIN_SIZE = 6_000
VALIDATION_SIZE = 1_500
GRID_SIZE = 100
MAX_EPOCHS = 10_000
EARLY_STOPPING_PATIENCE = 100
BATCH_SIZE = 1_000
NORMALIZATION_SAMPLES = 1_000
METHODS = ("cindes", "rfcde", "maf")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("spherical", "elliptical"), required=True)
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; replication r uses seed + r.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument(
        "--protocol",
        choices=("fixed", "resampled"),
        default="resampled",
        help="Use fixed Algorithm 1 fake samples or SADE minibatch resampling.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/table1"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small smoke experiment; quick results are not full-scale.",
    )
    return parser.parse_args()


def scenario_from_name(name: str):
    if name == "spherical":
        return spherical_gaussian_mixture()
    if name == "elliptical":
        return elliptical_gaussian_mixture()
    raise ValueError(f"unknown scenario: {name}")


def run_replication(
    scenario_name: str,
    base_seed: int,
    replication_index: int,
    device: str,
    quick: bool,
    protocol: str,
    method: str = "cindes",
) -> dict[str, object]:
    seed = base_seed + replication_index
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    scenario = scenario_from_name(scenario_name)

    train_size = 512 if quick else TRAIN_SIZE
    validation_size = 128 if quick else VALIDATION_SIZE
    grid_size = 25 if quick else GRID_SIZE

    source_sample_size = 640 if quick else SOURCE_SAMPLE_SIZE
    all_responses = scenario.sample(source_sample_size, rng).astype(np.float32)
    training_responses = all_responses[:train_size]
    validation_responses = all_responses[train_size:train_size + validation_size]

    grid = evaluation_grid(
        all_responses.min(axis=0),
        all_responses.max(axis=0),
        grid_size=grid_size,
    )
    if method == "cindes":
        optimization = OptimizationConfig(
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=0.0,
            scheduler_gamma=0.999,
            fake_response_strategy=("fixed_global" if protocol == "fixed" else "batch_resampled"),
            reset_seed_per_candidate=(protocol == "fixed"),
        )
        validation = ValidationConfig(reference_samples=NORMALIZATION_SAMPLES)
        if quick:
            optimization = replace(optimization, max_epochs=5, patience=2,
                                   weight_decay_candidates=(1e-4,))
            validation = replace(validation, reference_samples=64)
        estimator = ExplicitCINDES(
            network=NetworkConfig(), optimization=optimization,
            validation=validation, seed=seed, device=device,
        )
        estimator.fit(y=training_responses, y_valid=validation_responses)
        if estimator.reference_ is None:
            raise RuntimeError("fitted estimator did not retain its reference distribution")
        estimated_density = estimator.density(
            grid, normalize=True,
            reference_samples=64 if quick else NORMALIZATION_SAMPLES,
            seed=seed + 20_000,
        )
        details = {
            "selected_weight_decay": estimator.best_weight_decay_,
            "validation_nll": estimator.best_validation_nll_,
            "reference_lower": estimator.reference_.lower.tolist(),
            "reference_upper": estimator.reference_.upper.tolist(),
            "selection_history": estimator.selection_history_,
            "estimator_config": estimator.get_config(),
        }
    elif method == "rfcde":
        # A constant context lets the shared conditional baseline fit p(y).
        x_train = np.zeros((train_size, 1), dtype=np.float32)
        estimator = RandomForestDensityEstimator(
            n_estimators=12 if quick else 200,
            max_depth=4 if quick else 12,
            seed=42,
        )
        estimator.fit(x=x_train, y=training_responses, rng=np.random.RandomState(seed))
        estimated_density = estimator.conditional_density_grid(
            x=np.zeros((1, 1), dtype=np.float32), y=grid,
            normalization_samples=64 if quick else NORMALIZATION_SAMPLES,
            seed=seed + 30_000,
            lower=all_responses.min(axis=0), upper=all_responses.max(axis=0),
        ).reshape(-1)
        details = {"estimator_config": estimator.get_config()}
    elif method == "maf":
        x_train = np.zeros((train_size, 1), dtype=np.float32)
        x_valid = np.zeros((validation_size, 1), dtype=np.float32)
        estimator = ConditionalMAF(
            depth=3, width=64, max_epochs=3 if quick else MAX_EPOCHS,
            patience=1 if quick else EARLY_STOPPING_PATIENCE, batch_size=BATCH_SIZE,
            weight_decay_candidates=(1e-4,) if quick else (1e-3, 5e-4, 2e-4, 1e-4, 0.0),
            seed=seed, device=device,
        )
        estimator.fit(x=x_train, y=training_responses, x_valid=x_valid, y_valid=validation_responses)
        estimated_density = estimator.conditional_density_grid(
            x=np.zeros((1, 1), dtype=np.float32), y=grid,
        ).reshape(-1)
        details = {
            "selected_weight_decay": estimator.best_weight_decay_,
            "validation_nll": estimator.best_validation_nll_,
            "selection_history": estimator.selection_history_,
            "estimator_config": estimator.get_config(),
        }
    else:
        raise ValueError(f"unknown method: {method}")
    true_density = scenario.density(grid)
    metric = mean_absolute_density_error(estimated_density, true_density)

    return {
        "seed": seed,
        "scenario": scenario_name,
        "method": method,
        "quick": quick,
        "protocol": protocol,
        "replication_index": replication_index,
        "seed_contract": seed_contract(base_seed, replication_index),
        "train_size": train_size,
        "validation_size": validation_size,
        "source_sample_size": source_sample_size,
        "grid_size": grid_size,
        "empirical_tv": metric,
        "evaluation_lower": all_responses.min(axis=0).tolist(),
        "evaluation_upper": all_responses.max(axis=0).tolist(),
        **details,
    }


def save_results(
    results: list[dict[str, object]],
    arguments: argparse.Namespace,
    seeds: list[int],
) -> Path:
    summary = {}
    for method in dict.fromkeys(result["method"] for result in results):
        metric_values = [float(result["empirical_tv"]) for result in results if result["method"] == method]
        summary[method] = {
            "empirical_tv_mean": mean(metric_values),
            "empirical_tv_standard_deviation": stdev(metric_values) if len(metric_values) > 1 else 0.0,
        }
    payload = {
        "experiment": "table1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "seed_schedule": {
            "rule": "actual_seed = base_seed + replication_index",
            "base_seed": arguments.seed,
            "selected_seeds": seeds,
        },
        "runtime": runtime_metadata(),
        "results": results,
        "summary": summary,
    }

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    mode = "quick" if arguments.quick else "full"
    filename = (
        f"table1_{arguments.method}_{arguments.scenario}_{arguments.protocol}_{mode}_"
        f"seed{arguments.seed}_n{arguments.replications}.json"
    )
    output_path = arguments.output_dir / filename
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    arguments = parse_arguments()
    if arguments.replications < 1:
        raise ValueError("--replications must be positive")
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative")

    seeds = replication_seeds(arguments.seed, arguments.replications)
    results = []
    for replication_index, seed in enumerate(seeds):
        methods = METHODS if arguments.method == "all" else (arguments.method,)
        for method in methods:
            result = run_replication(
                scenario_name=arguments.scenario,
                base_seed=arguments.seed,
                replication_index=replication_index,
                device=arguments.device,
                quick=arguments.quick,
                protocol=arguments.protocol,
                method=method,
            )
            results.append(result)
            print(
                f"[{method.upper()}] scenario={arguments.scenario} seed={seed} "
                f"empirical_tv={result['empirical_tv']:.6f}"
            )

    output_path = save_results(results, arguments, seeds)
    print(f"[CINDES] results={output_path}")


if __name__ == "__main__":
    main()
