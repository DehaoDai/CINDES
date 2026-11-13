"""Run the univariate-response conditional density experiments in Table 2."""

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from cindes.config import NetworkConfig, OptimizationConfig, ValidationConfig
from cindes.data import conditional_covariates, conditional_scenario
from cindes.estimators import (
    ConditionalMAF,
    ExplicitCINDES,
    LinCDEEstimator,
    RandomForestDensityEstimator,
)
from cindes.metrics import mean_absolute_density_error
from cindes.reproducibility import (
    replication_seeds,
    runtime_metadata,
    seed_contract,
    seed_everything,
)

VALIDATION_RATIO = 0.25
TEST_COVARIATES = 500
TEST_RESPONSES = 500
MAX_EPOCHS = 1_000
EARLY_STOPPING_PATIENCE = 200
BATCH_SIZE = 64
NORMALIZATION_SAMPLES = 1_000
SAMPLE_SIZES = (500, 2_000, 8_000)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("nonlinear", "additive", "mixture"),
        required=True,
    )
    parser.add_argument("--sample-size", type=int, choices=SAMPLE_SIZES, default=500)
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; replication r uses seed + r.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--method",
        choices=("cindes", "rfcde", "maf", "lincde"),
        default="cindes",
    )
    parser.add_argument(
        "--protocol",
        choices=("fixed", "resampled"),
        default="resampled",
    )
    parser.add_argument(
        "--data-layout",
        choices=("archive", "declared"),
        default="archive",
        help="Reproduce the archived numerical split or use the declared sample sizes.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/table2"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small smoke experiment; quick results are not full-scale.",
    )
    return parser.parse_args()


def run_replication(
    scenario_name: str,
    sample_size: int,
    base_seed: int,
    replication_index: int,
    method: str,
    device: str,
    protocol: str,
    data_layout: str,
    quick: bool,
) -> dict[str, object]:
    seed = base_seed + replication_index
    seed_everything(seed)
    test_covariates = 20 if quick else TEST_COVARIATES
    test_responses = 25 if quick else TEST_RESPONSES
    normalization_samples = 64 if quick else NORMALIZATION_SAMPLES

    if data_layout == "archive":
        rng = np.random.RandomState(seed)
        layout_sample_size = 256 if quick else sample_size
        layout_validation_size = int(VALIDATION_RATIO * layout_sample_size)
        generated_size = layout_sample_size + layout_validation_size + test_covariates
        if scenario_name == "additive":
            x_all = conditional_covariates(scenario_name, generated_size, rng)
            scenario = conditional_scenario(scenario_name, rng)
            y_all = scenario.sample_responses(x_all, rng)[:, None].astype(np.float32)
            x_all = x_all.astype(np.float32)
        else:
            scenario = conditional_scenario(scenario_name, rng)
            x_all, y_all = scenario.sample(generated_size, rng)
        train_validation_size = generated_size - test_covariates
        train_size = int(0.4 * train_validation_size)
        validation_size = int(0.1 * train_validation_size)
        x_train = x_all[:train_size]
        y_train = y_all[:train_size]
        x_valid = x_all[train_size : train_size + validation_size]
        y_valid = y_all[train_size : train_size + validation_size]
        x_test = x_all[-test_covariates:]
        rng_name = "MT19937"
    elif data_layout == "declared":
        rng = np.random.default_rng(seed)
        scenario = conditional_scenario(scenario_name, rng)
        train_size = 128 if quick else sample_size
        validation_size = int(VALIDATION_RATIO * train_size)
        x_train, y_train = scenario.sample(train_size, rng)
        x_valid, y_valid = scenario.sample(validation_size, rng)
        x_test = scenario.sample_covariates(test_covariates, rng).astype(np.float32)
        generated_size = train_size + validation_size + test_covariates
        rng_name = "PCG64"
    else:
        raise ValueError(f"unknown data layout: {data_layout}")

    y_test = rng.uniform(
        scenario.response_lower,
        scenario.response_upper,
        size=(test_responses, 1),
    ).astype(np.float32)

    repeated_x = np.repeat(x_test, test_responses, axis=0)
    tiled_y = np.tile(y_test, (test_covariates, 1))
    true_density = scenario.density(repeated_x, tiled_y).reshape(
        test_covariates,
        test_responses,
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
        validation = ValidationConfig(
            reference_samples=normalization_samples,
            batch_size=128,
        )
        if quick:
            optimization = replace(
                optimization,
                max_epochs=5,
                patience=2,
                weight_decay_candidates=(1e-4,),
            )
        estimator = ExplicitCINDES(
            network=NetworkConfig(),
            optimization=optimization,
            validation=validation,
            seed=seed,
            device=device,
        )
        estimator.fit(
            x=x_train,
            y=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
        )
        estimated_density = estimator.conditional_density_grid(
            x=x_test,
            y=y_test,
            reference_samples=normalization_samples,
            seed=seed + 20_000,
        )
        method_details = {
            "selected_weight_decay": estimator.best_weight_decay_,
            "validation_nll": estimator.best_validation_nll_,
            "selection_history": estimator.selection_history_,
            "estimator_config": estimator.get_config(),
        }
    elif method == "rfcde":
        estimator = RandomForestDensityEstimator(
            n_estimators=12 if quick else 200,
            max_depth=4 if quick else 12,
            seed=42,
        )
        estimator.fit(x=x_train, y=y_train, rng=rng)
        estimated_density = estimator.conditional_density_grid(
            x=x_test, y=y_test, normalization_samples=normalization_samples,
            seed=seed + 30_000,
            lower=np.array([scenario.response_lower]),
            upper=np.array([scenario.response_upper]),
        )
        method_details = {
            "normalization_samples": normalization_samples,
            "estimator_config": estimator.get_config(),
        }
    elif method == "maf":
        estimator = ConditionalMAF(
            depth=3, width=64, max_epochs=3 if quick else 200,
            patience=1 if quick else 2, batch_size=500,
            weight_decay_candidates=(1e-4,) if quick else (1e-3, 5e-4, 2e-4, 1e-4, 0.0),
            seed=seed, device=device,
        )
        estimator.fit(x=x_train, y=y_train, x_valid=x_valid, y_valid=y_valid)
        estimated_density = estimator.conditional_density_grid(x=x_test, y=y_test)
        method_details = {
            "selected_weight_decay": estimator.best_weight_decay_,
            "validation_nll": estimator.best_validation_nll_,
            "selection_history": estimator.selection_history_,
            "estimator_config": estimator.get_config(),
        }
    elif method == "lincde":
        estimator = LinCDEEstimator(n_trees=10 if quick else 100, seed=seed)
        estimator.fit(x=x_train, y=y_train)
        raw_density = estimator.raw_conditional_density_grid(x=x_test, y=y_test)
        estimated_density = estimator.conditional_density_grid(
            x=x_test,
            y=y_test,
            normalization_samples=normalization_samples,
            seed=seed + 30_000,
        )
        method_details = {
            "raw_package_empirical_tv": mean_absolute_density_error(
                raw_density,
                true_density,
            ),
            "normalization_samples": normalization_samples,
            "estimator_config": estimator.get_config(),
        }
    else:
        raise ValueError(f"unknown method: {method}")
    empirical_tv = mean_absolute_density_error(estimated_density, true_density)

    contract = seed_contract(base_seed, replication_index)
    contract["numpy_bit_generator"] = rng_name
    if method == "lincde":
        contract["evaluation_normalization_seed"] = seed + 30_000
        contract["r_seed"] = seed

    return {
        "seed": seed,
        "replication_index": replication_index,
        "seed_contract": contract,
        "scenario": scenario_name,
        "method": method,
        "requested_sample_size": sample_size,
        "train_size": train_size,
        "validation_size": validation_size,
        "generated_size": generated_size,
        "test_covariates": test_covariates,
        "test_responses": test_responses,
        "protocol": protocol if method == "cindes" else "method defaults",
        "data_layout": data_layout,
        "quick": quick,
        "empirical_tv": empirical_tv,
        "additive_function_indices": list(scenario.additive_function_indices),
        **method_details,
    }


def save_results(
    results: list[dict[str, object]],
    arguments: argparse.Namespace,
    seeds: list[int],
) -> Path:
    values = [float(result["empirical_tv"]) for result in results]
    payload = {
        "experiment": "table2",
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
        "summary": {
            "empirical_tv_mean": mean(values),
            "empirical_tv_standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        },
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    mode = "quick" if arguments.quick else "full"
    protocol_label = arguments.protocol if arguments.method == "cindes" else "default"
    filename = (
        f"{arguments.method}_{arguments.scenario}_n{arguments.sample_size}_"
        f"{protocol_label}_{arguments.data_layout}_{mode}_"
        f"seed{arguments.seed}_r{arguments.replications}.json"
    )
    path = arguments.output_dir / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    arguments = parse_arguments()
    if arguments.replications < 1:
        raise ValueError("--replications must be positive")
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative")

    seeds = replication_seeds(arguments.seed, arguments.replications)
    results = []
    for replication_index, seed in enumerate(seeds):
        result = run_replication(
            scenario_name=arguments.scenario,
            sample_size=arguments.sample_size,
            base_seed=arguments.seed,
            replication_index=replication_index,
            method=arguments.method,
            device=arguments.device,
            protocol=arguments.protocol,
            data_layout=arguments.data_layout,
            quick=arguments.quick,
        )
        results.append(result)
        print(
            f"[{arguments.method.upper()}] scenario={arguments.scenario} "
            f"requested_n={arguments.sample_size} train_n={result['train_size']} "
            f"seed={seed} empirical_tv={result['empirical_tv']:.6f}"
        )
    path = save_results(results, arguments, seeds)
    print(f"[{arguments.method.upper()}] results={path}")


if __name__ == "__main__":
    main()
