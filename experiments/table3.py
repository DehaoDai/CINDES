"""Run the multivariate-response conditional density experiment in Table 3."""

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from cindes.config import NetworkConfig, OptimizationConfig, ValidationConfig
from cindes.data import multivariate_conditional_scenario
from cindes.estimators import (
    ConditionalMAF,
    ExplicitCINDES,
    RandomForestDensityEstimator,
)
from cindes.metrics import box_integrated_total_variation, mean_absolute_density_error
from cindes.reproducibility import (
    replication_seeds,
    runtime_metadata,
    seed_contract,
    seed_everything,
)

COVARIATE_DIMENSION = 16
RESPONSE_DIMENSIONS = (4, 8, 12, 16)
VALIDATION_RATIO = 0.25
TEST_COVARIATES = 500
TEST_RESPONSES = 500
NORMALIZATION_SAMPLES = 1_000
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 2
BATCH_SIZE = 500
SAMPLE_SIZES = (500, 2_000, 8_000)
METHODS = ("cindes", "rfcde", "maf")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, choices=SAMPLE_SIZES, default=500)
    parser.add_argument("--response-dimension", type=int, choices=RESPONSE_DIMENSIONS, default=4)
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; replication r uses seed + r.",
    )
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--data-layout",
        choices=("archive", "declared"),
        default="archive",
        help="Reproduce the archived split or use the sample sizes stated in the article.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/table3"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small smoke experiment; quick results are not full-scale.",
    )
    return parser.parse_args()


def _split_data(
    x: np.ndarray,
    y: np.ndarray,
    train_size: int,
    validation_size: int,
    test_size: int,
) -> tuple[np.ndarray, ...]:
    if train_size + validation_size + test_size > x.shape[0]:
        raise ValueError("requested split is larger than the generated data")
    validation_stop = train_size + validation_size
    return (
        x[:train_size],
        y[:train_size],
        x[train_size:validation_stop],
        y[train_size:validation_stop],
        x[-test_size:],
    )


def _clone_random_state(rng: np.random.RandomState) -> np.random.RandomState:
    clone = np.random.RandomState()
    clone.set_state(rng.get_state())
    return clone


def _evaluate_cindes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    device: str,
    quick: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    optimization = OptimizationConfig(
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=0.0,
        scheduler_gamma=0.999,
        fake_response_strategy="batch_resampled",
        reset_seed_per_candidate=False,
    )
    validation = ValidationConfig(
        reference_samples=NORMALIZATION_SAMPLES,
        batch_size=128,
    )
    if quick:
        optimization = replace(
            optimization,
            max_epochs=3,
            patience=1,
            weight_decay_candidates=(1e-4,),
        )
        validation = replace(validation, reference_samples=64, batch_size=32)

    estimator = ExplicitCINDES(
        network=NetworkConfig(depth=3, width=64),
        optimization=optimization,
        validation=validation,
        seed=seed,
        device=device,
    )
    estimator.fit(x=x_train, y=y_train, x_valid=x_valid, y_valid=y_valid)
    density = estimator.conditional_density_grid(
        x=x_test,
        y=y_test,
        reference_samples=validation.reference_samples,
        seed=seed + 20_000,
    )
    details = {
        "selected_weight_decay": estimator.best_weight_decay_,
        "validation_nll": estimator.best_validation_nll_,
        "selection_history": estimator.selection_history_,
        "estimator_config": estimator.get_config(),
    }
    return density, details


def _evaluate_rfcde(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_rng: np.random.RandomState,
    seed: int,
    quick: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    estimator = RandomForestDensityEstimator(
        n_estimators=12 if quick else 200,
        max_depth=4 if quick else 12,
        seed=42,
    )
    estimator.fit(x=x_train, y=y_train, rng=model_rng)
    normalization_samples = 64 if quick else NORMALIZATION_SAMPLES
    density = estimator.conditional_density_grid(
        x=x_test,
        y=y_test,
        normalization_samples=normalization_samples,
        seed=seed + 30_000,
    )
    details = {
        "estimator_config": estimator.get_config(),
        "normalization_samples": normalization_samples,
        "reference_lower": estimator.reference_lower_.tolist(),
        "reference_upper": estimator.reference_upper_.tolist(),
    }
    return density, details


def _evaluate_maf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    device: str,
    quick: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    estimator = ConditionalMAF(
        depth=3,
        width=64,
        learning_rate=1e-3,
        max_epochs=3 if quick else MAX_EPOCHS,
        patience=1 if quick else EARLY_STOPPING_PATIENCE,
        batch_size=BATCH_SIZE,
        weight_decay_candidates=(1e-4,) if quick else (1e-3, 5e-4, 2e-4, 1e-4, 0.0),
        scheduler_gamma=0.999,
        seed=seed,
        device=device,
    )
    estimator.fit(x=x_train, y=y_train, x_valid=x_valid, y_valid=y_valid)
    density = estimator.conditional_density_grid(x=x_test, y=y_test)
    details = {
        "selected_weight_decay": estimator.best_weight_decay_,
        "validation_nll": estimator.best_validation_nll_,
        "selection_history": estimator.selection_history_,
        "estimator_config": estimator.get_config(),
    }
    return density, details


def run_replication(
    sample_size: int,
    base_seed: int,
    replication_index: int,
    methods: tuple[str, ...],
    device: str,
    data_layout: str,
    quick: bool,
    response_dimension: int = 4,
) -> dict[str, object]:
    if response_dimension not in RESPONSE_DIMENSIONS:
        raise ValueError(f"unsupported response dimension: {response_dimension}")
    seed = base_seed + replication_index
    seed_everything(seed)
    rng = np.random.RandomState(seed)
    test_covariates = 12 if quick else TEST_COVARIATES
    test_responses = 16 if quick else TEST_RESPONSES
    layout_sample_size = 128 if quick else sample_size
    requested_validation_size = int(VALIDATION_RATIO * layout_sample_size)
    generated_size = layout_sample_size + requested_validation_size + test_covariates

    x_all = rng.uniform(0.0, 1.0, size=(generated_size, COVARIATE_DIMENSION))
    if data_layout == "archive":
        scenario_rng = rng
        weight_matrix_protocol = "redrawn_each_replication_after_covariates"
    elif data_layout == "declared":
        scenario_rng = np.random.RandomState(base_seed)
        weight_matrix_protocol = "fixed_across_replications_from_base_seed"
    else:
        raise ValueError(f"unknown data layout: {data_layout}")
    scenario = multivariate_conditional_scenario(
        scenario_rng,
        covariate_dimension=COVARIATE_DIMENSION,
        response_dimension=response_dimension,
    )
    y_all = scenario.sample_responses(x_all, rng)
    if data_layout == "archive":
        train_validation_size = generated_size - test_covariates
        train_size = int(0.4 * train_validation_size)
        validation_size = int(0.1 * train_validation_size)
    elif data_layout == "declared":
        train_size = layout_sample_size
        validation_size = requested_validation_size
    x_train, y_train, x_valid, y_valid, x_test = _split_data(
        x_all,
        y_all,
        train_size,
        validation_size,
        test_covariates,
    )
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_valid = x_valid.astype(np.float32)
    y_valid = y_valid.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = rng.uniform(
        scenario.response_lower,
        scenario.response_upper,
        size=(test_responses, response_dimension),
    ).astype(np.float32)
    model_rng = _clone_random_state(rng)
    true_density = scenario.density_grid(x_test, y_test)

    method_results = {}
    for method in methods:
        if method == "cindes":
            estimated_density, details = _evaluate_cindes(
                x_train,
                y_train,
                x_valid,
                y_valid,
                x_test,
                y_test,
                seed,
                device,
                quick,
            )
        elif method == "rfcde":
            estimated_density, details = _evaluate_rfcde(
                x_train,
                y_train,
                x_test,
                y_test,
                _clone_random_state(model_rng),
                seed,
                quick,
            )
        elif method == "maf":
            estimated_density, details = _evaluate_maf(
                x_train,
                y_train,
                x_valid,
                y_valid,
                x_test,
                y_test,
                seed,
                device,
                quick,
            )
        else:
            raise ValueError(f"unknown method: {method}")

        reported_error = mean_absolute_density_error(estimated_density, true_density)
        integrated_tv = box_integrated_total_variation(
            estimated_density,
            true_density,
            lower=np.full(response_dimension, scenario.response_lower),
            upper=np.full(response_dimension, scenario.response_upper),
        )
        method_results[method] = {
            "mean_absolute_density_error": reported_error,
            "reported_empirical_tv": integrated_tv,
            "box_integrated_tv": integrated_tv,
            **details,
        }

    contract = seed_contract(base_seed, replication_index)
    contract["numpy_bit_generator"] = "MT19937"
    return {
        "seed": seed,
        "replication_index": replication_index,
        "seed_contract": contract,
        "requested_sample_size": sample_size,
        "train_size": train_size,
        "validation_size": validation_size,
        "generated_size": generated_size,
        "test_covariates": test_covariates,
        "test_responses": test_responses,
        "covariate_dimension": COVARIATE_DIMENSION,
        "response_dimension": response_dimension,
        "data_layout": data_layout,
        "weight_matrix_protocol": weight_matrix_protocol,
        "quick": quick,
        "weight_matrix": scenario.weight_matrix.tolist(),
        "methods": method_results,
    }


def _summarize(results: list[dict[str, object]], methods: tuple[str, ...]) -> dict:
    summary = {}
    for method in methods:
        reported_values = [
            float(result["methods"][method]["reported_empirical_tv"]) for result in results
        ]
        integrated_values = [
            float(result["methods"][method]["box_integrated_tv"]) for result in results
        ]
        summary[method] = {
            "reported_empirical_tv_mean": mean(reported_values),
            "reported_empirical_tv_standard_deviation": (
                stdev(reported_values) if len(reported_values) > 1 else 0.0
            ),
            "box_integrated_tv_mean": mean(integrated_values),
            "box_integrated_tv_standard_deviation": (
                stdev(integrated_values) if len(integrated_values) > 1 else 0.0
            ),
        }
    return summary


def save_results(
    results: list[dict[str, object]],
    arguments: argparse.Namespace,
    methods: tuple[str, ...],
    seeds: list[int],
) -> Path:
    payload = {
        "experiment": "table3",
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
        "reported_metric": "0.5 * 2**response_dimension * mean(abs(estimated_density - true_density))",
        "archive_metric": "mean(abs(estimated_density - true_density))",
        "runtime": runtime_metadata(),
        "results": results,
        "summary": _summarize(results, methods),
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    mode = "quick" if arguments.quick else "full"
    method_label = "all" if len(methods) == len(METHODS) else methods[0]
    filename = (
        f"table3_dy{arguments.response_dimension}_n{arguments.sample_size}_{method_label}_{arguments.data_layout}_"
        f"{mode}_seed{arguments.seed}_r{arguments.replications}.json"
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
    methods = METHODS if arguments.method == "all" else (arguments.method,)
    seeds = replication_seeds(arguments.seed, arguments.replications)
    results = []
    for replication_index, seed in enumerate(seeds):
        result = run_replication(
            sample_size=arguments.sample_size,
            base_seed=arguments.seed,
            replication_index=replication_index,
            methods=methods,
            device=arguments.device,
            data_layout=arguments.data_layout,
            quick=arguments.quick,
            response_dimension=arguments.response_dimension,
        )
        results.append(result)
        values = " ".join(
            f"{method}={result['methods'][method]['reported_empirical_tv']:.6f}"
            for method in methods
        )
        print(
            f"[Table 3] requested_n={arguments.sample_size} "
            f"train_n={result['train_size']} seed={seed} {values}"
        )
    path = save_results(results, arguments, methods, seeds)
    print(f"[Table 3] results={path}")


if __name__ == "__main__":
    main()
