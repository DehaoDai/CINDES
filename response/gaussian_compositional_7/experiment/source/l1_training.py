"""Generic neural CINDES and DSM with a true normalized L1 weight penalty.

No target graph, pairwise feature map, target potentials, target scores or
reference-CDF features enter fitting. Every hidden layer is an ordinary
fully connected ReLU or Softplus layer. L1 applies to all Linear weight entries, including
the output layer, and excludes biases. Adam is used directly on the penalized
objective: near-zero weights are measured, but exact sparsity is not claimed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

import training_loop as loop


WIDTH = 64
DEPTH = 2
TIME_FREQUENCIES = 8
R = 25.0
T = 3.0
T_MIN = 0.001
VALIDATION_DRAWS = 8
L1_GRID = (0.0, 0.01, 0.1, 0.3, 1.0)


def activation_layer(name, beta=1.0):
    if name not in ("relu", "softplus"):
        raise ValueError("activation must be relu or softplus")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("softplus_beta must be finite and positive")
    return nn.ReLU() if name == "relu" else nn.Softplus(beta=float(beta), threshold=20.0)


class GenericMLP(nn.Module):
    """A fully connected raw-input network; output dimension is explicit."""

    def __init__(self, d_in, d_out, width=WIDTH, depth=DEPTH,
                 activation="relu", softplus_beta=1.0):
        super().__init__()
        sizes = [d_in] + [width] * depth
        layers = []
        for index in range(depth):
            layers.extend([nn.Linear(sizes[index], sizes[index + 1]),
                           activation_layer(activation, softplus_beta)])
        layers.append(nn.Linear(width, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, y):
        return self.net(y)


class GenericDSM(nn.Module):
    """Time-conditioned full score, optionally with the known Gaussian score."""

    def __init__(self, d, parameterization, width=WIDTH, depth=DEPTH,
                 n_time_frequencies=TIME_FREQUENCIES, activation="relu", softplus_beta=1.0):
        super().__init__()
        self.parameterization = parameterization
        self.register_buffer("freqs", 2.0 ** torch.arange(n_time_frequencies).float())
        self.mlp = GenericMLP(d + 2 * n_time_frequencies, d, width=width, depth=depth,
                              activation=activation, softplus_beta=softplus_beta)

    def forward(self, y, t):
        if t.ndim != 2 or t.shape != (len(y), 1):
            raise ValueError("DSM requires one scalar time per observation, shape (batch,1)")
        angles = t * self.freqs[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        output = self.mlp(torch.cat([y, embedding], dim=1))
        return output - y if self.parameterization == "residual" else output


def model_spec(method, d, parameterization=None, *, activation="relu", softplus_beta=1.0):
    if method not in ("cindes", "dsm"):
        raise ValueError("method must be cindes or dsm")
    loop._positive_int("d", int(d))
    activation_layer(activation, softplus_beta)
    if method == "cindes":
        if parameterization not in (None, "classifier"):
            raise ValueError("CINDES uses parameterization='classifier'")
        parameterization = "classifier"
    elif parameterization not in ("direct", "residual"):
        raise ValueError("DSM parameterization must be direct or residual")
    return {
        "family": "small_l1_mlp", "method": method, "d": int(d),
        "width": WIDTH, "depth": DEPTH, "activation": activation,
        "softplus_beta": float(softplus_beta), "softplus_threshold": 20.0,
        "input_features": "raw", "parameterization": parameterization,
        "n_time_frequencies": TIME_FREQUENCIES if method == "dsm" else 0,
        "structural_information": "none; generic fully connected network",
    }


def build_model(spec):
    """Rebuild a model using only its portable saved architecture specification."""
    if spec.get("family") != "small_l1_mlp":
        raise ValueError("Expected a small_l1_mlp checkpoint specification")
    if spec.get("activation") not in ("relu", "softplus") or spec.get("input_features") != "raw":
        raise ValueError("This study uses generic raw-input ReLU or Softplus networks")
    beta = float(spec.get("softplus_beta", 1.0))
    activation_layer(spec["activation"], beta)
    if float(spec.get("softplus_threshold", 20.0)) != 20.0:
        raise ValueError("This study fixes the numerical Softplus threshold at 20")
    for name in ("d", "width", "depth"):
        loop._positive_int(name, spec[name])
    if spec["method"] == "cindes":
        if spec["parameterization"] != "classifier" or spec["n_time_frequencies"] != 0:
            raise ValueError("Invalid classifier specification")
        return GenericMLP(spec["d"], 1, width=spec["width"], depth=spec["depth"],
                          activation=spec["activation"], softplus_beta=beta)
    if spec["method"] != "dsm" or spec["parameterization"] not in ("direct", "residual"):
        raise ValueError("Invalid DSM specification")
    loop._positive_int("n_time_frequencies", spec["n_time_frequencies"])
    return GenericDSM(spec["d"], spec["parameterization"], width=spec["width"],
                      depth=spec["depth"], n_time_frequencies=spec["n_time_frequencies"],
                      activation=spec["activation"], softplus_beta=beta)


def linear_weights(model):
    """Return every Linear weight; no biases or fixed Fourier buffers."""
    entries = [(name + ".weight", layer.weight)
               for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]
    if not entries:
        raise ValueError("L1 requires at least one Linear weight matrix")
    return entries


def mean_absolute_weight(model):
    entries = linear_weights(model)
    count = sum(weight.numel() for _, weight in entries)
    return sum(weight.abs().sum() for _, weight in entries) / count


def penalized_objective(model, data_loss, l1_lambda):
    """Data loss + lambda times mean absolute Linear weight, exactly as stated."""
    if l1_lambda == 0.0:
        # Avoid changing the arithmetic or gradient path of the no-L1 baseline.
        return data_loss
    return data_loss + l1_lambda * mean_absolute_weight(model)


@torch.no_grad()
def weight_diagnostics(model):
    """Descriptive coefficient shrinkage, never an inferred target graph."""
    entries = linear_weights(model)
    all_weights = torch.cat([weight.detach().reshape(-1) for _, weight in entries])
    absolute = all_weights.abs()
    result = {
        "weight_mean_abs": float(absolute.double().mean().item()),
        "weight_l1_sum": float(absolute.double().sum().item()),
        "weight_fraction_abs_le_1e_3": float((absolute <= 1e-3).double().mean().item()),
        "weight_fraction_abs_le_1e_2": float((absolute <= 1e-2).double().mean().item()),
        "weight_fraction_exact_zero": float((all_weights == 0).double().mean().item()),
    }
    for index, (_, weight) in enumerate(entries):
        abs_layer = weight.detach().abs()
        result[f"layer_{index:02d}_weight_mean_abs"] = float(abs_layer.double().mean().item())
        result[f"layer_{index:02d}_weight_fraction_abs_le_1e_3"] = float(
            (abs_layer <= 1e-3).double().mean().item())
        result[f"layer_{index:02d}_weight_fraction_abs_le_1e_2"] = float(
            (abs_layer <= 1e-2).double().mean().item())
    return result


def _stream_seeds(method, d, seed):
    # Same model/parameterization shapes share identical random streams across
    # every lambda. Distinct dimensions and training seeds get distinct streams.
    namespace = 30_000_000 + 100_000 * int(d) + 100 * int(seed)
    if method == "cindes":
        return {"initialization": namespace + 1, "permutations": namespace + 2,
                "refreshed_negative_draws": namespace + 3,
                "validation_negative_bank": namespace + 4,
                "validation_density_bank": namespace + 5}
    return {"initialization": namespace + 11, "permutations": namespace + 12,
            "training_times": namespace + 13, "training_noise": namespace + 14,
            "validation_times": namespace + 15, "validation_noise": namespace + 16}


def train_model(train, val, *, method, parameterization=None, l1_lambda,
                seed, output_dir, device="cpu", verbose=False, max_updates=None,
                validation_checks=100, batch_size=256, activation="relu", softplus_beta=1.0,
                validation_criterion="bce", density_normalizer_size=65536,
                retain_validation_states=False):
    """Fit a fixed configuration and select only by unpenalized validation loss.

    CINDES batches are drawn from the pooled 1:2 positive/reference labels;
    every reference observation is freshly simulated. DSM's data loss is
    normalized by dimension before the identical normalized L1 term is added.
    """
    if method not in ("cindes", "dsm"):
        raise ValueError("method must be cindes or dsm")
    if validation_criterion not in ("bce", "density_nll"):
        raise ValueError("CINDES validation_criterion must be bce or density_nll")
    loop._positive_int("density_normalizer_size", density_normalizer_size)
    if not isinstance(retain_validation_states, bool):
        raise ValueError("retain_validation_states must be boolean")
    if isinstance(l1_lambda, bool) or not math.isfinite(l1_lambda) or l1_lambda < 0:
        raise ValueError("l1_lambda must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 1_000_000:
        raise ValueError("seed must be an integer between zero and one million")
    if max_updates is None:
        max_updates = 20000 if method == "cindes" else 60000
    train, val, target = loop._inputs(train, val, device)
    n, d = train.shape
    spec = model_spec(method, d, parameterization, activation=activation, softplus_beta=softplus_beta)
    seeds = _stream_seeds(method, d, seed)
    config = loop._config(method, train, val, seed, target, max_updates,
                          batch_size, validation_checks, seeds)
    config.update({
        "protocol": "small_dimension_generic_activation_l1_training_v1",
        "model_spec": spec, "network_width": WIDTH, "network_depth": DEPTH,
        "activation": activation, "softplus_beta": float(softplus_beta), "feature_map": "raw",
        "learning_rate": .001, "weight_decay": 0.0,
        "l1_lambda": float(l1_lambda),
        "l1_penalty_definition": "lambda * sum(abs(W)) / P, over every Linear weight entry; biases excluded",
        "l1_optimization": "Adam gradient on penalized objective; no proximal thresholding or pruning",
        "sparsity_interpretation": "Small weights are descriptive; no exact sparsity or graph recovery claim",
        "selection_uses_target_truth": False,
        "validation_penalty_included": False,
        "validation_criterion": "unpenalized held-out data loss",
        "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
        "same_rng_across_l1_activation_and_dsm_parameterizations": True,
        "retain_validation_states": retain_validation_states,
    })
    config["source_sha256"][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    model = loop._initialize(lambda: build_model(spec), seeds["initialization"], target)
    entries = linear_weights(model)
    config["weight_parameter_count"] = sum(weight.numel() for _, weight in entries)
    config["total_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    config["penalized_weight_names"] = [name for name, _ in entries]

    if method == "cindes":
        val_q = torch.randn((2 * len(val), d), generator=loop._generator(seeds["validation_negative_bank"]))
        density_q = torch.randn((density_normalizer_size, d),
                                generator=loop._generator(seeds["validation_density_bank"]))
        config.update({
            "R": R, "fake_ratio": 2, "negative_mode": "refresh",
            "training_objective": "mean_pooled_BCE_plus_normalized_L1_weight_penalty",
            "validation_objective": ("normalized_density_NLL_omitting_common_Gaussian_term"
                                     if validation_criterion == "density_nll" else
                                     "unpenalized_mean_pooled_BCE_class_prior_1_to_2"),
            "validation_criterion": validation_criterion,
            "validation_negative_bank_fingerprint": loop._fingerprint(val_q),
            "density_normalizer_size": density_normalizer_size,
            "density_normalizer_bank_fingerprint": loop._fingerprint(density_q),
            "batch_semantics": "batch_size is total observations from pooled target/reference labels",
        })
        z = torch.cat([train, torch.zeros((2 * n, d), dtype=train.dtype)])
        labels = torch.cat([torch.ones(n), torch.zeros(2 * n)])
        validation_z = torch.cat([val, val_q]).to(target)
        validation_labels = torch.cat([torch.ones(len(val)), torch.zeros(len(val_q))]).to(target)
        negative_rng = loop._generator(seeds["refreshed_negative_draws"])

        def loss_for_batch(net, indices):
            points, y = z[indices], labels[indices]
            mask = indices >= n
            n_negative = int(mask.sum().item())
            if n_negative:
                points[mask] = torch.randn((n_negative, d), generator=negative_rng, dtype=points.dtype)
            raw = net(points.to(target)).squeeze(-1)
            data_loss = F.binary_cross_entropy_with_logits(raw.clamp(-R, R), y.to(target))
            return penalized_objective(net, data_loss, l1_lambda), len(indices) - n_negative, n_negative

        def validate(net):
            positive_loss = negative_loss = clipped = 0.0
            largest_logit = 0.0
            positive_logit_sum = 0.0
            for start in range(0, len(validation_z), 1024):
                raw = net(validation_z[start:start + 1024]).squeeze(-1)
                y = validation_labels[start:start + 1024]
                errors = F.binary_cross_entropy_with_logits(raw.clamp(-R, R), y, reduction="none")
                positive_loss += float(errors[y == 1].double().sum().item())
                positive_logit_sum += float(raw.clamp(-R, R)[y == 1].double().sum().item())
                negative_loss += float(errors[y == 0].double().sum().item())
                clipped += float((raw.abs() >= R).sum().item())
                largest_logit = max(largest_logit, float(raw.abs().max().item()))
            bce = (positive_loss + negative_loss) / len(validation_z)
            sum_w = sum_w2 = max_w = 0.0
            for start in range(0, len(density_q), 8192):
                logits = net(density_q[start:start + 8192].to(target)).squeeze(-1).clamp(-R, R).double()
                weights = logits.exp()
                sum_w += float(weights.sum().item())
                sum_w2 += float(weights.square().sum().item())
                max_w = max(max_w, float(weights.max().item()))
            logz = math.log(sum_w / len(density_q))
            density_nll = logz - positive_logit_sum / len(val)
            gaussian_nll = float((.5 * val.double().square().sum(dim=1) + .5 * d * math.log(2 * math.pi)).mean().item())
            return {
                "validation_loss": density_nll if validation_criterion == "density_nll" else bce,
                "validation_bce": bce,
                "validation_density_nll_omit_common_log_phi": density_nll,
                "validation_density_nll_full": density_nll + gaussian_nll,
                "validation_log_normalizer": logz,
                "validation_normalizer_ess": sum_w * sum_w / sum_w2,
                "validation_normalizer_max_normalized_weight": max_w / sum_w,
                "validation_positive_bce": positive_loss / len(val),
                "validation_negative_bce": negative_loss / len(val_q),
                "validation_clipped_fraction": clipped / len(validation_z),
                "validation_largest_abs_raw_logit": largest_logit,
                **weight_diagnostics(net),
            }

        data_length = len(z)
    else:
        config.update({
            "T": T, "t_min": T_MIN, "dsm_parameterization": spec["parameterization"],
            "dsm_time_frequencies": TIME_FREQUENCIES,
            "validation_draws_per_point": VALIDATION_DRAWS,
            "training_objective": "mean_over_observations_AND_coordinates_squared_sigma_score_plus_epsilon_plus_normalized_L1",
            "validation_objective": "unpenalized_mean_over_observations_AND_coordinates_squared_sigma_score_plus_epsilon",
            "time_distribution": "log_uniform_between_t_min_and_T",
            "residual_base_score": "minus_original_y" if spec["parameterization"] == "residual" else None,
            "batch_semantics": "batch_size is target observations, with one fresh time/noise draw each",
        })
        y_train = train.to(target)
        clean = val.repeat_interleave(VALIDATION_DRAWS, dim=0)
        val_t = torch.exp(torch.rand((len(clean), 1), generator=loop._generator(seeds["validation_times"]))
                          * (math.log(T) - math.log(T_MIN)) + math.log(T_MIN))
        val_epsilon = torch.randn(clean.shape, generator=loop._generator(seeds["validation_noise"]))
        val_sigma = torch.sqrt(-torch.expm1(-2.0 * val_t))
        val_y = torch.exp(-val_t) * clean + val_sigma * val_epsilon
        config["validation_bank_fingerprints"] = {
            "t": loop._fingerprint(val_t), "epsilon": loop._fingerprint(val_epsilon),
            "y_t": loop._fingerprint(val_y),
        }
        val_t, val_epsilon, val_sigma, val_y = (x.to(target) for x in (val_t, val_epsilon, val_sigma, val_y))
        time_rng = loop._generator(seeds["training_times"])
        noise_rng = loop._generator(seeds["training_noise"])

        def loss_for_batch(net, indices):
            y0 = y_train[indices.to(target)]
            t = torch.exp(torch.rand((len(indices), 1), generator=time_rng)
                          * (math.log(T) - math.log(T_MIN)) + math.log(T_MIN)).to(target)
            epsilon = torch.randn((len(indices), d), generator=noise_rng).to(target)
            sigma = torch.sqrt(-torch.expm1(-2.0 * t))
            yt = torch.exp(-t) * y0 + sigma * epsilon
            data_loss = ((sigma * net(yt, t) + epsilon) ** 2).mean()
            return penalized_objective(net, data_loss, l1_lambda), len(indices), 0

        def validate(net):
            total = 0.0
            for start in range(0, len(val_y), 1024):
                sl = slice(start, start + 1024)
                errors = ((val_sigma[sl] * net(val_y[sl], val_t[sl]) + val_epsilon[sl]) ** 2).mean(dim=1)
                total += float(errors.double().sum().item())
            value = total / len(val_y)
            return {"validation_loss": value, "validation_loss_per_coordinate": value,
                    "validation_loss_sum_coordinates": value * d, **weight_diagnostics(net)}

        data_length = n

    return loop._run(model, loss_for_batch, validate, data_length,
                     loop._generator(seeds["permutations"]), config,
                     output_dir, target, verbose)


def refit_fixed(train, *, method, parameterization=None, l1_lambda, seed,
                output_dir, stop_updates, scheduler_updates=None, activation="relu",
                softplus_beta=1.0, device="cpu", batch_size=256, verbose=False):
    """Refit on all supplied observations for a previously selected duration.

    No validation observations, target quantities, or generation metrics enter.
    The cosine horizon is scheduler_updates, even when stopping earlier. This
    preserves the schedule prefix selected from the fold training trajectories.
    The caller must freeze lambda and stop_updates before calling this routine.
    Returns (eval_model, metadata); final weights are saved as model.pt.
    """
    if method not in ("cindes", "dsm"):
        raise ValueError("method must be cindes or dsm")
    if isinstance(l1_lambda, bool) or not math.isfinite(l1_lambda) or l1_lambda < 0:
        raise ValueError("l1_lambda must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 1_000_000:
        raise ValueError("seed must be an integer between zero and one million")
    loop._positive_int("batch_size", batch_size)
    if isinstance(stop_updates, bool) or not isinstance(stop_updates, int) or stop_updates < 0:
        raise ValueError("stop_updates must be a nonnegative integer, including initialization at zero")
    if scheduler_updates is None:
        scheduler_updates = 20000 if method == "cindes" else 60000
    loop._positive_int("scheduler_updates", scheduler_updates)
    if stop_updates > scheduler_updates:
        raise ValueError("stop_updates cannot exceed the cosine scheduler horizon")
    train = torch.as_tensor(train).detach().to(device="cpu", dtype=torch.float32).contiguous()
    if train.ndim != 2 or min(train.shape) < 1 or not bool(torch.isfinite(train).all()):
        raise ValueError("Training observations must be a finite nonempty matrix")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Use a CPU or CUDA device")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    n, d = train.shape
    spec = model_spec(method, d, parameterization,
                      activation=activation, softplus_beta=softplus_beta)
    seeds = _stream_seeds(method, d, seed)
    model = loop._initialize(lambda: build_model(spec), seeds["initialization"], target)
    entries = linear_weights(model)
    source_files = {"l1_training.py": Path(__file__),
                    "training_loop.py": Path(loop.__file__),
                    "subgaussian_methods.py": Path(loop.base.__file__)}
    used_seeds = {k: seeds[k] for k in ("initialization", "permutations")}
    used_seeds.update({k: seeds[k] for k in
                      (("refreshed_negative_draws",) if method == "cindes" else
                       ("training_times", "training_noise"))})
    config = {
        "protocol": "fixed_duration_after_cross_validation_refit_v1",
        "method": method, "model_spec": spec, "d": d, "n_train": n,
        "n_validation": 0, "seed": seed, "device": str(target), "dtype": "float32",
        "network_width": WIDTH, "network_depth": DEPTH, "activation": activation,
        "softplus_beta": float(softplus_beta), "input_features": "raw",
        "optimizer": "Adam", "learning_rate": .001, "weight_decay": 0.0,
        "l1_lambda": float(l1_lambda),
        "l1_penalty_definition": "lambda * sum(abs(W)) / P, over every Linear weight entry; biases excluded",
        "l1_optimization": "Adam gradient on penalized objective; no proximal thresholding or pruning",
        "stop_updates": stop_updates, "scheduler_updates": scheduler_updates,
        "scheduler": "CosineAnnealingLR_per_optimizer_update_T_max_scheduler_updates",
        "selection_rule": "lambda_and_duration_supplied_by_caller_frozen_before_refit; no validation selection",
        "batch_size": batch_size, "rng_stream_seeds": used_seeds,
        "training_input_fingerprint": loop._fingerprint(train),
        "selection_uses_target_truth": False,
        "weight_parameter_count": sum(weight.numel() for _, weight in entries),
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "penalized_weight_names": [name for name, _ in entries],
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in source_files.items()},
        "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
    }
    if method == "cindes":
        config.update({"R": R, "fake_ratio": 2, "negative_mode": "refresh",
                       "training_objective": "mean_pooled_BCE_plus_normalized_L1_weight_penalty",
                       "batch_semantics": "batch_size is total observations from pooled target/reference labels"})
        z = torch.cat([train, torch.zeros((2 * n, d), dtype=train.dtype)])
        labels = torch.cat([torch.ones(n), torch.zeros(2 * n)])
        negative_rng = loop._generator(seeds["refreshed_negative_draws"])
        data_length = len(z)

        def loss_for_batch(indices):
            points, y = z[indices], labels[indices]
            mask = indices >= n
            n_negative = int(mask.sum().item())
            if n_negative:
                points[mask] = torch.randn((n_negative, d), generator=negative_rng, dtype=points.dtype)
            raw = model(points.to(target)).squeeze(-1)
            data_loss = F.binary_cross_entropy_with_logits(raw.clamp(-R, R), y.to(target))
            return penalized_objective(model, data_loss, l1_lambda), len(indices) - n_negative, n_negative
    else:
        config.update({"T": T, "t_min": T_MIN, "dsm_parameterization": spec["parameterization"],
                       "dsm_time_frequencies": TIME_FREQUENCIES,
                       "training_objective": "mean_over_observations_AND_coordinates_squared_sigma_score_plus_epsilon_plus_normalized_L1",
                       "time_distribution": "log_uniform_between_t_min_and_T",
                       "batch_semantics": "batch_size is target observations, with one fresh time/noise draw each"})
        y_train = train.to(target)
        time_rng = loop._generator(seeds["training_times"])
        noise_rng = loop._generator(seeds["training_noise"])
        data_length = n

        def loss_for_batch(indices):
            y0 = y_train[indices.to(target)]
            t = torch.exp(torch.rand((len(indices), 1), generator=time_rng)
                          * (math.log(T) - math.log(T_MIN)) + math.log(T_MIN)).to(target)
            epsilon = torch.randn((len(indices), d), generator=noise_rng).to(target)
            sigma = torch.sqrt(-torch.expm1(-2.0 * t))
            yt = torch.exp(-t) * y0 + sigma * epsilon
            data_loss = ((sigma * model(yt, t) + epsilon) ** 2).mean()
            return penalized_objective(model, data_loss, l1_lambda), len(indices), 0

    directory = Path(output_dir)
    with loop._locked(directory):
        metadata_path = directory / "metadata.json"
        checkpoint_path = directory / "model.pt"
        history_path = directory / "training_history.jsonl"
        if metadata_path.exists():
            old = json.loads(metadata_path.read_text())
            if old.get("config") != config or old.get("status") != "complete":
                raise ValueError("Existing refit is incomplete or has a different configuration; use a new directory")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if payload.get("config") != config or payload.get("update") != stop_updates:
                raise ValueError("Refit checkpoint configuration/update mismatch")
            model.load_state_dict(payload["state_dict"], strict=True)
            return model.eval(), dict(old, reused=True)
        if checkpoint_path.exists() or history_path.exists():
            raise ValueError("Unrecognized refit artifacts; use a new directory")
        initial_hash = loop._state_hash(model)
        batch_hash = hashlib.sha256()
        permutation_rng = loop._generator(seeds["permutations"])
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_updates)
        checks = {math.ceil(k * stop_updates / 20) for k in range(1, 21)}
        updates = epoch_index = positives_seen = negatives_seen = 0
        loss_total = 0.0
        loss_count = 0
        batches_per_epoch = math.ceil(data_length / batch_size)
        started = time.perf_counter()
        epoch_fraction = 0.0

        def metadata(status):
            return {"status": status, "config": config, "actual_updates": updates,
                    "actual_epochs": epoch_fraction, "positive_occurrences": positives_seen,
                    "negative_occurrences": negatives_seen,
                    "initial_state_sha256": initial_hash,
                    "batch_schedule_sha256": batch_hash.hexdigest(),
                    "wall_seconds": time.perf_counter() - started,
                    "timing_scope": "all refit operations including progress serialization; no validation",
                    "checkpoint": "model.pt", "reused": False}

        loop._json(metadata_path, metadata("in_progress"))
        model.train()
        while updates < stop_updates:
            epoch_index += 1
            order = torch.randperm(data_length, generator=permutation_rng, device="cpu")
            for batch_index, start in enumerate(range(0, data_length, batch_size)):
                indices = order[start:start + batch_size]
                batch_hash.update(indices.numpy().tobytes())
                loss, positives, negatives = loss_for_batch(indices)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite refit training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                updates += 1
                positives_seen += positives
                negatives_seen += negatives
                loss_total += float(loss.detach().item()) * len(indices)
                loss_count += len(indices)
                epoch_fraction = epoch_index - 1 + (batch_index + 1) / batches_per_epoch
                if updates in checks:
                    loop._sync(target)
                    row = {"update": updates, "epoch": epoch_fraction,
                           "training_loss_since_previous_check": loss_total / loss_count,
                           "learning_rate_next_update": float(optimizer.param_groups[0]["lr"]),
                           "positive_occurrences": positives_seen, "negative_occurrences": negatives_seen}
                    with history_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, allow_nan=False) + "\n")
                    loss_total = 0.0
                    loss_count = 0
                    loop._json(metadata_path, metadata("in_progress"))
                    if verbose:
                        print(f"[{method} {activation}] fixed refit update={updates}/{stop_updates}", flush=True)
                if updates == stop_updates:
                    break
        loop._sync(target)
        model.eval()
        payload = {"format_version": 2, "method": method, "config": config,
                   "model_config": spec,
                   "state_dict": {name: tensor.detach().cpu().clone()
                                  for name, tensor in model.state_dict().items()},
                   "update": updates, "epoch": epoch_fraction,
                   "initial_state_sha256": initial_hash,
                   "batch_schedule_sha256": batch_hash.hexdigest(),
                   "selection": "fixed hyperparameters and duration; no refit validation"}
        loop._atomic(checkpoint_path, lambda fh: torch.save(payload, fh), binary=True)
        result = metadata("complete")
        result["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        result["final_state_sha256"] = loop._state_hash(model)
        result["weight_diagnostics"] = weight_diagnostics(model)
        loop._json(metadata_path, result)
        return model, result


def objective_gradient_check():
    """Independently verify the added derivative on weights and excluded biases."""
    results = []
    for method, parameterization in (("cindes", "classifier"), ("dsm", "direct"), ("dsm", "residual")):
        model = loop._initialize(lambda: build_model(model_spec(method, 3, parameterization)), 823, "cpu").double()
        # Include an exact-zero entry to check the documented zero subgradient.
        with torch.no_grad():
            next(parameter for parameter in model.parameters() if parameter.ndim == 2).flatten()[0] = 0.
        y = torch.tensor([[.2, -.3, .4], [.1, .7, -.2], [-.8, .4, .1], [.6, -.5, -.2]], dtype=torch.float64)
        if method == "cindes":
            labels = torch.tensor([1., 0., 0., 1.], dtype=torch.float64)
            data_loss = F.binary_cross_entropy_with_logits(model(y).squeeze(-1).clamp(-R, R), labels)
        else:
            times = torch.tensor([[.001], [.03], [.8], [3.]], dtype=torch.float64)
            epsilon = torch.tensor([[.1, -.2, .3], [.4, -.2, -.1], [.7, .5, -.4], [-.8, .3, .6]], dtype=torch.float64)
            sigma = torch.sqrt(-torch.expm1(-2 * times))
            data_loss = ((sigma * model(y, times) + epsilon)**2).mean()
        named = list(model.named_parameters())
        parameters = [parameter for _, parameter in named]
        data_grads = torch.autograd.grad(data_loss, parameters, retain_graph=True)
        zero_objective = penalized_objective(model, data_loss, 0.0)
        zero_grads = torch.autograd.grad(zero_objective, parameters, retain_graph=True)
        if zero_objective is not data_loss or any(not torch.equal(a, b) for a, b in zip(data_grads, zero_grads)):
            raise AssertionError("The lambda-zero baseline changed")
        coefficient = .3
        penalized = penalized_objective(model, data_loss, coefficient)
        grads = torch.autograd.grad(penalized, parameters)
        # Matrix-valued trainable parameters are exactly the Linear weights in
        # this fixed architecture; do not use the penalty's enumeration helper.
        p = sum(parameter.numel() for parameter in parameters if parameter.ndim == 2)
        errors = {}
        for (name, parameter), before, after in zip(named, data_grads, grads):
            expected = coefficient * parameter.detach().sign() / p if parameter.ndim == 2 else torch.zeros_like(parameter)
            error = float((after - before - expected).abs().max().item())
            if error > 1e-14:
                raise AssertionError((method, parameterization, name, error))
            errors[name] = error
        results.append({"method": method, "parameterization": parameterization,
                        "dimension": 3, "lambda": coefficient, "weight_count": p,
                        "lambda_zero_gradient_bitwise_unchanged": True,
                        "max_derivative_errors_by_parameter": errors,
                        "max_derivative_error": max(errors.values())})
    return {"check": "Added L1 derivative is lambda*sign(W)/P for every Linear weight and zero for all biases",
            "dtype": "float64", "tolerance": 1e-14, "results": results,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Run the objective-gradient check; no data fitting")
    parser.add_argument("--save-check", type=Path)
    args = parser.parse_args()
    if not args.check_only:
        parser.error("This module exposes train_model; use --check-only for its independent implementation check")
    result = objective_gradient_check()
    if args.save_check:
        args.save_check.parent.mkdir(parents=True, exist_ok=True)
        args.save_check.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
