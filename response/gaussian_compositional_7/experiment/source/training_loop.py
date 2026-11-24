"""Validation-selected CINDES/DSM fits with unchanged networks and objectives.

The caller supplies disjoint positive training and validation observations;
this module does not choose a validation fraction or split the data. Each fit
runs its complete, exact update budget. Selection uses only a fixed held-out
classification or denoising bank, never an analytic target score or density.
Both methods receive 100 validation opportunities plus initialization by
default, at common fractions of their respective update budgets.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch
from torch.nn import functional as F

import subgaussian_methods as base


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _inputs(Y_train, Y_val, device):
    train = torch.as_tensor(Y_train).detach().to(device="cpu", dtype=torch.float32).contiguous()
    val = torch.as_tensor(Y_val).detach().to(device="cpu", dtype=torch.float32).contiguous()
    if (train.ndim != 2 or val.ndim != 2 or train.shape[1] != val.shape[1]
            or min(len(train), len(val), train.shape[1]) < 1):
        raise ValueError("Training and validation inputs must be nonempty matrices with equal dimension")
    if not bool(torch.isfinite(train).all()) or not bool(torch.isfinite(val).all()):
        raise ValueError("Training and validation inputs must be finite")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Use a CPU or CUDA device")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return train, val, target


def _fingerprint(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}


def _state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _initialize(factory, seed, device):
    # Use the usual Linear initialization, with an isolated CPU RNG state.
    # No training draw or initialization modifies another fit's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(_generator(seed).get_state())
        model = factory()
    return model.to(device)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _atomic(path, writer, binary=False):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
                mode="wb" if binary else "w",
                **({} if binary else {"encoding": "utf-8"})) as fh:
            temporary = Path(fh.name)
            writer(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _json(path, value):
    _atomic(path, lambda fh: json.dump(value, fh, indent=2, allow_nan=False))


@contextmanager
def _locked(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another training process is using {output_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _config(method, train, val, seed, device, max_updates, batch_size,
            validation_checks, stream_seeds):
    for name, value in (("max_updates", max_updates), ("batch_size", batch_size),
                        ("validation_checks", validation_checks)):
        _positive_int(name, value)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63 - 20000:
        raise ValueError("seed must be a nonnegative integer below 2**63-20000")
    checks = sorted({math.ceil(k * max_updates / validation_checks)
                     for k in range(1, validation_checks + 1)})
    files = {Path(__file__).name: Path(__file__),
             "subgaussian_methods.py": Path(inspect.getfile(base))}
    return {
        "protocol": "validation_selected_improvement_sweep_v1", "method": method,
        "n_train": len(train), "n_val": len(val), "d": train.shape[1], "seed": seed,
        "device": str(device), "dtype": "float32", "network_width": 128, "network_depth": 3,
        "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.00001,
        "max_updates": max_updates, "batch_size": batch_size,
        "scheduler": "CosineAnnealingLR_per_optimizer_update_T_max_max_updates",
        "scheduler_difference_from_original": "Original schedule advanced once per epoch",
        "validation_checks_requested": validation_checks,
        "validation_update_schedule": checks,
        "selection_rule": "minimum_validation_loss_including_initialization_no_patience",
        "validation_split": "supplied_by_caller_no_automatic_split",
        "training_input_fingerprint": _fingerprint(train),
        "validation_input_fingerprint": _fingerprint(val),
        "rng_stream_seeds": stream_seeds,
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in files.items()},
    }


def _run(model, loss_for_batch, validate, data_length, permutation_rng, config,
         output_dir, device, verbose):
    """Common exact-budget loop; loss_for_batch returns (loss, positives, negatives)."""
    output_dir = Path(output_dir)
    retain_states = bool(config.get("retain_validation_states", False))
    with _locked(output_dir):
        metadata_path = output_dir / "metadata.json"
        artifacts = [output_dir / name for name in ("history.jsonl", "best.pt", "last.pt")]
        if retain_states:
            artifacts.append(output_dir / "trajectory.pt")
        if metadata_path.exists():
            old = json.loads(metadata_path.read_text())
            if old.get("config") != config:
                raise ValueError(f"Different configuration/data in {output_dir}; use a new output directory")
            if old.get("status") != "complete":
                raise ValueError(f"Incomplete fit in {output_dir}; use a new output directory to restart")
            if retain_states and not (output_dir / "trajectory.pt").exists():
                raise ValueError("A complete fit is missing its requested validation trajectory")
            payload = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
            if payload.get("config") != config:
                raise ValueError("Best checkpoint configuration does not match metadata")
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            old = dict(old, reused=True)
            if verbose:
                print(f"REUSED {config['method']} best checkpoint at update {old['best_update']}", flush=True)
            return model, old
        if any(path.exists() for path in artifacts):
            raise ValueError(f"Unrecognized existing training artifacts in {output_dir}; use a new directory")

        max_updates, batch_size = config["max_updates"], config["batch_size"]
        checks = set(config["validation_update_schedule"])
        batches_per_epoch = math.ceil(data_length / batch_size)
        opt = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
                               weight_decay=float(config["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_updates)
        initial_hash = _state_hash(model)
        batch_hash = hashlib.sha256()
        best_loss, best_update, best_epoch = float("inf"), None, None
        updates = epoch_index = positives_seen = negatives_seen = check_index = 0
        training_seconds = validation_seconds = 0.0
        started = time.perf_counter()
        last_metrics = None
        retained_states = {}

        def record(epoch_fraction, train_loss):
            nonlocal best_loss, best_update, best_epoch, validation_seconds, check_index, last_metrics
            _sync(device)
            val_started = time.perf_counter()
            was_training = model.training
            model.eval()
            with torch.no_grad():
                metrics = validate(model)
            model.train(was_training)
            _sync(device)
            validation_seconds += time.perf_counter() - val_started
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError("Non-finite validation metric")
            last_metrics = metrics
            improved = metrics["validation_loss"] < best_loss
            if improved:
                best_loss, best_update, best_epoch = metrics["validation_loss"], updates, epoch_fraction
            row = {
                "check": check_index, "update": updates, "epoch": epoch_fraction,
                "epoch_index": epoch_index, "training_loss_since_previous_check": train_loss,
                "learning_rate_next_update": float(opt.param_groups[0]["lr"]),
                "positive_occurrences": positives_seen, "negative_occurrences": negatives_seen,
                "selected_as_best": improved, **metrics,
            }
            with (output_dir / "history.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, allow_nan=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            payload = {
                "format_version": 1, "method": config["method"], "config": config,
                "model_config": dict(config.get("model_spec", {
                    "d": config["d"], "width": config.get("network_width", 128),
                    "depth": config.get("network_depth", 3), "dsm_parameterization": "direct"})),
                "state_dict": {name: tensor.detach().cpu().clone()
                               for name, tensor in model.state_dict().items()},
                "update": updates, "epoch": epoch_fraction,
                "validation_loss": metrics["validation_loss"], "validation_metrics": metrics,
                "initial_state_sha256": initial_hash,
                "batch_schedule_sha256": batch_hash.hexdigest(),
            }
            if retain_states:
                retained_states[updates] = payload["state_dict"]
            if improved:
                _atomic(output_dir / "best.pt", lambda fh: torch.save(payload, fh), binary=True)
            _atomic(output_dir / "last.pt", lambda fh: torch.save(payload, fh), binary=True)
            check_index += 1
            _json(metadata_path, metadata("in_progress", epoch_fraction))
            if verbose:
                print(f"[{config['method']}] update={updates}/{max_updates} epoch={epoch_fraction:.4f} "
                      f"validation={metrics['validation_loss']:.7g} best={best_loss:.7g}"
                      + (" NEW_BEST" if improved else ""), flush=True)

        def metadata(status, epoch_fraction):
            return {
                "status": status, "config": config, "method": config["method"],
                "actual_updates": updates, "actual_epochs": epoch_fraction,
                "best_update": best_update, "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "last_validation_loss": None if last_metrics is None else last_metrics["validation_loss"],
                "last_validation_metrics": last_metrics,
                "validation_evaluations": check_index,
                "initial_state_sha256": initial_hash,
                "batch_schedule_sha256": batch_hash.hexdigest(),
                "positive_occurrences": positives_seen, "negative_occurrences": negatives_seen,
                "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                "wall_seconds": time.perf_counter() - started,
                "timing_scope": "training_seconds excludes validation and checkpoint serialization",
                "best_checkpoint": "best.pt", "last_checkpoint": "last.pt",
                "torch_version": str(torch.__version__), "reused": False,
                "retained_validation_states": len(retained_states),
                "trajectory_checkpoint": "trajectory.pt" if retain_states else None,
            }

        record(0.0, None)
        model.train()
        accumulated_loss = 0.0
        accumulated_count = 0
        epoch_fraction = 0.0
        segment_started = time.perf_counter()
        while updates < max_updates:
            epoch_index += 1
            order = torch.randperm(data_length, generator=permutation_rng, device="cpu")
            for batch_index, start in enumerate(range(0, data_length, batch_size)):
                indices = order[start:start + batch_size]
                batch_hash.update(indices.numpy().tobytes())
                loss, positive_count, negative_count = loss_for_batch(model, indices)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite training loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                scheduler.step()
                updates += 1
                positives_seen += positive_count
                negatives_seen += negative_count
                accumulated_loss += float(loss.detach().item()) * len(indices)
                accumulated_count += len(indices)
                epoch_fraction = epoch_index - 1 + (batch_index + 1) / batches_per_epoch
                if updates in checks:
                    _sync(device)
                    training_seconds += time.perf_counter() - segment_started
                    record(epoch_fraction, accumulated_loss / accumulated_count)
                    accumulated_loss, accumulated_count = 0.0, 0
                    segment_started = time.perf_counter()
                if updates == max_updates:
                    break
        if retain_states:
            trajectory = {"format_version": 1, "model_config": config["model_spec"],
                          "config": config, "states": retained_states}
            _atomic(output_dir / "trajectory.pt", lambda fh: torch.save(trajectory, fh), binary=True)
        result = metadata("complete", epoch_fraction)
        _json(metadata_path, result)
        best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(best["state_dict"], strict=True)
        return model.eval(), result


def train_cindes_validated(Y_train, Y_val, *, negative_mode="fixed", R=25.0,
                           max_updates=20000, batch_size=256, fake_ratio=2,
                           seed=0, device="cpu", output_dir=Path("cindes_validated"),
                           verbose=True, validation_checks=100):
    """Fit the original classifier and select by fixed 1:fake_ratio validation BCE.

    Fixed and refresh modes share exactly the same initialization, positive
    occurrences, labels, and shuffled batch indices. Refresh replaces only the
    negative rows of each minibatch by fresh N(0,I) observations.
    """
    if negative_mode not in ("fixed", "refresh"):
        raise ValueError("negative_mode must be 'fixed' or 'refresh'")
    _positive_int("fake_ratio", fake_ratio)
    if not math.isfinite(R) or R <= 0:
        raise ValueError("R must be positive and finite")
    train, val, target = _inputs(Y_train, Y_val, device)
    seeds = {"initialization": seed, "training_negative_bank": seed + 1,
             "refreshed_negative_draws": seed + 1, "permutations": seed + 101,
             "validation_negative_bank": seed + 10001}
    config = _config("cindes", train, val, seed, target, max_updates, batch_size,
                     validation_checks, seeds)
    n, d = train.shape
    fixed_negatives = torch.randn((fake_ratio * n, d), generator=_generator(seeds["training_negative_bank"]))
    val_negatives = torch.randn((fake_ratio * len(val), d), generator=_generator(seeds["validation_negative_bank"]))
    config.update({"negative_mode": negative_mode, "R": float(R), "fake_ratio": fake_ratio,
                   "validation_objective": "BCE_with_class_weights_1_to_fake_ratio",
                   "initial_negative_bank_fingerprint": _fingerprint(fixed_negatives),
                   "validation_negative_bank_fingerprint": _fingerprint(val_negatives)})
    z = torch.cat([train, fixed_negatives]).to(target)
    labels = torch.cat([torch.ones(n), torch.zeros(fake_ratio * n)]).to(target)
    validation_z = torch.cat([val, val_negatives]).to(target)
    validation_labels = torch.cat([torch.ones(len(val)), torch.zeros(len(val_negatives))]).to(target)
    refresh_rng = _generator(seeds["refreshed_negative_draws"])
    model = _initialize(lambda: base.MLP(d, 1), seeds["initialization"], target)

    def loss_for_batch(net, indices):
        ids = indices.to(target)
        points, y = z[ids], labels[ids]
        negative_positions = indices >= n
        n_negative = int(negative_positions.sum().item())
        if negative_mode == "refresh" and n_negative:
            points[negative_positions.to(target)] = torch.randn(
                (n_negative, d), generator=refresh_rng).to(target)
        logits = net(points).squeeze(-1).clamp(-R, R)
        return F.binary_cross_entropy_with_logits(logits, y), len(indices) - n_negative, n_negative

    def validate(net):
        positive_loss = negative_loss = correct = clipped = 0.0
        for start in range(0, len(validation_z), 1024):
            raw = net(validation_z[start:start + 1024]).squeeze(-1)
            y = validation_labels[start:start + 1024]
            logits = raw.clamp(-R, R)
            losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            positive_loss += float(losses[y == 1].double().sum().item())
            negative_loss += float(losses[y == 0].double().sum().item())
            correct += float(((logits >= 0) == (y == 1)).sum().item())
            clipped += float((raw.abs() > R).sum().item())
        return {
            "validation_loss": (positive_loss + negative_loss) / len(validation_z),
            "validation_positive_bce": positive_loss / len(val),
            "validation_negative_bce": negative_loss / len(val_negatives),
            "validation_accuracy": correct / len(validation_z),
            "validation_clipped_fraction": clipped / len(validation_z),
        }

    return _run(model, loss_for_batch, validate, len(z), _generator(seeds["permutations"]),
                config, output_dir, target, verbose)


def train_dsm_validated(Y_train, Y_val, *, max_updates=60000, batch_size=256,
                        T=3.0, t_min=0.001, seed=0, device="cpu",
                        output_dir=Path("dsm_validated"), verbose=True,
                        validation_checks=100, validation_draws=4):
    """Fit direct-score DSM, selecting by a fixed held-out denoising bank.

    The bank has four independent log-uniform time/noise draws per supplied
    validation observation by default. The objective is the original
    mean(sum((sigma*score + epsilon)^2)); it is not an analytic score error.
    """
    _positive_int("validation_draws", validation_draws)
    if not math.isfinite(T) or not math.isfinite(t_min) or not 0 < t_min < T:
        raise ValueError("Require finite 0 < t_min < T")
    train, val, target = _inputs(Y_train, Y_val, device)
    seeds = {"initialization": seed + 10, "permutations": seed + 111,
             "training_times": seed + 112, "training_noise": seed + 113,
             "validation_times": seed + 10012, "validation_noise": seed + 10013}
    config = _config("dsm", train, val, seed, target, max_updates, batch_size,
                     validation_checks, seeds)
    config.update({"T": float(T), "t_min": float(t_min), "dsm_parameterization": "direct",
                   "dsm_time_frequencies": 8, "validation_draws_per_point": validation_draws,
                   "validation_objective": "mean_sum_coordinates_squared_sigma_score_plus_epsilon"})
    n, d = train.shape
    y_train = train.to(target)
    clean = val.repeat_interleave(validation_draws, dim=0)
    val_t = torch.exp(torch.rand((len(clean), 1), generator=_generator(seeds["validation_times"]))
                      * (math.log(T) - math.log(t_min)) + math.log(t_min))
    val_eps = torch.randn(clean.shape, generator=_generator(seeds["validation_noise"]))
    val_sigma = torch.sqrt(-torch.expm1(-2.0 * val_t))
    val_y = torch.exp(-val_t) * clean + val_sigma * val_eps
    config["validation_bank_fingerprints"] = {
        "t": _fingerprint(val_t), "epsilon": _fingerprint(val_eps), "y_t": _fingerprint(val_y)}
    val_t, val_eps, val_sigma, val_y = (x.to(target) for x in (val_t, val_eps, val_sigma, val_y))
    time_rng, noise_rng = _generator(seeds["training_times"]), _generator(seeds["training_noise"])
    model = _initialize(lambda: base.ScoreNet(d, gaussian_residual=False), seeds["initialization"], target)

    def loss_for_batch(net, indices):
        y0 = y_train[indices.to(target)]
        t = torch.exp(torch.rand((len(indices), 1), generator=time_rng)
                      * (math.log(T) - math.log(t_min)) + math.log(t_min)).to(target)
        epsilon = torch.randn((len(indices), d), generator=noise_rng).to(target)
        sigma = torch.sqrt(-torch.expm1(-2.0 * t))
        yt = torch.exp(-t) * y0 + sigma * epsilon
        loss = ((sigma * net(yt, t) + epsilon) ** 2).sum(dim=1).mean()
        return loss, len(indices), 0

    def validate(net):
        total = 0.0
        for start in range(0, len(val_y), 1024):
            sl = slice(start, start + 1024)
            errors = ((val_sigma[sl] * net(val_y[sl], val_t[sl]) + val_eps[sl]) ** 2).sum(dim=1)
            total += float(errors.double().sum().item())
        value = total / len(val_y)
        return {"validation_loss": value, "validation_loss_per_coordinate": value / d}

    return _run(model, loss_for_batch, validate, n, _generator(seeds["permutations"]),
                config, output_dir, target, verbose)
