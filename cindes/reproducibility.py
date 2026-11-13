"""Utilities for deterministic experiment execution."""

import os
import platform
import random
import sys

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch from one experiment seed."""

    if seed < 0:
        raise ValueError("seed must be non-negative")

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def replication_seeds(base_seed: int, replications: int) -> list[int]:
    """Return the ``base_seed + replication_index`` sequence."""

    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if replications < 1:
        raise ValueError("replications must be positive")
    return [base_seed + replication_index for replication_index in range(replications)]


def derive_seed(replication_seed: int, *stream_keys: int) -> int:
    """Derive a deterministic seed for an independent random stream."""

    if replication_seed < 0:
        raise ValueError("replication_seed must be non-negative")
    if any(key < 0 for key in stream_keys):
        raise ValueError("stream keys must be non-negative")
    sequence = np.random.SeedSequence([replication_seed, *stream_keys])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def seed_contract(
    base_seed: int,
    replication_index: int,
    evaluation_offset: int = 20_000,
) -> dict[str, object]:
    """Describe how one base seed controls every RNG in a replication."""

    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if replication_index < 0:
        raise ValueError("replication_index must be non-negative")
    seed = base_seed + replication_index
    return {
        "rule": "actual_seed = base_seed + replication_index",
        "base_seed": base_seed,
        "replication_index": replication_index,
        "replication_seed": seed,
        "python_random_seed": seed,
        "numpy_legacy_seed": seed,
        "numpy_generator_seed": seed,
        "numpy_bit_generator": "PCG64",
        "torch_seed": seed,
        "estimator_seed": seed,
        "evaluation_normalization_seed": seed + evaluation_offset,
    }


def runtime_metadata() -> dict[str, str]:
    """Return runtime versions stored with every experiment result."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "cudnn": str(torch.backends.cudnn.version() or "none"),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "none"
        ),
        "deterministic_algorithms": str(
            torch.are_deterministic_algorithms_enabled()
        ).lower(),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG",
            "unset",
        ),
    }
