"""Shared frozen protocol and filesystem helpers for the SCC study."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import uuid

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PACKAGE_ROOT / "outputs" / "full_study_v1"
METHODS = ("cindes", "dsm")
METRICS = ("tv1", "tv2", "llgap", "h_tv", "h_w1")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds this study/run lock: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def make_config(profile="full"):
    if profile not in ("full", "smoke"):
        raise ValueError("profile must be full or smoke")
    full = profile == "full"
    return {
        "protocol": "compositional_scc_cv20_v1",
        "profile": profile,
        "d": 7,
        "n_values": [500, 1000, 1500, 2000] if full else [500],
        "replications": 20 if full else 1,
        "first_fit_seed": 1000 if full else 9000,
        "dgp": {"d": 7, "pot_seed": 11, "amplitude": 1.5, "kappa": 1.0},
        "methods": list(METHODS),
        "network": {"width": 64, "depth": 2, "activation": "softplus", "beta": 5.0,
                    "cindes": "scalar classifier", "dsm": "Gaussian residual score",
                    "known_structure_used": False},
        "lambda_grid": [0.0, 0.01, 0.1, 0.3, 1.0] if full else [0.0, 0.1],
        "cv_folds": 3,
        "validation_checks": 100 if full else 2,
        "max_positive_epochs": 2000 if full else 2,
        "positive_batch": 256,
        "fake_ratio": 2,
        "normalizer_size": 65536 if full else 1024,
        "dsm_validation_draws": 8,
        "retain_cindes_cv_states": True,
        "training_device": "cpu",
        "generation_device": "cuda" if full else "cpu_or_cuda",
        "generation_batch": 128 if full else 4,
        "n_gen": 10000 if full else 64,
        "shard_size": 1000 if full else 32,
        "R": 25.0, "K": 5000 if full else 20,
        "T": 3.0, "delta": 0.001, "M": 200 if full else 4,
        "terminal_correction": "none", "support_clipping": False,
        "last_reverse_step_noise": True,
        "sampler": "original antithetic moment score, Gaussian-reference exponential Euler",
        "cuda_precision": "float32; TF32 disabled",
        "reference_banks": 8 if full else 4,
        "reference_bank_size": 131072 if full else 1024,
        "reference_seed_start": 97000000 if full else 97200000,
        "baseline_reps": 20 if full else 3,
        "baseline_seed_start": 97100000 if full else 97300000,
        "tv_bins": 20,
        "primary_metrics": ["tv1", "tv2", "llgap"],
        "diagnostic_metrics": ["h_tv", "h_w1"],
        "histograms": "fixed standard-normal quantiles including infinite tails",
        "tv2_pairs": "six pairs entering H, not asserted to be conditional-independence edges",
        "llgap": "signed reference-minus-generated mean analytic target logpotential",
        "seed_rules": {
            "fit_seed": "first_fit_seed + task_id - 1",
            "data": "fit_seed",
            "fold_partition": "121000000 + fit_seed",
            "cv_training": "100*fit_seed + fold",
            "full_training": "fit_seed",
            "reverse_argument": "98000000 + 10000*fit_seed + shard",
            "actual_reverse": "reverse_argument + 999",
            "particles": "99000000 + 10000*fit_seed + shard",
        },
        "selection": "joint lambda/common check minimizing validation-size-weighted mean fold loss; lower lambda then earlier check breaks exact ties",
        "refit": "all n observations; common selected positive-epoch fraction; original full cosine horizon retained",
        "fairness": "same observed data, CV partitions and positive-data exposure cap; selected durations and computation can differ",
    }


def task_specs(config):
    return [
        {"task_id": i*config["replications"]+r+1, "n": n,
         "replication": r+1,
         "seed": config["first_fit_seed"]+i*config["replications"]+r}
        for i, n in enumerate(config["n_values"])
        for r in range(config["replications"])
    ]


def task_spec(task_id, config):
    rows = task_specs(config)
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 1 <= task_id <= len(rows):
        raise ValueError(f"task_id must be between 1 and {len(rows)}")
    return rows[task_id-1]


def run_dir(root, spec):
    return Path(root) / "runs" / f"n_{spec['n']:04d}" / f"rep_{spec['replication']:02d}_seed_{spec['seed']}"


def source_hashes():
    paths = sorted(PACKAGE_ROOT.glob("*.py")) + sorted((PACKAGE_ROOT / "source").glob("*.py"))
    return {str(p.relative_to(PACKAGE_ROOT)): sha(p) for p in paths}


def initialize(root, profile):
    root = Path(root).resolve()
    config = make_config(profile)
    if profile == "smoke" and root == DEFAULT_ROOT.resolve():
        raise ValueError("Smoke tests must use a separate output root")
    path = root / "study_manifest.json"
    with locked(root / ".manifest.lock"):
        if path.exists():
            manifest = load_manifest(root, profile)
            if not (root / "task_manifest.csv").is_file():
                write_task_csv(root, manifest)
            return manifest
        core = {"config": config, "tasks": task_specs(config), "source_sha256": source_hashes()}
        manifest = {**core, "study_id": digest(core), "created_at_utc": utc_now()}
        snapshot = root / "source_snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        snapshot_files = [PACKAGE_ROOT / name for name in core["source_sha256"]]
        for pattern in ("*.sh", "*.qsub", "*.txt", "*.md"):
            snapshot_files.extend(sorted(PACKAGE_ROOT.glob(pattern)))
        snapshot_hashes = {}
        for source in snapshot_files:
            relative = source.relative_to(PACKAGE_ROOT)
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            snapshot_hashes[str(relative)] = sha(destination)
        atomic_json(snapshot / "snapshot_manifest.json", snapshot_hashes)
        atomic_json(path, manifest)
        write_task_csv(root, manifest)
        return manifest


def write_task_csv(root, manifest):
    import csv
    path = Path(root) / "task_manifest.csv"
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "n", "replication", "seed"])
        writer.writeheader()
        writer.writerows(manifest["tasks"])
    os.replace(temporary, path)


def load_manifest(root, profile=None):
    manifest = read_json(Path(root) / "study_manifest.json")
    config = manifest["config"]
    if profile is not None and config["profile"] != profile:
        raise ValueError("Output root belongs to a different profile")
    if config != make_config(config["profile"]):
        raise ValueError("Frozen configuration differs from this code")
    if manifest["tasks"] != task_specs(config):
        raise ValueError("Task manifest changed")
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("Frozen Python source files changed; restore saved sources or use a new output root")
    core = {k: manifest[k] for k in ("config", "tasks", "source_sha256")}
    if digest(core) != manifest["study_id"]:
        raise ValueError("Study manifest identity mismatch")
    return manifest


def horizons(n, config):
    sizes = [n//3 + (i < n % 3) for i in range(3)]
    return {"validation_sizes": sizes,
            "cv_updates": [config["max_positive_epochs"]*math.ceil((n-v)/config["positive_batch"]) for v in sizes],
            "full_updates": config["max_positive_epochs"]*math.ceil(n/config["positive_batch"])}
