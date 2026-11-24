"""Fixed settings for the bounded-tree study; no scientific imports required."""
from __future__ import annotations

METHODS = ("cindes", "dsm")
METRICS = ("tv1", "tv2", "llgap", "energy", "sw", "clip", "time")


def driver_settings(profile="full"):
    settings = {
        "d": 31, "coupling": 0.55, "unary": 0.3, "R": 25.0,
        "T": 3.0, "delta": 0.001, "M": 200, "K": 5000,
        "n_gen": 10000, "cindes_steps": 20000, "dsm_steps": 60000,
        "fake_ratio": 2, "generation_batch": 64,
    }
    if profile == "smoke":
        settings.update(d=7, M=4, K=20, n_gen=64,
                        cindes_steps=20, dsm_steps=20)
    elif profile != "full":
        raise ValueError(f"Unknown profile: {profile}")
    return settings


def expected_experiment_config(profile="full"):
    s = driver_settings(profile)
    return {
        "protocol": "tree_no_terminal_v1", "d": s["d"],
        "coupling": s["coupling"], "unary": s["unary"],
        "pot_seed": 11, "G": 200, "R": s["R"], "T": s["T"],
        "delta": s["delta"], "M": s["M"], "K": s["K"],
        "n_gen": s["n_gen"], "n_ref": 20000, "reference_seed": 12345,
        "cindes_steps_requested": s["cindes_steps"],
        "dsm_steps_requested": s["dsm_steps"],
        "training_batch": 256, "fake_ratio": s["fake_ratio"],
        "generation_batch": s["generation_batch"],
        "cindes_final": "none", "denoise_final": False,
    }


def expected_pairs(profile="full"):
    if profile == "full":
        return [(n, seed) for n in (500, 1000, 1500, 2000)
                for seed in range(20)]
    if profile == "smoke":
        return [(64, 0)]
    raise ValueError(f"Unknown profile: {profile}")


def default_root(profile="full"):
    expected_pairs(profile)  # Validate the name.
    return "outputs/bounded_n500_2000_v1" if profile == "full" else "outputs/bounded_n500_2000_smoke_v1"


def profile_settings(profile="full"):
    pairs = expected_pairs(profile)
    return {
        "profile": profile,
        "n_values": sorted({n for n, _ in pairs}),
        "seeds": sorted({seed for _, seed in pairs}),
        "expected_paired_tasks": len(pairs),
        "expected_method_results": 2 * len(pairs),
        "driver_settings": driver_settings(profile),
        "experiment_config": expected_experiment_config(profile),
    }
