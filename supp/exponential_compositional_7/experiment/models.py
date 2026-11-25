"""Generic CINDES and reference-aware DSM in common log-data coordinates.

Both methods know only the fixed product log-exponential reference. Neither
module imports the target tilt, graph, H function, target score, or test data.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

CINDES_VARIANTS = ("baseline",)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width=64, depth=2, beta=5.0):
        super().__init__()
        dims = [d_in] + [width] * depth
        layers = []
        for i in range(depth):
            layers.extend((nn.Linear(dims[i], dims[i + 1]), nn.Softplus(beta=beta)))
        layers.append(nn.Linear(width, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CINDES(nn.Module):
    def __init__(self, d, width=64, depth=2, beta=5.0):
        super().__init__()
        self.mlp = MLP(d, 1, width, depth, beta)
        self.upper_bound = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.upper_bound - F.softplus(self.mlp(x), beta=1.0)

    def logit_upper_bound(self):
        return self.upper_bound


class DSM(nn.Module):
    def __init__(self, d, width=64, depth=2, beta=5.0, n_time_frequencies=8):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_time_frequencies).float())
        self.mlp = MLP(d + 2 * n_time_frequencies, d, width, depth, beta)
        self._references = {}

    def reference_score(self, y, t):
        from reference import ReferenceScore
        key = str(y.device)
        if key not in self._references:
            self._references[key] = ReferenceScore(device=y.device)
        return self._references[key].score(y, t)

    def correction(self, y, t):
        if t.ndim != 2 or t.shape != (len(y), 1):
            raise ValueError("One time per observation is required")
        angles = t * self.freqs[None]
        return self.mlp(torch.cat((y, torch.sin(angles), torch.cos(angles)), dim=1))

    def forward(self, y, t):
        return self.reference_score(y, t) + self.correction(y, t)


def model_spec(method, d, dsm_parameterization="reference_residual", cindes_variant="baseline"):
    if method not in ("cindes", "dsm") or cindes_variant != "baseline":
        raise ValueError("Unknown method or CINDES variant")
    if method == "dsm" and dsm_parameterization != "reference_residual":
        raise ValueError("This fixed pilot uses reference-aware DSM")
    return {"family": "exponential_generic_log_coordinates_v1", "method": method,
            "d": int(d), "width": 64, "depth": 2, "activation": "softplus",
            "softplus_beta": 5.0, "input_features": "raw_log_coordinates",
            "parameterization": "learned_upper_logit" if method == "cindes" else dsm_parameterization,
            "n_time_frequencies": 0 if method == "cindes" else 8,
            "fixed_logit_truncation": None, "known_target_structure_used": False,
            "reference_distribution": "coordinatewise log of independent Exponential(rate=1)",
            "tail_correction": "none", "support_transform": "Z=log(Y), common to both methods"}


def build_model(spec):
    if spec.get("family") != "exponential_generic_log_coordinates_v1":
        raise ValueError("Unrecognized model specification")
    common = dict(d=spec["d"], width=spec["width"], depth=spec["depth"], beta=spec["softplus_beta"])
    if spec["method"] == "cindes":
        return CINDES(**common)
    if spec["method"] == "dsm":
        return DSM(**common, n_time_frequencies=spec["n_time_frequencies"])
    raise ValueError("Unknown method")


def linear_weights(model):
    return [layer.weight for layer in model.modules() if isinstance(layer, nn.Linear)]


def mean_absolute_weight(model):
    weights = linear_weights(model)
    return sum(w.abs().sum() for w in weights) / sum(w.numel() for w in weights)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def diagnostics(model):
    values = torch.cat([w.detach().reshape(-1) for w in linear_weights(model)]).abs()
    result = {"weight_mean_abs": float(values.double().mean()),
              "weight_fraction_abs_le_1e_3": float((values <= 1e-3).double().mean()),
              "weight_fraction_abs_le_1e_2": float((values <= 1e-2).double().mean())}
    if hasattr(model, "logit_upper_bound"):
        result["learned_logit_upper_bound"] = float(model.logit_upper_bound())
    return result


def load_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


restore_model = load_model
