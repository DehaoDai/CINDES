"""Evaluation-only target metrics; never imported by fitting or sampling."""
from pathlib import Path
import math
import sys

import numpy as np
from scipy.stats import wasserstein_distance

import study
sys.path.insert(0, str(study.PACKAGE_ROOT / "source"))
from compositional_dgp import BoundedCompositionalDGP

METRICS = study.METRICS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probabilities(points, h, dgp, edges):
    points = np.asarray(points, dtype=np.float64)
    b = len(edges)-1
    bins = np.searchsorted(edges[1:-1], points, side="right")
    hb = np.searchsorted(edges[1:-1], h, side="right")
    p1 = np.array([np.bincount(bins[:, j], minlength=b) for j in range(dgp.d)], dtype=np.float64)/len(points)
    p2 = np.array([np.bincount(bins[:, u]*b+bins[:, v], minlength=b*b).reshape(b, b)
                   for u, v in dgp.edges], dtype=np.float64)/len(points)
    ph = np.bincount(hb, minlength=b).astype(np.float64)/len(points)
    return p1, p2, ph


def jackknife_se(values):
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    return float(np.sqrt((count-1)/count*np.square(values-values.mean()).sum()))


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    sd = float(values.std(ddof=1)) if len(values)>1 else None
    return {"mean": float(values.mean()), "sd": sd,
            "se": None if sd is None else sd/math.sqrt(len(values)),
            "q025": float(np.quantile(values, .025)),
            "q975": float(np.quantile(values, .975)), "n_rep": len(values)}


def validate_reference(root, manifest):
    root = Path(root)
    marker = root/"reference/reference_complete.json"
    record = study.read_json(marker)
    require(record.get("status") == "complete", "Reference is incomplete")
    require(record["study_id"] == manifest["study_id"], "Reference study identity mismatch")
    require(record["profile"] == manifest["config"]["profile"], "Reference profile mismatch")
    require(record["n_reference"] == manifest["config"]["reference_banks"]*manifest["config"]["reference_bank_size"],
            "Reference size mismatch")
    require(record.get("files_sha256"), "Reference is missing content hashes")
    for name, expected in record["files_sha256"].items():
        require(study.sha(root/name) == expected, "Reference content changed: "+name)
    return record


class ReferenceMetrics:
    def __init__(self, archive_path, config):
        self.config = config
        self.dgp = BoundedCompositionalDGP(**config["dgp"])
        self.banks = config["reference_banks"]
        self.bank_size = config["reference_bank_size"]
        nr = self.banks*self.bank_size
        b = config["tv_bins"]
        with np.load(archive_path, allow_pickle=False) as bank:
            self.reference = bank["reference"].copy()
            self.logs = bank["reference_log_p_unnorm"].copy()
            self.h = bank["reference_H"].copy()
            self.bank_id = bank["reference_bank_id"].copy()
            self.edges = bank["bin_edges"].copy()
            self.p1 = bank["node_prob"].copy()
            self.p2 = bank["pair_prob"].copy()
            self.ph = bank["H_prob"].copy()
            self.p1_bank = bank["node_prob_by_bank"].copy()
            self.p2_bank = bank["pair_prob_by_bank"].copy()
            self.ph_bank = bank["H_prob_by_bank"].copy()
            self.baselines = bank["baseline"].copy()
            require(np.array_equal(bank["generating_pairs"], self.dgp.edges), "Reference generating pairs differ")
        require(self.reference.shape == (nr, config["d"]), "Reference shape mismatch")
        require(self.baselines.shape == (config["baseline_reps"], config["n_gen"], config["d"]), "Baseline shape mismatch")
        require(self.logs.shape == self.h.shape == self.bank_id.shape == (nr,), "Reference statistic shape mismatch")
        require(self.edges.shape == (b+1,) and np.isneginf(self.edges[0]) and np.isposinf(self.edges[-1]),
                "Infinite-tail histogram edges missing")
        require(np.array_equal(self.bank_id, np.repeat(np.arange(self.banks), self.bank_size)), "Bank membership mismatch")
        require(self.p1.shape == (config["d"], b) and self.p2.shape == (len(self.dgp.edges), b, b), "Histogram shape mismatch")
        for pooled, banks in ((self.p1, self.p1_bank), (self.p2, self.p2_bank), (self.ph, self.ph_bank)):
            require(np.allclose(pooled, banks.mean(axis=0), rtol=0, atol=2e-15), "Pooled reference probabilities changed")
        require(np.allclose(self.logs, self.dgp.log_p_unnorm(self.reference), rtol=0, atol=2e-13),
                "Reference logpotential disagrees with DGP")
        require(np.allclose(self.h, self.dgp.H(self.reference), rtol=0, atol=2e-13), "Reference H disagrees with DGP")
        self.mean_log = float(self.logs.mean())
        self.log_bank_means = self.logs.reshape(self.banks, self.bank_size).mean(axis=1)
        self.p1_delete = (self.banks*self.p1[None]-self.p1_bank)/(self.banks-1)
        self.p2_delete = (self.banks*self.p2[None]-self.p2_bank)/(self.banks-1)
        self.ph_delete = (self.banks*self.ph[None]-self.ph_bank)/(self.banks-1)
        self.log_delete = (self.banks*self.mean_log-self.log_bank_means)/(self.banks-1)
        self.h_delete = [self.h[self.bank_id != i] for i in range(self.banks)]

    def evaluate(self, values, reference_uncertainty=True):
        x = np.asarray(values, dtype=np.float64)
        require(x.ndim == 2 and x.shape[1] == self.config["d"] and len(x)>1 and np.isfinite(x).all(),
                "Generated observations must be a finite (n,d) matrix with n>1")
        h, logs = self.dgp.H(x), self.dgp.log_p_unnorm(x)
        p1, p2, ph = probabilities(x, h, self.dgp, self.edges)
        mean_log = float(logs.mean())
        result = {
            "tv1": float(.5*np.abs(p1-self.p1).sum(axis=1).mean()),
            "tv2": float(.5*np.abs(p2-self.p2).sum(axis=(1,2)).mean()),
            "llgap": self.mean_log-mean_log,
            "h_tv": float(.5*np.abs(ph-self.ph).sum()),
            "h_w1": float(wasserstein_distance(h, self.h)),
            "generated_mean_log_p_unnorm": mean_log,
            "generated_log_p_sd": float(logs.std(ddof=1)),
            "generated_mean_H": float(h.mean()), "generated_sd_H": float(h.std(ddof=1)),
            "conditional_llgap_mc_se": float(np.sqrt(self.logs.var(ddof=1)/len(self.logs)+logs.var(ddof=1)/len(logs))),
            "reference_mean_log_p_unnorm": self.mean_log,
            "tail3_coordinate": float(np.mean(np.abs(x)>3)),
            "tail4_coordinate": float(np.mean(np.abs(x)>4)),
            "tail3_any": float(np.mean(np.max(np.abs(x), axis=1)>3)),
            "max_absolute_coordinate": float(np.max(np.abs(x))),
        }
        if reference_uncertainty:
            delete = {
                "tv1": [.5*np.abs(p1-p).sum(axis=1).mean() for p in self.p1_delete],
                "tv2": [.5*np.abs(p2-p).sum(axis=(1,2)).mean() for p in self.p2_delete],
                "llgap": [p-mean_log for p in self.log_delete],
                "h_tv": [.5*np.abs(ph-p).sum() for p in self.ph_delete],
                "h_w1": [wasserstein_distance(h,p) for p in self.h_delete],
            }
            result["reference_delete_one_bank_metrics"] = {k:[float(v) for v in a] for k,a in delete.items()}
            result["reference_only_jackknife_mc_se"] = {k:jackknife_se(a) for k,a in delete.items()}
        return result

    def reference_summary(self):
        rows = [{"baseline_replication": i, "n": len(x), **self.evaluate(x, reference_uncertainty=False)}
                for i,x in enumerate(self.baselines)]
        return {
            "n_reference": len(self.reference), "reference_banks": self.banks,
            "reference_bank_size": self.bank_size,
            "reference_mean_H": float(self.h.mean()),
            "reference_mean_log_p_unnorm": self.mean_log,
            "reference_log_potential_mean_iid_mc_se": float(self.logs.std(ddof=1)/np.sqrt(len(self.logs))),
            "reference_log_potential_mean_bank_mc_se": float(self.log_bank_means.std(ddof=1)/np.sqrt(self.banks)),
            "baseline_summary": {m:describe([row[m] for row in rows]) for m in METRICS},
            "baseline_per_replication": rows,
            "baseline_description": "Independent exact-target samples at n_gen; evaluation-only finite-sample calibration, never subtracted.",
            "tv2_generating_pairs": [list(p) for p in self.dgp.edges],
            "tv2_interpretation": "Pairs entering H; not asserted to be conditional-independence edges after outer tanh.",
            "llgap_interpretation": "Signed target-logpotential moment difference; not KL, not inherently nonnegative.",
            "reference_uncertainty": "Delete-one-bank jackknife conditional on model samples; approximate for nonsmooth distances.",
        }
