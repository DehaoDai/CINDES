"""Evaluate a completed paired run; no fitting or model selection."""
from pathlib import Path
import time
import numpy as np
import study
from metrics import ReferenceMetrics, validate_reference


def evaluate_run(root, run_path, manifest, spec, model_records, generation_records, environment):
    root, run_path = Path(root), Path(run_path)
    cfg = manifest["config"]
    reference_record = validate_reference(root,manifest)
    if set(model_records) != set(study.METHODS) or set(generation_records) != set(study.METHODS):
        raise ValueError("Evaluation requires both completed methods")
    arrays={}
    for method in study.METHODS:
        record=generation_records[method]
        path=run_path/record["samples"]
        if study.sha(path) != record["samples_sha256"]:
            raise ValueError("Generated sample hash mismatch")
        x=np.load(path,allow_pickle=False)
        if x.shape != (cfg["n_gen"],cfg["d"]) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("Generated sample shape/dtype/finiteness mismatch")
        arrays[method]=x
    start=time.perf_counter()
    reference=ReferenceMetrics(root/"reference/reference_bank.npz",cfg)
    methods={}
    for method in study.METHODS:
        model,gen=model_records[method],generation_records[method]
        metrics=reference.evaluate(arrays[method],reference_uncertainty=True)
        methods[method]={
            **spec,"method":method,"profile":cfg["profile"],"study_id":manifest["study_id"],
            "n_gen":cfg["n_gen"],"d":cfg["d"],"selected_lambda":model["selected_lambda"],
            "updates":model["updates"],"training_seconds":model["training_seconds"],
            "generation_seconds":gen["generation_seconds"],
            "n_params":model.get("n_params",model.get("parameter_count")),
            "positive_occurrences":model.get("positive_occurrences"),
            "negative_occurrences":model.get("negative_occurrences"),
            **metrics,"model_record":model,"generation_record":gen,
        }
    return {
        "study_id":manifest["study_id"],"profile":cfg["profile"],**spec,
        "methods":methods,"environment":environment,
        "reference_complete_sha256":study.sha(root/"reference/reference_complete.json"),
        "reference_bank_sha256":reference_record["files_sha256"]["reference/reference_bank.npz"],
        "reference_n":cfg["reference_banks"]*cfg["reference_bank_size"],
        "evaluation_seconds":time.perf_counter()-start,
        "primary_metrics":cfg["primary_metrics"],"diagnostic_metrics":cfg["diagnostic_metrics"],
        "metric_notes":{"llgap":"Signed target-logpotential moment difference, not KL.",
                        "tv2":"Six generating pairs in H; no claim of conditional-independence graph.",
                        "baselines":"Exact-target calibration is never subtracted."},
    }
