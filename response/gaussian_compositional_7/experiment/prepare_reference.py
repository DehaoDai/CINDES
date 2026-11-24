"""Create the shared evaluation-only target bank once, with resumable safety."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np
import scipy
from scipy.special import ndtri

import study
from metrics import ReferenceMetrics, probabilities, validate_reference
sys.path.insert(0, str(study.PACKAGE_ROOT/"source"))
from compositional_dgp import BoundedCompositionalDGP


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def prepare(root, profile):
    root = Path(root).resolve()
    manifest = study.initialize(root, profile)
    cfg = manifest["config"]
    with study.locked(root/".reference.lock"):
        folder = root/"reference"
        marker = folder/"reference_complete.json"
        if marker.exists():
            validate_reference(root, manifest)
            print("Verified existing completed reference.", flush=True)
            return
        if folder.exists() and any(folder.iterdir()):
            archive = root/"interrupted"/("reference_"+uuid.uuid4().hex)
            archive.parent.mkdir(parents=True, exist_ok=True)
            folder.rename(archive)
            print("Preserved incomplete reference under "+str(archive), flush=True)
        folder.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        dgp = BoundedCompositionalDGP(**cfg["dgp"])
        if dgp.edges != [(3,1),(3,4),(1,5),(1,2),(4,0),(4,6)] or dgp.signs.tolist() != [-1,-1,-1,-1,1,1]:
            raise ValueError("The fixed DGP design changed")
        if abs(dgp.v-.3942944903978412)>1e-13:
            raise ValueError("The H normalization changed")
        b, size, d = cfg["reference_banks"], cfg["reference_bank_size"], cfg["d"]
        edges = ndtri(np.linspace(0., 1., cfg["tv_bins"]+1))
        reference = np.empty((b*size,d),dtype=np.float64)
        hs, logs = np.empty(b*size), np.empty(b*size)
        p1s, p2s, phs, bank_records = [], [], [], []
        for i in range(b):
            seed = cfg["reference_seed_start"]+i
            x = dgp.sample(size,np.random.default_rng(seed))
            h, ell = dgp.H(x), dgp.log_p_unnorm(x)
            sl = slice(i*size,(i+1)*size)
            reference[sl],hs[sl],logs[sl] = x,h,ell
            p1,p2,ph = probabilities(x,h,dgp,edges)
            p1s.append(p1);p2s.append(p2);phs.append(ph)
            bank_records.append({"bank":i,"seed":seed,"n":size,"sampling":dgp.last_sample_stats,
                                 "sample_sha256":array_sha(x),"mean_H":float(h.mean()),
                                 "mean_log_p_unnorm":float(ell.mean())})
            print(f"Reference bank {i+1}/{b} complete",flush=True)
        baseline = np.empty((cfg["baseline_reps"],cfg["n_gen"],d),dtype=np.float64)
        baseline_stats = []
        for i in range(cfg["baseline_reps"]):
            seed = cfg["baseline_seed_start"]+i
            baseline[i] = dgp.sample(cfg["n_gen"],np.random.default_rng(seed))
            baseline_stats.append({"replication":i,"seed":seed,"sampling":dgp.last_sample_stats,
                                   "sample_sha256":array_sha(baseline[i])})
        p1s,p2s,phs = map(np.stack,(p1s,p2s,phs))
        arrays = {
            "reference":reference,"reference_H":hs,"reference_log_p_unnorm":logs,
            "reference_bank_id":np.repeat(np.arange(b,dtype=np.int64),size),
            "bin_edges":edges,"node_prob":p1s.mean(0),"pair_prob":p2s.mean(0),"H_prob":phs.mean(0),
            "node_prob_by_bank":p1s,"pair_prob_by_bank":p2s,"H_prob_by_bank":phs,
            "baseline":baseline,"generating_pairs":np.asarray(dgp.edges,dtype=np.int64),
        }
        path = folder/"reference_bank.npz"
        tmp = path.with_name(path.name+".tmp."+uuid.uuid4().hex)
        with tmp.open("wb") as handle:
            np.savez_compressed(handle,**arrays)
        os.replace(tmp,path)
        summary = ReferenceMetrics(path,cfg).reference_summary()
        study.atomic_json(folder/"reference_summary.json",summary)
        study.atomic_json(folder/"reference_audit.json",{
            "status":"complete","study_id":manifest["study_id"],"profile":profile,
            "evaluation_only":True,"model_outputs_read":False,
            "dgp_config":dgp.config,"bank_records":bank_records,"baseline_records":baseline_stats,
            "arrays":{k:{"shape":list(v.shape),"dtype":str(v.dtype),"sha256":array_sha(v)} for k,v in arrays.items()},
            "environment":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"host":platform.node()},
            "elapsed_seconds":time.perf_counter()-start,
        })
        files = [path,folder/"reference_summary.json",folder/"reference_audit.json"]
        study.atomic_json(marker,{
            "status":"complete","study_id":manifest["study_id"],"profile":profile,
            "n_reference":b*size,"reference_banks":b,"baseline_reps":cfg["baseline_reps"],
            "n_gen":cfg["n_gen"],"completed_at_utc":study.utc_now(),
            "elapsed_seconds":time.perf_counter()-start,
            "files_sha256":{str(p.relative_to(root)):study.sha(p) for p in files},
        })
        print(f"REFERENCE_DONE: {b*size} target observations; {cfg['baseline_reps']} calibration datasets.",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=study.DEFAULT_ROOT)
    parser.add_argument("--profile",choices=["full","smoke"],default="full")
    args=parser.parse_args()
    prepare(args.root,args.profile)
