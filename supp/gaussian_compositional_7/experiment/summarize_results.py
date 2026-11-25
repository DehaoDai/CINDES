"""Validate paired SCC runs and export tables; --status is a light metadata check."""
import argparse
import csv
import math
from pathlib import Path
import statistics
import sys
import study


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows, fieldnames):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def descriptive(values):
    values=[float(x) for x in values]
    if not values:
        return {"n_rep":0,"mean":None,"sd":None,"se":None}
    sd=statistics.stdev(values) if len(values)>1 else None
    return {"n_rep":len(values),"mean":statistics.mean(values),"sd":sd,
            "se":None if sd is None else sd/math.sqrt(len(values))}


def jackknife(values):
    center=statistics.mean(values)
    return math.sqrt((len(values)-1)/len(values)*sum((x-center)**2 for x in values))


def validate_identity(row,spec,manifest):
    cfg=manifest["config"]
    require(row.get("status")=="complete","Completion record is not complete")
    require(row.get("study_id")==manifest["study_id"],"Study identity mismatch")
    require(row.get("profile")==cfg["profile"],"Smoke/full profile mismatch")
    for key,value in spec.items():
        require(row.get(key)==value,f"Run {key} differs from task manifest")
    require(set(row.get("methods",{}))==set(study.METHODS),"Missing or duplicate method")
    require(row.get("reference_n")==cfg["reference_banks"]*cfg["reference_bank_size"],"Reference size mismatch")
    for method,m in row["methods"].items():
        require(m.get("method")==method and m.get("n_gen")==cfg["n_gen"],"Method/sample count mismatch")
        for key,value in spec.items():
            require(m.get(key)==value,f"Method {key} differs from task")
        require(m.get("profile")==cfg["profile"] and m.get("study_id")==manifest["study_id"],"Method identity mismatch")
        for metric in study.METRICS:
            require(isinstance(m.get(metric),(float,int)) and math.isfinite(m[metric]),"Invalid metric "+metric)
        require(m["selected_lambda"] in cfg["lambda_grid"],"Penalty outside frozen grid")
        require(isinstance(m["updates"],int) and m["updates"]>=0,"Invalid update count")


def inventory(root,manifest):
    expected={study.run_dir(root,spec)/"run_complete.json":spec for spec in manifest["tasks"]}
    records,missing,invalid=[],[],[]
    for path,spec in expected.items():
        if not path.exists():
            missing.append({**spec,"reason":"run_complete.json missing"})
            continue
        try:
            row=study.read_json(path)
            validate_identity(row,spec,manifest)
            records.append((path,row))
        except Exception as exc:
            invalid.append({**spec,"reason":str(exc)})
    for path in (root/"runs").glob("n_*/rep_*/run_complete.json"):
        if path not in expected:
            invalid.append({"task_id":None,"n":None,"replication":None,"seed":None,
                            "reason":"Unexpected completed run: "+str(path.relative_to(root))})
    return records,missing,invalid


def status_record(root,manifest,records,missing,invalid,hashes_checked=False):
    per_n=[]
    for n in manifest["config"]["n_values"]:
        per_n.append({"n":n,"complete_pairs":sum(row["n"]==n for _,row in records),
                      "expected_pairs":manifest["config"]["replications"]})
    return {"status":"complete" if not missing and not invalid else "incomplete",
            "profile":manifest["config"]["profile"],"study_id":manifest["study_id"],
            "completed_pairs":len(records),"expected_pairs":len(manifest["tasks"]),
            "per_n":per_n,"missing_tasks":missing,"invalid_tasks":invalid,
            "checked_file_hashes":hashes_checked,
            "note":"--status checks completion metadata only; final aggregation additionally verifies all recorded files.",
            "checked_at_utc":study.utc_now()}


def verify_files(root,path,row,manifest):
    import numpy as np
    folder=path.parent
    hashes=row.get("files_sha256")
    require(isinstance(hashes,dict) and bool(hashes),"Missing run file hashes")
    for name,expected in hashes.items():
        candidate=(folder/name).resolve()
        require(candidate.is_relative_to(folder.resolve()),"Run file path escapes its directory")
        require(study.sha(candidate)==expected,"Changed run evidence file: "+name)
    require(row["reference_complete_sha256"]==study.sha(root/"reference/reference_complete.json"),
            "Run used a different reference completion record")
    for method,m in row["methods"].items():
        model,gen=m["model_record"],m["generation_record"]
        for name,expected in ((model["checkpoint"],model["checkpoint_sha256"]),
                              (gen["samples"],gen["samples_sha256"])):
            require(hashes.get(name)==expected,"Essential file absent or inconsistent in evidence hashes: "+name)
        x=np.load(folder/gen["samples"],allow_pickle=False,mmap_mode="r")
        require(x.shape==(manifest["config"]["n_gen"],manifest["config"]["d"]) and x.dtype==np.float32,
                "Saved sample shape/dtype mismatch")
        require(np.isfinite(x).all(),"Nonfinite saved generated samples")
        delete=m.get("reference_delete_one_bank_metrics",{})
        require(set(delete)==set(study.METRICS),"Missing reference uncertainty traces")
        for values in delete.values():
            require(len(values)==manifest["config"]["reference_banks"] and all(math.isfinite(v) for v in values),
                    "Invalid reference uncertainty traces")


def scalar_method_rows(records):
    rows=[]
    for path,run in records:
        for method in study.METHODS:
            m=run["methods"][method]
            row={k:v for k,v in m.items() if not isinstance(v,(dict,list))}
            row["run_record"]=str(path)
            row["evaluation_seconds_pair"]=run.get("evaluation_seconds")
            model=m["model_record"]
            row["checkpoint"]=model["checkpoint"]
            row["checkpoint_sha256"]=model["checkpoint_sha256"]
            row["samples"]=m["generation_record"]["samples"]
            row["samples_sha256"]=m["generation_record"]["samples_sha256"]
            row["generation_device"]=m["generation_record"].get("runtime_signature",{}).get("generation_device")
            row["selected_check"]=model.get("selected_check")
            row["cv_sum_fit_wall_seconds"]=model.get("cv_sum_fit_wall_seconds")
            row["cv_stage_elapsed_seconds_this_attempt"]=run.get("timing",{}).get("cv",{}).get("elapsed_seconds_this_attempt")
            row["task_elapsed_seconds_this_attempt"]=run.get("timing",{}).get("task_elapsed_seconds_this_attempt")
            rows.append(row)
    return rows


def latex_table(summaries,metrics,title,label,partial,profile):
    headings={"tv1":"TV1","tv2":"TV2","llgap":"Signed LLgap","h_tv":"$H$-TV","h_w1":"$H$-Wasserstein-$1$"}
    caption=title
    if partial:
        caption="PARTIAL STUDY: "+caption+" Actual completed replication counts are shown."
    if profile=="smoke":
        caption="SMOKE TEST ONLY; these are workflow checks, not experiment results. "+caption
    lines=[r"% Requires \usepackage{booktabs}.",r"\begin{table}[htbp]",r"\centering",
           r"\caption{"+caption+r" Entries are mean $\pm$ sample standard deviation over independent fitted datasets.}",
           r"\label{"+label+"}",r"\begin{tabular}{rrl"+"c"*len(metrics)+"}",r"\toprule",
           "$n$ & Reps & Method & "+" & ".join(headings[m] for m in metrics)+r" \\",r"\midrule"]
    for row in summaries:
        if not row["n_rep"]:
            continue
        values=[]
        for metric in metrics:
            mean,sd=row[metric+"_mean"],row[metric+"_sd"]
            values.append("$"+f"{mean:.5f}"+(r" \pm "+f"{sd:.5f}" if sd is not None else r"\;(\mathrm{SD:\ --})")+"$")
        lines.append(f"{row['n']} & {row['n_rep']} & {row['method'].upper()} & "+" & ".join(values)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",
              r"\par\smallskip\footnotesize LLgap is signed and is not a divergence. TV2 uses the six generating pairs.",
              r"\end{table}",""]
    return "\n".join(lines)


def summarize(root,profile="full",allow_partial=False,status_only=False):
    root=Path(root).resolve()
    manifest=study.load_manifest(root,profile)
    cfg=manifest["config"]
    folder=root/"summary"
    folder.mkdir(parents=True,exist_ok=True)
    records,missing,invalid=inventory(root,manifest)
    state=status_record(root,manifest,records,missing,invalid)
    study.atomic_json(folder/"status.json",state)
    write_csv(folder/"missing_tasks.csv",missing+invalid,["task_id","n","replication","seed","reason"])
    if status_only:
        import json
        print(json.dumps(state,indent=2))
        return 0
    if invalid:
        raise ValueError(f"{len(invalid)} invalid/mixed completed run(s); inspect summary/status.json")
    if missing and not allow_partial:
        raise ValueError(f"Only {len(records)}/{len(manifest['tasks'])} complete paired runs. Missing tasks are in summary/missing_tasks.csv. No full-study tables were written.")
    if not records:
        raise ValueError("No completed paired runs to summarize")
    from metrics import validate_reference
    reference_complete=validate_reference(root,manifest)
    reference_summary=study.read_json(root/"reference/reference_summary.json")
    for path,row in records:
        verify_files(root,path,row,manifest)
    partial=bool(missing)
    if partial:
        folder=root/"summary_partial"
        folder.mkdir(parents=True,exist_ok=True)
    scalar=scalar_method_rows(records)
    summaries,paired,timing=[],[],[]
    for n in cfg["n_values"]:
        runs=[r for _,r in records if r["n"]==n]
        for method in study.METHODS:
            rows=[r["methods"][method] for r in runs]
            entry={"n":n,"method":method,"n_rep":len(rows),"expected_replications":cfg["replications"],
                   "partial_study":partial,"profile":profile}
            for metric in study.METRICS:
                desc=descriptive([row[metric] for row in rows])
                for key in ("mean","sd","se"):
                    entry[metric+"_"+key]=desc[key]
                if rows:
                    deletes=[statistics.mean(row["reference_delete_one_bank_metrics"][metric][i] for row in rows)
                             for i in range(cfg["reference_banks"])]
                    entry[metric+"_reference_only_jackknife_mc_se"]=jackknife(deletes)
                else:
                    entry[metric+"_reference_only_jackknife_mc_se"]=None
            summaries.append(entry)
            for measure in ("training_seconds","generation_seconds"):
                timing.append({"n":n,"method":method,"timing_metric":measure,**descriptive([r[measure] for r in rows]),
                               "profile":profile,"partial_study":partial})
        for metric in study.METRICS:
            diffs=[r["methods"]["cindes"][metric]-r["methods"]["dsm"][metric] for r in runs]
            desc=descriptive(diffs)
            item={"n":n,"metric":metric,"definition":"CINDES_minus_DSM",**desc,
                  "cindes_numerically_smaller":sum(x<0 for x in diffs),
                  "dsm_numerically_smaller":sum(x>0 for x in diffs),"ties":sum(x==0 for x in diffs),
                  "profile":profile,"partial_study":partial,
                  "interpretation":"For LLgap, numerical ordering is not a distance ordering; inspect signs and closeness to zero."}
            if runs:
                deltas=[statistics.mean(r["methods"]["cindes"]["reference_delete_one_bank_metrics"][metric][i]
                                        -r["methods"]["dsm"]["reference_delete_one_bank_metrics"][metric][i] for r in runs)
                        for i in range(cfg["reference_banks"])]
                item["reference_only_jackknife_mc_se"]=0.0 if metric=="llgap" else jackknife(deltas)
                if metric=="llgap":
                    item["cindes_closer_to_zero"]=sum(abs(r["methods"]["cindes"][metric])<abs(r["methods"]["dsm"][metric]) for r in runs)
                    item["dsm_closer_to_zero"]=sum(abs(r["methods"]["dsm"][metric])<abs(r["methods"]["cindes"][metric]) for r in runs)
            else:
                item["reference_only_jackknife_mc_se"]=None
            paired.append(item)
    scalar_fields=list(dict.fromkeys(key for row in scalar for key in row))
    write_csv(folder/"per_run.csv",scalar,scalar_fields)
    write_csv(folder/"summary.csv",summaries,list(summaries[0]))
    write_csv(folder/"paired_differences.csv",paired,list(dict.fromkeys(k for r in paired for k in r)))
    write_csv(folder/"timing.csv",timing,list(timing[0]))
    import json
    with (folder/"summary_all_runs.jsonl").open("w",encoding="utf-8") as handle:
        for _,row in records:
            handle.write(json.dumps(row,allow_nan=False)+"\n")
    result={"status":"partial" if partial else "complete","profile":profile,"study_id":manifest["study_id"],
            "n_complete_pairs":len(records),"n_expected_pairs":len(manifest["tasks"]),
            "summaries":summaries,"paired_differences":paired,"timing":timing,
            "reference_and_baseline":reference_summary,
            "reference_complete_sha256":study.sha(root/"reference/reference_complete.json"),
            "run_record_sha256":{str(p.relative_to(root)):study.sha(p) for p,_ in records},
            "missing_tasks":missing,"sd_ddof":1,
            "interpretation":"Replications are independently fitted datasets; generation shards are not replications. Reference-only MC uncertainty is separate. Baselines are never subtracted."}
    study.atomic_json(folder/"summary.json",result)
    (folder/"primary_table.tex").write_text(latex_table(summaries,("tv1","tv2","llgap"),
        "Primary generation metrics for the Gaussian target with a bounded compositional log-density tilt and $d=7$.",
        "tab:cindes_scc_primary",partial,profile))
    (folder/"H_table.tex").write_text(latex_table(summaries,("h_tv","h_w1"),
        "Prespecified diagnostics of the compositional statistic $H$; used only for evaluation.",
        "tab:cindes_scc_H",partial,profile))
    study.atomic_json(folder/"status.json",status_record(root,manifest,records,missing,invalid,hashes_checked=True))
    print(f"Validated {len(records)} complete paired runs ({2*len(records)} method results); profile={profile}; partial={partial}.")
    print("n     method    metric      reps       mean            SD            SE")
    for row in summaries:
        if not row["n_rep"]:
            continue
        for metric in study.METRICS:
            sd="--" if row[metric+"_sd"] is None else f"{row[metric+'_sd']:.8g}"
            se="--" if row[metric+"_se"] is None else f"{row[metric+'_se']:.8g}"
            print(f"{row['n']:<5} {row['method']:<9} {metric:<11} {row['n_rep']:<5} {row[metric+'_mean']:>13.8g} {sd:>13} {se:>13}")
    print("Wrote summary CSV/JSON, paired comparisons, timings and LaTeX tables under "+str(folder))
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=study.DEFAULT_ROOT)
    parser.add_argument("--profile",choices=["full","smoke"],default="full")
    parser.add_argument("--status",action="store_true",help="Light completion-metadata inventory; no array loading or file-hash sweep")
    parser.add_argument("--allow-partial",action="store_true",help="Write clearly labeled summary_partial outputs; never ignore corrupt records")
    args=parser.parse_args()
    try:
        sys.exit(summarize(args.root,args.profile,args.allow_partial,args.status))
    except (ValueError,FileNotFoundError,KeyError) as exc:
        print("SUMMARY NOT COMPLETE: "+str(exc),file=sys.stderr)
        sys.exit(2)
