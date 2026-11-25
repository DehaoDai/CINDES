"""Exercise the real shell submission workflow using a local fake scheduler.

No command contacts SCC. The fake qsub records argument vectors, returns
synthetic job IDs, and can fail after accepting an earlier stage.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("submit_scc.sh", "scc_env.sh", "prepare_reference.qsub",
           "full_array.qsub", "summarize.qsub")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    checks = []
    subprocess.run(["bash", "-n", *[str(ROOT / f) for f in SCRIPTS]], check=True)
    checks.append("Bash syntax for all five scripts")
    with tempfile.TemporaryDirectory(prefix="cindes_submission_check_") as directory:
        root = Path(directory)
        package = root / "package with spaces"
        package.mkdir()
        for name in SCRIPTS:
            shutil.copyfile(ROOT / name, package / name)
        (package / "study_config.py").write_text(
            "import sys\np=sys.argv[sys.argv.index('--profile')+1]\n"
            "print({'n1000':20,'full':80,'smoke':1}[p])\n")
        for name in ("prepare_reference.py", "run_experiment.py", "make_return_bundle.py"):
            (package / name).write_text("# Placeholder: never executed by submission checks.\n")
        (package / "summarize_results.py").write_text(
            "import json,sys\nprint(json.dumps({'status_invocation':sys.argv[1:]}))\n")
        bindir = root / "bin"
        bindir.mkdir()
        fake = bindir / "qsub"
        fake.write_text("""#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
p=Path(os.environ['FAKE_QSUB_CALLS'])
previous=p.read_text().splitlines() if p.exists() else []
args=sys.argv[1:]
with p.open('a') as f:f.write(json.dumps(args)+'\\n')
stage=next(x for x in args if x.endswith('.qsub'))
if os.environ.get('FAKE_QSUB_FAIL')==stage:
    print('Synthetic scheduler rejection',file=sys.stderr)
    sys.exit(17)
if os.environ.get('FAKE_QSUB_BAD_REPLY')==stage:
    print('Ambiguous output; submission state unknown')
    sys.exit(0)
job=str(9100001+len(previous))
if stage=='full_array.qsub':job+='.'+args[args.index('-t')+1]+':1'
print(job)
""")
        fake.chmod(0o755)
        calls = root / "calls.jsonl"
        env = os.environ.copy()
        env.update(PATH=str(bindir) + os.pathsep + env["PATH"],
                   PYTHON_BIN=sys.executable, FAKE_QSUB_CALLS=str(calls))
        env.pop("CINDES_ENV_SCRIPT", None)
        env.pop("SCC_PROJECT", None)

        def invoke(*arguments, overrides=None):
            run_env = env.copy()
            run_env.update(overrides or {})
            return subprocess.run(["bash", str(package / "submit_scc.sh"), *arguments],
                                  cwd=root, env=run_env, text=True, capture_output=True)

        def recorded():
            return [json.loads(x) for x in calls.read_text().splitlines()] if calls.exists() else []

        output = root / "outputs with spaces"
        result = invoke("--root", str(output), "--dry-run")
        require(result.returncode == 0 and "No jobs submitted" in result.stdout, result.stderr)
        require(not calls.exists() and not output.exists(), "Dry run mutated output or called scheduler")
        checks.append("Dry run submits nothing and creates no output directory")

        result = invoke("--root", str(output))
        require(result.returncode == 0, result.stderr)
        rows = recorded()
        require(len(rows) == 3, "Expected preparation, array, and summary submissions")
        reference, array, summary = rows
        require(array[array.index("-hold_jid") + 1] == "9100001", "Array dependency incorrect")
        require(summary[summary.index("-hold_jid") + 1] == "9100002", "Summary dependency incorrect")
        require(array[array.index("-t") + 1] == "1-20", "Default range incorrect")
        require(array[array.index("-tc") + 1] == "2", "Concurrency limit incorrect")
        require(all("-V" not in row for row in rows), "Unexpected whole-environment export")
        require(reference[-3:] == [str(output), "n1000", "-"], "Script arguments lost path boundaries")
        require(summary[-4:] == [str(output), "n1000", "-", "0"], "Strict summary flag incorrect")
        checks.append("Default paired array uses separate argv, correct dependencies, two-task cap, and paths with spaces")

        result = invoke("--root", str(output))
        require(result.returncode != 0 and len(recorded()) == 3, "Unrequested duplicate submission was accepted")
        result = invoke("--root", str(output), "--resume")
        require(result.returncode == 0 and len(recorded()) == 6, "Explicit resume did not submit one complete chain")
        checks.append("Accidental duplicate is refused; explicit resume records a fresh chain")

        count = len(recorded())
        result = invoke("--root", str(output), "--status")
        require(result.returncode == 0 and "status_invocation" in result.stdout, result.stderr)
        require(len(recorded()) == count, "Status command contacted scheduler")
        checks.append("Status calls only the summarizer")

        partial = root / "pilot"
        result = invoke("--profile", "full", "--pilot", "--root", str(partial))
        require(result.returncode == 0, result.stderr)
        rows = recorded()[-3:]
        require(rows[1][rows[1].index("-t") + 1] == "1-1", "Pilot range incorrect")
        require(rows[2][-1] == "1" and rows[2][-3] == "full", "Pilot partial summary flag incorrect")
        checks.append("Pilot preserves full profile and explicitly requests a partial summary")

        result = invoke("--profile", "full", "--root", str(root / "full"))
        require(result.returncode == 0, result.stderr)
        array = recorded()[-2]
        require(array[array.index("-t") + 1] == "1-80", "Full-study range incorrect")
        checks.append("Full profile submits 80 paired tasks")

        failed = root / "failed"
        count = len(recorded())
        result = invoke("--root", str(failed), overrides={"FAKE_QSUB_FAIL": "full_array.qsub"})
        require(result.returncode == 17, "Scheduler failure code was not retained")
        require(len(recorded()) == count + 2, "Failed array submission continued to summary")
        record = next((failed / "submissions").glob("submission_*.txt")).read_text()
        require("reference_job=" in record and "stage=array" in record and "status=interrupted_or_failed" in record,
                "Partial submission record is incomplete")
        require("Existing jobs were not cancelled" in result.stderr, "Partial submission not explained")
        checks.append("Array rejection preserves reference ID and stops before summary without cancellation")

        count = len(recorded())
        result = invoke("--root", str(root / "bad_reply"),
                        overrides={"FAKE_QSUB_BAD_REPLY": "prepare_reference.qsub"})
        require(result.returncode != 0 and len(recorded()) == count + 1,
                "Ambiguous scheduler response was retried or propagated")
        require("may already have been submitted" in result.stderr, "Ambiguous submission not explained")
        checks.append("Unparseable scheduler output stops without automatic retry")

        count = len(recorded())
        for arguments in (("--tasks", "0-1"), ("--tasks", "1-21"),
                          ("--profile", "unknown"), ("--pilot", "--tasks", "1-1")):
            result = invoke(*arguments)
            require(result.returncode != 0, "Invalid CLI input accepted: " + repr(arguments))
        require(len(recorded()) == count, "Invalid CLI input reached scheduler")
        checks.append("Invalid task ranges and conflicting options fail before submission")

        activation = root / "activation.sh"
        activation.write_text('printf "activation_unset=%s\\n" "$CINDES_INTENTIONALLY_UNSET"\n')
        run_env = env.copy()
        run_env.update(NSLOTS="4", CUDA_VISIBLE_DEVICES="7")
        run_env.pop("CINDES_INTENTIONALLY_UNSET", None)
        code = ('set -u; source "$1" "$2"; '
                'printf "CHECK:%s:%s:%s\\n" "$OMP_NUM_THREADS" "$CINDES_CPU_THREADS" "$CUDA_VISIBLE_DEVICES"')
        result = subprocess.run(["bash", "-c", code, "bash", str(package / "scc_env.sh"), str(activation)],
                                env=run_env, text=True, capture_output=True)
        require(result.returncode == 0 and "CHECK:4:4:7" in result.stdout, result.stderr)
        checks.append("Activation tolerates unset variables, respects allocated threads, and preserves assigned GPU")

        python_calls = root / "python_calls.jsonl"
        fake_python = bindir / "python"
        fake_python.write_text("""#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
with Path(os.environ['FAKE_PYTHON_CALLS']).open('a') as f:
    f.write(json.dumps(sys.argv[1:])+'\\n')
if ('summarize_results.py' in sys.argv and '--allow-partial' not in sys.argv
        and os.environ.get('FAKE_STRICT_SUMMARY_FAILURE')=='1'):
    sys.exit(3)
""")
        fake_python.chmod(0o755)
        batch_env = run_env.copy()
        batch_env.update(SGE_O_WORKDIR=str(package), SGE_TASK_ID="7", JOB_ID="900001",
                         FAKE_PYTHON_CALLS=str(python_calls))

        def batch(script, *arguments, fail_summary=False):
            python_calls.unlink(missing_ok=True)
            this_env = batch_env.copy()
            this_env["FAKE_STRICT_SUMMARY_FAILURE"] = "1" if fail_summary else "0"
            completed = subprocess.run(["bash", str(package / script), str(output), "n1000",
                                        str(activation), *arguments], env=this_env, text=True, capture_output=True)
            invocations = [json.loads(x) for x in python_calls.read_text().splitlines()] if python_calls.exists() else []
            return completed, invocations

        result, rows = batch("prepare_reference.qsub")
        require(result.returncode == 0 and len(rows) == 1 and rows[0][1] == "prepare_reference.py", result.stderr)
        result, rows = batch("full_array.qsub")
        require(result.returncode == 0 and len(rows) == 1 and rows[0][1] == "run_experiment.py", result.stderr)
        require(rows[0][rows[0].index("--task-id")+1] == "7" and rows[0][-2:] == ["--device", "cuda"],
                "GPU array task CLI is incorrect")
        checks.append("Preparation and array wrappers forward exact root, profile, task ID, and CUDA choice")

        result, rows = batch("summarize.qsub", "0")
        require(result.returncode == 0 and len(rows) == 2, result.stderr)
        require(all("--allow-partial" not in row for row in rows), "Complete summary silently allowed missing tasks")
        result, rows = batch("summarize.qsub", "0", fail_summary=True)
        require(result.returncode == 3 and len(rows) == 3, "Strict summary failure was hidden or diagnostics not saved")
        require(all("--allow-partial" in row for row in rows[1:]), "Failure did not save partial reports and bundle")
        result, rows = batch("summarize.qsub", "1", fail_summary=True)
        require(result.returncode == 0 and len(rows) == 2 and all("--allow-partial" in row for row in rows),
                "Intentional pilot summary was incorrectly required to be complete")
        checks.append("Summary wrapper is strict, preserves failure status after diagnostic bundle, and handles intentional pilots")

    record = {"status": "passed", "scheduler_calls": "local fake qsub only; no SCC submission",
              "checks": checks, "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                             for name in SCRIPTS}}
    destination = ROOT / "verification/submission_check.json"
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
