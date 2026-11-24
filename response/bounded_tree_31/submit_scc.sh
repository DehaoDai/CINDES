#!/bin/bash
# Submit the bounded d=31 tree study at n=500,1000,1500,2000.
# SCC array range and concurrency are directives in a saved submission copy.
set -euo pipefail
mode=submit
project="${SCC_PROJECT:-dgmgrp}"
output_argument=""
tasks=""
pilot=0
resume=0
environment_script="${CINDES_ENV_SCRIPT:--}"
python_bin="${PYTHON_BIN:-python3}"
threads="${CINDES_CPU_THREADS:-4}"

usage() {
    cat <<'HELP'
Usage: bash submit_scc.sh [options]

Default: 80 paired tasks, 20 seeds each at n=500,1000,1500,2000, at most two GPUs
at once. A full submission also schedules a strict summary and results ZIP.

  --root DIRECTORY       Output directory; overrides CINDES_OUTPUT_ROOT
  --project NAME         SCC project; overrides SCC_PROJECT (default dgmgrp)
  --pilot                Submit task 1: n=500, seed=0
  --tasks FIRST-LAST     Submit a contiguous range within 1-80
  --resume               Check earlier job IDs are absent from qstat, then
                         resubmit; compatible completed outputs are reused
  --dry-run              Print the plan; create no files and submit no jobs
  --status               Print result status without submitting jobs
  --env-script FILE      Source this environment script in each batch job
  -h, --help             Show this help

Default output: outputs/bounded_n500_2000_v1 beside this script.
Pilot/subset submissions omit the full-study summary job. After the pilot has
left the queue, use --resume without --pilot to submit the full study.
PYTHON_BIN defaults to python3. CINDES_CPU_THREADS may be 1,2,3,4 (default 4).
Array range/concurrency options are never passed on the qsub command line.
HELP
}
need_value() {
    [[ $# -ge 2 && -n "$2" ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) need_value "$@"; output_argument="$2"; shift 2 ;;
        --project) need_value "$@"; project="$2"; shift 2 ;;
        --tasks) need_value "$@"; [[ -z "$tasks" ]] || { printf 'Specify --tasks once.\n' >&2; exit 2; }; tasks="$2"; shift 2 ;;
        --pilot) pilot=1; shift ;;
        --resume) resume=1; shift ;;
        --env-script) need_value "$@"; environment_script="$2"; shift 2 ;;
        --dry-run) [[ "$mode" == submit ]] || { printf 'Choose one of --dry-run and --status.\n' >&2; exit 2; }; mode=dry; shift ;;
        --status) [[ "$mode" == submit ]] || { printf 'Choose one of --dry-run and --status.\n' >&2; exit 2; }; mode=status; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$project" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf 'Invalid SCC project name.\n' >&2; exit 2; }
[[ "$threads" =~ ^[1-4]$ ]] || { printf 'CINDES_CPU_THREADS must be between 1 and 4.\n' >&2; exit 2; }
command -v "$python_bin" >/dev/null || { printf 'Cannot find Python executable: %s\n' "$python_bin" >&2; exit 2; }
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_argument="${output_argument:-${CINDES_OUTPUT_ROOT:-$script_directory/outputs/bounded_n500_2000_v1}}"
# Resolve user paths before changing the working directory.
output_root="$("$python_bin" -B -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$output_argument")"
if [[ "$environment_script" != - ]]; then
    environment_script="$("$python_bin" -B -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$environment_script")"
    [[ -f "$environment_script" ]] || { printf 'Missing activation script: %s\n' "$environment_script" >&2; exit 2; }
fi
for value in "$output_root" "$environment_script" "$python_bin"; do
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || { printf 'Paths and executable names must not contain line breaks.\n' >&2; exit 2; }
done
cd "$script_directory"
for file in study_config.py run_experiment.py summarize_results.py make_results_bundle.py \
            tree_array.qsub summarize.qsub scc_env.sh; do
    [[ -f "$file" ]] || { printf 'Missing package file: %s\n' "$file" >&2; exit 2; }
done
task_count="$("$python_bin" -B -c 'from study_config import expected_pairs; print(len(expected_pairs("full")))')"
[[ "$task_count" == 80 ]] || { printf 'Expected the fixed 80-task bounded study, got %s tasks.\n' "$task_count" >&2; exit 2; }
if (( pilot )); then
    [[ -z "$tasks" ]] || { printf 'Use either --pilot or --tasks.\n' >&2; exit 2; }
    tasks=1-1
fi
tasks="${tasks:-1-80}"
[[ "$tasks" =~ ^([1-9][0-9]?)-([1-9][0-9]?)$ ]] || { printf 'Use a task range such as 1-80 or 21-40.\n' >&2; exit 2; }
first_task="${BASH_REMATCH[1]}"; last_task="${BASH_REMATCH[2]}"
(( first_task <= last_task && last_task <= task_count )) || { printf 'Task range must lie within 1-80.\n' >&2; exit 2; }
if [[ "$mode" == status ]]; then
    exec "$python_bin" -B summarize_results.py --root "$output_root" --profile full --status
fi
# Validate and construct the exact array script before any submission or write.
# Requiring one valid range and one cap avoids silently submitting a malformed
# template. The replacement always installs this request's range and cap two.
array_text="$("$python_bin" -B - tree_array.qsub "$tasks" <<'PY_ARRAY'
from pathlib import Path
import re
import sys
text = Path(sys.argv[1]).read_text(encoding='utf-8')
for option, pattern, value in (
    ('t', r'[1-9][0-9]*-[1-9][0-9]*', sys.argv[2]),
    ('tc', r'[1-9][0-9]*', '2'),
):
    expression = r'(?m)^#\$[ \t]+-' + option + r'(?:[ \t]+[^\r\n]*)?$'
    found = re.findall(expression, text)
    if len(found) != 1 or re.fullmatch(r'#\$[ \t]+-' + option + r'[ \t]+' + pattern + r'[ \t]*', found[0]) is None:
        raise SystemExit('tree_array.qsub must contain exactly one valid -' + option + ' directive.')
    text = re.sub(expression, lambda match: '#$ -' + option + ' ' + value, text)
print(text, end='')
PY_ARRAY
)"
bash -n <<< "$array_text"
bash -n summarize.qsub scc_env.sh
base_command=(qsub -terse -P "$project" -o "$output_root/logs")
if [[ "$mode" == dry ]]; then
    printf 'Project: %s\nOutput directory: %s\nTasks: %s\n' "$project" "$output_root" "$tasks"
    printf 'Generated array directives: #$ -t %s; #$ -tc 2\n' "$tasks"
    printf 'Saved submission scripts and logs will be under %s/submissions/.\n' "$output_root"
    printf '%q ' "${base_command[@]}" '<GENERATED_ARRAY_SCRIPT>' "$output_root" "$environment_script" "$python_bin" "$threads"; printf '\n'
    if [[ "$tasks" == 1-80 ]]; then
        printf '%q ' "${base_command[@]}" -hold_jid '<ARRAY_JOB_ID>' summarize.qsub "$output_root" "$environment_script" "$python_bin"; printf '\n'
    else
        printf 'Partial array: no full-study summary job will be submitted.\n'
    fi
    if (( resume )); then printf 'On submission, recorded job IDs must be absent from a readable qstat XML response.\n'; fi
    printf 'No files created; no jobs submitted.\n'
    exit 0
fi
command -v qsub >/dev/null || { printf 'qsub is unavailable; run this on BU SCC.\n' >&2; exit 2; }
command -v flock >/dev/null || { printf 'flock is required to protect submission records.\n' >&2; exit 2; }
mkdir -p "$output_root/logs" "$output_root/submissions"
exec 9>"$output_root/.submission.lock"
flock -n 9 || { printf 'Another submission for this output directory is in progress.\n' >&2; exit 2; }
shopt -s nullglob
previous_submissions=("$output_root"/submissions/submission_*.txt)
shopt -u nullglob
if (( ${#previous_submissions[@]} && ! resume )); then
    printf 'This output directory already has a submission record.\n' >&2
    printf 'After its jobs leave qstat, use --resume to reuse checked outputs, or select a new --root.\n' >&2
    exit 2
fi
if (( ${#previous_submissions[@]} && resume )); then
    command -v qstat >/dev/null || { printf 'Cannot verify earlier jobs: qstat is unavailable. No new jobs submitted.\n' >&2; exit 2; }
    : "${USER:?USER must identify the SCC account for the resume check.}"
    resume_record="$output_root/submissions/resume_check_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    printf 'qstat -xml -u %q\n' "$USER" > "$resume_record.command.txt"
    if ! qstat -xml -u "$USER" > "$resume_record.stdout.xml" 2> "$resume_record.stderr.txt"; then
        printf 'Cannot read scheduler status. No new jobs submitted. Inspect %s.stderr.txt and retry when qstat works.\n' "$resume_record" >&2
        exit 2
    fi
    "$python_bin" -B - "$resume_record.stdout.xml" "${previous_submissions[@]}" <<'PY_RESUME'
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
try:
    root = ET.parse(sys.argv[1]).getroot()
except (ET.ParseError, OSError) as exc:
    raise SystemExit('Cannot parse qstat XML; no new jobs submitted. Inspect the saved scheduler response. ' + str(exc))
if root.tag.split('}')[-1] != 'job_info':
    raise SystemExit('Unexpected qstat XML root; no new jobs submitted. Inspect the saved scheduler response.')
active = {node.text.strip() for node in root.iter()
          if node.tag.split('}')[-1] == 'JB_job_number' and node.text}
previous = set()
for name in sys.argv[2:]:
    text = Path(name).read_text(encoding='utf-8')
    if '_unparsed_response=' in text:
        raise SystemExit('An earlier qsub response was not recognized. Inspect ' + name +
                         ' and its stdout plus qstat before resolving the missing job ID. No new jobs submitted.')
    previous.update(re.findall(r'(?m)^(?:array|summary)_job=([0-9]+)$', text))
still_active = sorted(previous & active, key=int)
if still_active:
    raise SystemExit('Earlier jobs remain queued or running: ' + ', '.join(still_active) +
                     '. Wait until they leave qstat, then retry --resume. No new jobs submitted.')
print('Resume check: no previously recorded job IDs remain in the scheduler.')
PY_RESUME
fi
submission_log="$output_root/submissions/submission_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
( set -o noclobber; printf 'project=%s\nprofile=full\noutput_root=%s\ntasks=%s\nresume=%s\nstatus=submitting\n' \
    "$project" "$output_root" "$tasks" "$resume" > "$submission_log" )
submission_complete=0
submission_stage=array_script
last_job_id=""
on_exit() {
    local code=$?
    if (( ! submission_complete )); then
        printf 'status=interrupted_or_failed\nstage=%s\nexit_code=%s\n' "$submission_stage" "$code" >> "$submission_log"
        printf 'Submission did not finish; any accepted jobs remain submitted.\nRecord: %s\n' "$submission_log" >&2
        printf 'Inspect this record and qstat before retrying with --resume.\n' >&2
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
array_script="${submission_log%.txt}.array.qsub"
( set -o noclobber; printf '%s\n' "$array_text" > "$array_script" )
printf 'array_script=%s\n' "$array_script" >> "$submission_log"
submit_one() {
    local label="$1"; shift
    local response code candidate=""
    printf '%s_command=' "$label" >> "$submission_log"
    printf '%q ' "$@" >> "$submission_log"; printf '\n' >> "$submission_log"
    if "$@" > "$submission_log.$label.stdout" 2> "$submission_log.$label.stderr"; then code=0; else code=$?; fi
    response="$(cat "$submission_log.$label.stdout")"
    response="${response#"${response%%[![:space:]]*}"}"
    response="${response%"${response##*[![:space:]]}"}"
    if [[ "$response" =~ ^([0-9]+)(\.[0-9]+(-[0-9]+)?(:[0-9]+)?)?$ ]]; then
        candidate="${BASH_REMATCH[1]}"
        last_job_id="$candidate"
        printf '%s_job=%s\n' "$label" "$candidate" | tee -a "$submission_log"
    elif [[ -n "$response" || "$code" == 0 ]]; then
        printf '%s_unparsed_response=%q\n' "$label" "$response" >> "$submission_log"
    fi
    printf '%s_submission_exit_code=%s\n' "$label" "$code" >> "$submission_log"
    if [[ -s "$submission_log.$label.stderr" ]]; then cat "$submission_log.$label.stderr" >&2; fi
    if (( code )); then
        printf 'qsub failed at the %s stage. Any job ID it reported is preserved in the record.\n' "$label" >&2
        return "$code"
    fi
    if [[ -z "$candidate" ]]; then
        printf 'qsub returned an unrecognized response; a job may already have been submitted. Inspect the saved stdout and qstat.\n' >&2
        return 2
    fi
}
submission_stage=array
submit_one array "${base_command[@]}" "$array_script" "$output_root" "$environment_script" "$python_bin" "$threads"
array_job="$last_job_id"
if [[ "$tasks" == 1-80 ]]; then
    submission_stage=summary
    submit_one summary "${base_command[@]}" -hold_jid "$array_job" \
        summarize.qsub "$output_root" "$environment_script" "$python_bin"
fi
submission_complete=1
printf 'status=submitted\n' >> "$submission_log"
printf 'Submitted %s paired tasks; at most 2 GPU tasks run concurrently.\n' "$((last_task-first_task+1))"
printf 'Submission record: %s\n' "$submission_log"
printf 'Scheduler status: qstat -u "$USER"\n'
printf 'Experiment status: bash %q --root %q --status\n' "$script_directory/submit_scc.sh" "$output_root"
if [[ "$tasks" == 1-80 ]]; then
    printf 'After all 80 pairs succeed, the summary job creates %s/summary/results_for_tables.zip\n' "$output_root"
fi
