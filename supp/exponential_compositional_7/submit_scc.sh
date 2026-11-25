#!/bin/bash
# Submit reference preparation, one paired GPU array, and the summary job.
set -euo pipefail

mode=submit
profile=n1000
project="${SCC_PROJECT:-dgmgrp}"
output_argument=""
tasks=""
pilot=0
resume=0
environment_script="${CINDES_ENV_SCRIPT:--}"
python_bin="${PYTHON_BIN:-python3}"

usage() {
    cat <<'HELP'
Usage: bash submit_scc.sh [options]

Default: 20 paired replications at n=1000, at most two GPU tasks at once.
Each task fits CINDES and DSM sequentially on the same allocated GPU.

  --profile n1000|full|smoke   n1000: 20 tasks; full: 80 tasks; smoke: 1 task
  --root DIRECTORY           Output directory; default outputs/exponential_PROFILE_v1
  --project NAME             SCC project; default dgmgrp
  --pilot                    Submit task 1 of the selected profile
  --tasks FIRST-LAST         Submit a contiguous part of the selected profile
  --resume                   Resubmit after previous jobs have left the queue;
                             completed compatible outputs are checked and reused
  --dry-run                  Print the three qsub commands without submitting
  --status                   Print experiment status without submitting
  --env-script FILE          Source another activation script in each batch job
  -h, --help                 Show this help

The full profile uses n=500,1000,1500,2000, with 20 paired replications each.
The smoke profile verifies execution and must not be used for scientific tables.
The script uses the successfully tested miniconda / academic-ml/fall-2026 /
fall-2026-pyt environment unless --env-script is supplied.
HELP
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) need_value "$@"; profile="$2"; shift 2 ;;
        --root) need_value "$@"; output_argument="$2"; shift 2 ;;
        --project) need_value "$@"; project="$2"; shift 2 ;;
        --tasks) need_value "$@"; tasks="$2"; shift 2 ;;
        --env-script) need_value "$@"; environment_script="$2"; shift 2 ;;
        --pilot) pilot=1; shift ;;
        --resume) resume=1; shift ;;
        --dry-run) [[ "$mode" == submit ]] || { printf 'Choose only one of --dry-run and --status.\n' >&2; exit 2; }; mode=dry; shift ;;
        --status) [[ "$mode" == submit ]] || { printf 'Choose only one of --dry-run and --status.\n' >&2; exit 2; }; mode=status; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$profile" == n1000 || "$profile" == full || "$profile" == smoke ]] || {
    printf 'Profile must be n1000, full, or smoke.\n' >&2; exit 2;
}
[[ "$project" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf 'Invalid SCC project name.\n' >&2; exit 2; }
command -v "$python_bin" >/dev/null || { printf 'Cannot find Python executable: %s\n' "$python_bin" >&2; exit 2; }
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory"
for file in study_config.py prepare_reference.py run_experiment.py summarize_results.py make_return_bundle.py \
            prepare_reference.qsub full_array.qsub summarize.qsub scc_env.sh; do
    [[ -f "$file" ]] || { printf 'Missing package file: %s\n' "$file" >&2; exit 2; }
done
task_count="$("$python_bin" study_config.py --profile "$profile" --task-count)"
[[ "$task_count" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid task count from study_config.py: %s\n' "$task_count" >&2; exit 2; }
if (( pilot )); then
    [[ -z "$tasks" ]] || { printf 'Use either --pilot or --tasks.\n' >&2; exit 2; }
    tasks=1-1
fi
tasks="${tasks:-1-$task_count}"
[[ "$tasks" =~ ^([1-9][0-9]*)-([1-9][0-9]*)$ ]] || {
    printf 'Use a task range such as 1-20 or 1-1.\n' >&2; exit 2;
}
first_task="${BASH_REMATCH[1]}"
last_task="${BASH_REMATCH[2]}"
(( first_task <= last_task && last_task <= task_count )) || {
    printf 'Task range must lie within 1-%s for profile %s.\n' "$task_count" "$profile" >&2; exit 2;
}
output_argument="${output_argument:-outputs/exponential_${profile}_v1}"
output_root="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$output_argument")"
if [[ "$environment_script" != - ]]; then
    environment_script="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$environment_script")"
    [[ -f "$environment_script" ]] || { printf 'Missing activation script: %s\n' "$environment_script" >&2; exit 2; }
fi
for value in "$output_root" "$environment_script"; do
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || { printf 'Paths must not contain line breaks.\n' >&2; exit 2; }
done
if [[ "$mode" == status ]]; then
    exec "$python_bin" summarize_results.py --root "$output_root" --profile "$profile" --status
fi
allow_partial=0
if (( first_task != 1 || last_task != task_count )); then allow_partial=1; fi
base_command=(qsub -terse -P "$project" -o "$output_root/logs")
reference_command=("${base_command[@]}" prepare_reference.qsub "$output_root" "$profile" "$environment_script")
if [[ "$mode" == dry ]]; then
    printf 'Project: %s\nProfile: %s\nOutput directory: %s\nTasks: %s\n' "$project" "$profile" "$output_root" "$tasks"
    printf '%q ' "${reference_command[@]}"; printf '\n'
    printf '%q ' "${base_command[@]}" -hold_jid '<REFERENCE_JOB_ID>' -t "$tasks" -tc 2 \
        full_array.qsub "$output_root" "$profile" "$environment_script"; printf '\n'
    printf '%q ' "${base_command[@]}" -hold_jid '<ARRAY_JOB_ID>' \
        summarize.qsub "$output_root" "$profile" "$environment_script" "$allow_partial"; printf '\n'
    printf 'No jobs submitted.\n'
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
if (( ${#previous_submissions[@]} > 0 && ! resume )); then
    printf 'This output directory already has a submission record.\n' >&2
    printf 'After its jobs leave qstat, use --resume to continue checked outputs, or use a new --root.\n' >&2
    exit 2
fi
submission_log="$output_root/submissions/submission_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
printf 'project=%s\nprofile=%s\noutput_root=%s\ntasks=%s\nresume=%s\nstatus=submitting\n' \
    "$project" "$profile" "$output_root" "$tasks" "$resume" > "$submission_log"
submission_complete=0
submission_stage=reference
reference_job=""
array_job=""
summary_job=""
on_exit() {
    local rc=$?
    if (( ! submission_complete )); then
        printf 'status=interrupted_or_failed\nstage=%s\nexit_code=%s\n' "$submission_stage" "$rc" >> "$submission_log"
        printf 'Submission did not finish. Existing jobs were not cancelled.\nRecord: %s\n' "$submission_log" >&2
        printf 'Check qstat -u "$USER" before retrying with --resume.\n' >&2
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

submit_one() {
    local label="$1"
    shift
    local response candidate code
    printf '%s_command=' "$label" >> "$submission_log"
    printf '%q ' "$@" >> "$submission_log"
    printf '\n' >> "$submission_log"
    if "$@" > "$submission_log.$label.stdout" 2> "$submission_log.$label.stderr"; then
        code=0
    else
        code=$?
        printf '%s_submission_exit_code=%s\n' "$label" "$code" >> "$submission_log"
        cat "$submission_log.$label.stderr" >&2
        printf 'qsub failed at the %s stage. Inspect the recorded response before retrying.\n' "$label" >&2
        return "$code"
    fi
    response="$(cat "$submission_log.$label.stdout")"
    response="${response#"${response%%[![:space:]]*}"}"
    response="${response%"${response##*[![:space:]]}"}"
    if [[ "$response" =~ ^([0-9]+)(\.[0-9]+(-[0-9]+)?(:[0-9]+)?)?$ ]]; then
        candidate="${BASH_REMATCH[1]}"
    else
        printf '%s_unparsed_response=%q\n' "$label" "$response" >> "$submission_log"
        printf 'qsub returned an unrecognized response; the job may already have been submitted.\n' >&2
        printf 'Inspect %s.%s.stdout and qstat before retrying.\n' "$submission_log" "$label" >&2
        return 2
    fi
    last_job_id="$candidate"
    printf '%s_job=%s\n' "$label" "$candidate" | tee -a "$submission_log"
    if [[ -s "$submission_log.$label.stderr" ]]; then cat "$submission_log.$label.stderr" >&2; fi
}

submit_one reference "${reference_command[@]}"
reference_job="$last_job_id"
submission_stage=array
submit_one array "${base_command[@]}" -hold_jid "$reference_job" -t "$tasks" -tc 2 \
    full_array.qsub "$output_root" "$profile" "$environment_script"
array_job="$last_job_id"
submission_stage=summary
submit_one summary "${base_command[@]}" -hold_jid "$array_job" \
    summarize.qsub "$output_root" "$profile" "$environment_script" "$allow_partial"
summary_job="$last_job_id"
submission_complete=1
printf 'status=submitted\n' >> "$submission_log"
printf 'Submitted %s paired tasks for profile %s; at most 2 GPU tasks run concurrently.\n' \
    "$((last_task-first_task+1))" "$profile"
printf 'Submission record: %s\n' "$submission_log"
printf 'Scheduler status: qstat -u "$USER"\n'
printf 'Experiment status: bash %q --profile %q --root %q --status\n' "$script_directory/submit_scc.sh" "$profile" "$output_root"
printf 'Strict summary: python3 %q --root %q --profile %q\n' "$script_directory/summarize_results.py" "$output_root" "$profile"

