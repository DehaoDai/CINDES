#!/bin/bash
# Submit reference preparation, the paired array, then complete-study tables.
# The Python experiment is unchanged; this wrapper makes the SCC account and
# output location explicit for a new user.
set -euo pipefail

case "${1:-submit}" in
    -h|--help)
        cat <<'USAGE'
Usage: SCC_PROJECT=your_project bash experiment/submit_scc.sh [--dry-run]

Run from any directory. The default output is experiment/outputs/full_study_v1.
Set CINDES_OUTPUT_ROOT to choose a different output directory. A relative value
is interpreted relative to the directory from which you invoke this script.
Set CINDES_ENV_SCRIPT to an absolute activation-script path to use your own
Python environment instead of the recorded BU SCC module environment.
PYTHON_BIN may name the Python executable (default: python3).
--dry-run prints the submission commands and does not submit any jobs.
USAGE
        exit 0 ;;
    submit|--dry-run) mode="${1:-submit}" ;;
    *) printf 'Usage: SCC_PROJECT=your_project bash submit_scc.sh [--dry-run]\n' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { printf 'Too many arguments.\n' >&2; exit 2; }
: "${SCC_PROJECT:?Set SCC_PROJECT to your BU SCC project allocation.}"
project="$SCC_PROJECT"
python_bin="${PYTHON_BIN:-python3}"
command -v "$python_bin" >/dev/null || { printf 'Python executable not found: %s\n' "$python_bin" >&2; exit 2; }
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_argument="${CINDES_OUTPUT_ROOT:-$script_directory/outputs/full_study_v1}"
output_root="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$output_argument")"
environment_script="${CINDES_ENV_SCRIPT:-}"
if [[ -n "$environment_script" ]]; then
    environment_script="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$environment_script")"
    [[ -f "$environment_script" ]] || { printf 'Environment script does not exist: %s\n' "$environment_script" >&2; exit 2; }
fi
# Grid Engine separates -v assignments with commas.
for value in "$output_root" "$environment_script" "$python_bin"; do
    [[ "$value" != *','* && "$value" != *$'\n'* && "$value" != *$'\r'* ]] || {
        printf 'SCC environment values must not contain commas or line breaks.\n' >&2; exit 2;
    }
done
job_environment="CINDES_OUTPUT_ROOT=$output_root,PYTHON_BIN=$python_bin"
if [[ -n "$environment_script" ]]; then
    job_environment+=",CINDES_ENV_SCRIPT=$environment_script"
fi
cd "$script_directory"
for file in prepare_reference.qsub full_array.qsub summarize.qsub scc_env.sh; do
    [[ -f "$file" ]] || { printf 'Missing %s\n' "$file" >&2; exit 2; }
done
if [[ "$mode" == "--dry-run" ]]; then
    printf 'Project: %s\nOutput directory: %s\n' "$project" "$output_root"
    printf 'qsub -terse -P %q -v %q prepare_reference.qsub\n' "$project" "$job_environment"
    printf 'qsub -terse -P %q -v %q -hold_jid <REFERENCE_JOB_ID> full_array.qsub\n' "$project" "$job_environment"
    printf 'qsub -terse -P %q -v %q -hold_jid <ARRAY_JOB_ID> summarize.qsub\n' "$project" "$job_environment"
    exit 0
fi
command -v qsub >/dev/null || { printf 'qsub is not available; run this on BU SCC.\n' >&2; exit 2; }
mkdir -p submissions
submission_log="submissions/submission_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
printf 'project=%s\noutput_root=%s\n' "$project" "$output_root" > "$submission_log"
parse_id() {
    local candidate="${1%%.*}"
    [[ "$candidate" =~ ^[0-9]+$ ]] || {
        printf 'Could not parse qsub response: %s\n' "$1" >&2; return 1;
    }
    printf '%s' "$candidate"
}
reference_reply=$(qsub -terse -P "$project" -v "$job_environment" prepare_reference.qsub)
reference_job=$(parse_id "$reference_reply")
printf 'reference_job=%s\n' "$reference_job" | tee -a "$submission_log"
array_reply=$(qsub -terse -P "$project" -v "$job_environment" -hold_jid "$reference_job" full_array.qsub)
array_job=$(parse_id "$array_reply")
printf 'array_job=%s\n' "$array_job" | tee -a "$submission_log"
summary_reply=$(qsub -terse -P "$project" -v "$job_environment" -hold_jid "$array_job" summarize.qsub)
summary_job=$(parse_id "$summary_reply")
printf 'summary_job=%s\n' "$summary_job" | tee -a "$submission_log"
printf 'Submitted 80 paired tasks, with at most 2 GPU tasks running concurrently.\n'
printf 'Submission record: %s\n' "$submission_log"
printf 'Check scheduler: qstat -u "$USER"\n'
printf 'Check completion: %q %q --root %q --profile full --status\n' "$python_bin" "$script_directory/summarize_results.py" "$output_root"
