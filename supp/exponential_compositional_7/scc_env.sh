#!/bin/bash
# Source in a scheduled job, passing an optional activation script as argument 1.
set -eo pipefail
set +u
activation_script="${1:--}"
if [[ "$activation_script" != - && -n "$activation_script" ]]; then
    [[ -f "$activation_script" ]] || { printf 'Missing activation script: %s\n' "$activation_script" >&2; return 2; }
    source "$activation_script"
else
    type module >/dev/null 2>&1 || { printf 'SCC module command is unavailable; use a login-shell batch job.\n' >&2; return 2; }
    if module is-loaded python3 >/dev/null 2>&1; then module unload python3; fi
    module load miniconda
    module load academic-ml/fall-2026
    conda activate fall-2026-pyt
fi
# Some activation scripts read unset variables; enable nounset only afterwards.
set -u
: "${NSLOTS:?This batch job must request CPU slots with -pe omp.}"
[[ "$NSLOTS" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid NSLOTS value.\n' >&2; return 2; }
export CINDES_CPU_THREADS="$NSLOTS"
export OMP_NUM_THREADS="$NSLOTS"
export MKL_NUM_THREADS="$NSLOTS"
export OPENBLAS_NUM_THREADS="$NSLOTS"
export NUMEXPR_NUM_THREADS="$NSLOTS"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# CUDA_VISIBLE_DEVICES is provided by SCC and must not be overwritten.
printf 'Host: %s; job: %s; task: %s; CPU slots: %s\n' \
    "${HOSTNAME:-unknown}" "${JOB_ID:-none}" "${SGE_TASK_ID:-none}" "$NSLOTS"
date
if type module >/dev/null 2>&1; then module list; fi

