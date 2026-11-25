#!/bin/bash
# Source in a batch job. Override with CINDES_ENV_SCRIPT for another environment.
# Do not enable nounset until module/conda activation has finished.
set -eo pipefail
set +u
if [[ -n "${CINDES_ENV_SCRIPT:-}" ]]; then
    [[ -f "$CINDES_ENV_SCRIPT" ]] || { printf 'Missing activation script: %s\n' "$CINDES_ENV_SCRIPT" >&2; return 2; }
    source "$CINDES_ENV_SCRIPT"
else
    if module is-loaded python3 >/dev/null 2>&1; then
        module unload python3
    fi
    module load miniconda
    module load academic-ml/fall-2026
    conda activate fall-2026-pyt
fi
set -u
: "${NSLOTS:?This job must request CPU slots with -pe omp.}"
[[ "$NSLOTS" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid NSLOTS.\n' >&2; return 2; }
export CINDES_CPU_THREADS="${CINDES_CPU_THREADS:-$NSLOTS}"
[[ "$CINDES_CPU_THREADS" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid CINDES_CPU_THREADS.\n' >&2; return 2; }
(( CINDES_CPU_THREADS <= NSLOTS )) || { printf 'CPU thread cap exceeds allocated slots.\n' >&2; return 2; }
export OMP_NUM_THREADS="$CINDES_CPU_THREADS"
export MKL_NUM_THREADS="$CINDES_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CINDES_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CINDES_CPU_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
printf 'Host %s; job %s; task %s; slots %s; requested CPU threads %s\n' \
    "${HOSTNAME:-unknown}" "${JOB_ID:-none}" "${SGE_TASK_ID:-none}" "$NSLOTS" "$CINDES_CPU_THREADS"
date
if type module >/dev/null 2>&1; then module list; fi
