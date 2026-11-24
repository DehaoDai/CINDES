#!/bin/bash
# Source inside a batch job. The default is the recorded BU SCC environment.
# Set CINDES_ENV_SCRIPT to an existing activation script to use another one.
set -eo pipefail
if [[ -n "${CINDES_ENV_SCRIPT:-}" ]]; then
    [[ -f "$CINDES_ENV_SCRIPT" ]] || { printf 'Missing environment script: %s\n' "$CINDES_ENV_SCRIPT" >&2; exit 2; }
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
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'Host: %s; job: %s; task: %s; slots: %s\n' \
    "${HOSTNAME:-unknown}" "${JOB_ID:-none}" "${SGE_TASK_ID:-none}" "${NSLOTS:-1}"
date
if type module >/dev/null 2>&1; then
    module list
fi
