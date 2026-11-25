#!/bin/bash
# Run the unchanged experiment sequentially without a batch scheduler.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
Usage: bash experiment/run_local.sh <smoke|full> <output-directory>

smoke: reduced one-replication implementation check on CPU.
full: all 80 paired replications using CUDA for sample generation.

The output directory is relative to your current working directory. Reusing
the same directory resumes verified completed stages. Use separate output
directories for smoke and full profiles. Optional environment variables:
PYTHON_BIN=python3; CINDES_CV_WORKERS=2 (smoke) or 4 (full).
Run full only on a workstation or compute allocation with a usable CUDA GPU.
USAGE
    exit 0
fi
[[ $# == 2 ]] || { printf 'Usage: bash experiment/run_local.sh <smoke|full> <output-directory>\n' >&2; exit 2; }
profile="$1"
case "$profile" in
    smoke) device=cpu; default_workers=2 ;;
    full) device=cuda; default_workers=4 ;;
    *) printf 'Profile must be smoke or full.\n' >&2; exit 2 ;;
esac
python_bin="${PYTHON_BIN:-python3}"
command -v "$python_bin" >/dev/null || { printf 'Python executable not found: %s\n' "$python_bin" >&2; exit 2; }
workers="${CINDES_CV_WORKERS:-$default_workers}"
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { printf 'CINDES_CV_WORKERS must be a positive integer.\n' >&2; exit 2; }
output_root="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$2")"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$output_root/driver_logs"
driver_log="$output_root/driver_logs/local_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
printf 'Profile: %s\nOutput: %s\nCV workers: %s\n' "$profile" "$output_root" "$workers" | tee -a "$driver_log"
"$python_bin" - "$profile" <<'PY' 2>&1 | tee -a "$driver_log"
import json
import platform
import sys
import numpy
import scipy
import torch
profile = sys.argv[1]
available = torch.cuda.is_available()
print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                  "numpy": numpy.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
                  "cuda_runtime": torch.version.cuda, "cuda_available": available,
                  "gpu": torch.cuda.get_device_name(0) if available else None}, indent=2))
if profile == "full" and not available:
    raise SystemExit("The full profile requires a usable CUDA GPU. Use the smoke profile for a CPU check.")
PY
"$python_bin" -u prepare_reference.py --root "$output_root" --profile "$profile" 2>&1 | tee -a "$driver_log"
task_count="$("$python_bin" -c 'import study, sys; print(len(study.task_specs(study.make_config(sys.argv[1]))))' "$profile")"
for ((task=1; task<=task_count; task++)); do
    "$python_bin" -u run_task.py --root "$output_root" --profile "$profile" \
        --task-id "$task" --cv-workers "$workers" --device "$device" 2>&1 | tee -a "$driver_log"
done
"$python_bin" -u summarize_results.py --root "$output_root" --profile "$profile" 2>&1 | tee -a "$driver_log"
"$python_bin" -u make_return_bundle.py --root "$output_root" --profile "$profile" 2>&1 | tee -a "$driver_log"
if [[ "$profile" == "full" ]]; then
    "$python_bin" -u ../reproduce_tables.py --root "$output_root" --output "$output_root/journal_tables" 2>&1 | tee -a "$driver_log"
fi
printf 'Completed %s profile. Driver log: %s\n' "$profile" "$driver_log"
