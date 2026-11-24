#!/bin/bash
# Run from any directory: bash run_local.sh smoke [output_directory]
#                       bash run_local.sh full  [output_directory]
set -euo pipefail
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:-full}"
case "$profile" in
    -h|--help)
        printf 'Usage: bash run_local.sh {smoke|full} [output_directory]\n'
        printf 'smoke is one small CPU check; full runs 80 paired tasks sequentially on a CUDA GPU.\n'
        exit 0 ;;
    smoke|full) ;;
    *) printf 'Expected profile smoke or full.\n' >&2; exit 2 ;;
esac
[[ $# -le 2 ]] || { printf 'Too many arguments.\n' >&2; exit 2; }
python_bin="${PYTHON_BIN:-python3}"
if [[ "$profile" == full ]]; then default_output=outputs/bounded_n500_2000_v1; else default_output=outputs/bounded_n500_2000_smoke_v1; fi
output_root="${2:-$script_directory/$default_output}"
# Resolve a user-supplied relative path before changing the working directory.
output_root="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$output_root")"
cd "$script_directory"
if [[ "$profile" == "smoke" ]]; then
    CUDA_VISIBLE_DEVICES="" CINDES_CPU_THREADS="${CINDES_CPU_THREADS:-1}" \
        "$python_bin" run_experiment.py --profile smoke --all --root "$output_root"
else
    "$python_bin" run_experiment.py --profile full --all --root "$output_root" --require-cuda
fi
"$python_bin" summarize_results.py --profile "$profile" --root "$output_root"
"$python_bin" make_results_bundle.py --profile "$profile" --root "$output_root"
