# Packaging checks

`packaging_validation.json` records the checks performed on this assembled archive.
The CPU smoke run completed both methods, cross-validation, refitting, all 400
reverse steps, evaluation, and strict summary validation. Exporting its result
bundle and rebuilding its four CSV/LaTeX summary files reproduced the files exactly.

Python and shell syntax, the canonical 80-task plan, and original scientific hashes
were checked. The portable driver and corrected SCC launcher were checked in dry-run
mode. No SCC jobs or full 80-pair scientific reruns were performed during packaging.
The recorded environment is for this CPU check, not the authors' final SCC runs.

Historical checks from the original code package are separately retained under
`../provenance/original_verification/`.

## Final SCC metadata verification

`final_run_metadata_verification.json` records the successful comparison of the
authors' final SCC metadata against this archive. All 14 critical scientific
files/assets, the full configuration, and 80 distinct paired-task identities match.
All 23 recorded source entries match their retained copies, with the original
README preserved under `provenance/original_package/`. The final task plan retains
the original seeds for the two rerun tasks. This verifies provenance and planned
run identities; the final per-replication outputs were not supplied for an
independent recomputation of their reported aggregate statistics.
