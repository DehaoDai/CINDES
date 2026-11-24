# Verification of this reproducibility package

The reports distinguish archived experimental results, explicitly synthetic test fixtures, and the reduced CPU check. The full GPU study was not rerun while preparing the archive.

## Archived numerical results

`check_reported_results.py` checks all 80 real archived pairs, their file hashes, expected settings, parameter and update counts, and their agreement with the previously verified summaries. It reproduces all 56 mean, sample-standard-deviation, and standard-error rows for the four sample sizes. No records are excluded. The report is `reported_results_check.json`.

```bash
python3 verification/check_reported_results.py
```

This checks the aggregation of saved metrics. The original generated observations are absent from the supplied historical archives, so it cannot independently recalculate those metrics from observations.

## Protocol and reporting behavior

`check_protocol_and_reporting.py` checks the preserved scientific configuration and source hashes, all 80 task assignments, the new output roots, correct sample-SD calculations, reuse of complete results, and rejection of missing or duplicate results. It uses synthetic records in a temporary directory solely to test reporting. Those values are deleted when the check ends and are not included in the scientific tables. The report is `protocol_and_reporting_check.json`.

```bash
python3 verification/check_protocol_and_reporting.py
```

`historical_protocol.json` retains the configuration and source comparison used when the successful small-sample package was prepared. Its historical-size list describes that original baseline comparison, not the set of tasks in the current release. The current study is fixed to n=500,1000,1500,2000.

## SCC submission behavior

`check_scc_submission.py` exercises the actual shell launcher with simulated `qsub` and `qstat` commands in temporary directories. It checks the 80-task range, two-task concurrency cap, summary dependency, pilot and subset submissions, quoting, resume protection, locks, and preservation of accepted job IDs when a later submission fails. No real scheduler jobs are submitted. The report is `scheduler_check.json`.

```bash
python3 verification/check_scc_submission.py
```

## Actual reduced CPU run

Both methods completed the package's smoke profile: n=64, d=7, K=20, M=4, 64 generated observations, and 20 requested optimization updates. The check verified finite generated arrays on the support, completion metadata, summaries, archive creation, and reuse of the completed pair. The resulting values are not part of `reported_results/`. `cpu_smoke_check.json` records the actual commands, environment, and checks performed.

To perform a new installation check:

```bash
bash run_local.sh smoke
```

## File integrity

`SHA256SUMS` lists the released files and their digests. On Linux, verify a freshly extracted copy with `sha256sum -c SHA256SUMS`; on macOS use `shasum -a 256 -c SHA256SUMS`. Regenerating summaries or verification reports can update those files. The scientific-source hash guard is separate and remains active for new runs.
