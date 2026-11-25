# CINDES versus DSM: exponential-tailed compositional experiment

This archive contains the implementation of the seven-dimensional exponential-tailed
experiment, including data generation, model selection and fitting, diffusion
sampling, evaluation, and table generation. The scientific Python files and the
supplied numerical lookup table are unchanged from the original experiment package.
The corrected SCC launcher is included separately from the portable driver.

The full profile uses 20 paired replications at each sample size
**500, 1000, 1500, and 2000** (80 paired runs). Each pair fits CINDES and DSM to
the same observations. `reported_results/` contains the final aggregate statistics
reported by the authors, clearly separated from outputs of new runs.

**Verified provenance:** the scientific source hashes, full configuration, and all
80 task identities match the final SCC run metadata supplied by the authors.
The final plan retains the original replication seeds, including seed 4001 for
task 2 (n=500) and seed 4009 for task 50 (n=1500). The three supplied metadata
files are included under `provenance/final_scc_run/`. See
`REPRODUCIBILITY_NOTES.md` for the verification scope.

## 1. Installation

Use Python 3.10 or newer on Linux, macOS, or WSL. The run-management code uses
Unix file locking. CUDA is optional for the small execution check; a CUDA GPU is
recommended for the full study.

From the extracted archive directory:

```bash
cd CINDES_exponential_reproducibility
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements are NumPy >= 1.24, SciPy >= 1.10, and PyTorch >= 2.6. Use a
PyTorch build compatible with the intended CPU or CUDA environment. There are
no external datasets or pretrained model weights to download.

Keep `experiment/reference_score_table.npz` in place. This supplied numerical
table is required by the reference-score implementation and is covered by the
scientific source hashes. It is not a fitted CINDES or DSM model.

## 2. Small execution check

```bash
python reproduce.py --profile smoke --device cpu --threads 1
```

This runs data preparation, both estimators, cross-validation, final fitting,
sampling, metric evaluation, and strict summarization. It uses one dataset of
size 48, two training epochs, reduced particle and sample counts, and all 400
reverse steps. Its output is explicitly labeled test-only and must not be used
as scientific evidence. The default output is `outputs/exponential_smoke_v1/`.

To check the available commands without running the experiment:

```bash
python reproduce.py --profile full --device cuda --threads 4 --dry-run
```

## 3. Reproduce the full study

```bash
python reproduce.py --profile full --device cuda --threads 4
```

This executes the 80 paired tasks sequentially on the selected device and writes
strict summaries after completion. It is a full training and sampling workload,
not a quick table-formatting command. The default root is
`outputs/exponential_full_v1/`. Completed compatible stages are checked and reused
when the same command is run again.

The `n1000` profile runs only the 20 pairs at sample size 1000. To execute a selected
subset of the full profile sequentially:

```bash
python reproduce.py --profile full --device cuda --threads 4 --tasks 1-20
```

The full task ranges are 1-20 for n=500, 21-40 for n=1000, 41-60 for n=1500,
and 61-80 for n=2000. A subset run ends with a status report. Use the strict
summary after all tasks have finished. For parallel execution, use the SCC
launcher, which prepares the reference before starting its dependent task array.
For other schedulers, prepare the reference once and then launch the underlying
`run_experiment.py` entrypoint for distinct task IDs. Do not launch multiple
`reproduce.py` processes concurrently against one root: each starts preparation.

The underlying entrypoints are also available:

```bash
python prepare_reference.py --root outputs/exponential_full_v1 --profile full
python run_experiment.py --root outputs/exponential_full_v1 --profile full \
    --task-id 1 --device cuda --threads 4
python summarize_results.py --root outputs/exponential_full_v1 --profile full --status
```

Run these commands from the package directory. The portable driver always passes
absolute output paths to avoid differences in relative-path handling.

For SCC, use the recovered corrected launcher:

```bash
bash submit_scc_fixed.sh --profile full --root outputs/exponential_full_v1 \
    --project YOUR_SCC_PROJECT --dry-run
```

Remove `--dry-run` to submit. This schedules reference preparation, a dependent
GPU task array, and a dependent summary job. Supply `--env-script /absolute/path/env.sh`
to activate a different Python environment. The supplied default environment is
specific to the authors' SCC installation. Use `--resume` for a previously submitted
root only after its earlier jobs have left the queue. Scheduler templates request
one GPU, four CPU slots, at least 16 GB GPU memory, and a 12-hour limit per task;
at most two GPU tasks run concurrently. No jobs are submitted by extracting this archive.

## 4. Produce the tables

After all 80 paired tasks finish:

```bash
python summarize_results.py --root outputs/exponential_full_v1 --profile full
```

The strict summary validates both methods and their saved artifacts for every
expected replication. Its principal outputs are:

| Output under `summary/` | Contents |
| --- | --- |
| `summary.csv` | Accuracy means, sample SDs, SEs, and actual replication counts |
| `runtime.csv` | Computation-time summaries |
| `accuracy_table.tex` | LaTeX accuracy table |
| `runtime_table.tex` | LaTeX computation-time table |
| `paired_differences.csv`, `paired_summary.csv` | Paired method comparisons |
| `per_run.csv`, `run_records.jsonl` | Per-run records |
| `export.json` | Validated records for portable table regeneration |

For an explicitly incomplete study, `--allow-partial` reports only completed pairs
and labels the actual counts. A status check alone uses `--status`.

To export the completed study's table records and provenance:

```bash
python make_return_bundle.py --root outputs/exponential_full_v1 --profile full
```

This creates `outputs/exponential_full_v1/summary/results_for_tables.zip`.
It contains summaries, per-run metric records, model-selection information, and
recorded source and environment metadata. Add `--include-samples` to include
the large sample arrays and `--include-checkpoints` to include fitted model
checkpoints. Both may be supplied together. The default bundle excludes these
large artifacts.

## 5. Scientific specification

### Distribution

For positive observations y in (0,infinity)^7, the density is proportional to

```text
q_Y(y) * exp(1.5 * tanh(H(y))),  where q_Y(y) = exp(-sum_j y_j).
u_j = 1 - 2*exp(-y_j)
H(y) = (3/sqrt(6)) * (-u4*u2 - u4*u5 - u2*u6 - u2*u3 + u5*u1 + u5*u7).
```

Training observations and the independent evaluation sample are generated by
rejection sampling from product Exp(1), accepting with probability
`exp(1.5*tanh(H(y)) - 1.5)`. Both methods fit and sample in z=log(y) coordinates,
with known reference `q_Z(z)=exp(sum_j(z_j-exp(z_j)))`. Generated observations
are mapped back by y=exp(z). The generating pairs and H are used only for data
generation and evaluation, not supplied to either fitted estimator.

### Model fitting and sampling

| Setting | Full profile |
| --- | --- |
| Dimension | 7 |
| Sample sizes and pairs | 500, 1000, 1500, 2000; 20 at each size |
| Hidden network layers | 2 layers of 64 Softplus units; beta=5 |
| Penalty values | 0, 0.01, 0.1, 0.3, 1, 3 |
| Selection | 3-fold cross-validation of penalty and training duration |
| Training cap | 2,000 positive-data epochs |
| Optimizer | Adam; initial learning rate 0.001; batch size 256 |
| CINDES selection criterion | Held-out normalized-density negative log-likelihood |
| DSM selection criterion | Held-out denoising loss |
| Diffusion horizon / early cutoff | T=3, delta=0.001 |
| Reverse discretization | 400 exponential-Euler steps |
| CINDES score particles | 1,000 antithetic particles with a reference-score control variate |
| Initialization | Forward-diffused reference distribution at T |
| Generated observations | 5,000 per fitted model, in ten generation shards |
| Independent target-reference sample | 524,288 observations, seed 53000001 |

The learned CINDES logit is `b-softplus(g_theta(z))`, with learned b; there is
no fixed R=25 truncation. DSM uses a residual correction to the known diffused
reference score. Both models are refitted on all observations after selection.
The final reverse step includes noise. There is no terminal denoising correction
or support clipping.

### Evaluation and timing

- TV1: mean histogram TV discrepancy over the seven coordinate marginals.
- TV2: mean histogram TV discrepancy over the six generating coordinate pairs.
- LLgap: signed reference-minus-generated mean unnormalized target log density;
  agreement is assessed by closeness to zero.
- H-TV: histogram TV discrepancy between the generated and reference H values.
- H-W1: one-dimensional empirical Wasserstein-1 discrepancy between those H values.

TV1 and TV2 use 20 Exp(1)-quantile bins per coordinate, including the unbounded
last bin. H-TV uses 20 standard-normal-quantile bins including both infinite tails.
Tables report mean and sample SD over fitted replications, not over generation shards.

`training_and_generation_seconds` includes CV training, CV validation, final
refitting, and generation. It excludes data preparation, serialization, metric
evaluation, and queueing. Hardware and software versions are saved by each run.

## 6. File organization and verification

- `experiment/`: unchanged scientific implementation and reference lookup table.
- `study_config.py`: canonical profiles, task ordering, and seed rules.
- `configuration/`: readable copies of the canonical full configuration and task plan.
- `reproduce.py`: portable orchestration of the original entrypoints.
- `submit_scc_fixed.sh` and `*.qsub`: optional SCC launch infrastructure.
- `reported_results/`: final author-reported aggregate values and tables.
- `provenance/`: original package metadata and historical audit records.
- `verification/`: checks performed while assembling this archive.
- `SHA256SUMS.txt`: checksums for the assembled package.

The saved output manifest freezes scientific source hashes, configuration, and task
identity. Use a new output root after changing scientific settings or source files.
See `REPRODUCIBILITY_NOTES.md` for the source-verification and numerical-reproduction scope.
