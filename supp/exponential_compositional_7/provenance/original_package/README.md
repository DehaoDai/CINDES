# CINDES and DSM: exponential-tail experiments on BU SCC

This package runs new paired experiments for the centered exponential example.
It uses the training procedure from the pilot and the sampler selected by the
subsequent numerical check: forward-noised reference initialization, 400 reverse
steps, and 1,000 Monte Carlo particles for each CINDES score evaluation.

The default study uses dimension seven, sample size 1,000, and 20 independent
datasets. Both methods receive the same dataset in each replication. An optional
full study uses sample sizes 500, 1,000, 1,500, and 2,000, with 20 replications at
each size. The package fits new models; it does not reuse the pilot checkpoints.

## 1. Upload and submit from your VS Code SCC terminal

Upload `CINDES_exponential_SCC_overnight.zip` into:

```text
/projectnb/dgmgrp/CINDES_Experiments_DM/
```

In the terminal connected to SCC, run:

```bash
cd /projectnb/dgmgrp/CINDES_Experiments_DM
unzip CINDES_exponential_SCC_overnight.zip
cd cindes_exponential_scc
```

Use the Python environment that worked in the earlier experiments. Unloading
`python3` first avoids its conflict with the `miniconda` module:

```bash
if module is-loaded python3 >/dev/null 2>&1; then
    module unload python3
fi
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
```

Choose one of these submission commands.

**Twenty paired replications at n=1,000:**

```bash
bash submit_scc.sh
```

**Twenty paired replications at each of n=500, 1,000, 1,500, and 2,000:**

```bash
bash submit_scc.sh --profile full
```

The submission script uses project `dgmgrp`. It submits a CPU job to prepare
the independent evaluation sample, a dependent GPU array, and a final CPU job
to validate and summarize the results. It prints and saves all three job IDs.
Each array task fits and samples both methods sequentially on its assigned GPU.
At most two GPU tasks run simultaneously; the remaining tasks wait in the queue.
The experiment continues after you disconnect from VS Code.

Each GPU task requests four CPU slots, at least 16 GB of GPU memory, GPU compute
capability at least 8.0, and a 12-hour time limit. This limit is per task, not the
elapsed duration of the whole array. Queue availability and GPU type determine
when the full study finishes, so an overnight completion time is not guaranteed.
The package includes the fixed reference-score interpolation table; it does not
download data or install software while the array runs.

To inspect the submission commands without submitting jobs:

```bash
bash submit_scc.sh --dry-run
```

Use `--profile full --dry-run` for the full study. The helper passes `-t` and its
range as separate arguments to `qsub`, using the Grid Engine array syntax.

## 2. Check progress

For scheduler status:

```bash
qstat -u "$USER"
```

For the default experiment's completed and missing replications:

```bash
bash submit_scc.sh --status
```

For the full study:

```bash
bash submit_scc.sh --profile full --status
```

The status command reads saved records and does not train a model. Logs for
individual array tasks are in the corresponding output directory's `logs/`
folder. The exact stage within a replication is also recorded in its
`progress.json` file.

The output roots are:

| Profile | Paired replications | Output directory |
| --- | ---: | --- |
| `n1000` (default) | 20 | `outputs/exponential_n1000_v1` |
| `full` | 80 | `outputs/exponential_full_v1` |
| `smoke` (execution check only) | 1 small test | `outputs/exponential_smoke_v1` |

## 3. Print and download the tables

The final scheduled job normally creates the summary automatically. You can
print it again without refitting or generating any observations:

```bash
python3 summarize_results.py \
    --root outputs/exponential_n1000_v1 \
    --profile n1000
```

For the full study:

```bash
python3 summarize_results.py \
    --root outputs/exponential_full_v1 \
    --profile full
```

This checks that every expected replication has both methods and that the
saved data, models, generated samples, and metric records have not changed.
The default summary requires a complete study. It does not silently discard
failed or missing replications. Summary calculations use only Python's standard
library, so they do not need a GPU or a PyTorch import.

The `summary/` folder contains the accuracy and runtime tables in CSV and LaTeX,
the full-precision per-run records, paired method differences, and the completion
status. Every accuracy metric is summarized by its mean, sample standard
deviation (`ddof=1`), and standard error. The LaTeX tables report mean plus or
minus SD. LLgap is reported as a signed diagnostic and is not automatically
ranked by its sign.

Create or refresh the file to upload for table preparation:

```bash
python3 make_return_bundle.py \
    --root outputs/exponential_n1000_v1 \
    --profile n1000
ls -lh outputs/exponential_n1000_v1/summary/results_for_tables.zip
```

For the full study, replace the root with `outputs/exponential_full_v1` and the
profile with `full`. Download the resulting `summary/results_for_tables.zip`
through VS Code or SCC OnDemand. It contains the tables, all per-run metric
records, CV histories and selections, timing/environment records, logs, and
source information needed to check and reconstruct the tables.

The complete raw output remains on SCC. The lightweight table bundle excludes
large NumPy arrays and PyTorch checkpoints by default. To include those too:

```bash
python3 make_return_bundle.py \
    --root outputs/exponential_n1000_v1 \
    --profile n1000 \
    --include-samples --include-checkpoints
```

The bundle includes a small standalone analysis copy. After extracting it,
its `summary/export.json` can regenerate the tables without access to SCC:

```bash
python3 analysis_code/summarize_results.py \
    --from-export summary/export.json \
    --output regenerated_tables
```

## 4. Resume an interrupted study

Wait until the earlier jobs for this study have left the queue, then run:

```bash
bash submit_scc.sh --resume
```

For the full study:

```bash
bash submit_scc.sh --profile full --resume
```

The submission script resubmits the array. Each task first checks whether its
complete result can be reused. For a partial task, it reuses completed CV fits,
the final refit, and completed generation chunks after validating their saved
identities and hashes. An unfinished fit or sampling chunk is preserved in an
`interrupted/` directory and restarted. Training is checkpointed at the level
of a completed CV fit or refit, not at each optimizer update.

Keep the package and output folders in place while jobs are running or being
resumed. Do not edit the scientific code or settings within an existing study.
An incompatible source, profile, environment, or completed artifact produces an
error instead of mixing results. Use a new `--root` for a different study.
Completed tasks can be inspected without a GPU. Partial tasks require the same
CPU/GPU mode, thread count, and NumPy/SciPy/PyTorch versions as before.

If a submission fails after one or two jobs have been accepted, the accepted
job IDs remain in `submissions/`. Inspect that record and `qstat` before retrying.
The script does not automatically cancel accepted jobs. If a full array ends
with missing results, its summary job saves clearly labelled partial diagnostics
and returns a nonzero status. Use `--allow-partial` explicitly for an interim
summary; it is not a replacement for completing the 20 paired replications.

## 5. Scientific protocol

### Generating distribution

For positive observations in dimension seven, the reference is

\[
q_Y(y)=\exp\left(-\sum_{j=1}^7y_j\right),\qquad y_j>0.
\]

Let \(\psi(y)=1-2e^{-y}\) and

\[
H(y)=\frac{3}{\sqrt6}\sum_{(j,k)\in E}s_{jk}\psi(y_j)\psi(y_k),
\qquad
p_Y(y)=Z^{-1}q_Y(y)\exp\{1.5\tanh H(y)\}.
\]

The generating pairs are `(3,1), (3,4), (1,5), (1,2), (4,0), (4,6)` in zero-based
indices, with signs `-1,-1,-1,-1,+1,+1`. These were fixed using potential seed 11.
The bounded tilt preserves exponential tails. The outer tanh makes this a
compositional example; the pairs do not imply a tree Markov property.

Training and reference observations are drawn independently by rejection
sampling: propose from the product Exp(1) law and accept with probability
`exp(1.5*tanh(H(y))-1.5)`. There is no support gridding or MCMC burn-in.
For each sample size, replication labels 4000 through 4019 identify fresh
datasets. The actual data/fitting seed is `100000*n + replication_seed`, and
the fold seed is that number plus 700000. This separates the datasets across
sample sizes while keeping them shared by the two methods within a replication.

### Fitting

Both methods train in log coordinates \(Z=\log Y\), using the same known reference
\(q_Z(z)=\exp\{\sum_j(z_j-e^{z_j})\}\). They use fully connected networks with
two hidden layers of 64 Softplus units. CINDES has a scalar output of the form
\(b-\operatorname{softplus}(g_\theta(z))\); DSM adds a seven-dimensional learned
correction to the known reference score. The estimators receive neither the
generating pairs nor the signs, component function, or target score.

Each method separately selects its L1 penalty and training duration by
three-fold cross-validation under a common cap of 2,000 positive-data epochs.
The penalty grid is `0, 0.01, 0.1, 0.3, 1, 3`. The penalty averages absolute
Linear-layer weights and excludes biases. CINDES uses held-out normalized
density negative log likelihood, with 65,536 reference draws per fold for the
normalizer. DSM uses held-out, unpenalized denoising loss with eight fixed
noise/time draws per observation. Each selected model is then refitted using
all observations. Validation is checked at 100 common fractions of the training
cap; the refit retains the selected prefix of the same cosine learning-rate
schedule. Adam uses initial learning rate 0.001, batch size 256, and zero weight
decay. There is no fixed logit truncation R and no added tail correction.

### Sampling

Both reverse processes start from the known forward-noised reference:

\[
Z_T=e^{-T}\log E+\sqrt{1-e^{-2T}}\,\xi,
\quad E_j\stackrel{\mathrm{iid}}{\sim}\operatorname{Exp}(1),
\quad \xi\sim N(0,I_7),
\]

with the two draws independent. This is \(q_T\), not the unknown target terminal
law. The samplers use \(T=3\), \(\delta=0.001\), 400 exponential-Euler steps, and
5,000 generated observations. The final reverse step includes noise. There is
no terminal posterior-sampling, denoising, or support-clipping correction.
Generated values are transformed back by \(Y=\exp Z\).

CINDES uses 1,000 antithetic particles for each score evaluation and the same
reference-score control variate as in the pilot. The known reference posterior
is evaluated by 128-node adaptive Gauss--Legendre quadrature. DSM uses the same
reference score through the supplied cubic interpolation table, with quadrature
fallback outside the table. These are numerical approximations. The table is
evaluated on the CPU even when the learned network is on the GPU; the associated
transfers remain part of the measured computation.

Generation is split into ten chunks of 500 observations. Both methods share
initialization and reverse-noise streams within each chunk. The 400-step sampler
retains the audited OU bridge construction from the diagnostic; its marginal
noise law is unchanged. CINDES's particle streams are recorded separately.
Target metrics are evaluated only after both methods finish fitting and sampling.

### Metrics and computation time

A separate sample of 524,288 target observations, generated with seed 53000001,
is shared by all evaluation runs. It is never used for model selection.

- TV1 averages histogram TV over the seven coordinate marginals.
- TV2 averages two-dimensional histogram TV over the six generating pairs.
  Both use 20 Exp(1)-quantile bins per coordinate, including the infinite tail.
- LLgap is the reference mean unnormalized target log density minus its mean
  over generated observations. The common normalizing constant cancels. This
  is signed; a lower value is not automatically a better fit.
- H-TV compares the generated and reference histograms of H(Y), using 20 fixed
  standard-normal-quantile bins with infinite end bins.
- H-W1 is the one-dimensional empirical Wasserstein-1 distance between those
  two H(Y) samples. The H metrics assess this specified dependence summary.

The outputs separate CV training, CV validation, refit training, generation,
serialization, and metric-evaluation durations. The main training-and-generation
total includes CV training and validation, refit training, and generation. It
excludes serialization, evaluation, queue time, and other setup costs. Task and
attempt elapsed times are saved separately. Completed-stage timings omit work
lost in an interrupted unfinished stage, whose logs and attempts are retained.

Within an uninterrupted replication, both methods run sequentially on the same
allocated GPU; the first method alternates by seed parity. GPU models may vary
across replications and after resubmission. Each stage records its environment
so this variation can be identified when interpreting runtimes.

## 6. Local execution check

Use Python 3.10 or newer with NumPy, SciPy, and PyTorch. The requirements file
lists minimum dependencies; the output records the actual versions used.

```bash
python prepare_reference.py --root outputs/local_smoke --profile smoke
python run_experiment.py --root outputs/local_smoke --profile smoke \
    --task-id 1 --device cpu --threads 1
python summarize_results.py --root outputs/local_smoke --profile smoke
python make_return_bundle.py --root outputs/local_smoke --profile smoke
```

The smoke profile has one tiny dataset, two epochs, and reduced Monte Carlo
budgets. It checks execution only and is explicitly labelled in its tables.
It retains all 400 reverse steps to exercise the selected sampler. Do not use
these test values as scientific results. A CUDA preflight in each actual SCC
task checks a matrix operation, reference quadrature, posterior sampling, and
reference-score interpolation before fitting begins. This package was prepared
without submitting jobs to your SCC account.

## 7. Files and provenance

`experiment/` contains unchanged copies of the pilot's scientific implementation
and the selected sampler. `protocol_origin.json` records their hashes and the
source archive hashes. The study saves a source snapshot and a frozen identity
before generating data. An existing output root rejects changed scientific
code or configuration. The `verification/` folder records the package checks;
the diagnostic's sampler decision is included for provenance.

Raw outputs are organized by sample size and replication seed:

```text
outputs/exponential_n1000_v1/
    manifest.json
    study_config.json
    task_plan.json
    source_snapshot/
    reference/
    logs/
    submissions/
    runs/n_1000/seed_4000/
        data.npz
        environment/
        attempts/
        methods/cindes/
        methods/dsm/
        result.json
    summary/
```

The SCC scripts follow BU's documentation for
[batch submission](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/),
[arrays and dependencies](https://www.bu.edu/tech/support/research/system-usage/running-jobs/advanced-batch/),
and [GPU jobs](https://www.bu.edu/tech/support/research/software-and-programming/gpu-computing/).
They preserve SCC's assigned `CUDA_VISIBLE_DEVICES` value. The environment setup
uses the `miniconda`, `academic-ml/fall-2026`, and `fall-2026-pyt` configuration
that worked in your preceding SCC runs. For a different environment, supply an
activation script through `bash submit_scc.sh --env-script /absolute/path/env.sh`.
