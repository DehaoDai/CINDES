# CINDES and DSM: bounded binary-tree experiment

This archive contains the code and saved results for the bounded experiment in dimension 31, with sample sizes 500, 1000, 1500, and 2000. We use 20 replications at each sample size, with seeds 0 through 19. Both methods are fitted to the same dataset within each replication. The complete study contains 80 paired replications and 160 method fits.

You can reproduce the reported tables directly from the included results, or generate new datasets and rerun both methods. The instructions below explain each use. The full simulations use a CUDA GPU; a separate, small CPU check confirms that the installation works.

## 1. Reproduce the reported tables

Unzip the archive and open a terminal in `cindes_bounded_n500_2000_reproducibility`. Run:

```bash
python3 summarize_results.py --root reported_results --profile full
```

This command uses only the Python standard library. It validates the 160 saved method results, prints their summaries, and writes the following files in `reported_results/summary/`:

| File | Contents |
| --- | --- |
| `accuracy_table.tex` | TV1, TV2, and LLgap, as mean plus or minus sample standard deviation |
| `runtime_table.tex` | Combined training and generation time, in seconds |
| `tables.tex` | A standalone LaTeX document containing both tables |
| `summary.csv` | Unrounded means, standard deviations, standard errors, and counts for all metrics |
| `per_run.csv` | One row for each method and replication |
| `summary_all_runs.jsonl` | The complete saved method records |
| `paired_differences.csv` | CINDES-minus-DSM differences within each paired replication |
| `validation_report.json` | Completeness, configuration, and input checks |

The command should report 80 complete pairs, with 20 at each sample size. Standard deviations use denominator 19. The tables use standard deviations, not standard errors, and retain every replication, including the two larger CINDES LLgap values at n=1000.

Both tables use `\small`. Their labels are `tab:cindes_bounded_tree` and `tab:cindes_bounded_runtime`. To include them in a manuscript, load `booktabs` and `amsmath` in the preamble and use:

```latex
\input{reported_results/summary/accuracy_table.tex}
\input{reported_results/summary/runtime_table.tex}
```

Check the archived records against the previously verified summaries with:

```bash
python3 verification/check_reported_results.py
```

## 2. Install the dependencies and run a small check

Model fitting and generation require Python 3.10 or later, NumPy, and PyTorch. The launchers use Unix file locking and support Linux, macOS, and Linux under WSL. A CUDA-enabled PyTorch installation is needed for the full GPU study. On BU SCC, use the environment described in Section 3.

For a new local Python environment, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` specifies the required APIs; it is not an exact software lock file for the original cluster runs. The observed software versions and hardware are described in `docs/ENVIRONMENT.md`.

Run the small CPU check with:

```bash
bash run_local.sh smoke
```

This uses one paired replication with n=64, dimension 7, 20 requested training updates, four reverse steps, 20 CINDES particles, and 64 generated observations per method. Its results are written to `outputs/bounded_n500_2000_smoke_v1/`. They are an installation check and must not be used in the scientific tables.

## 3. Rerun the full study on BU SCC

Upload and extract this directory inside your SCC project, then open a terminal in the extracted directory. Issue the commands below from that directory. Replace `YOUR_SCC_PROJECT` with your project name; the project used for the reported experiments was `dgmgrp`.

Inspect the study and submission plans without starting jobs:

```bash
python3 run_experiment.py --profile full --dry-run
bash submit_scc.sh --project YOUR_SCC_PROJECT --dry-run
```

Submit the full study with:

```bash
bash submit_scc.sh --project YOUR_SCC_PROJECT
```

This submits an array of 80 paired tasks and a dependent summary job. Each task requests one GPU and four CPU slots. At most two array tasks run concurrently. The assignments are:

| Array tasks | Sample size | Seeds |
| --- | ---: | --- |
| 1–20 | 500 | 0–19 |
| 21–40 | 1000 | 0–19 |
| 41–60 | 1500 | 0–19 |
| 61–80 | 2000 | 0–19 |

The default batch setup unloads a conflicting `python3` module, loads `miniconda` and `academic-ml/fall-2026`, and activates `fall-2026-pyt`. These are the module names used in the successful SCC runs. If the available environment changes, put the appropriate activation commands in a shell file and pass its path:

```bash
bash submit_scc.sh --project YOUR_SCC_PROJECT --env-script /absolute/path/activate_environment.sh
```

The GPU check runs in a separate process that exits before fitting starts. The launcher retains the fix used in the successful SCC study. Array ranges and concurrency are written into the submitted job script, preserving compatibility with the SCC `qsub` behavior encountered during these experiments.

To run one full-setting pair first, use the pilot command in place of the full-submission command above:

```bash
bash submit_scc.sh --project YOUR_SCC_PROJECT --pilot
```

The pilot uses n=500 and seed 0 with the full scientific settings. After its job has left the queue, check that the result-status command below reports one complete paired replication. Then submit the remaining study with:

```bash
bash submit_scc.sh --project YOUR_SCC_PROJECT --resume
```

The launcher validates and reuses complete results. It checks that previously recorded jobs have left the scheduler before accepting a resumed submission. An interrupted method fit restarts from the beginning; completed, valid method results are retained.

Check scheduler and result status with:

```bash
qstat -u "$USER"
python3 summarize_results.py --root outputs/bounded_n500_2000_v1 --profile full --status
```

The full study is complete when the validator reports:

```text
Validated 80 of 80 complete paired replications (160 of 160 method results).
  n=500: 20 complete pairs
  n=1000: 20 complete pairs
  n=1500: 20 complete pairs
  n=2000: 20 complete pairs
```

The dependent summary job creates `outputs/bounded_n500_2000_v1/summary/results_for_tables.zip`. Regenerate this archive with:

```bash
python3 make_results_bundle.py --root outputs/bounded_n500_2000_v1 --profile full
```

New simulations are written under `outputs/`; the included historical records remain in `reported_results/`. Use a different `--root` when starting an independent rerun.

## 4. Run on another CUDA machine

From the package directory, with NumPy and a compatible CUDA PyTorch build installed, run:

```bash
bash run_local.sh full
```

This executes all 80 pairs sequentially on the available CUDA GPU and builds the summaries and results archive. To run one specified pair, use:

```bash
python3 run_experiment.py --profile full --n 2000 --seed 0 --require-cuda
```

`run_experiment.py` is the entry point for this study. The scientific files in `experiment/` are preserved unchanged, including their older standalone defaults; the wrapper supplies the four sample sizes and fixed settings explicitly.

## 5. Fixed scientific settings

| Setting | Value |
| --- | --- |
| Dimension and support | 31; the unit cube |
| Graph | Balanced binary tree |
| Potential seed, coupling, unary scale | 11, 0.55, 0.3 |
| Grid cells per coordinate | 200 |
| Training sample sizes | 500, 1000, 1500, 2000 |
| Replications per sample size | 20, with seeds 0–19 |
| Networks | Three hidden layers of 128 ReLU units |
| CINDES logit truncation R | 25 |
| CINDES particles K per score evaluation | 5000 |
| Reverse horizon T, cutoff delta, steps M | 3, 0.001, 200 |
| Requested optimization updates | 20,000 for CINDES; 60,000 for DSM |
| Training batch size | 256 |
| Optimizer | Adam; learning rate 0.001; weight decay 0.00001; cosine schedule |
| Added L1 penalty or cross-validation | None |
| Generated observations per method | 10,000 |
| Evaluation reference observations | 20,000, with seed 12345 |
| Generation batch size | 64 |
| Additional terminal correction | None |

Training budgets are rounded to whole epochs. The actual CINDES update counts are 19,998, 20,004, 19,998, and 19,992 for the four sample sizes, respectively. DSM uses 60,000 updates at each sample size.

Data are sampled from the midpoint-grid approximation of the bounded tree density, with uniform sampling inside selected cells. Both estimators use fully connected networks and are not supplied with the tree structure. The ordinary last reverse step is deterministic, and the final generated coordinates are projected onto the unit cube. Omitting the additional terminal correction does not remove that support projection. Details appear in `docs/IMPLEMENTATION.md`.

TV1 averages histogram total variation discrepancies over the 31 coordinate marginals. TV2 averages them over the 30 parent–child marginals. LLgap is the reference-minus-generated mean target log-potential; it is signed and is not a KL divergence. Runtime includes training and generation, excluding data simulation and metric evaluation. Summaries also retain energy distance, sliced Wasserstein distance, the boundary-near fraction, and parameter and update counts.

## 6. Source files and reproducibility checks

The three files under `experiment/` implement the DGP, estimation, generation, and evaluation. Their SHA-256 digests match those recorded in the successful small-sample runs. The wrappers manage execution and outputs without altering the scientific procedure.

The included n=500,1000,1500 records come from the successful corrected SCC runs. The n=2000 records come from the earlier bounded study. Their recorded scientific settings agree. The older n=2000 archive lacks execution-time source hashes and per-run environment records; this distinction is retained in `reported_results/provenance.json`.

Fresh runs save method results, environment and source metadata, completion status, logs, and NumPy sample files. Each sample file contains the full generated sample and an 8000-observation subset of the evaluation reference; metric calculations use the full 20,000-observation reference. The included historical results do not contain generated arrays or trained networks, so rebuilding their tables recomputes the aggregation of saved metrics.

Before release, the source and settings were checked, all 160 included method results were validated, the submission scripts were exercised with simulated scheduler commands, and both methods completed the small CPU check. Verification reports distinguish these checks from a full GPU rerun. The full 80-pair study was not rerun while preparing this archive.

Fixed seeds do not guarantee bitwise agreement across different PyTorch releases or CPU/GPU hardware. GPU models varied across replications, so runtimes depend on the execution environment. That environment is recorded for each fresh run. See `docs/ENVIRONMENT.md` and `verification/README.md` for the recorded versions and repeatable checks.
