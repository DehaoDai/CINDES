# CINDES and denoising score matching: an experiment with unbounded observations

This package reproduces the accuracy and computation-time tables for the seven-dimensional compositional experiment. We compare CINDES with denoising score matching (DSM) at sample sizes 500, 1000, 1500, and 2000, using 20 independent paired replications at each sample size. Both methods use the same observations within a replication.

The package provides two ways to reproduce the results. The first rebuilds the reported tables from the saved results of the 80 completed replications. The second generates new observations, fits both methods, generates samples, and evaluates the resulting distributions. All observations are simulated by the supplied code; no external dataset is needed.

## 1. Reproduce the reported tables

Unzip the archive, open a terminal in the resulting directory, and run:

```bash
python3 reproduce_tables.py --check
```

This command uses only the Python standard library. It does not require PyTorch, a GPU, or access to a computing cluster. It checks that the supplied records contain all 80 paired replications, recalculates the means and sample standard deviations from the individual results, and compares the output with the reported values.

The tables are written to `tables/`:

| File | Contents |
| --- | --- |
| `accuracy_table.tex` | TV1, TV2, LLgap, H-TV, and H-Wasserstein-1 |
| `runtime_table.tex` | Final refitting, sample generation, their sum, and cumulative cross-validation fitting time |
| `tables.tex` | A standalone LaTeX document containing both tables |
| `accuracy_summary.csv` | Numerical accuracy summaries |
| `runtime_summary.csv` | Numerical timing summaries |
| `validation_report.json` | Results of the numerical checks |

Each entry is the mean plus or minus the sample standard deviation across the 20 replications at the corresponding sample size. The sample standard deviation uses denominator 19. The table labels are `tab:cindes_accuracy_full` and `tab:cindes_runtime_full`, matching the references in the response report. Bold entries identify the reported mean comparisons; they do not indicate a significance test.

To compile the standalone document, use a LaTeX installation with the packages listed in `tables.tex`:

```bash
cd tables
pdflatex tables.tex
pdflatex tables.tex
cd ..
```

The two compilations allow LaTeX to resolve the table references. LaTeX is optional: the CSV files and the table fragments are produced directly by Python.

### What this first command reproduces

The supplied `reported_results/` directory contains individual method outcomes and timing records from the completed study, together with the configuration and provenance needed to interpret them. The first command therefore reproduces the aggregation used in the tables. It does not refit the networks or recalculate the metrics from the original generated observations.

The compact results do not contain the original observation arrays, fitted-network checkpoints, or all individual cross-validation worker files. The complete experiment below recreates these objects and saves them in a new output directory. In particular, the supplied cross-validation timing is the recorded sum for each method and replication, rather than a collection of the 15 separate worker timings.

## 2. Install the numerical dependencies

For a new simulation, use Linux, Python, NumPy, SciPy, and PyTorch. The experiment uses Unix file locking, so it should be run under Linux or a Linux environment such as WSL rather than native Windows. The small implementation check can run on a CPU. The full experiment uses a CUDA-capable NVIDIA GPU for sample generation.

If you are using BU SCC, the submission scripts activate the software environment described in Section 5. You do not need a separate local installation to submit those jobs.

On another Linux machine, first create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build appropriate for your GPU and driver using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For a CPU-only implementation check, a CPU build of PyTorch is sufficient:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy scipy
```

Before starting the full experiment, check the active environment:

```bash
python -c "import numpy, scipy, torch; print('NumPy:', numpy.__version__); print('SciPy:', scipy.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The full experiment requires `CUDA available: True` on the machine or compute node that will run it. The software versions recorded in the original study and in the package checks are given in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). The generic dependency file does not install the original SCC environment or guarantee identical floating-point results across different installations.

## 3. Run a small implementation check

From the package directory, run:

```bash
bash experiment/run_local.sh smoke outputs/smoke_test
```

This command runs the complete sequence at reduced settings: reference generation, data generation, cross-validation, refitting, sample generation, evaluation, and summary creation. It uses one replication with 500 observations, two candidate penalties, at most two training epochs, 20 CINDES particles, four reverse steps, and 64 generated observations per method. These settings are for checking execution and do not reproduce the scientific tables.

The command prints its progress and saves a log under `outputs/smoke_test/driver_logs/`. A successful run ends with `Completed smoke profile`. Keep this directory separate from the full experiment.

The following command checks the sampler arithmetic and restart logic without fitting the full study:

```bash
python3 experiment/qa_checks.py --static --sampler --runtime
```

The original Python verification file is retained unchanged. Its historical `--submission` option checks the original project-specific submission arguments. To check the journal submission wrapper using a local replacement for the scheduler, run `python3 verification/journal_audit.py --mock-scheduler`. This check does not submit any cluster jobs.

## 4. Run the full experiment on one GPU

On a Linux workstation or an allocated compute node with a CUDA GPU, run:

```bash
bash experiment/run_local.sh full outputs/new_full_study
```

This command runs all 80 paired replications sequentially, using the same scientific configuration and seeds as the reported experiment. Cross-validation runs on CPUs with up to four workers, and sample generation runs on the GPU. The script checks that CUDA is available before starting the study.

The full experiment retains the observations, cross-validation histories, fitted networks, generated samples, and evaluation records. Allow several gigabytes of free disk space. Execution time depends on the CPU, GPU, and available resources.

To use a particular Python interpreter or a different number of CPU workers, set the corresponding variables before running the command:

```bash
export PYTHON_BIN=python3
export CINDES_CV_WORKERS=4
```

The number of workers affects scheduling, not the prescribed sample sizes, seeds, or training budgets. Keep the scientific Python files unchanged while an experiment is running: the program records their hashes and refuses to combine results from different source versions.

After completion, the full output directory contains the original experiment summaries under `summary/` and the two journal tables under `journal_tables/`.

## 5. Run the full experiment on BU SCC

Upload and unzip the package in your SCC project directory. Open that directory in your VS Code terminal connected to SCC. Set your project name and a new output directory:

```bash
export SCC_PROJECT=YOUR_SCC_PROJECT
export CINDES_OUTPUT_ROOT="$PWD/outputs/full_study_v1"
bash experiment/submit_scc.sh --dry-run
```

Replace `YOUR_SCC_PROJECT` with the name of a project you are authorized to use. The dry run prints the submission commands without submitting jobs. To submit the study, run:

```bash
bash experiment/submit_scc.sh
```

The driver submits three jobs in order: a CPU job to prepare the shared reference sample, an array of 80 paired experiment tasks, and a CPU job to validate the completed study and build the summaries. Each array task fits and evaluates both CINDES and DSM for one dataset.

The array requests four CPU slots and one GPU per task, with at most two GPU tasks running at once. Its GPU request requires compute capability at least 8.0 and at least 16 GB of GPU memory. The 12-hour limit applies to each array task, not to the entire study. Queue availability determines when the tasks can start.

The supplied `experiment/scc_env.sh` uses the environment recorded for the original SCC run:

```bash
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
```

The script first unloads a conflicting `python3` module if it is present. If this named environment is unavailable, adapt the activation commands in `scc_env.sh` to an installed environment containing the required numerical packages and CUDA-enabled PyTorch. The scientific Python source files can remain unchanged.

Once the jobs have been submitted successfully, they continue independently of the VS Code terminal. Check their scheduler status with:

```bash
qstat -u "$USER"
```

Check how many paired replications have completed with:

```bash
python3 experiment/summarize_results.py --root outputs/full_study_v1 --profile full --status
```

This status command uses only the standard library. The reference job must first have created the study manifest. A job disappearing from the scheduler does not by itself establish successful completion; the summary checks the saved experiment records.

The submission scripts follow the [SCC guidance on array jobs and dependencies](https://www.bu.edu/tech/support/research/system-usage/running-jobs/advanced-batch/) and [GPU allocations](https://www.bu.edu/tech/support/research/software-and-programming/gpu-computing/).

## 6. Rebuild tables after a new full run

The full-run drivers create the tables automatically. They can also be rebuilt from the package directory:

```bash
python3 reproduce_tables.py --root outputs/new_full_study --output outputs/new_full_study/journal_tables
```

For the SCC example, replace `outputs/new_full_study` with `outputs/full_study_v1`. The table builder requires all four sample sizes and 20 complete paired replications at each sample size. It does not silently treat an incomplete study as the reported experiment.

Use `--check` with the supplied records to verify the published values. A fresh full run on a different numerical environment can give different fitted models and generated samples, so its tables are rebuilt without imposing equality to the historical results.

To rerun the experiment's more extensive integrity checks and original summary exporter, use the numerical environment on a compute node:

```bash
python3 experiment/summarize_results.py --root outputs/new_full_study --profile full
```

This exporter checks the saved artifacts as well as the completion records. It requires the complete output of a rerun, including the arrays and checkpoints, rather than the compact supplied results.

## 7. Experimental design and reported measures

The target density is a bounded nonlinear tilt of a seven-dimensional standard Gaussian density. It has full support on the real coordinate space and sub-Gaussian tails. The scalar function H combines six signed products of transformed coordinates; an outer hyperbolic tangent introduces the compositional dependence. The exact density, component functions, and sampling method are defined in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

Both methods use fully connected networks with two hidden layers of 64 Softplus units. For each method, three-fold cross-validation selects the normalized L1 penalty and the number of training epochs, allowing at most 2000 epochs. The selected model is then refitted using all observations. Neither estimator is given H, its generating coordinate pairs, or the analytic target density during fitting or model selection.

CINDES uses a fixed standard Gaussian reference distribution, classifier-logit bound R = 25, and 5000 antithetic particles per score evaluation. Both methods use T = 3, delta = 0.001, and 200 reverse steps, with no additional terminal sampling, final denoising, or support clipping. Each fitted method produces 10000 observations.

The five accuracy measures are TV1, TV2, LLgap, H-TV, and H-Wasserstein-1. TV1 averages histogram discrepancies over the seven coordinates. TV2 averages them over the six pairs entering H. H-TV and H-Wasserstein-1 compare the generated and reference distributions of H. LLgap is a signed difference of mean target log-potentials, not a divergence. All recorded LLgaps in the reported study are positive. Definitions and implementation details are given in `docs/IMPLEMENTATION.md`.

The common evaluation sample contains 1048576 independent target observations. It is used only for evaluation. The procedure also records target-versus-target calibration results separately; these are not subtracted from the method results.

## 8. How to read the computation times

Final refit is the recorded elapsed time for fitting the selected model to all observations on a CPU. Generation is the recorded elapsed time for producing 10000 samples on the GPU. The mean and standard deviation of their sum are computed from the sum within each replication.

The cross-validation column adds the recorded times of the 15 candidate fits for each method. Up to four fits were executed simultaneously on CPUs, so this sum describes cumulative fitting time rather than the elapsed duration of the parallel stage. The fits for both methods were scheduled together, and the total elapsed cross-validation stage is recorded separately.

Both methods generated samples on the same allocated GPU within each replication. The GPU hardware varied across replications in the original study. The supplied environment records document these allocations. Runtime comparisons therefore describe the recorded experiment and will change on other hardware.

## 9. Saved outputs and resuming an interrupted run

A full output directory contains `study_manifest.json`, `task_manifest.csv`, `source_snapshot/`, `reference/`, `runs/`, and `summary/`. Each run retains its observations and folds, selected tuning parameters, trained models, generated samples, timings, metrics, and completion checks. Driver and worker logs show where an interrupted task stopped.

To resume, use the same driver command and output directory. Completed stages are checked and reused. An incomplete fit restarts from its fixed seed; sample generation resumes from complete shards when the numerical environment is compatible. If an unfinished run moves to a different GPU type or numerical environment, the generation stage is restarted for both methods using the saved fitted models. Corrupted or inconsistent records cause an error rather than being silently combined.

The reported study fixes all seeds, uses deterministic PyTorch algorithms, and records its numerical environment. Exact equality across different PyTorch versions, devices, or platforms is not guaranteed; see the [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html). The supplied recorded outcomes allow the reported table aggregation to be reproduced independently of these differences.

## 10. Package contents and verification

| Location | Purpose |
| --- | --- |
| `README.md` | Execution instructions |
| `reproduce_tables.py` | Rebuild the two tables using only the standard library |
| `reported_results/` | Individual recorded outcomes, configuration, and provenance |
| `expected_tables/` | Reference values used to verify the tables |
| `experiment/` | Original scientific Python code and execution scripts |
| `docs/IMPLEMENTATION.md` | Mathematical model and implementation details |
| `docs/ENVIRONMENT.md` | Recorded and locally checked software environments |
| `verification/` | Independent numerical and execution checks |

The 14 Python source files used by the experiment are preserved byte for byte. Their hashes match those recorded in the original SCC study. The README, journal table builder, and execution wrappers provide the additional instructions and exports needed for this submission; they do not change the statistical procedure.

Run the independent package audit with:

```bash
python3 verification/journal_audit.py
```

The verification report distinguishes checks of the supplied completed study from the reduced CPU execution check. The original full study was run on SCC; preparing this archive does not rerun all 80 GPU tasks.
