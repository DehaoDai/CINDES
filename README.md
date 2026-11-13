# CINDES reproduction code

Code to rerun the CINDES main-paper and reviewer-response experiments. This repository ships **no datasets, results, generated tables, or tests**. New outputs go to ignored local paths. Full neural-network runs require substantial compute.

## Setup

From the repository root (Python 3.11 recommended):

```bash
conda env create -f environment.yml
conda activate cindes
python -m pip install -e .
```

Alternatively use a Python virtual environment and `python -m pip install -e .`. Table 2's LinCDE method additionally needs `python -m pip install -e '.[lincde]'` and the LinCDE R package, which is not bundled. Run full neural experiments inside GPU jobs on SCC. `--quick` checks the code path with reduced settings and is **not** a paper run.

## Main paper — Tables 1–3

The maintained code is in `cindes/` and `experiments/`. Each replication uses `seed + replication_index`; JSON outputs contain the configuration, per-run metric, mean, and sample SD. The following commands use 100 replications and write to ignored `results/main/` paths.

### Table 1: unconditional Gaussian mixtures

```bash
for scenario in spherical elliptical; do
  python -m experiments.table1 --scenario "$scenario" --method all \
    --protocol resampled --replications 100 --device cuda \
    --output-dir results/main/table1
done
python -m experiments.summarize --table 1 \
  --input-dir results/main/table1 --output results/main/table1.csv
```

`--method all` runs CINDES, RFCDE, and MAF on the same generated data. The archived `NNestimation/exp_uncond.py` uses 15,000 generated observations, the first 6,000 for training, and the next 1,500 for validation; this runner follows those sizes. CINDES follows the archived minibatch-resampling protocol. The archived RFCDE/MAF source was found in `NNestimation/models/`, but this maintained runner still uses shared conditional estimators with a constant covariate; treat those two paths as reconstructed baselines. The full MAF schedule is 10,000 epochs with patience 100.

### Table 2: univariate conditional density

```bash
for scenario in nonlinear additive mixture; do
  for n in 500 2000 8000; do
    for method in cindes rfcde maf lincde; do
      python -m experiments.table2 --scenario "$scenario" --sample-size "$n" \
        --method "$method" --data-layout archive --protocol resampled \
        --replications 100 --device cuda --output-dir results/main/table2
    done
  done
done
python -m experiments.summarize --table 2 \
  --input-dir results/main/table2 --output results/main/table2.csv
```

`--data-layout archive` follows the old numerical split, which trains on approximately half the requested `n`. `--data-layout declared` uses the manuscript's literal training size. The archived I(c) mixture implementation also differs from the printed formula. Keep results from these protocols in separate directories.

### Table 3: multivariate conditional density

```bash
for dy in 4 8 12 16; do
  for n in 500 2000 8000; do
    python -m experiments.table3 --response-dimension "$dy" --sample-size "$n" \
      --method all --data-layout archive --replications 100 --device cuda \
      --output-dir results/main/table3
  done
done
python -m experiments.summarize --table 3 \
  --input-dir results/main/table3 --output results/main/table3.csv
```

This runs CINDES, RFCDE, and MAF. The reported box-integrated TV equals `0.5 × 2^dy × mean absolute density error` on `[-1,1]^dy`. The archived layout redraws the weight matrix each replication; `--data-layout declared` fixes it as described in the paper. Label new results by the chosen protocol.

## Main paper — Tables 4–5, Light Chamber

The archived `NNestimation/app_lightchamber.py` and sensor CSV were found locally. Put `uniform_reference.csv` at `data/light_chamber/uniform_reference.csv` (ignored by Git); its source path in `NNestimation` is `dataset/lightchamber/lt_interventions_standard_v1/uniform_reference.csv`. The training runner uses 3,000 sampled rows per replication, standardizes each sensor column, takes 40% of the pre-test pool for training, 10% for validation, and the last 500 rows for testing. It starts with seed 1000 and writes the matrices expected by the aggregation commands below. Full runs are computationally expensive.

```bash
mkdir -p data/light_chamber
# Supply uniform_reference.csv in the path above before running.
for method in baseline cindes rfcde maf lincde; do
  python -m experiments.light_chamber_train --table 4 --method "$method" \
    --data data/light_chamber/uniform_reference.csv --sample-size 3000 \
    --replications 100 --device cuda --output-dir data/light_chamber/table4
done
for method in baseline cindes rfcde maf; do
  python -m experiments.light_chamber_train --table 5 --method "$method" \
    --data data/light_chamber/uniform_reference.csv --sample-size 3000 \
    --replications 100 --device cuda --output-dir data/light_chamber/table5
done
```

The maintained CINDES implementation uses a uniform response reference; the archived Light Chamber script selected a Gaussian reference. Each replication now draws independently from the original CSV; the archived script accidentally resampled its previous subset on later replications. RFCDE uses the archived Gaussian fake-response reference, while LinCDE needs the optional R dependency. New NLLs are rerun measurements, so they may differ from historical table entries.

You can also aggregate existing per-replication NLL matrices. Place them under local `data/` (ignored by Git):

| Table | Required filenames | Input shape and rule |
| --- | --- | --- |
| 4 | `result_BaseLine.csv`, `result_NCDE.csv`, `result_RFCDE.csv`, `result_MAF.csv`, `result_LinCDE.csv` | Each `12 × 100`; apply the 1.5-IQR filter separately by target/method, then mean and sample SD |
| 5 | `result_BaseLine.csv`, `result_CINDES.csv`, `result_RFCDE.csv`, `result_MAF.csv` | Each `4 × 100`; unfiltered mean and sample SD |

```bash
python -m experiments.light_chamber \
  --input-dir data/light_chamber/table4 --output-dir results/main/table4
python -m experiments.light_chamber_multi \
  --input-dir data/light_chamber/table5 --output-dir results/main/table5
```

## Reviewer response — chain and tree MRFs

The original SCC code is in `response/`. Its scripts use local imports, so submit jobs **from that directory**. The job files assume BU SCC `qsub` and a Conda environment named `cindes`. They write ignored JSONL outputs in `response/`.

```bash
cd response
qsub run_chain16.qsub    # Response Table 1: 16D chain, 10 seeds
qsub run_scc_d32.qsub   # Response Table 3: 32D chain, 10 seeds
qsub run_jointtv.qsub   # Table 3 joint TV, 3 seeds
qsub run_tree.qsub      # Response Table 4: 31D tree, 10 seeds
cd ..

python response/summarize.py --table 1 --input response/results.jsonl \
  --output results/response/table1.csv
python response/summarize.py --table 3 --input response/results_d32.jsonl \
  --joint response/jointtv_results.jsonl --output results/response/table3.csv
python response/summarize.py --table 4 --input response/results_tree.jsonl \
  --output results/response/table4.csv
```

The response PDF also contains a second 16D chain table (Table 2), but its distinct run configuration was not recoverable. The 32D joint-TV script runs three seeds although the PDF caption says ten. The tree joint-TV column has no corresponding code here. Reruns of the available scripts are new measurements and are not guaranteed to equal historical PDF entries.

## Quick code check

```bash
python -m experiments.table1 --scenario spherical --method all --quick --replications 1
python -m experiments.table2 --scenario nonlinear --sample-size 500 --method cindes --quick
python -m experiments.table3 --response-dimension 4 --sample-size 500 --method cindes --quick
python -m experiments.light_chamber_train --table 4 --method cindes --quick \
  --sample-size 600 --replications 1 --row 0 \
  --data data/light_chamber/uniform_reference.csv --output-dir results/light_smoke
python response/cindes_vs_dsm_mrf.py --help
```

These commands write only ignored local output; this repository includes no recorded results.
