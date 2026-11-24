# CINDES reproduction code

Code to rerun the CINDES main-paper and reviewer-response experiments. The main-paper runners include the small Light Chamber sensor CSV but no saved results. The reviewer-response studies in [`response/`](response/) include recorded results, tables, and verification checks. New outputs go to ignored local paths. Full neural-network runs require substantial compute.

## Citation

This repository accompanies [CINDES: Classification induced neural density estimator and simulator](https://arxiv.org/abs/2510.00367). If you use this code, please cite our paper:

```bibtex
@misc{dai2025cindes,
  title={CINDES: Classification induced neural density estimator and simulator},
  author={Dehao Dai and Jianqing Fan and Yihong Gu and Debarghya Mukherjee},
  year={2025},
  eprint={2510.00367},
  archivePrefix={arXiv},
  primaryClass={stat.ML},
  url={https://arxiv.org/abs/2510.00367}
}
```

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
    --output-dir results/main/table1 --figure-data-dir results/main/figure1
done
python -m experiments.summarize --table 1 \
  --input-dir results/main/table1 --output results/main/table1.csv
python -m experiments.figure1 --input-dir results/main/figure1 \
  --output results/main/figure1.pdf
```

`--method all` runs CINDES, RFCDE, and MAF on the same generated data. The archived `NNestimation/exp_uncond.py` uses 15,000 generated observations, the first 6,000 for training, and the next 1,500 for validation; this runner follows those sizes. CINDES follows the archived minibatch-resampling protocol. The archived RFCDE/MAF source was found in `NNestimation/models/`, but this maintained runner still uses shared conditional estimators with a constant covariate; treat those two paths as reconstructed baselines. The full MAF schedule is 10,000 epochs with patience 100. Figure 1 uses the first replication (seed 0), plotting the true density and all three estimates on each scenario's 100 × 100 grid. To reproduce only Figure 1, set `--replications 1` in the loop above.

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

The sensor CSV is included at `data/light_chamber/uniform_reference.csv`, copied from `NNestimation/dataset/lightchamber/lt_interventions_standard_v1/uniform_reference.csv`. The training runner uses 3,000 sampled rows per replication, standardizes each sensor column, takes 40% of the pre-test pool for training, 10% for validation, and the last 500 rows for testing. It starts with seed 1000. Full runs are computationally expensive.

```bash
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

After the training runs finish, generate Tables 4 and 5:

```bash
python -m experiments.light_chamber \
  --input-dir data/light_chamber/table4 --output-dir results/main/table4
python -m experiments.light_chamber_multi \
  --input-dir data/light_chamber/table5 --output-dir results/main/table5
```

## Reviewer response — current experiments

See [`response/README.md`](response/README.md) for the experiment map, table-reproduction commands, and the distinctions between the current PDF tables and earlier draft prose. The current studies are:

- [`response/bounded_tree_31/`](response/bounded_tree_31/): bounded 31D binary-tree model, 20 paired replications at each of four sample sizes; saved results and Table 1.
- [`response/gaussian_compositional_7/`](response/gaussian_compositional_7/): unbounded 7D Gaussian-tail compositional model, 20 paired replications at each of four sample sizes; saved results and Tables 2–3.

## Quick code check

```bash
for scenario in spherical elliptical; do
  python -m experiments.table1 --scenario "$scenario" --method all --quick \
    --replications 1 --figure-data-dir results/figure1_quick
done
python -m experiments.figure1 --quick --input-dir results/figure1_quick \
  --output results/figure1_quick.pdf
python -m experiments.table2 --scenario nonlinear --sample-size 500 --method cindes --quick
python -m experiments.table3 --response-dimension 4 --sample-size 500 --method cindes --quick
python -m experiments.light_chamber_train --table 4 --method cindes --quick \
  --sample-size 600 --replications 1 --row 0 \
  --data data/light_chamber/uniform_reference.csv --output-dir results/light_smoke
```

These main-paper quick checks write only ignored local output. The response packages include recorded results.
