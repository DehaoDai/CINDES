# Supplementary reviewer-response experiments

The three studies compare CINDES and DSM on paired simulated datasets. Each package retains its own scientific code, configuration, reported results, and verification material in the original relative layout. Run commands from the relevant package directory. The table numbers below follow the latest reviewer-response PDF.

| Study | PDF table | Design | Saved results |
| --- | --- | --- | --- |
| [`bounded_tree_31/`](bounded_tree_31/) | Table 1 | Bounded 31D binary tree, maximal clique size 2; `n = 500, 1000, 1500, 2000`, 20 paired replications per size | [Accuracy table](bounded_tree_31/reported_results/summary/accuracy_table.tex), [runtime table](bounded_tree_31/reported_results/summary/runtime_table.tex), and individual run records |
| [`gaussian_compositional_7/`](gaussian_compositional_7/) | Table 2 | Unbounded 7D compositional density with sub-Gaussian tails; same four sizes and 20 paired replications | [Accuracy table](gaussian_compositional_7/tables/accuracy_table.tex), [runtime table](gaussian_compositional_7/tables/runtime_table.tex), and individual run records |
| [`exponential_compositional_7/`](exponential_compositional_7/) | Table 3 | Positive 7D compositional density with exponential tails; same four sizes and 20 paired replications | [Accuracy and runtime tables](exponential_compositional_7/reported_results/reported_tables.tex) and aggregate statistics |

The bounded study generates 10,000 observations per fitted model and evaluates coordinate and tree-edge histogram marginals. The sub-Gaussian study generates 10,000 observations and also evaluates the compositional statistic $H$. The exponential study operates in log coordinates, uses 1,000 score particles and 400 reverse steps, and generates 5,000 observations per fitted model. At `n = 2000` its mean TV1 is slightly lower for DSM, while CINDES has lower mean TV2, H-TV, and H-W1. The reported standard deviations are across 20 replications.

## Verify the supplied records and tables

From the repository root:

```bash
cd supp/bounded_tree_31
mkdir -p outputs
cp -R reported_results outputs/table_check
python3 summarize_results.py --root outputs/table_check --profile full
python3 verification/check_reported_results.py

cd ../gaussian_compositional_7
python3 reproduce_tables.py --check --output outputs/table_check
python3 verification/journal_audit.py

cd ../exponential_compositional_7
shasum -a 256 -c SHA256SUMS.txt
python3 reproduce.py --profile full --device cuda --threads 4 --dry-run
```

The first two packages contain per-replication records, so their saved tables can be recalculated without model fitting. The exponential package contains final aggregate values and verified SCC source, configuration, and task metadata, but **not** the 80 individual final-run outcomes. Its reported statistics cannot be independently recomputed from the supplied archive. The dry run inspects its full task plan without starting the experiment. Each package's own README explains how to perform a new scientific run; write new outputs under its ignored `outputs/` directory. The exponential package README preserves its original archive name in its initial `cd` example; inside this repository, use `cd supp/exponential_compositional_7` instead.
