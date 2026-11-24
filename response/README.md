# Reviewer-response experiments

The two current studies are self-contained reproducibility packages. Their scientific source, saved per-replication records, generated tables, provenance, and verification files are kept in their original relative layout. Run each package's commands from its own directory.

| Study | PDF location | Training sizes and repetitions | Reported outputs |
| --- | --- | --- | --- |
| [`bounded_tree_31/`](bounded_tree_31/) | Table 1 and bounded-domain discussion | `n = 500, 1000, 1500, 2000`; 20 paired replications each | [`accuracy_table.tex`](bounded_tree_31/reported_results/summary/accuracy_table.tex), [`runtime_table.tex`](bounded_tree_31/reported_results/summary/runtime_table.tex) |
| [`gaussian_compositional_7/`](gaussian_compositional_7/) | Tables 2–3 and unbounded light-tailed discussion | `n = 500, 1000, 1500, 2000`; 20 paired replications each | [`runtime_table.tex`](gaussian_compositional_7/tables/runtime_table.tex), [`accuracy_table.tex`](gaussian_compositional_7/tables/accuracy_table.tex) |

Both studies compare CINDES with DSM on the same dataset within each replication. The bounded tree has ambient dimension 31 and maximum clique size 2; TV1 and TV2 summarize coordinate and tree-edge histogram marginals. The unbounded study has dimension 7 and a bounded nonlinear tilt of a Gaussian reference; its extra H-TV and H-W1 metrics measure the compositional statistic. The reported standard deviations are across the 20 replications, not standard errors. The supplied records are metric and timing outputs; they do not include the fitted networks or generated observations.

## Rebuild and check the saved tables

```bash
cd response/bounded_tree_31
mkdir -p outputs
cp -R reported_results outputs/table_check
python3 summarize_results.py --root outputs/table_check --profile full
python3 verification/check_reported_results.py

cd ../gaussian_compositional_7
python3 reproduce_tables.py --check --output outputs/table_check
python3 verification/journal_audit.py
```

These commands validate and aggregate the saved records without training, writing the rebuilt tables under ignored `outputs/table_check/` so the archived files stay byte-for-byte intact. See each package's own `README.md` for a CPU smoke check and instructions for a full CUDA rerun. New runs belong under its `outputs/` directory, separate from the included `reported_results/`.

## PDF version notes

The supplied response PDF is a draft with older experiment descriptions mixed with the current tables. In particular, its earlier 31D tree discussion mentions `n = 16,000`, and the paragraph following Table 1 mentions `n = 2,000, 4,000, 6,000, 8,000` and an `n = 8,000` result. Those statements do not describe the saved Table 1 study. For the current bounded study use the four sample sizes, settings, and values in [`bounded_tree_31/README.md`](bounded_tree_31/README.md) and its Table 1. The PDF's Tables 2–3 agree with the Gaussian package's saved tables.
