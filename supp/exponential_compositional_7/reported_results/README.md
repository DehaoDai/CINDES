# Final reported aggregate results

These files record the final 80-pair summary provided by the authors: 20 paired
replications for each of n=500, 1000, 1500, and 2000.

- `accuracy_summary.csv`: means, sample SDs, and SEs for the five accuracy metrics.
- `runtime_summary.csv`: means and sample SDs for the five recorded timing quantities.
- `reported_aggregates.json`: the same data and supplied text representations.
- `reported_tables.tex`: formatted accuracy and computation-time tables.
- `accuracy_source.txt`, `runtime_source.txt`: supplied aggregate rows used to build the tables.

These are transcribed aggregate values, not per-replication records or results
of the archive's smoke check. They do not permit reconstruction of paired
confidence intervals. The final task manifest is included under
`../provenance/final_scc_run/`, and its scientific source/configuration/task identity
matches this archive. Raw per-replication outputs and final SCC environment records
were not supplied; see `../REPRODUCIBILITY_NOTES.md` for the verification scope.

At n=2000, the reported H-TV values are 0.0573455145 for CINDES and
0.0823961032 for DSM; H-W1 values are 0.0676192537 and 0.110016084.
TV1 is slightly smaller for DSM (0.027932111 versus 0.0282725378).
The reported total times of 468.418 and 582.438 seconds include cross-validation
training and validation, final refitting, and generation.
