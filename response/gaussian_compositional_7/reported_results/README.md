# Recorded results

This directory contains the measurements from the completed experiment used in the two reported tables. There are 80 paired replications: 20 datasets at each of the sample sizes 500, 1000, 1500, and 2000. Each `run_complete.json` contains the results for both CINDES and DSM fitted to the same dataset.

Run `python reproduce_tables.py --check` from the package directory to reconstruct the tables. This calculation uses only Python's standard library. It recomputes means, sample standard deviations, and standard errors from the individual replication records; it does not take the displayed means and standard deviations as input.

## Contents of a replication record

The `methods` field contains the five accuracy criteria, the final-refit time, and the generation time for each method. Its `model_record` also records the chosen penalty, training duration, number of parameters, and `cv_sum_fit_wall_seconds`. The last quantity is the sum of the 15 cross-validation fit times recorded by the experiment for that method. The table script takes the mean and sample standard deviation of these recorded sums across the 20 datasets. The compact results archive does not contain the 15 separate worker timing files.

The records also retain the data and reference fingerprints, fitted-model and generated-sample fingerprints, DGP specification, cross-validation selections, numerical diagnostics, software versions, and GPU type. Hostnames, scheduler job identifiers, device allocation numbers, and attempt timestamps have been removed because they are not needed to reproduce the tables. The numerical measurements have not been changed.

## What is and is not included

The original compact results archive contains evaluation records, rather than the observed datasets, generated samples, reference arrays, or fitted checkpoints. Those arrays and models are therefore not included here. The experimental code regenerates them when the study is rerun. Rebuilding these tables from the supplied records and rerunning the experiment from simulated observations are distinct reproducibility steps; instructions for both are given in the main README.

The reference calibration results are retained in `reference/reference_summary.json`. These were computed using independent exact-target samples of the same size as the generated samples. They are not subtracted from any reported metric.

## Provenance

`provenance.json` records the SHA-256 hash and size of the source archive, the original hashes of the relevant archive members, and the hashes of the projected records included here. All 815 file hashes listed in the original archive manifest were checked before preparing this directory. `original_bundle_manifest.json` preserves that original manifest. Its entries describe the original archive and are not a list of the files in this smaller projected results directory.

`study_manifest.json` preserves the original study specification, task roster, and the hashes of the 14 Python source files used to generate the results. The table script requires all 80 expected paired replications and rejects a mixture of protocols, missing methods, duplicate tasks, or inconsistent shared-reference records.
