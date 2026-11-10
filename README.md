# Supplementary table scripts

Scripts used for Tables S5-S7 of the supplementary material.

| Script | Output |
|---|---|
| `29.publication_grouped.py` | Table S5 (publication-grouped validation) |
| `30.composition_errors.py` | Table S7 (composition-level errors) |
| `31.pooled_metrics.py` | Table S6 (pooled metrics, fold sizes, baselines) |
| `32.baselines.py` | Mean and linear-regression baselines |

The scripts read the datasets from the repository root and write their output to this folder.
Run them from here, for example:

    python 31.pooled_metrics.py
