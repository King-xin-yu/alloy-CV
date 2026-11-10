# alloy-CV

Data and code for the paper:

Composition-grouped cross-validation exposes uneven generalization failure in Cu–Ni–Si–Ce–La alloy property models

The repository has the two curated datasets, one script per figure and per supplementary table, and the result files behind the numbers quoted in the paper.

## Data

| File | Content |
|---|---|
| `dataset_A.csv` | Full-route set, 448 records, 20 features. The `Source` column gives the source publication of each record. |
| `dataset_B.csv` | Cold-rolled route with no annealing step, 290 records, 17 features. `Source` column as above. |
| `data/dataset_A_with_source.xlsx` | Same records, Source column kept in spreadsheet form. |
| `data/dataset_B_with_source.xlsx` | Same for dataset B. |

Missing values: a continuous processing field that is not reported is set to `-1`. Whether a step was performed is stored separately in the 0/1 indicator variables, which never take `-1`. So `-1` means "not reported", not "not done".

## Results

| File | Content |
|---|---|
| `results/cv_reps.csv` | Pooled out-of-fold metrics for the ten randomized assignments, both protocols. |
| `results/cv_folds_first_assignment.csv` | Per-fold R2 of the first randomized assignment (Tables S10-S11). |
| `results/cv_per_fold_R2.csv` | Per-fold R2 for all runs reported. |
| `results/r12_table.csv` | Pooled summary by dataset, target and protocol (Table S6). |
| `results/r12_baselines.csv` | Mean predictor and linear regression baselines. |
| `results/nested_svr.csv` | Nested versus fixed-default hyperparameters for the SVR. |
| `results/publication_grouped.csv` | Publication-grouped validation (Table S5). |
| `results/composition_errors.csv` | Composition-level out-of-fold errors (Table S7). |
| `results/log_offset_sensitivity.csv` | Grouped-CV scores for log offsets 1e-9, 1e-6 and 1e-3. |

## Scripts

`alloycv_common.py` holds the shared code: data loading, feature lists, model pipelines and search grids, the two CV protocols, the NSGA-II search with bootstrap, and one draw function per panel. Scripts 1-28 produce the figures of the main text, 29-33 the supplementary tables.

| Script | Output |
|---|---|
| `29.publication_grouped.py` | Table S5 |
| `30.composition_errors.py` | Table S7 |
| `31.pooled_metrics.py` | Table S6 |
| `32.baselines.py` | Baseline metrics |
| `33.per_fold_summary.py` | Per-fold summary |
| `fig_r1_5_nested.py` | Nested SVR comparison (Fig. S3b) |

## Running

    pip install -r requirements.txt
    python 1.workflow.py

Paths in the scripts are relative to the repository root. Figures go to `figures/` and tables to `results/`; both are regenerated, so the figures themselves are not committed.

## Reproducing all figures

    python make_figures.py

`make_figures.py` composes the multi-panel figures (Figure_1 to Figure_10, plus Figure_S1, S2 and S6) from the per-panel draw functions in `alloycv_common.py`. The numbered scripts (1.workflow.py ... 33.*.py) still produce individual panels. Figure 10 additionally reads `results/cv_per_fold_R2.csv` and `results/composition_errors.csv`.

## Notes

- Random state is fixed at 42 for the models; the ten CV assignments use seeds 2000-2009.
- The log transform uses an offset of 1e-9. Scores for 1e-6 and 1e-3 are in `results/log_offset_sensitivity.csv` and do not change the ranking of the two protocols.
- Records are transcribed from the cited papers. No measurement was repeated by the authors.
- DOI is given where the source provides one; theses and some Chinese journals do not have a DOI.

## License

MIT. See `LICENSE`.
