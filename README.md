# alloy-CV

Data and code for the paper:

Composition-grouped cross-validation exposes uneven generalization failure in Cu–Ni–Si–Ce–La alloy property models

The repository has the two curated datasets, the scripts that draw each figure and each supplementary table, and the result files behind the numbers quoted in the paper.

## Data

| File | Content |
|---|---|
| `dataset_A.csv` | Full-route set, 448 records, 20 features. The `Source` column gives the source publication of each record. |
| `dataset_B.csv` | Cold-rolled route with no annealing step, 290 records, 17 features. `Source` column as above. |
| `dataset_A_with_source.xlsx` | Same records, Source column kept in spreadsheet form. |
| `dataset_B_with_source.xlsx` | Same for dataset B. |

Missing values: a continuous processing field that is not reported is set to `-1`. Whether a step was performed is stored separately in the 0/1 indicator variables, which never take `-1`. So `-1` means "not reported", not "not done".

## Results

| File | Content |
|---|---|
| `cv_per_fold_R2.csv` | Per-fold R2 for all runs reported (Figures 4 and 10). |
| `composition_errors.csv` | Composition-level out-of-fold errors (Table S7). |
| `four_model_test_r2.csv` | Held-out test R2 of the four models (Figure 3). |

The CSV files behind Tables S5 and S6 are produced by the scripts listed below.

## Scripts

`alloycv_common.py` holds the shared code: data loading, feature lists, model pipelines and search grids, the two CV protocols, the NSGA-II search with bootstrap, and one draw function per panel.

| Script | Output |
|---|---|
| `1.workflow.py` to `28.RF_vs_SVR.py` | one panel each; `make_figures.py` assembles them |
| `29.publication_grouped.py` | Table S5 |
| `30.composition_errors.py` | Table S7 |
| `31.pooled_metrics.py` | Table S6 |
| `32.baselines.py` | Baseline metrics |

## Running

    pip install -r requirements.txt
    python make_figures.py

Paths are relative to the repository root. Scripts write to `figures/` when that folder exists and to the repository root otherwise; `make_figures.py` recreates the ten main-text figures plus Figures S1, S2 and S6. Individual panels can also be drawn with the numbered scripts.

## Notes

- Random state is fixed at 42 for the models; the ten CV assignments use seeds 2000-2009.
- The log transform uses an offset of 1e-9. Scores for 1e-6 and 1e-3 are in `log_offset_sensitivity.csv` and do not change the ranking of the two protocols.
- Records are transcribed from the cited papers. No measurement was repeated by the authors.
- DOI is given where the source provides one; theses and some Chinese journals do not have a DOI.

## License

MIT. See `LICENSE`.
