# alloy-CV

Code and data for the paper: "Random cross-validation overstates generalization: composition-grouped CV reveals a shortfall in Cu-Ni-Si-Ce-La alloy ML models".

The repository exposes the data plus one runnable script per analysis output.

## Layout

```
alloycv_common.py    shared code: data loading, feature lists, models and their
                     search grids, both CV protocols, NSGA-II and bootstrap,
...
dataset_A.csv        full-route dataset (n = 448, 20 features)
dataset_B.csv        cold-rolled-only dataset (n = 290, 17 features)
figures/             created here when you run a script (not tracked by git)
results/             cv_per_fold_R2.csv - the per-fold R2 values behind the cross-validation statistics
requirements.txt     pinned package versions
```

## Requirements

Python 3.12 (tested with 3.12.13). Package versions are pinned in
`requirements.txt` for exact reproduction:

```
pip install -r requirements.txt
```

## Reproducing the results


```
python 7.test_R2_UTS.py
```

Each script is self-contained: it imports `alloycv_common`, builds a single-Axes
figure, draws that analysis, writes `<script-name>.png` and `<script-name>.pdf` at
600 dpi into `figures/`, and prints one confirmation line. Scripts that need the
model comparison, the cross-validation results or the NSGA-II design search run
that computation on demand, so the first run of a given script can take a few
minutes. Run the scripts in any order; `figures/` then holds one file pair per


The scripts are numbered 1 to 28. Each one draws is a single analysis and writes
`figures/<script name>.png` and `.pdf`; 
figures ).

| --- | --- | --- |
| `1.workflow.py` | workflow |
| `2.composition_Ni.py` | composition Ni |
| `3.composition_Si.py` | composition Si |
| `4.composition_Ce.py` | composition Ce |
| `5.composition_La.py` | composition La |
| `6.correlation_matrix.py` | correlation matrix |
| `7.test_R2_UTS.py` | test R2 UTS |
| `8.test_R2_conductivity.py` | test R2 conductivity |
| `9.cv_UTS.py` | cv UTS |
| `10.cv_conductivity.py` | cv conductivity |
| `11.true_vs_pred_A_UTS.py` | true vs pred A UTS |
| `12.true_vs_pred_B_UTS.py` | true vs pred B UTS |
| `13.true_vs_pred_A_conductivity.py` | true vs pred A conductivity |
| `14.true_vs_pred_B_conductivity.py` | true vs pred B conductivity |
| `15.shap_UTS.py` | shap UTS |
| `16.shap_conductivity.py` | shap conductivity |
| `17.permutation_UTS.py` | permutation UTS |
| `18.permutation_conductivity.py` | permutation conductivity |
| `19.pdp_cold_rolling_strain.py` | pdp cold rolling strain |
| `20.pdp_aging_time.py` | pdp aging time |
| `21.pdp_Ni.py` | pdp Ni |
| `22.pareto_front.py` | pareto front |
| `23.convergence.py` | convergence |
| `24.candidates_Ni_Si.py` | candidates Ni Si |
| `25.candidates_Ce_La.py` | candidates Ce La |
| `26.tree_schematic.py` | tree schematic |
| `27.kolev_comparison.py` | kolev comparison |
| `28.RF_vs_SVR.py` | RF vs SVR |

## Data

- `dataset_A.csv` - full-route set (n = 448): samples without hot rolling, with homogenization annealing (20 features).
- `dataset_B.csv` - cold-rolled-only set (n = 290): samples whose processing records carry no annealing step (solution/aging/cold-rolling only, 17 features). The label refers to the retained processing route; a cold-rolling step is reported for 133 of the 290 samples, and samples without a reported cold-rolling strain are kept with the step encoded as -1.

Columns: Ni, Si, Ce, La (wt.%), cold_rolling_strain, solution_T, solution_t, aging_T, aging_t, UTS (MPa), conductivity (%IACS), is_* binary flags, log_* transforms, precipitation_index. Dataset A additionally carries the annealing-related features (annealing_T, annealing_t) and a hardness property column, which are removed from dataset B. Both files have a header row.

## Scope

The repository contains the curated datasets, the per-fold cross-validation
values reported as `results/cv_per_fold_R2.csv`, and the Python scripts for model
training, cross-validation, SHAP analysis and figure generation, matching the
Data availability statement of the manuscript. The manuscript tables themselves
are not part of the repository.

## License

MIT
