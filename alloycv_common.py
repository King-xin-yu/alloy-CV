# -*- coding: utf-8 -*-
"""Shared analysis and drawing code for the alloy-CV data repository.

This module holds everything the per-panel scripts need: dataset loading, the
feature lists, the model pipelines with their randomized-search grids, the two
cross-validation protocols, the NSGA-II surrogate optimization with its
bootstrap, and one draw function per panel of the paper.

Every draw function takes a matplotlib Axes as its first argument and draws a
single panel onto it, so a composition script outside the repository can lay
the panels out without touching the analysis code::

    import matplotlib.pyplot as plt
    import alloycv_common as ac

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ac.panel_test_r2_uts(ax)
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import (GroupKFold, KFold, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).parent
DATASET_A = BASE_DIR / "dataset_A.csv"
DATASET_B = BASE_DIR / "dataset_B.csv"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)   # the panel scripts write their output here

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.unicode_minus": False,
                     "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight"})

RS = 42
CB = "#2b6cb0"           # blue: dataset A / the reference model
CR = "#c53030"           # red: dataset B / grouped CV
GY = "#9aa3ad"           # grey: reference clouds and uncertainty
CB_LAYOUT = "#4477AA"    # blue of the grouped/random CV box plots (Figures 5-6)
CR_LAYOUT = "#CC6677"    # red of the grouped/random CV box plots (Figures 5-6)

FEATURES_A = ["Ni", "Si", "Ce", "La", "cold_rolling_strain", "annealing_T", "annealing_t",
              "solution_T", "solution_t", "aging_T", "aging_t", "log_Ni", "log_Si",
              "log_Ce", "log_La", "precipitation_index", "is_cold_rolled",
              "is_annealed", "is_solution", "is_aged"]
FEATURES_B = ["Ni", "Si", "Ce", "La", "cold_rolling_strain",
              "solution_T", "solution_t", "aging_T", "aging_t", "log_Ni", "log_Si",
              "log_Ce", "log_La", "precipitation_index", "is_cold_rolled",
              "is_solution", "is_aged"]
TARGETS = ["UTS", "conductivity"]
CONT_A = [i for i, c in enumerate(FEATURES_A) if not c.startswith("is_")]
CONT_B = [i for i, c in enumerate(FEATURES_B) if not c.startswith("is_")]


def no_ticks(ax):
    """Inward ticks on all four sides, as in the published panels."""
    ax.tick_params(direction="in", top=True, right=True)


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

_DATA_CACHE = {}
_TUNE_CACHE = {}
_MEMO = {}


def _memo(key, func):
    if key not in _MEMO:
        _MEMO[key] = func()
    return _MEMO[key]


def dataset_A():
    """Full-route dataset (n = 448, 20 features)."""
    if "A" not in _DATA_CACHE:
        _DATA_CACHE["A"] = pd.read_csv(DATASET_A)
    return _DATA_CACHE["A"]


def dataset_B():
    """Cold-rolled-only dataset (n = 290, 17 features)."""
    if "B" not in _DATA_CACHE:
        _DATA_CACHE["B"] = pd.read_csv(DATASET_B)
    return _DATA_CACHE["B"]


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def model_dict():
    """The four regressors compared in the paper."""
    return {
        "RF": RandomForestRegressor(n_estimators=400, random_state=RS, n_jobs=-1),
        "SVR": SVR(),
        "XGB": XGBRegressor(n_estimators=400, random_state=RS, verbosity=0),
        "GBR": GradientBoostingRegressor(n_estimators=400, random_state=RS),
    }


def make_pipeline(features, model):
    """Standardise the continuous features, pass the binary flags through."""
    num = [c for c in features if not c.startswith("is_")]
    pre = ColumnTransformer([("s", StandardScaler(), num)], remainder="passthrough")
    return Pipeline([("p", pre), ("m", model)])


def _cache_key(Xtr, ytr, features, name):
    h = hashlib.md5()
    h.update(name.encode())
    h.update("|".join(features).encode())
    h.update(np.asarray(Xtr, dtype=float).tobytes())
    h.update(np.asarray(ytr, dtype=float).tobytes())
    return (name, h.hexdigest())


def tuned_rf(Xtr, ytr, features):
    """5-fold randomized search over the RF grid described in the paper
    (n_estimators 200-600, max_depth None/10/20/30, min_samples_leaf 1/2/4)."""
    ck = _cache_key(Xtr, ytr, features, "RF_tuned")
    if ck in _TUNE_CACHE:
        return _TUNE_CACHE[ck]
    pipe = make_pipeline(features, RandomForestRegressor(random_state=RS, n_jobs=-1))
    grid = {"m__n_estimators": [200, 300, 400, 500, 600],
            "m__max_depth": [None, 10, 20, 30],
            "m__min_samples_leaf": [1, 2, 4]}
    search = RandomizedSearchCV(pipe, grid, n_iter=30, cv=5, scoring="r2",
                                random_state=RS, n_jobs=-1)
    search.fit(Xtr, ytr)
    _TUNE_CACHE[ck] = search
    return search


GRIDS = {
    "SVR": {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.01, 0.1, 1], "epsilon": [0.01, 0.05, 0.1]},
    "XGB": {"n_estimators": [200, 400, 600], "max_depth": [3, 4, 5, 6], "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0]},
    "GBR": {"n_estimators": [200, 400, 600], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [3, 4, 5]},
}


def tuned_model(Xtr, ytr, features, name):
    """Tune a single model via 5-fold randomized search (30 configurations),
    matching the four-model comparison of the paper. Results are cached, so a
    given (dataset, target, model) combination is tuned only once per run."""
    ck = _cache_key(Xtr, ytr, features, name)
    if ck in _TUNE_CACHE:
        return _TUNE_CACHE[ck]
    if name == "XGB":
        base_model = XGBRegressor(random_state=RS, objective="reg:squarederror", tree_method="hist", n_jobs=-1, verbosity=0)
    else:
        base_model = model_dict()[name]
    pipe = make_pipeline(features, base_model)
    grid = {"m__" + k: v for k, v in GRIDS[name].items()}
    search = RandomizedSearchCV(pipe, grid, n_iter=30, cv=5, scoring="r2", random_state=RS, n_jobs=-1)
    search.fit(Xtr, ytr)
    _TUNE_CACHE[ck] = search
    return search


def four_model_scores(df, features, target):
    """Held-out test R2 of the four models for one dataset and target. All four
    are tuned by the same 5-fold randomized search protocol (30 configurations
    each): RF uses the grid described in the paper; SVR/XGB/GBR use the grids of
    the original four-model comparison."""
    sub = df[df[target].notna()]
    Xtr, Xte, ytr, yte = train_test_split(sub[features], sub[target], test_size=0.2, random_state=RS)
    out = {"RF": tuned_rf(Xtr, ytr, features).score(Xte, yte)}
    for name in ["SVR", "XGB", "GBR"]:
        out[name] = tuned_model(Xtr, ytr, features, name).score(Xte, yte)
    return out


def cv_scores(df, features, target, grouped, model=None):
    """10-fold CV R2: random folds (KFold) or composition-grouped folds
    (GroupKFold over the unique [Ni, Si, Ce, La] compositions)."""
    sub = df[df[target].notna()]
    X, y = sub[features], sub[target].to_numpy()
    g = sub.groupby(["Ni", "Si", "Ce", "La"]).ngroup().to_numpy()
    pipe = make_pipeline(features, model if model is not None else RandomForestRegressor(n_estimators=400, random_state=RS, n_jobs=-1))
    if grouped:
        cv = GroupKFold(min(10, len(np.unique(g))))
        return cross_val_score(pipe, X, y, cv=cv, groups=g, scoring="r2", n_jobs=-1)
    return cross_val_score(pipe, X, y, cv=KFold(10, shuffle=True, random_state=RS), scoring="r2", n_jobs=-1)


def grouped_cv_repeated(name, df, features, target, repeats=10, seed0=1000, model=None):
    """Per-fold R2 of repeated randomized composition-grouped 10-fold CV.

    GroupKFold is deterministic and balances folds by group size, which ties fold
    membership to the row order of the compositions; shuffling the unique
    compositions into the folds removes that dependence. Returns the pooled
    per-fold R2 values (repeats x 10 folds).
    """
    def compute():
        sub = df[df[target].notna()]
        X, y = sub[features], sub[target].to_numpy(dtype=float)
        g = sub.groupby(["Ni", "Si", "Ce", "La"]).ngroup().to_numpy()
        groups = np.unique(g)
        out = []
        for r in range(repeats):
            rng = np.random.default_rng(seed0 + r)
            perm = rng.permutation(groups)
            fold_of = {gg: i % 10 for i, gg in enumerate(perm)}
            fold = np.array([fold_of[gg] for gg in g])
            pred = np.full(y.shape, np.nan)
            for k in range(10):
                te = fold == k
                if not te.any():
                    continue
                pipe = make_pipeline(features, model if model is not None else RandomForestRegressor(
                    n_estimators=400, random_state=RS, n_jobs=-1))
                pipe.fit(X[~te], y[~te])
                pred[te] = pipe.predict(X[te])
            for k in range(10):
                te = fold == k
                if te.any():
                    res = float(np.sum((y[te] - pred[te]) ** 2))
                    tot = float(np.sum((y[te] - y[te].mean()) ** 2))
                    out.append(1.0 - res / tot)
        return np.array(out)
    return _memo(f"grouped_rep_{name}_{target}_{type(model).__name__ if model is not None else 'RF'}", compute)


def cv_results():
    """Both CV protocols for the four (dataset, target) combinations."""
    def compute():
        out = {}
        for name, df, feats in [("A", dataset_A(), FEATURES_A), ("B", dataset_B(), FEATURES_B)]:
            for t in TARGETS:
                rr = cv_scores(df, feats, t, grouped=False)
                rg = grouped_cv_repeated(name, df, feats, t)
                out[(name, t)] = (rr, rg)
        return out
    return _memo("cv_results", compute)


def best_rf(df, features, target):
    """Tuned random forest for one dataset and target, with the held-out split."""
    sub = df[df[target].notna()]
    Xtr, Xte, ytr, yte = train_test_split(sub[features], sub[target], test_size=0.2, random_state=RS)
    search = tuned_rf(Xtr, ytr, features)
    return search.best_estimator_, Xte, yte, search.score(Xte, yte)


# --------------------------------------------------------------------------
# NSGA-II design search
# --------------------------------------------------------------------------

def derive_B(Xc):
    """Expand the nine design variables into the 17 features of dataset B."""
    Ni, Si, Ce, La = Xc[:, 0], Xc[:, 1], Xc[:, 2], Xc[:, 3]
    cr, sT, st, gT, gt = Xc[:, 4], Xc[:, 5], Xc[:, 6], Xc[:, 7], Xc[:, 8]
    return np.column_stack([Ni, Si, Ce, La, cr, sT, st, gT, gt,
                            np.log(Ni + 1e-9), np.log(Si + 1e-9), np.log(Ce + 1e-9),
                            np.log(La + 1e-9), np.full_like(Ni, 3.0), np.ones_like(Ni),
                            np.ones_like(Ni), np.ones_like(Ni)])


def surrogate_models():
    """Gradient-boosting surrogates for UTS and conductivity, with the scalers
    of their continuous features."""
    def compute():
        B = dataset_B()
        models = {}
        for t, key in [("UTS", "uts"), ("conductivity", "cond")]:
            sub = B[B[t].notna()]
            X = sub[FEATURES_B].to_numpy()
            y = sub[t].to_numpy()
            sc = StandardScaler()
            sc.fit(X[:, CONT_B])
            Xs = X.copy()
            Xs[:, CONT_B] = sc.transform(X[:, CONT_B])
            m = GradientBoostingRegressor(random_state=RS, n_estimators=400, learning_rate=0.05, max_depth=4)
            m.fit(Xs, y)
            models[key] = m
            models[key + "_sc"] = sc
        return models
    return _memo("surrogates", compute)


class AlloyProblem(Problem):
    """Maximise UTS and conductivity over composition and processing variables,
    subject to the three domain-knowledge constraints of the paper."""

    def __init__(self, models):
        super().__init__(n_var=9, n_obj=2, n_constr=3,
                         xl=np.array([0, 0.4, 0, 0, 20, 850, 0.5, 350, 0.5]),
                         xu=np.array([8, 1.2, 0.15, 0.15, 80, 950, 3, 500, 6]))
        self.models = models

    def _evaluate(self, X, out, *a, **k):
        F = derive_B(X)
        Fs = F.copy()
        Fs[:, CONT_B] = self.models["uts_sc"].transform(F[:, CONT_B])
        uts = self.models["uts"].predict(Fs)
        Fs2 = F.copy()
        Fs2[:, CONT_B] = self.models["cond_sc"].transform(F[:, CONT_B])
        cond = self.models["cond"].predict(Fs2)
        out["F"] = np.column_stack([-uts, -cond])
        out["G"] = np.column_stack([X[:, 2] + X[:, 3] - 0.15,
                                    2 * X[:, 1] - X[:, 0],
                                    X[:, 7] - X[:, 5] + 300])


class _HistCB(Callback):
    """Mean objective values per generation."""

    def __init__(self):
        super().__init__()
        self.u = []
        self.c = []

    def notify(self, algorithm, **k):
        F = algorithm.pop.get("F")
        self.u.append(float(-F[:, 0].mean()))
        self.c.append(float(-F[:, 1].mean()))


def run_nsga2():
    """Run the NSGA-II design search. Returns (result, callback)."""
    def compute():
        cb = _HistCB()
        res = minimize(AlloyProblem(surrogate_models()),
                       NSGA2(pop_size=100, sampling=FloatRandomSampling(),
                             crossover=SBX(prob=0.9, eta=15),
                             mutation=PM(prob=0.1, eta=20), eliminate_duplicates=True),
                       ("n_gen", 200), seed=RS, callback=cb, verbose=False)
        return res, cb
    return _memo("nsga2", compute)


def bootstrap_pareto():
    """Bootstrap prediction uncertainty of the Pareto front (60 replications),
    one learner per target (gradient boosting for both). Returns
    (std_uts, std_cond, std_uts_dense, n_dense)."""
    def compute():
        res, _ = run_nsga2()
        n_bs = 60
        rng = np.random.default_rng(RS)
        pred_uts = np.zeros((n_bs, len(res.X)))
        pred_cond = np.zeros((n_bs, len(res.X)))
        Fpf = derive_B(res.X)
        B = dataset_B()
        dense = B[(B["UTS"].notna()) & (B["Ce"] == 0) & (B["La"] == 0)][FEATURES_B].to_numpy()
        pred_uts_dense = np.zeros((n_bs, len(dense)))
        for b in range(n_bs):
            for t, arr in [("UTS", pred_uts), ("conductivity", pred_cond)]:
                sub = B[B[t].notna()]
                X = sub[FEATURES_B].to_numpy()
                y = sub[t].to_numpy()
                idx = rng.integers(0, len(y), len(y))
                sc = StandardScaler()
                sc.fit(X[idx][:, CONT_B])
                Xs = X.copy()
                Xs[:, CONT_B] = sc.transform(X[:, CONT_B])
                Fs = Fpf.copy()
                Fs[:, CONT_B] = sc.transform(Fpf[:, CONT_B])
                mm = GradientBoostingRegressor(random_state=b, n_estimators=400, learning_rate=0.05, max_depth=4)
                mm.fit(Xs[idx], y[idx])
                arr[b] = mm.predict(Fs)
                if t == "UTS":
                    Ds = dense.copy()
                    Ds[:, CONT_B] = sc.transform(dense[:, CONT_B])
                    pred_uts_dense[b] = mm.predict(Ds)
        return pred_uts.std(axis=0), pred_cond.std(axis=0), pred_uts_dense.std(axis=0), len(dense)
    return _memo("bootstrap", compute)


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def panel_workflow(ax):
    """workflow of the study."""
    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-1.0, 11.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=9):
        ax.add_patch(mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                     boxstyle="round,pad=0.15", linewidth=1.2, edgecolor="black", facecolor=fc, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="black", zorder=2))

    box(0, 10.4, 5.2, 0.95, "Literature data compilation\nCu-Ni-Si-Ce-La alloys", "#dbe7f5")
    arrow(-0.9, 9.9, -1.9, 9.0); arrow(0.9, 9.9, 1.9, 9.0)
    box(-1.95, 8.55, 3.1, 1.25, "Dataset A (full-route, n = 448)\nwith annealing records\n20 features", "#dbe7f5", fs=8)
    box(1.95, 8.55, 3.1, 1.25, "Dataset B (cold-rolled-only, n = 290)\nannealing records removed\n17 features", "#f5d9d9", fs=8)
    arrow(-1.95, 7.9, -0.7, 7.05); arrow(1.95, 7.9, 0.7, 7.05)
    box(0, 6.6, 5.6, 0.95, "Feature engineering\nlog transforms, precipitation index, binary indicators", "#e8eef5")
    arrow(0, 6.1, 0, 5.45)
    box(0, 5.0, 4.6, 0.95, "Four regression models\nRF, SVR, XGBoost, GBR (scikit-learn)", "#e8eef5")
    arrow(-0.9, 4.5, -1.9, 3.85); arrow(0.9, 4.5, 1.9, 3.85)
    box(-1.95, 3.4, 3.1, 1.1, "Random 10-fold CV\n(split by sample)", "#dbe7f5", fs=8)
    box(1.95, 3.4, 3.1, 1.1, "Composition-grouped CV\n(split by unique composition)", "#f5d9d9", fs=8)
    arrow(-1.95, 2.8, -1.95, 2.15); arrow(1.95, 2.8, 1.95, 2.15)
    box(-1.95, 1.6, 3.1, 1.1, "SHAP interpretation\n(feature importance)", "#dbe7f5", fs=8)
    box(1.95, 1.6, 3.1, 1.1, "NSGA-II optimization\n(3 domain-knowledge constraints)", "#f5d9d9", fs=8)
    arrow(-1.95, 1.05, -0.7, 0.25); arrow(1.95, 1.05, 0.7, 0.25)
    box(0, -0.25, 5.6, 0.95, "Generalization assessment and design guidance", "#e2e2e2")


def _draw_composition_boxplot(ax, element, letter):
    A, B = dataset_A(), dataset_B()
    bp = ax.boxplot([A[element].values, B[element].values], positions=[1, 2], widths=0.5,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color="black"),
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, color in zip(bp["boxes"], [CB, CR]):
        patch.set_facecolor(color); patch.set_alpha(0.8); patch.set_edgecolor("black")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["A", "B"])
    ax.set_ylabel(f"{element} (wt.%)")
    ax.text(-0.18, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)


def panel_ni(ax):
    """Ni content of datasets A and B."""
    _draw_composition_boxplot(ax, "Ni", "a")


def panel_si(ax):
    """Si content of datasets A and B."""
    _draw_composition_boxplot(ax, "Si", "b")


def panel_ce(ax):
    """Ce content of datasets A and B."""
    _draw_composition_boxplot(ax, "Ce", "c")


def panel_la(ax):
    """La content of datasets A and B."""
    _draw_composition_boxplot(ax, "La", "d")


def panel_corr_matrix(ax):
    """correlation matrix of the 17 features of dataset B."""
    corr = dataset_B()[FEATURES_B].corr()
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES_B))); ax.set_xticklabels(FEATURES_B, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(FEATURES_B))); ax.set_yticklabels(FEATURES_B, fontsize=8)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)


def _draw_four_model_bars(ax, target, letter):
    r2 = {"A": four_model_scores(dataset_A(), FEATURES_A, target),
          "B": four_model_scores(dataset_B(), FEATURES_B, target)}
    names = ["RF", "SVR", "XGB", "GBR"]
    x = np.arange(4); w = 0.38
    va = [r2["A"][n] for n in names]
    vb = [r2["B"][n] for n in names]
    ax.bar(x - w/2, va, w, color=CB, label="A (full-route)", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, vb, w, color=CR, label="B (cold-rolled-only)", edgecolor="black", linewidth=0.5)
    for i, v in enumerate(va):
        ax.text(i - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    for i, v in enumerate(vb):
        ax.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    ba = int(np.argmax(va)); bb = int(np.argmax(vb))
    ax.text(ba - w/2, va[ba] + 0.06, "\u2605", ha="center", va="bottom", fontsize=13, fontfamily="DejaVu Sans")
    ax.text(bb + w/2, vb[bb] + 0.06, "\u2605", ha="center", va="bottom", fontsize=13, fontfamily="DejaVu Sans")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel(f"{target} test R\u00b2" if letter == "a" else "Conductivity test R\u00b2")
    ax.set_ylim(0, 1.1)
    if letter == "a":
        # place the legend above the axes: inside the axes it collides with the star
        # markers and the value labels of the tallest bars.
        ax.legend(frameon=False, fontsize=7, loc="lower center",
                  bbox_to_anchor=(0.5, 1.005), ncol=2)
    ax.text(-0.18, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)
    no_ticks(ax)


def panel_test_r2_uts(ax):
    """held-out test R2 of the four models for UTS."""
    _draw_four_model_bars(ax, "UTS", "a")


def panel_test_r2_conductivity(ax):
    """held-out test R2 of the four models for conductivity."""
    _draw_four_model_bars(ax, "conductivity", "b")


def _draw_cv_boxplot(ax, target, letter=None):
    cv = cv_results()
    a_rr, _ = cv[("A", target)]
    b_rr, _ = cv[("B", target)]
    a_rg = grouped_cv_repeated("A", dataset_A(), FEATURES_A, target)
    b_rg = grouped_cv_repeated("B", dataset_B(), FEATURES_B, target)
    data = [a_rr, a_rg, b_rr, b_rg]; pos = [1, 2, 3, 4]
    bp = ax.boxplot(data, positions=pos, widths=0.45, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], [CB_LAYOUT, CR_LAYOUT, CB_LAYOUT, CR_LAYOUT]):
        patch.set_facecolor(c); patch.set_alpha(0.8); patch.set_edgecolor("black")
    for p, vals in zip(pos, data):
        ax.scatter(p + np.random.default_rng(0).normal(0, 0.045, len(vals)), vals,
                   s=8, color="gray", zorder=3, alpha=0.30)
    for p, vals in zip(pos, data):
        med = float(np.median(vals)); iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        ax.text(p, 1.04, f"Med {med:.2f}\nIQR {iqr:.2f}", ha="center", va="bottom", fontsize=7, color="black")
    for p, vals in zip([2, 4], [a_rg, b_rg]):
        neg = float((vals < 0).mean() * 100)
        ax.text(p, 0.20, f"negative folds {neg:.0f}%\nworst {vals.min():.1f}", ha="center",
                va="top", fontsize=6.5, color=CR_LAYOUT)
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels(["A\nrandom\n10 folds", "A\ngrouped\n10 x 10", "B\nrandom\n10 folds", "B\ngrouped\n10 x 10"])
    ax.set_ylabel(f"{target} $R^2$ (per fold)")
    ax.set_ylim(-8, 1.35); ax.set_yticks([-6, -4, -2, 0, 1])
    ax.tick_params(direction="in", top=False, right=False)
    if letter:
        ax.text(-0.18, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)


def panel_cv_uts(ax, letter=None):
    """random vs grouped 10-fold CV R2 for UTS."""
    _draw_cv_boxplot(ax, "UTS", letter)


def panel_cv_conductivity(ax, letter=None):
    """random vs grouped 10-fold CV R2 for conductivity."""
    _draw_cv_boxplot(ax, "conductivity", letter)


def _draw_true_vs_pred(ax, name, target, letter):
    df = dataset_A() if name == "A" else dataset_B()
    feats = FEATURES_A if name == "A" else FEATURES_B
    pipe, Xte, yte, _ = best_rf(df, feats, target)
    pred = pipe.predict(Xte)
    ax.scatter(yte, pred, s=8, color=CB, alpha=0.7)
    lim = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    ax.plot(lim, lim, color="black", lw=0.8)
    ax.set_xlabel(f"True {target}"); ax.set_ylabel(f"Predicted {target}")
    ax.text(-0.18, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)
    no_ticks(ax)


def panel_truevspred_a_uts(ax):
    """true vs predicted UTS, dataset A."""
    _draw_true_vs_pred(ax, "A", "UTS", "a")


def panel_truevspred_b_uts(ax):
    """true vs predicted UTS, dataset B."""
    _draw_true_vs_pred(ax, "B", "UTS", "b")


def panel_truevspred_a_cond(ax):
    """true vs predicted conductivity, dataset A."""
    _draw_true_vs_pred(ax, "A", "conductivity", "c")


def panel_truevspred_b_cond(ax):
    """true vs predicted conductivity, dataset B."""
    _draw_true_vs_pred(ax, "B", "conductivity", "d")


def _draw_shap(ax, target, letter):
    import shap
    pipe, Xte, _, _ = best_rf(dataset_B(), FEATURES_B, target)
    Xte_tr = pipe.named_steps["p"].transform(Xte)
    explainer = shap.TreeExplainer(pipe.named_steps["m"])
    sv = explainer.shap_values(pd.DataFrame(Xte_tr, columns=FEATURES_B))
    plt.sca(ax)
    shap.summary_plot(sv, pd.DataFrame(Xte_tr, columns=FEATURES_B), show=False, max_display=12)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.text(-0.18, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)
    ax.text(0.5, -0.26, target, transform=ax.transAxes, ha="center", va="top", fontsize=11)


def panel_shap_uts(ax):
    """SHAP summary for UTS (dataset B)."""
    _draw_shap(ax, "UTS", "a")


def panel_shap_conductivity(ax):
    """SHAP summary for conductivity (dataset B)."""
    _draw_shap(ax, "conductivity", "b")


def _draw_perm_importance(ax, target, letter):
    comp = ["Ni", "Si", "Ce", "La", "log_Ni", "log_Si", "log_Ce", "log_La"]
    proc = ["cold_rolling_strain", "solution_T", "solution_t", "aging_T", "aging_t"]
    pipe, Xte, yte, _ = best_rf(dataset_B(), FEATURES_B, target)
    Xte_tr = pipe.named_steps["p"].transform(Xte)
    r = permutation_importance(pipe.named_steps["m"], Xte_tr, yte, n_repeats=20, random_state=RS, n_jobs=-1)
    imp = r.importances_mean
    ci = sum(imp[FEATURES_B.index(c)] for c in comp if c in FEATURES_B)
    pi = sum(imp[FEATURES_B.index(c)] for c in proc if c in FEATURES_B)
    oi = imp.sum() - ci - pi
    ax.bar([0, 1, 2], [ci, pi, oi], 0.55, color=[GY, CR, "#c9ced6"], edgecolor="black", linewidth=0.5)
    for i, v in enumerate([ci, pi, oi]):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Composition", "Processing", "Others"], fontsize=8)
    ax.set_ylabel("Permutation importance")
    ax.set_ylim(0, max(ci, pi, oi) * 1.25)
    ax.text(-0.18, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)
    ax.text(0.5, -0.14, target, transform=ax.transAxes, ha="center", va="top", fontsize=11)
    no_ticks(ax)


def panel_perm_uts(ax):
    """permutation importance by feature group for UTS."""
    _draw_perm_importance(ax, "UTS", "a")


def panel_perm_conductivity(ax):
    """permutation importance by feature group for conductivity."""
    _draw_perm_importance(ax, "conductivity", "b")


def _draw_pdp(ax, feature, letter, ylabel=False):
    pipe, _, _, _ = best_rf(dataset_B(), FEATURES_B, "UTS")
    sub = dataset_B()[dataset_B()["UTS"].notna()]
    Xtr = sub[FEATURES_B]
    Xtr_tr = pipe.named_steps["p"].transform(Xtr)
    _sc = pipe.named_steps["p"].named_transformers_["s"]
    _num = [c for c in FEATURES_B if not c.startswith("is_")]
    fi = FEATURES_B.index(feature)
    pdp = partial_dependence(pipe.named_steps["m"], Xtr_tr, [fi], grid_resolution=40)
    gv = pdp["grid_values"][0]
    j = _num.index(feature); gv_raw = gv * _sc.scale_[j] + _sc.mean_[j]
    ax.plot(gv_raw, pdp["average"][0], color=CB, lw=1.6)
    ax.set_xlabel({"cold_rolling_strain": "Cold-rolling strain (%)",
                   "aging_t": "Aging time (h)", "Ni": "Ni (wt.%)"}[feature])
    ax.set_ylabel("Predicted UTS (MPa)" if ylabel else "")
    ax.text(-0.18, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=13)
    no_ticks(ax)


def panel_pdp_strain(ax, letter="a", ylabel=True):
    """UTS response to cold-rolling strain."""
    _draw_pdp(ax, "cold_rolling_strain", letter, ylabel)


def panel_pdp_agingtime(ax, letter="b", ylabel=False):
    """UTS response to aging time."""
    _draw_pdp(ax, "aging_t", letter, ylabel)


def panel_pdp_ni(ax, letter="c", ylabel=False):
    """UTS response to Ni content."""
    _draw_pdp(ax, "Ni", letter, ylabel)


def panel_pareto(ax, letter="a"):
    """Pareto front with bootstrap prediction uncertainty."""
    res, _ = run_nsga2()
    std_uts, std_cond, _, _ = bootstrap_pareto()
    ax.errorbar(-res.F[:, 0], -res.F[:, 1], xerr=std_uts, yerr=std_cond,
                fmt="none", ecolor=GY, alpha=0.6, elinewidth=0.6, zorder=1)
    ax.scatter(-res.F[:, 0], -res.F[:, 1], s=14, color=CB, alpha=0.8, edgecolor="none", zorder=2)
    ax.set_xlabel("UTS (MPa)"); ax.set_ylabel("Conductivity (% IACS)")
    if letter:
        ax.text(-0.18, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
    no_ticks(ax)


def panel_convergence(ax, letter="b"):
    """convergence of the mean objectives over the generations."""
    _, cb = run_nsga2()
    gens = np.arange(1, len(cb.u) + 1)
    ax.plot(gens, np.array(cb.u)/max(cb.u), color=CB, label="Normalized mean UTS")
    ax.plot(gens, np.array(cb.c)/max(cb.c), color=CR, label="Normalized mean conductivity")
    ax.set_xlabel("Generation"); ax.set_ylabel("Normalized mean objective")
    if letter:
        ax.text(-0.18, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
    ax.legend(frameon=False, fontsize=7)
    no_ticks(ax)


def _draw_candidates(ax, col_x, col_y, idx_x, idx_y, xlabel, ylabel, letter, legend):
    res, _ = run_nsga2()
    cand = res.X
    B = dataset_B()
    ax.scatter(B[col_x], B[col_y], s=6, color=GY, alpha=0.6, label="Training")
    ax.scatter(cand[:, idx_x], cand[:, idx_y], s=16, color=CR, alpha=0.9, label="Candidates")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False, fontsize=7)
    ax.text(-0.18, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)
    no_ticks(ax)


def panel_cand_ni_si(ax, letter="a"):
    """candidate compositions in the Ni-Si plane."""
    _draw_candidates(ax, "Ni", "Si", 0, 1, "Ni (wt.%)", "Si (wt.%)", letter, True)


def panel_cand_ce_la(ax, letter="b"):
    """candidate compositions in the Ce-La plane."""
    _draw_candidates(ax, "Ce", "La", 2, 3, "Ce (wt.%)", "La (wt.%)", letter, False)


def panel_tree_schematic(ax):
    """schematic of a held-out composition in a sparse region of the
    training envelope (illustrative)."""
    rng = np.random.default_rng(0)
    # dense training cloud with one sparsely sampled region (upper right)
    tr = rng.uniform(-1.5, 1.5, (48, 2))
    tr = tr[~((tr[:, 0] > 0.55) & (tr[:, 1] > 0.55))]
    # held-out composition inside the envelope but with few/mixed neighbours
    te = np.array([[0.95, 1.05], [1.25, 0.85], [0.85, 1.30], [1.05, 1.15]])
    ax.scatter(tr[:, 0], tr[:, 1], s=14, color=GY, edgecolor="black", linewidth=0.4, label="Training")
    ax.scatter(te[:, 0], te[:, 1], s=30, facecolor="none", edgecolor=CR, linewidth=1.3, label="Held-out (sparse region)")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(1.75 * np.cos(th), 1.75 * np.sin(th), color="black", lw=1.0, linestyle="--")
    ax.text(0.15, 0.20, "Interpolation region", fontsize=7, color="#555555")
    ax.text(1.05, 1.55, "sparse region:", fontsize=7, color=CR)
    ax.text(1.05, 1.42, "few/mixed neighbors", fontsize=7, color=CR)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_xlabel("Composition dimension 1"); ax.set_ylabel("Composition dimension 2")
    no_ticks(ax)


def panel_kolev_compare(ax, letter=None):
    """random vs grouped CV R2 for the Kolev study and for this work.
    The broken y axis is drawn as two inset axes inside the supplied Axes."""
    cv = cv_results()
    kolev = {"Kolev hardness": (0.855, 0.438), "Kolev conductivity": (0.840, 0.293),
             "This work UTS (B)": (np.median(cv[("B", "UTS")][0]), np.median(cv[("B", "UTS")][1])),
             "This work conductivity (B)": (np.median(cv[("B", "conductivity")][0]), np.median(cv[("B", "conductivity")][1]))}
    keys = list(kolev.keys())
    ax.set_axis_off()
    top = ax.inset_axes([0.0, 0.53, 1.0, 0.47])
    bot = ax.inset_axes([0.0, 0.0, 1.0, 0.47])
    x = np.arange(4); w = 0.36
    top.bar(x - w/2, [kolev[k][0] for k in keys], w, color=CB, label="Random CV R\u00b2", edgecolor="black", linewidth=0.5)
    top.bar(x + w/2, [kolev[k][1] for k in keys], w, color=CR, label="Grouped CV R\u00b2", edgecolor="black", linewidth=0.5)
    for i, k in enumerate(keys):
        v0, v1 = kolev[k]
        top.text(i - w/2, v0 + 0.03, f"{v0:.2f}", ha="center", va="bottom", fontsize=7)
        if v1 >= 0:
            top.text(i + w/2, v1 + 0.03, f"{v1:.2f}", ha="center", va="bottom", fontsize=7, color=CR)
    top.set_ylim(0, 1.25)
    top.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    top.spines["bottom"].set_visible(False)
    top.tick_params(bottom=False, labelbottom=False)  # x axis is shared with the lower inset
    bot.bar(x - w/2, [kolev[k][0] for k in keys], w, color=CB, edgecolor="black", linewidth=0.5)
    bot.bar(x + w/2, [kolev[k][1] for k in keys], w, color=CR, edgecolor="black", linewidth=0.5)
    for i, k in enumerate(keys):
        v1 = kolev[k][1]
        if v1 < 0:
            bot.text(i + w/2, v1 - 0.9, f"{v1:.1f}", ha="center", va="top", fontsize=7, color=CR)
    bot.set_ylim(-21, 0)
    bot.set_yticks([-20, -15, -10, -5])
    bot.spines["top"].set_visible(False)
    bot.set_xticks(x)
    bot.set_xticklabels(keys, fontsize=6.5)
    d = 0.015
    for a_, edge in [(top, 0.0), (bot, 1.0)]:
        for xpos in (0.01, 0.99):
            a_.plot([xpos - d, xpos + d], [edge - d, edge + d], transform=a_.transAxes,
                    color="black", clip_on=False, lw=0.8)
    top.legend(frameon=False, fontsize=8, loc="upper right")
    no_ticks(top); no_ticks(bot)
    if letter:
        ax.text(-0.10, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)


def panel_rf_vs_svr(ax, letter=None):
    """grouped-CV median R2 of the random forest and of a tuned
    RBF-kernel SVR."""
    cv = cv_results()
    svr_med = {}
    for name, df, feats in [("A", dataset_A(), FEATURES_A), ("B", dataset_B(), FEATURES_B)]:
        for t in TARGETS:
            sub = df[df[t].notna()]
            Xtr, _, ytr, _ = train_test_split(sub[feats], sub[t], test_size=0.2, random_state=RS)
            best_svr = tuned_model(Xtr, ytr, feats, "SVR").best_estimator_.named_steps["m"]
            svr_med[(name, t)] = np.median(grouped_cv_repeated(name, df, feats, t, model=best_svr))
    rf_med = {k: np.median(v[1]) for k, v in cv.items()}
    keys = [("A", "UTS"), ("B", "UTS"), ("A", "conductivity"), ("B", "conductivity")]
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w/2, [rf_med[k] for k in keys], w, color=CB, label="Random forest", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, [svr_med[k] for k in keys], w, color=CR, label="RBF-kernel SVR (tuned)", edgecolor="black", linewidth=0.5)
    for i, k in enumerate(keys):
        rv = rf_med[k]; sv = svr_med[k]
        ax.text(i - w/2, rv + 0.04, f"{rv:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w/2, sv + 0.04, f"{sv:.2f}", ha="center", va="bottom", fontsize=7, color=CR)
    ax.set_ylim(0, 0.95)
    ax.set_xticks(x); ax.set_xticklabels([f"{k[0]}\n{k[1]}" for k in keys], fontsize=7)
    ax.set_ylabel("Grouped-CV median R\u00b2")
    ax.legend(frameon=False, fontsize=8)
    no_ticks(ax)
    if letter:
        ax.text(-0.18, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)
