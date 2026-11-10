# -*- coding: utf-8 -*-
"""Compose the multi-panel figures of the paper.

Each numbered script draws a single panel; this script places the panels
in their final layout and writes the figures to the output directory.

Run:  python make_figures.py [output_dir]
Output defaults to figures/ in this repository.
"""
import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import alloycv_common as ac  # noqa: E402

RES = REPO / "results" if (REPO / "results").is_dir() else REPO
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def embolden(fig):
    """Bold every text object and increase its size by one point (Figs. 5-9)."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        t.set_fontweight("bold")
        t.set_fontsize(t.get_fontsize() + 1)


def save(fig, name, pad=None, w_pad=None):
    kw = {}
    if pad is not None:
        kw["pad"] = pad
    if w_pad is not None:
        kw["w_pad"] = w_pad
    fig.tight_layout(**kw)
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    gc.collect()
    print("wrote", name)


# 1 workflow --------------------
fig, ax = plt.subplots(figsize=(6.8, 8.2))
ac.panel_workflow(ax)
save(fig, "Figure_1", pad=0.4)

# ---------------- Fig. 2 composition distributions (a-d) ----------------
fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.2))
ac.panel_ni(axes[0, 0]); ac.panel_si(axes[0, 1])
ac.panel_ce(axes[1, 0]); ac.panel_la(axes[1, 1])
save(fig, "Figure_2")

# ---------------- Fig. 3 held-out test R2 of the four models (a-b) ----------------
fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.6))
ac.panel_test_r2_uts(axes[0]); ac.panel_test_r2_conductivity(axes[1])
save(fig, "Figure_3")

# ---------------- Fig. 4 CV distributions, UTS and conductivity (a-b) ----------------
fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.0))
ac.panel_cv_uts(axes[0], "a"); ac.panel_cv_conductivity(axes[1], "b")
save(fig, "Figure_4")

# ---------------- Fig. 5 true vs predicted (a-d) ----------------
fig, axes = plt.subplots(2, 2, figsize=(7.55, 6.6))
ac.panel_truevspred_a_uts(axes[0, 0]); ac.panel_truevspred_b_uts(axes[0, 1])
ac.panel_truevspred_a_cond(axes[1, 0]); ac.panel_truevspred_b_cond(axes[1, 1])
embolden(fig); save(fig, "Figure_5")

# ---------------- Fig. 6 SHAP (a-b) ----------------
fig, axes = plt.subplots(2, 1, figsize=(7.08, 9.44))
ac.panel_shap_uts(axes[0]); ac.panel_shap_conductivity(axes[1])
# shap.summary_plot resets the figure size; restore ours before saving
fig.set_size_inches(7.08, 9.44)
embolden(fig); save(fig, "Figure_6")

# ---------------- Fig. 7 permutation importance (a-b) ----------------
fig, axes = plt.subplots(1, 2, figsize=(8.02, 3.78))
ac.panel_perm_uts(axes[0]); ac.panel_perm_conductivity(axes[1])
embolden(fig); save(fig, "Figure_7")

# ---------------- Fig. 8 partial dependence (a-c) ----------------
fig, axes = plt.subplots(1, 3, figsize=(9.68, 3.3))
ac.panel_pdp_strain(axes[0]); ac.panel_pdp_agingtime(axes[1]); ac.panel_pdp_ni(axes[2])
embolden(fig); save(fig, "Figure_8", w_pad=2.6)

# ---------------- Fig. 9 NSGA-II: front, convergence, candidates (a-d) ----------------
fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.55))
ac.panel_pareto(axes[0, 0]); ac.panel_convergence(axes[0, 1])
ac.panel_cand_ni_si(axes[1, 0], "c"); ac.panel_cand_ce_la(axes[1, 1], "d")
embolden(fig); save(fig, "Figure_9")

# ---------------- Fig. 10 generalization of the failure (a-b) ----------------
_CB, _CR = "#4477AA", "#CC6677"
_cv = pd.read_csv(RES / "cv_per_fold_R2.csv")
_cv = _cv[_cv.dataset == "B"].groupby(["target", "protocol"])["r2"].median()
_rand = [0.855, 0.840, _cv.loc[("UTS", "random")], _cv.loc[("conductivity", "random")]]
_grp = [0.438, 0.293, _cv.loc[("UTS", "grouped")], _cv.loc[("conductivity", "grouped")]]
_keys = ["Kolev\nhardness", "Kolev\nconductivity", "This work\nUTS (B)", "This work\nconductivity (B)"]
comp = pd.read_csv(RES / "composition_errors.csv")
fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4))
ax = axes[0]; xx = np.arange(len(_keys)); w = 0.36
ax.bar(xx - w / 2, _rand, w, color=_CB, edgecolor="black", linewidth=0.5, label="Random CV")
ax.bar(xx + w / 2, _grp, w, color=_CR, edgecolor="black", linewidth=0.5, label="Grouped CV")
for k in range(len(_keys)):
    ax.text(k - w / 2, _rand[k] + 0.015, "%.2f" % _rand[k], ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(k + w / 2, _grp[k] + 0.015, "%.2f" % _grp[k], ha="center", va="bottom", fontsize=8, fontweight="bold", color=_CR)
ax.set_xticks(xx); ax.set_xticklabels(_keys, fontsize=8.5, fontweight="bold")
ax.set_ylabel("Median cross-validation R\u00b2", fontsize=11, fontweight="bold"); ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.tick_params(top=False, right=False, direction="in", labelsize=10)
for _t in ax.get_yticklabels(): _t.set_fontweight("bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False, handletextpad=0.6,
          columnspacing=1.2, borderaxespad=0.2, prop={"family": "Times New Roman", "weight": "bold", "size": 9})
ax.text(0.03, 0.86, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
ax = axes[1]
_ymax = comp[(comp.target == "UTS") & (comp.dataset.isin(["A", "B"]))]["mae"].max()
for _ds, _col in [("A", _CB), ("B", _CR)]:
    _d = comp[(comp.dataset == _ds) & (comp.target == "UTS")]
    ax.scatter(_d["k"], _d["mae"], s=14, facecolor=_col, edgecolor="black", linewidth=0.4, zorder=3, label="Dataset " + _ds)
    for _j, (_, _r) in enumerate(_d.nlargest(2, "mae").sort_values("k").iterrows()):
        _big = _r["k"] > 20
        _xy = (-14, 24) if _big else (14, -18 if _j == 0 else 8)
        _ha = "right" if _big else "left"
        ax.annotate("%.1fNi-%.2fSi" % (_r["Ni"], _r["Si"]), (_r["k"], _r["mae"]), textcoords="offset points",
                    xytext=_xy, ha=_ha, fontsize=8.2, fontweight="bold", color=_col,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=_col, shrinkA=1, shrinkB=2))
ax.set_xscale("log"); ax.set_xlim(0.85, 60); ax.set_ylim(0, _ymax * 1.28)
ax.set_xticks([1, 2, 5, 10, 20, 40]); ax.set_xticklabels(["1", "2", "5", "10", "20", "40"], fontsize=10, fontweight="bold")
ax.set_xlabel("Records per composition", fontsize=11, fontweight="bold"); ax.set_ylabel("Out-of-fold MAE (MPa)", fontsize=11, fontweight="bold")
ax.tick_params(top=False, right=False, direction="in", labelsize=10)
for _t in ax.get_yticklabels(): _t.set_fontweight("bold")
ax.legend(loc="upper right", frameon=False, handletextpad=0.6, borderaxespad=0.3, labelspacing=0.35,
          markerscale=1.2, prop={"family": "Times New Roman", "weight": "bold", "size": 9})
ax.text(0.03, 0.92, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
fig.tight_layout(w_pad=2.2)
save(fig, "Figure_10")

# ---------------- Supplementary S6 random forest vs tuned SVR ----------------
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ac.panel_rf_vs_svr(ax, None)
fig.tight_layout()
save(fig, "Figure_S6")
print("wrote Figure_S6")

# ---------------- Supplementary S1 / S2 ----------------
fig, ax = plt.subplots(figsize=(7, 6))
ac.panel_corr_matrix(ax)
save(fig, "Figure_S1")

# two panels: the measured composition space next to the interpolation schematic
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
_A, _B = ac.dataset_A(), ac.dataset_B()
axes[0].scatter(_A["Si"], _A["Ni"], s=8, color=ac.CB_LAYOUT, alpha=0.45, label="A (full-route)")
axes[0].scatter(_B["Si"], _B["Ni"], s=8, color=ac.CR_LAYOUT, alpha=0.55, label="B (cold-rolled-only)")
axes[0].set_xlabel("Si (wt.%)"); axes[0].set_ylabel("Ni (wt.%)")
axes[0].set_title("(a) measured composition space", fontsize=9)
axes[0].legend(frameon=False, fontsize=7, loc="upper right")
axes[0].tick_params(direction="in", top=False, right=False)
ac.panel_tree_schematic(axes[1])
axes[1].set_title("(b) interpolation envelope (schematic)", fontsize=9)
save(fig, "Figure_S2")

print("done ->", OUT)
