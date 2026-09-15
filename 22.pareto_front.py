# -*- coding: utf-8 -*-
"""Pareto front with bootstrap prediction uncertainty.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "22.pareto_front"

fig, ax = plt.subplots(figsize=(4.6, 2.8))
ac.panel_pareto(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
