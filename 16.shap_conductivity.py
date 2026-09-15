# -*- coding: utf-8 -*-
"""SHAP summary for conductivity (dataset B).
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "16.shap_conductivity"

fig, ax = plt.subplots(figsize=(6, 4))
ac.panel_shap_conductivity(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
