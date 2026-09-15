# -*- coding: utf-8 -*-
"""SHAP summary for UTS (dataset B).
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "15.shap_UTS"

fig, ax = plt.subplots(figsize=(6, 4))
ac.panel_shap_uts(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
