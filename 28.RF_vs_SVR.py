# -*- coding: utf-8 -*-
"""grouped-CV median R2 of the random forest and of a tuned RBF-kernel SVR.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "28.RF_vs_SVR"

fig, ax = plt.subplots(figsize=(4.6, 3.4))
ac.panel_rf_vs_svr(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
