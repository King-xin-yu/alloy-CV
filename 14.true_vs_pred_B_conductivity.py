# -*- coding: utf-8 -*-
"""true vs predicted conductivity, dataset B.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "14.true_vs_pred_B_conductivity"

fig, ax = plt.subplots(figsize=(3.2, 2.8))
ac.panel_truevspred_b_cond(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
