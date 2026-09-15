# -*- coding: utf-8 -*-
"""true vs predicted UTS, dataset A.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "11.true_vs_pred_A_UTS"

fig, ax = plt.subplots(figsize=(3.2, 2.8))
ac.panel_truevspred_a_uts(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
