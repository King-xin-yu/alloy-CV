# -*- coding: utf-8 -*-
"""correlation matrix of the 17 features of dataset B.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "6.correlation_matrix"

fig, ax = plt.subplots(figsize=(7, 6))
ac.panel_corr_matrix(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
