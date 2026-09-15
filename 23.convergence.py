# -*- coding: utf-8 -*-
"""convergence of the mean objectives over the generations.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "23.convergence"

fig, ax = plt.subplots(figsize=(4.6, 2.8))
ac.panel_convergence(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
