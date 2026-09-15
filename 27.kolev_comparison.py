# -*- coding: utf-8 -*-
"""random vs grouped CV R2 for the Kolev study and for this work.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "27.kolev_comparison"

fig, ax = plt.subplots(figsize=(5.8, 4.9))
ac.panel_kolev_compare(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
