# -*- coding: utf-8 -*-
"""predicted UTS against cold-rolling strain.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "19.pdp_cold_rolling_strain"

fig, ax = plt.subplots(figsize=(2.9, 2.8))
ac.panel_pdp_strain(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
