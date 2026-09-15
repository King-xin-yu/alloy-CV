# -*- coding: utf-8 -*-
"""predicted UTS against aging time.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "20.pdp_aging_time"

fig, ax = plt.subplots(figsize=(2.9, 2.8))
ac.panel_pdp_agingtime(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
