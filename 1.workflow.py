# -*- coding: utf-8 -*-
"""workflow of the study.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "1.workflow"

fig, ax = plt.subplots(figsize=(6.8, 8.2))
ac.panel_workflow(ax)
fig.tight_layout(pad=0.4)
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
