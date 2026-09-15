# -*- coding: utf-8 -*-
"""schematic of a held-out composition in a sparse region of the training envelope.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "26.tree_schematic"

fig, ax = plt.subplots(figsize=(4.4, 4.2))
ac.panel_tree_schematic(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
