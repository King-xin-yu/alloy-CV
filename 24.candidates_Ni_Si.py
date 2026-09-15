# -*- coding: utf-8 -*-
"""candidate compositions in the Ni-Si plane.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "24.candidates_Ni_Si"

fig, ax = plt.subplots(figsize=(3.6, 3.2))
ac.panel_cand_ni_si(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
