# -*- coding: utf-8 -*-
"""Ni content of datasets A and B.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "2.composition_Ni"

fig, ax = plt.subplots(figsize=(3.2, 2.6))
ac.panel_ni(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
