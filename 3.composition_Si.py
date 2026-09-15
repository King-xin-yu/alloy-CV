# -*- coding: utf-8 -*-
"""Si content of datasets A and B.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "3.composition_Si"

fig, ax = plt.subplots(figsize=(3.2, 2.6))
ac.panel_si(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
