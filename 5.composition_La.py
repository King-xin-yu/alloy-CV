# -*- coding: utf-8 -*-
"""La content of datasets A and B.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "5.composition_La"

fig, ax = plt.subplots(figsize=(3.2, 2.6))
ac.panel_la(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
