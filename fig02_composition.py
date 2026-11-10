# -*- coding: utf-8 -*-
"""composition distributions of datasets A and B (Fig. 2)."""

import matplotlib as mpl
import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "fig02_composition"

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["font.size"] = 10

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
ac.panel_ni(axes[0, 0])
ac.panel_si(axes[0, 1])
ac.panel_ce(axes[1, 0])
ac.panel_la(axes[1, 1])

fig.tight_layout(pad=0.6, w_pad=1.2, h_pad=0.8)
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"), bbox_inches="tight", pad_inches=0.02)
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
