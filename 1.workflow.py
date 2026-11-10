# -*- coding: utf-8 -*-
"""workflow of the study."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text

import alloycv_common as ac

NAME = "1.workflow"

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["font.size"] = 12

fig, ax = plt.subplots(figsize=(8.2, 9.6))
ac.panel_workflow(ax)

for t in fig.findobj(Text):
    t.set_fontweight("bold")
    t.set_fontsize(12)

fig.tight_layout(pad=0.4)
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"), bbox_inches="tight", pad_inches=0.02)
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
