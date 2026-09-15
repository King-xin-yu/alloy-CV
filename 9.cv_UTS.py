# -*- coding: utf-8 -*-
"""random vs composition-grouped 10-fold CV R2 for UTS.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "9.cv_UTS"

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ac.panel_cv_uts(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
