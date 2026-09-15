# -*- coding: utf-8 -*-
"""permutation importance by feature group for UTS.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "17.permutation_UTS"

fig, ax = plt.subplots(figsize=(3.4, 3.2))
ac.panel_perm_uts(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
