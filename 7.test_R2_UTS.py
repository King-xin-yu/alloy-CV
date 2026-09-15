# -*- coding: utf-8 -*-
"""held-out test R2 of the four models for UTS.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "7.test_R2_UTS"

fig, ax = plt.subplots(figsize=(5.6, 2.8))
ac.panel_test_r2_uts(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
