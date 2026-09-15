# -*- coding: utf-8 -*-
"""held-out test R2 of the four models for conductivity.
"""

import matplotlib.pyplot as plt

import alloycv_common as ac

NAME = "8.test_R2_conductivity"

fig, ax = plt.subplots(figsize=(5.6, 2.8))
ac.panel_test_r2_conductivity(ax)
fig.tight_layout()
fig.savefig(ac.FIG_DIR / (NAME + ".png"), dpi=600)
fig.savefig(ac.FIG_DIR / (NAME + ".pdf"))
print(NAME + ": wrote figures/" + NAME + ".png and figures/" + NAME + ".pdf")
