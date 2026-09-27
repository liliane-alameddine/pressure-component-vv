"""Séance du 14 août, partie 9.4 : les trois figures de Kirsch."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytic import kirsch_stresses, kirsch_hole_hoop, kirsch_decay

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG, exist_ok=True)
a, S = 10.0, 100.0

th = np.linspace(0, 2 * np.pi, 361)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(np.degrees(th), kirsch_hole_hoop(th, S) / S)
ax.axhline(3, color="k", ls=":", lw=0.8); ax.axhline(-1, color="k", ls=":", lw=0.8)
ax.set_xlabel("angle depuis la traction (degrés)"); ax.set_ylabel("sigma_theta / sigma")
ax.set_xticks(range(0, 361, 45)); ax.grid(True, lw=0.3)
ax.set_title("Bord du trou : 3 à 90 et 270 degrés, -1 dans l'axe")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "kirsch_bord.png"), dpi=150)

x = np.linspace(1, 20, 400)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.semilogy(x, kirsch_decay(x) - 1, label="perturbation sigma_theta / sigma - 1, theta = 90")
for xr, lab in ((3, "3 a : 7.4 %"), (5, "5 a : 2.2 %"), (7.27, "7.27 a : 1 %")):
    ax.axvline(xr, color="k", ls=":", lw=0.8)
    ax.text(xr + 0.15, 0.3, lab, fontsize=8, rotation=90, va="center")
ax.set_xlabel("r / a"); ax.set_ylabel("perturbation"); ax.grid(True, which="both", lw=0.3)
ax.legend(fontsize=8); ax.set_title("Saint-Venant chiffré : la perturbation décroît en 1/r^2")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "kirsch_decroissance.png"), dpi=150)

R, T = np.meshgrid(np.linspace(a, 4 * a, 200), np.linspace(0, 2 * np.pi, 361))
_, s_tt, _ = kirsch_stresses(R, T, a, S)
fig, ax = plt.subplots(figsize=(5.4, 5.0))
cf = ax.contourf(R / a * np.cos(T), R / a * np.sin(T), s_tt / S, levels=np.linspace(-1, 3, 33), cmap="RdBu_r")
ax.add_patch(plt.Circle((0, 0), 1.0, color="white")); ax.set_aspect("equal")
fig.colorbar(cf, ax=ax, label="sigma_theta / sigma")
ax.set_title("Champ circonférentiel, traction horizontale"); ax.set_xlabel("x / a"); ax.set_ylabel("y / a")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "kirsch_champ.png"), dpi=150)
print("figures :", sorted(f for f in os.listdir(FIG) if f.startswith("kirsch")))
