"""Séance du 13 août, partie 7.4 : les trois figures de Lamé, sur le composant du projet
(a = 50, b = 100 mm, p = 20 MPa) et sur le rapport des rayons."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytic import lame_stresses, lame_von_mises, yield_pressure

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG, exist_ok=True)
a, b, p = 50.0, 100.0, 20.0
r = np.linspace(a, b, 300)

s_rr, s_tt = lame_stresses(r, a, b, p)
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.plot(r, s_rr, label="sigma_rr, vaut -p à l'alésage et 0 au rebord")
ax.plot(r, s_tt, label="sigma_theta, maximale à l'alésage")
ax.plot(r, s_rr + s_tt, "k:", label="somme, constante 2 C1")
ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("r (mm)"); ax.set_ylabel("MPa")
ax.grid(True, lw=0.3); ax.legend(fontsize=8)
ax.set_title("Lamé, a = 50, b = 100 mm, p = 20 MPa")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lame_contraintes.png"), dpi=150)

fig, ax = plt.subplots(figsize=(6.2, 4.0))
for case, lab in (("open", "extrémités ouvertes"), ("closed", "extrémités fermées"),
                  ("plane_strain", "déformations planes, retenu")):
    ax.plot(r, lame_von_mises(r, a, b, p, case=case), label=lab)
ax.set_xlabel("r (mm)"); ax.set_ylabel("sigma_vm (MPa)"); ax.grid(True, lw=0.3)
ax.legend(fontsize=8); ax.set_title("Von Mises selon la condition d'extrémité")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lame_von_mises_extremites.png"), dpi=150)

ratios = np.linspace(1.02, 6.0, 400)
s_hoop = np.array([lame_stresses(a, a, a * x, 1.0)[1] for x in ratios])
p_y = np.array([yield_pressure(a, a * x, 250.0) for x in ratios])
fig, ax1 = plt.subplots(figsize=(6.2, 4.0))
ax1.plot(ratios, s_hoop, label="sigma_theta(a) / p")
ax1.axhline(1.0, color="k", ls="--", lw=0.8, label="asymptote : la pression elle-même")
ax1.set_ylim(0, 8); ax1.set_xlabel("b / a"); ax1.set_ylabel("sigma_theta(a) / p")
ax2 = ax1.twinx()
ax2.plot(ratios, p_y, color="C3", label="pression d'amorçage, sigma_y = 250 MPa")
ax2.axhline(250 / np.sqrt(3), color="C3", ls=":", lw=0.8)
ax2.set_ylabel("p_y (MPa)", color="C3")
ax1.grid(True, lw=0.3); ax1.legend(loc="upper center", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
ax1.set_title("Le plafond : épaissir sature, p_y tend vers 0.577 sigma_y")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lame_plafond_epaisseur.png"), dpi=150)
print("figures :", sorted(os.listdir(FIG)))
