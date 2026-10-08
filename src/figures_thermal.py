"""Séance du 15 août, partie 8.4 : les trois figures thermiques, sur le composant du
projet (a = 50, b = 100 mm, p = 20 MPa) avec un écart de 50 K, intérieur chaud."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytic import (temperature_field, thermal_stresses, thermal_axial_free,
                          combined_stresses, lame_von_mises)
from src.stress import von_mises
from src.stress import von_mises_field

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG, exist_ok=True)
a, b, p = 50.0, 100.0, 20.0
E, nu, al = 210000.0, 0.3, 1.2e-5
Ta, Tb = 350.0, 300.0
Tref = 0.5 * (Ta + Tb)
r = np.linspace(a, b, 300)

fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(r, temperature_field(r, a, b, Ta, Tb), label="logarithmique, exacte")
ax.plot(r, Ta + (Tb - Ta) * (r - a) / (b - a), "k--", label="linéaire naïve")
ax.set_xlabel("r (mm)"); ax.set_ylabel("T (K)"); ax.grid(True, lw=0.3); ax.legend()
ax.set_title("Profil permanent : la surface traversée croît avec r")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "thermique_profil.png"), dpi=150)

s_rr, s_tt = thermal_stresses(r, a, b, Ta, Tb, E, nu, al)
s_zz = thermal_axial_free(r, a, b, Ta, Tb, E, nu, al)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(r, s_rr, label="sigma_rr, nulle aux deux parois")
ax.plot(r, s_tt, label="sigma_theta, change de signe")
ax.plot(r, s_zz, "--", label="sigma_zz, libre axialement : = sigma_rr + sigma_theta")
ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("r (mm)"); ax.set_ylabel("MPa")
ax.grid(True, lw=0.3); ax.legend(fontsize=8); ax.set_title("Contraintes thermiques, 50 K, intérieur chaud")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "thermique_contraintes.png"), dpi=150)

fig, ax = plt.subplots(figsize=(6.2, 3.8))
vm_p = lame_von_mises(r, a, b, p)
for (TA, TB), lab in (((Ta, Tb), "pression + intérieur chaud"), ((Tb, Ta), "pression + intérieur froid")):
    sr, st, sz = combined_stresses(r, a, b, p, TA, TB, E, nu, al)
    S = np.zeros((len(r), 3, 3)); S[:, 0, 0] = sr; S[:, 1, 1] = st; S[:, 2, 2] = sz
    ax.plot(r, von_mises_field(S), label=lab)
ax.plot(r, vm_p, "k--", label="pression seule")
ax.set_xlabel("r (mm)"); ax.set_ylabel("sigma_vm (MPa)"); ax.grid(True, lw=0.3); ax.legend(fontsize=8)
ax.set_title("50 K : même l'intérieur chaud aggrave l'alésage, par excès de compression")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "thermique_superposition.png"), dpi=150)
dts = np.linspace(0, 60, 241)
vb = []
for d in dts:
    sb = combined_stresses(a, a, b, p, 300.0 + d, 300.0, E, nu, al)
    vb.append(von_mises(np.diag([float(x) for x in sb])))
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(dts, vb, label="alésage, pression + intérieur chaud")
ax.axhline(float(lame_von_mises(a, a, b, p)), color="k", ls="--", label="pression seule, 46.3 MPa")
ax.set_xlabel("T_a - T_b (K)"); ax.set_ylabel("sigma_vm à l'alésage (MPa)"); ax.grid(True, lw=0.3)
ax.legend(fontsize=8); ax.set_title("Fenêtre de soulagement : jusqu'à 35 K, optimum vers 18 K")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "thermique_fenetre.png"), dpi=150)
print("figures :", sorted(f for f in os.listdir(FIG) if f.startswith("thermique")))
