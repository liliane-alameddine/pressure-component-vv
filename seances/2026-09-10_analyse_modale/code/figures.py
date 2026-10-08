"""Figures de la séance du 10 septembre, depuis les mêmes calculs que results/."""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q8_modal import (mesh, assemble, assemble_geometric, clamped_left, modal,
                      effective_masses, H, T, L, E)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

# 1. convergence des deux familles, log-log, avec les pentes 2 et 4
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "convergence_f1.csv"))))
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for cas, mk, lab in (("encastree-libre", "o-", "encastrée-libre : singularité de coin"),
                     ("libre-libre", "s-", "libre-libre : mode régulier")):
    r = [x for x in rows if x["cas"] == cas][:-1]          # le dernier niveau sert de base
    h = np.array([L / int(x["nx"]) for x in r]); e = np.abs([float(x["erreur_Hz"]) for x in r])
    ax.loglog(h, e, mk, label=lab)
    if cas == "libre-libre":
        ax.loglog(h, e[0] * (h / h[0])**4, "k:", label="pente 4 (ordre 2p)")
    else:
        ax.loglog(h, e[0] * (h / h[0])**2, "k--", label="pente 2")
ax.set_xlabel("h (mm)"); ax.set_ylabel("erreur sur f1 (Hz), Richardson")
ax.grid(True, which="both", lw=0.3); ax.legend(fontsize=8)
ax.set_title("Ordre 2p sur les fréquences : atteint sans singularité, réduit avec")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "convergence_ordre_2p.png"), dpi=150)

# 2. les trois premiers modes de flexion (déformée de la fibre moyenne)
xy, conn = mesh(40, 2); K, M, Ml = assemble(xy, conn); free = clamped_left(xy)
f, Phi = modal(K, M, free, n_modes=6)
my, _ = effective_masses(Phi, M, free, 1); mx, _ = effective_masses(Phi, M, free, 0)
flex = [i for i in range(6) if my[i] > mx[i]][:3]
mid = [k for k in range(len(xy)) if abs(xy[k, 1]) < 1e-9]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for n, i in enumerate(flex):
    U = np.zeros(K.shape[0]); U[free] = Phi[:, i]
    uy = U[1::2][mid]; uy = uy / np.abs(uy).max()
    ax.plot(xy[mid, 0], uy, label=f"mode {n + 1} : {f[i]:.1f} Hz")
ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("x (mm)"); ax.set_ylabel("u_y normalisé")
ax.legend(); ax.grid(True, lw=0.3); ax.set_title("Modes de flexion, poutre encastrée-libre Q8")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "modes_flexion.png"), dpi=150)

# 3. fréquence contre effort axial : traction relève, compression annule à la charge d'Euler
A = H * T; I = T * H**3 / 12; p_euler = np.pi**2 * E * I / (4 * L**2)
P = np.linspace(-0.98 * p_euler, 4 * p_euler, 26)
f1 = []
for pn in P:
    Kg = assemble_geometric(xy, conn, pn / A)
    f1.append(modal(K + Kg, M, free, n_modes=1)[0][0])
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.plot(P / p_euler, f1, "o-")
ax.axvline(-1, color="k", ls="--", lw=0.8); ax.axvline(0, color="k", lw=0.5)
ax.text(-0.95, max(f1) * 0.9, "flambement d'Euler : f1 = 0", fontsize=8)
ax.set_xlabel("effort axial / charge critique d'Euler (traction > 0)")
ax.set_ylabel("f1 (Hz)"); ax.grid(True, lw=0.3)
ax.set_title("Raideur géométrique : la précontrainte déplace la fréquence")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "frequence_vs_precontrainte.png"), dpi=150)
print("figures :", sorted(os.listdir(FIG)))
