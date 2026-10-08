import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.consolidation import yield_window_star, l2_and_energy_error, theta_y
ty = np.linspace(0, 2.0, 21)
fig, ax = plt.subplots(figsize=(6, 4))
for ba, c in ((1.5, "C0"), (2.0, "C1"), (3.0, "C2")):
    w = np.array([yield_window_star(ba, 0.3, t, n_p=1201) for t in ty])
    ax.plot(ty, w[:, 1], c, label=f"b/a = {ba}, p_hi")
    if np.nanmax(w[:, 0]) > 0: ax.plot(ty, w[:, 0], c + "--", label=f"b/a = {ba}, p_lo")
ax.axvline(theta_y(210000., .3, 1.2e-5, 50., 250.), color="k", lw=.8, ls=":")
ax.set_xlabel("Theta_y = E alpha dT / ((1 - nu) sigma_y)"); ax.set_ylabel("p / sigma_y")
ax.set_title("Pression admissible sans plastification, nu = 0.3"); ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(ROOT / "figures/consolidation_fenetre_adim.png", dpi=150)
ns = np.array([2, 4, 8, 16, 32, 64]); res = np.array([l2_and_energy_error(n, 210000., 100., 1000., 10.) for n in ns])
fig, ax = plt.subplots(figsize=(6, 4)); h = 1000. / ns
for k, lab in ((0, "L2, déplacement (ordre 2)"), (1, "énergie (ordre 1)"), (3, "contrainte au bord d'élément (ordre 1)")):
    ax.loglog(h, res[:, k] / res[0, k], "o-", label=lab)
ax.loglog(h, np.maximum(res[:, 2], 1e-16) / res[0, 3], "s--", label="contrainte au milieu (exacte)")
from matplotlib.ticker import NullFormatter; ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel("h (mm)"); ax.set_ylabel("erreur / erreur à n = 2"); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
ax.set_title("Barre 1D : l'erreur nodale est nulle, l'erreur de champ ne l'est pas")
fig.tight_layout(); fig.savefig(ROOT / "figures/consolidation_convergence_1d.png", dpi=150)
print("deux figures écrites")
