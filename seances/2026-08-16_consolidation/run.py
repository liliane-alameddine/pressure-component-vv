"""Exécution de la séance du 16 août, parties 3 à 6. Lancer depuis la racine du dépôt :
python -m seances.2026-08-16_consolidation.run n'est pas un nom de module valide, donc :
python seances/2026-08-16_consolidation/run.py"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.analytic import (lame_stresses, lame_displacement, lame_von_mises, kirsch_hole_hoop,
                          thermal_stresses, combined_stresses, yield_pressure)
from src.consolidation import *

print("=== Partie 3 : recoupement croisé Lamé / Kirsch ===")
sigma, a = 100.0, 10.0
s_rr_l, s_tt_l = lame_as_kirsch(a, a, sigma)
th = np.linspace(0, 2*np.pi, 73)
s_tt_k = kirsch_hole_hoop(th, sigma) + kirsch_hole_hoop(th - np.pi/2, sigma)
print(f"bord du trou : Lamé s_tt = {float(s_tt_l):.6f}, s_rr = {float(s_rr_l):.2e} ; "
      f"Kirsch s_tt min/max = {s_tt_k.min():.6f}/{s_tt_k.max():.6f}")
r = np.linspace(a, 20*a, 400)
gap = 0.0
for t in np.linspace(0, np.pi, 13):
    kr, kt, ks = kirsch_equibiaxial(r, t, a, sigma)
    lr, lt = lame_as_kirsch(r, a, sigma)
    gap = max(gap, np.abs(kr-lr).max(), np.abs(kt-lt).max(), np.abs(ks).max())
print(f"champ entier, r de a à 20a, 13 angles : écart max {gap:.2e} MPa")
for x in (2, 3, 5, 10, 100):
    b = x*a
    lt = float(lame_stresses(a, a, b, 0.0, -sigma)[1])
    print(f"  b/a = {x:4d} : Lamé fini s_tt(a) = {lt:8.3f}, écart à Kirsch "
          f"{100*(lt-2*sigma)/(2*sigma):7.4f} %  (formule a²/(b²-a²) = {100*lame_kirsch_gap(x):.4f} %)")
rr = np.linspace(50, 80, 100)
m = sum(lame_stresses(rr, 50, 80, 100.0)); t_ = sum(thermal_stresses(rr, 50, 80, 300, 200, 210000, .3, 12e-6))
print(f"trace mécanique : étendue {np.ptp(m):.2e} MPa (constante = {m[0]:.3f}) ; "
      f"trace thermique : de {t_[0]:.2f} à {t_[-1]:.2f} MPa")

print("\n=== Partie 4 : adimensionnement ===")
# 4a. Lamé : sigma/p ne dépend que de b/a. Composant et composant 10 fois plus grand, autre p.
s1 = np.array(lame_stresses(np.linspace(50, 100, 11), 50, 100, 20.0)) / 20.0
s2 = np.array(lame_stresses(np.linspace(500, 1000, 11), 500, 1000, 7.0)) / 7.0
print(f"Lamé, échelle x10 et p 20 -> 7 MPa : écart max sur sigma/p {np.abs(s1-s2).max():.1e}")
# 4b. Acier et aluminium au même Theta : même sigma/p partout, axial et von Mises compris.
cases = {"acier": dict(E=210000., alpha=1.2e-5, p=20.0, dT=50.0),
         "alu":   dict(E=70000.,  alpha=2.3e-5, p=5.0,  dT=None)}
nu = 0.3
Th = theta_p(210000., nu, 1.2e-5, 50.0, 20.0)
cases["alu"]["dT"] = Th * (1-nu) * 5.0 / (70000.*2.3e-5)
out = {}
for k, c in cases.items():
    rr = np.linspace(50, 100, 21)
    sr, st, sz = combined_stresses(rr, 50, 100, c["p"], c["dT"], 0.0, c["E"], nu, c["alpha"])
    vm = np.sqrt(.5*((st-sr)**2+(sr-sz)**2+(sz-st)**2))
    out[k] = np.array([sr, st, sz, vm]) / c["p"]
    print(f"  {k:5s} : E = {c['E']:.0f}, alpha = {c['alpha']:.1e}, p = {c['p']}, dT = {c['dT']:.3f} K, "
          f"Theta = {theta_p(c['E'], nu, c['alpha'], c['dT'], c['p']):.4f}, vm(a)/p = {out[k][3,0]:.5f}")
print(f"  écart max acier / aluminium sur sigma/p (4 composantes, 21 rayons) : "
      f"{np.abs(out['acier']-out['alu']).max():.1e}")
# 4c. Exercice 4.5 : p_y / sigma_y = F(b/a, nu, Theta_y)
print("  Exercice 4.5, fenêtre admissible p/sigma_y = [p_lo, p_hi] :")
print(f"  {'b/a':>4} {'Theta_y':>8} {'p_lo':>8} {'p_hi':>8}")
for ba in (1.5, 2.0, 3.0):
    for ty in (0.0, 0.36, 0.72, 1.2, 1.6):
        lo, hi = yield_window_star(ba, nu, ty)
        print(f"  {ba:4.1f} {ty:8.2f} {lo:8.4f} {hi:8.4f}")
ty_c = theta_y(210000., nu, 1.2e-5, 50.0, 250.0)
lo, hi = yield_window_star(2.0, nu, ty_c)
print(f"  composant (b/a = 2, dT = 50 K, Theta_y = {ty_c:.3f}) : p admissible de {250*lo:.2f} à {250*hi:.2f} MPa ;"
      f" sans thermique p_y = {yield_pressure(50, 100, 250):.2f} MPa")

print("\n=== Partie 5 : audit des unités (mm, N, MPa, K) ===")
ua = float(lame_displacement(50, 50, 100, 20.0)); st_a = float(lame_stresses(50, 50, 100, 20.0)[1])
sr_t, st_t = thermal_stresses(np.array([50., 100.]), 50, 100, 50, 0, 210000, .3, 1.2e-5)
print(f"  composant : s_tt(a) = {st_a:.2f} MPa, u(a) = {ua:.4e} mm, thermique 50 K : s_tt(a) = {st_t[0]:.1f}, s_tt(b) = {st_t[1]:.1f} MPa")
ua_bad = float(lame_displacement(50, 50, 100, 20.0, E=210.0))
st_bad = float(lame_stresses(50, 50, 100, 20.0)[1])
sr_tb, st_tb = thermal_stresses(np.array([50.]), 50, 100, 50, 0, 210, .3, 1.2e-5)
print(f"  piège E = 210 au lieu de 210000 : s_tt(a) = {st_bad:.2f} MPa (inchangé, Michell), "
      f"u(a) = {ua_bad:.3f} mm (x1000), thermique s_tt(a) = {st_tb[0]:.3f} MPa (/1000)")
sr_ta, st_ta = thermal_stresses(np.array([50.]), 50, 100, 50, 0, 210000, .3, 1.2e-2)
print(f"  piège alpha = 1.2e-2 : thermique s_tt(a) = {st_ta[0]:.0f} MPa (x1000)")

print("\n=== Partie 6 : convergence en norme du solveur 1D ===")
E, A, L, f = 210000.0, 100.0, 1000.0, 10.0
ns = [2, 4, 8, 16, 32, 64]
res = np.array([l2_and_energy_error(n, E, A, L, f) for n in ns])
p_l2, p_en, p_end = observed_orders(res[:, 0]), observed_orders(res[:, 1]), observed_orders(res[:, 3])
print(f"{'n':>4} {'err L2':>12} {'ordre':>6} {'err énergie':>12} {'ordre':>6} {'s milieu':>10} {'s bord':>10} {'ordre':>6}")
for i, n in enumerate(ns):
    o = lambda arr: "-" if i == 0 else f"{arr[i-1]:.3f}"
    print(f"{n:4d} {res[i,0]:12.4e} {o(p_l2):>6} {res[i,1]:12.4e} {o(p_en):>6} {res[i,2]:10.1e} {res[i,3]:10.4e} {o(p_end):>6}")
for n in (2, 8, 32):
    u, _ = solve_fixed_free(n, E, A, L, f)
    x = np.linspace(0, L, n+1)
    assert np.abs(u - exact_bar(x, E, A, L, f)[0]).max() < 1e-10
print("Erreur nodale nulle à 2, 8, 32 éléments : confirmé.")
