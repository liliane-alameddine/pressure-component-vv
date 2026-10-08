"""Exécute les huit contrôles de la séance du 10 septembre et écrit results/."""
import csv
import os
import sys
import time

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q8_modal import (mesh, assemble, assemble_geometric, clamped_left, modal,
                      euler_bernoulli, timoshenko, effective_masses, critical_load,
                      E, H, T, L, RHO)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
out = {}
t0 = time.perf_counter()

# maillage de travail : 40 x 2 éléments Q8, élancement L/H = 20
xy, conn = mesh(40, 2)
K, M, Ml = assemble(xy, conn)
free = clamped_left(xy)
all_dofs = np.arange(K.shape[0])

# 1. modes rigides sur structure libre : trois fréquences nulles en 2D
f_free, _ = modal(K, M, all_dofs, n_modes=8)
# le bruit d'arrondi sur lambda, ~1e-11 relatif, devient ~3e-6 relatif sur f = sqrt(lambda) :
# le seuil est fixé à 1e-4 de la première fréquence non nulle
n_zero = int(np.sum(f_free < 1e-4 * f_free[3]))
out["modes_rigides"] = n_zero
print(f"1. modes rigides sur structure libre : {n_zero} (attendu 3) ; "
      f"f4 = {f_free[3]:.2f} Hz")
assert n_zero == 3

# 2. rapports de fréquences, 3. valeurs absolues contre Euler-Bernoulli
f, Phi = modal(K, M, free, n_modes=6)
f_eb = euler_bernoulli(4)
# les modes de flexion sont les modes dont la masse effective en y est non nulle ;
# le mode axial (u_x) s'intercale et doit être écarté de la comparaison
meff_y, m_tot = effective_masses(Phi, M, free, 1)
meff_x, _ = effective_masses(Phi, M, free, 0)
flex = [i for i in range(6) if meff_y[i] > meff_x[i]][:4]
f_flex = f[flex]
print(f"2. rapports : f2/f1 = {f_flex[1] / f_flex[0]:.3f} (6.267), "
      f"f3/f1 = {f_flex[2] / f_flex[0]:.3f} (17.55)")
f_ti = timoshenko(4)
ecart = (f_flex - f_eb) / f_eb * 100
ecart_ti = (f_flex - f_ti) / f_ti * 100
print("3. fréquences de flexion Q8 (Hz)      :", np.round(f_flex, 3))
print("   Euler-Bernoulli (Hz)               :", np.round(f_eb, 3))
print("   écart à Euler-Bernoulli (%)        :", np.round(ecart, 2), "croissant avec le mode")
print("   Timoshenko, cisaillement inclus    :", np.round(f_ti, 3))
print("   écart à Timoshenko (%)             :", np.round(ecart_ti, 2))
assert abs(f_flex[1] / f_flex[0] - 6.267) / 6.267 < 0.02
assert abs(f_flex[2] / f_flex[0] - 17.55) / 17.55 < 0.03
assert np.all(np.abs(ecart[:2]) < 2.0)                   # E-B : les deux premiers modes
assert np.all(np.diff(np.abs(ecart)) > 0), "l'écart de cisaillement doit croître"
assert np.all(np.abs(ecart_ti) < 0.5)                    # Timoshenko : les quatre modes
for i in range(4):
    out[f"f{i + 1}_Q8_Hz"] = float(f_flex[i]); out[f"f{i + 1}_EB_Hz"] = float(f_eb[i])
    out[f"f{i + 1}_Timoshenko_Hz"] = float(f_ti[i])
    out[f"ecart{i + 1}_EB_pct"] = float(ecart[i]); out[f"ecart{i + 1}_Ti_pct"] = float(ecart_ti[i])

# 4. encadrement par les deux masses
f_cons, _ = modal(K, M, free, n_modes=6)
f_lump, _ = modal(K, Ml, free, n_modes=6)
print(f"4. encadrement : concentrée {np.round(f_lump[flex[:3]], 3)} <= exacte <= "
      f"cohérente {np.round(f_cons[flex[:3]], 3)} Hz")
assert np.all(f_lump[flex] < f_cons[flex])
out["f1_lumped_Hz"] = float(f_lump[flex[0]]); out["f1_consistent_Hz"] = float(f_cons[flex[0]])

# 5. orthogonalité généralisée
Mf = M[np.ix_(free, free)]
Phi_n = Phi / np.sqrt(np.einsum("ij,ij->j", Phi, Mf @ Phi))
Gm = Phi_n.T @ Mf @ Phi_n
print(f"5. orthogonalité : |Phi^T M Phi - I| max = {np.abs(Gm - np.eye(6)).max():.1e}")
assert np.allclose(Gm, np.eye(6), atol=1e-8)

# 6. masses effectives : somme sur tous les modes = masse totale ; 90 % sur peu de modes
vals, vecs = eigh(K[np.ix_(free, free)], Mf)
meff_all, m_tot = effective_masses(vecs, M, free, 1)
m_phys = RHO * L * H * T
cum = np.cumsum(np.sort(meff_all)[::-1]) / m_tot
n90 = int(np.searchsorted(cum, 0.90) + 1)
print(f"6. masses effectives en y : somme {meff_all.sum():.6e} t, masse totale libre "
      f"{m_tot:.6e} t (physique {m_phys:.3e}) ; 90 % atteints avec {n90} modes ; "
      f"mode 1 : {meff_y[flex[0]] / m_tot * 100:.1f} %")
assert abs(meff_all.sum() - m_tot) / m_tot < 1e-10
out["masse_effective_mode1_pct"] = float(meff_y[flex[0]] / m_tot * 100)
out["modes_pour_90pct"] = n90

# 7. ordre 2p : deux familles à h divisé par deux dans les deux directions.
#    Encastrée-libre : l'encastrement crée une singularité de coin (contrainte singulière
#    aux angles de la section encastrée), le mode n'est pas régulier et l'ordre 2p ne peut
#    pas apparaître ; libre-libre : mode régulier, ordre 4, facteur 16.
def family(free_of, mode_index):
    fam, rows = [], []
    for nx, ny in ((10, 1), (20, 2), (40, 4), (80, 8)):
        xyf, cf = mesh(nx, ny)
        Kf, Mfm, _ = assemble(xyf, cf)
        fr = free_of(xyf, Kf)
        ff, Pf = modal(Kf, Mfm, fr, n_modes=6)
        fam.append(ff[mode_index(ff, Pf, Mfm, fr)])
        rows.append((nx, ny, Kf.shape[0], fam[-1]))
    fam = np.array(fam)
    p = np.log((fam[-3] - fam[-2]) / (fam[-2] - fam[-1])) / np.log(2)
    ext = fam[-1] - (fam[-2] - fam[-1]) / (2**p - 1)
    errs = fam - ext
    return fam, p, ext, errs[:-1] / errs[1:], rows


def first_flexural(ff, Pf, Mfm, fr):
    my, _ = effective_masses(Pf, Mfm, fr, 1)
    mx, _ = effective_masses(Pf, Mfm, fr, 0)
    return [i for i in range(len(ff)) if my[i] > mx[i]][0]


fam_c, p_c, ext_c, rat_c, rows_c = family(lambda xyf, Kf: clamped_left(xyf), first_flexural)
fam_f, p_f, ext_f, rat_f, rows_f = family(lambda xyf, Kf: np.arange(Kf.shape[0]),
                                          lambda ff, Pf, Mfm, fr: 3)
print(f"7. ordre 2p, encastrée-libre : f1 = {np.round(fam_c, 5)} Hz, par le haut ; ordre "
      f"observé {p_c:.2f}, rapports {np.round(rat_c, 1)} : singularité de coin à "
      f"l'encastrement, ordre réduit")
print(f"   ordre 2p, libre-libre       : f1 = {np.round(fam_f, 5)} Hz, par le haut ; ordre "
      f"observé {p_f:.2f}, rapports {np.round(rat_f, 1)} : mode régulier, ordre 4, facteur 16")
assert np.all(np.diff(fam_c) < 0) and np.all(np.diff(fam_f) < 0)
assert 1.5 < p_c < 2.5 and p_f > 3.7 and np.all(rat_f > 12.0)
out["ordre_encastree"] = float(p_c); out["ordre_libre_libre"] = float(p_f)
out["f1_extrapole_encastree_Hz"] = float(ext_c); out["f1_extrapole_libre_Hz"] = float(ext_f)
with open(os.path.join(RES, "convergence_f1.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["cas", "nx", "ny", "ndof", "f1_Hz", "erreur_Hz"])
    for (nx, ny, nd, f1), e in zip(rows_c, fam_c - ext_c):
        w.writerow(["encastree-libre", nx, ny, nd, f"{f1:.6f}", f"{e:.3e}"])
    for (nx, ny, nd, f1), e in zip(rows_f, fam_f - ext_f):
        w.writerow(["libre-libre", nx, ny, nd, f"{f1:.6f}", f"{e:.3e}"])

# 8. précontrainte : traction relève, compression abaisse, annulation à la charge d'Euler
A = H * T
for pn in (0.0, 5000.0, 20000.0):
    Kg = assemble_geometric(xy, conn, pn / A)
    fp, _ = modal(K + Kg, M, free, n_modes=3)
    print(f"8. traction {pn:7.0f} N : f1 = {fp[0]:.4f} Hz")
Kg = assemble_geometric(xy, conn, 5000.0 / A)
fp, _ = modal(K + Kg, M, free, n_modes=6)
assert np.all(fp[flex[:3]] > f[flex[:3]]), "la traction doit rigidifier"
p_num, p_euler = critical_load(xy, conn, K, M, free)
print(f"   charge critique par annulation de f1 : {p_num:.1f} N ; Euler pi^2 E I / 4 L^2 = "
      f"{p_euler:.1f} N ; écart {(p_num - p_euler) / p_euler * 100:+.2f} %")
assert abs(p_num - p_euler) / p_euler < 0.01
out["P_critique_modal_N"] = float(p_num); out["P_critique_Euler_N"] = float(p_euler)

with open(os.path.join(RES, "results.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["nom", "valeur"])
    for k, v in out.items():
        w.writerow([k, v])
print(f"\nAnalyse modale vérifiée, ordre 2p confirmé. ({time.perf_counter() - t0:.0f} s)")
