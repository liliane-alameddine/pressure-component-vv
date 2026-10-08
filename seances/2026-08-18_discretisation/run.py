"""Exécution de la séance du 18 août. Lancer depuis la racine du dépôt :
python seances/2026-08-18_discretisation/run.py"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.fem1d import element_stiffness, consistent_load
from src.isoparametric import *

print("=== Partie 1 : les trois exigences ===")
for name, shape, nodes, deg in (("linéaire", shape_linear, [-1, 1], 1),
                                ("quadratique", shape_quadratic, [-1, 0, 1], 2)):
    print(f"  {name:12s} nodale {check_shape_properties(shape, nodes, deg)[0]:.1e}, partition "
          f"{check_shape_properties(shape, nodes, deg)[1]:.1e}, complétude degré {deg} "
          f"{check_shape_properties(shape, nodes, deg)[2]:.1e}")

print("\n=== Partie 5 : recoupement des trois voies, barre E = 210000, A = 100, L = 500 ===")
E, A, x1, x2 = 210000.0, 100.0, 0.0, 500.0
ke_elem = element_stiffness(E, A, x2 - x1)                    # 12 août, équilibre élémentaire
ke_var = E * A * (x2 - x1) * np.outer([-1 / 500, 1 / 500], [-1 / 500, 1 / 500])   # int B^T EA B dx
ke_iso = element_stiffness_isoparametric(E, A, x1, x2, n_gauss=1)
print(f"  voie élémentaire   : {ke_elem[0].tolist()}")
print(f"  voie variationnelle: {ke_var[0].tolist()}")
print(f"  voie isoparamétrique, 1 point de Gauss : {ke_iso[0].tolist()}")
print(f"  écart max entre les trois : {max(np.abs(ke_elem-ke_var).max(), np.abs(ke_elem-ke_iso).max()):.1e}")
for n in (2, 3):
    print(f"  {n} points : écart {np.abs(element_stiffness_isoparametric(E, A, x1, x2, n) - ke_elem).max():.1e} (aucun gain)")
print(f"  jacobien = {jacobian(shape_linear, [x1, x2], 0.0):.1f} = L_e / 2 ; mode rigide K 1 = {ke_iso @ np.ones(2)}")

print("\n=== Partie 6 : vecteur de charge cohérent, p = 10 N/mm ===")
fe = element_load_consistent(10.0, x1, x2)
print(f"  linéaire : {fe} (12 août : {consistent_load(1, 10.0, 500.0)})")
fq = load_quadratic(10.0, 500.0)
print(f"  quadratique : {fq}, somme {fq.sum():.1f} = p L, rapport médian / extrémité = {fq[1]/fq[0]:.3f}")

print("\n=== Partie 7 : points de Barlow ===")
print(f"  linéaire    : zéros de l'erreur de déformation {strain_error_zeros([-1, 1])}, Gauss ordre 1 {barlow_points(1)}")
print(f"  quadratique : {strain_error_zeros([-1, 0, 1])}, Gauss ordre 2 {barlow_points(2)}  (1/sqrt3 = {1/np.sqrt(3):.6f})")

print("\n=== Partie 8 : suffisance de rang ===")
for name, ng, ns, nd, nr in (("barre 2 noeuds, 1 pt", 1, 1, 2, 1), ("Q4, 1 pt", 1, 3, 8, 3),
                             ("Q4, 4 pts", 4, 3, 8, 3), ("Q8, 4 pts", 4, 3, 16, 3),
                             ("Q8, 9 pts", 9, 3, 16, 3), ("T3, 1 pt", 1, 3, 6, 3)):
    r, req, para = rank_sufficiency(ng, ns, nd, nr)
    print(f"  {name:22s} rang {r:2d} / requis {req:2d} : {para} mode(s) parasite(s)")

print("\n=== Partie 9 : conditionnement ===")
print(f"  {'n':>6} {'kappa numérique':>16} {'kappa exact':>14} {'kappa / n^2':>12}")
for n in (10, 20, 40, 80, 160, 320):
    k = condition_number(n)
    print(f"  {n:6d} {k:16.4e} {condition_number_exact(n):14.4e} {k/n**2:12.4f}")
print(f"  4 / pi^2 = {4/np.pi**2:.4f} ; n = 10000 : kappa = {condition_number_exact(10000):.2e}, "
      f"chiffres perdus {digits_lost(condition_number_exact(10000)):.1f} sur 16")

print("\n=== Exercices 2, 3, 4 ===")
for xm in (50.0, 20.0, 10.0):
    J = jacobian_quadratic_midnode(0.0, xm, 100.0, np.array([-1.0, 0.0, 1.0]))
    print(f"  ex. 2, noeud médian à {xm:4.0f} mm : J(-1, 0, 1) = {J} -> {'valide' if J.min() > 0 else 'INVALIDE'}")
lo, hi = midnode_valid_range(0.0, 100.0)
print(f"  ex. 2 : J > 0 partout ssi noeud médian dans ]{lo:.0f}, {hi:.0f}[ mm, le quart central ; "
      f"règle du tiers central ]{100/3:.1f}, {200/3:.1f}[ avec marge")
f = lambda xi: 1 + 2 * xi + 3 * xi**2 + 4 * xi**3
print(f"  ex. 3 : 1 point {quadrature(f, 1):.4f}, 2 points {quadrature(f, 2):.4f}, exact 4 "
      f"(le terme 3 xi^2 vaut 2, manqué par un point)")
kv1 = element_stiffness_variable_section(E, 100.0, 300.0, x1, x2, 1)
kv2 = element_stiffness_variable_section(E, 100.0, 300.0, x1, x2, 2)
print(f"  ex. 4 : section 100 -> 300, K[0,0] 1 point {kv1[0,0]:.3f}, 2 points {kv2[0,0]:.3f}, "
      f"section moyenne {E*200/500:.3f}")
