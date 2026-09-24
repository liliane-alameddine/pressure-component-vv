"""Tests du 12 août : cinématique, loi de comportement, hypothèses planes, solveur 1D."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.elasticity import (small_strain, green_lagrange, lame_constants, hooke_isotropic,
                            hooke_decoupled, plane_strain_sigma_zz, plane_stress_eps_zz,
                            plane_equivalent_constants, cylinder_axial_stress)
from src.fem1d import assemble, solve_fixed_free, element_stress, element_stiffness
from src.stress import von_mises

E, NU = 210000.0, 0.30


def test_rotation_rigide_exercice_1():
    th = np.radians(20.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    grad_u = R - np.eye(3)
    assert np.allclose(green_lagrange(grad_u), 0.0)                 # mesure exacte : nulle
    eps_lin = small_strain(grad_u)
    assert abs(eps_lin[0, 0] - (np.cos(th) - 1)) < 1e-12            # -6.0 % apparent
    assert abs(eps_lin[0, 0] + 0.0603) < 1e-3


def test_constantes_et_exemple_partie_5():
    G, lam, K = lame_constants(E, NU)
    assert abs(G - 80769.2) < 0.1 and abs(lam - 121153.8) < 0.1 and abs(K - 175000.0) < 1e-6
    eps = np.array([[1.0e-3, 2.0e-4, 0], [2.0e-4, -5.0e-4, 0], [0, 0, 0]])     # déf. planes
    s = hooke_isotropic(eps, E, NU)
    assert abs(s[0, 0] - 222.1) < 0.1 and abs(s[1, 1] + 20.2) < 0.1
    assert abs(s[2, 2] - 60.6) < 0.1 and abs(s[0, 1] - 32.3) < 0.1
    assert abs(s[2, 2] - NU * (s[0, 0] + s[1, 1])) < 1e-9             # sigma_zz = nu (sxx+syy)
    assert np.allclose(hooke_decoupled(eps, E, NU), s)                 # forme découplée


def test_von_mises_planes_contre_deformations_planes():
    eps = np.array([[1.0e-3, 2.0e-4, 0], [2.0e-4, -5.0e-4, 0], [0, 0, 0]])
    s = hooke_isotropic(eps, E, NU)
    vm_ps = von_mises(s)                                              # déformations planes
    s_cp = s.copy(); s_cp[2, 2] = 0.0                                 # si l'on croyait sigma_zz = 0
    vm_cp = von_mises(s_cp)
    assert vm_ps < vm_cp and (vm_cp - vm_ps) / vm_ps > 0.05           # écart mesurable


def test_hypotheses_planes_et_equivalence():
    assert abs(plane_strain_sigma_zz(222.1, -20.2, NU) - 60.57) < 0.01
    assert abs(plane_stress_eps_zz(100.0, 50.0, E, NU) + 0.3 * 150 / E) < 1e-15
    Ep, nup = plane_equivalent_constants(E, NU)
    assert abs(Ep - E / 0.91) < 1e-9 and abs(nup - 3 / 7) < 1e-12


def test_exercice_3_trois_conditions_extremite():
    p, a, b = 100.0, 50.0, 80.0
    s_open = cylinder_axial_stress(p, a, b, NU, "open")
    s_closed = cylinder_axial_stress(p, a, b, NU, "closed")
    s_ps = cylinder_axial_stress(p, a, b, NU, "plane_strain")
    assert s_open == 0.0 and abs(s_closed - 64.10) < 0.01 and abs(s_ps - 38.46) < 0.01
    assert abs(s_ps / s_closed - 2 * NU) < 1e-12                       # 0.6 pour nu = 0.3
    # Lamé à la paroi intérieure : sigma_r = -p, sigma_theta = p (a^2 + b^2)/(b^2 - a^2)
    s_r, s_t = -p, p * (a**2 + b**2) / (b**2 - a**2)
    vm = {c: von_mises(np.diag([s_r, s_t, s])) for c, s in
          (("open", s_open), ("closed", s_closed), ("plane_strain", s_ps))}
    assert vm["closed"] < vm["plane_strain"] < vm["open"]
    # 284.23, 285.39, 291.37 MPa : l'écart fermé / déformations planes vaut 0.4 %, non
    # « quelques pour cent » comme la réponse de la séance l'estimait ; ouvert / fermé 2.5 %
    gap = (vm["plane_strain"] - vm["closed"]) / vm["closed"]
    assert 0.003 < gap < 0.006
    assert (vm["open"] - vm["closed"]) / vm["closed"] > 0.02


def test_matrice_singuliere_exercice_4():
    ke = element_stiffness(E, 100.0, 250.0)
    assert np.allclose(ke @ np.ones(2), 0.0) and np.allclose(ke, ke.T)
    assert np.linalg.eigvalsh(ke).min() > -1e-9                         # semi-définie positive


def test_solveur_exact_aux_noeuds_et_superconvergence():
    A, L, f = 100.0, 1000.0, 10.0
    for n in (2, 4, 8, 16):
        u, R = solve_fixed_free(n, E, A, L, f)
        x = np.linspace(0, L, n + 1)
        assert np.abs(u - (f / (E * A)) * (L * x - x**2 / 2)).max() < 1e-12   # exact aux nœuds
        x_mid = (np.arange(n) + 0.5) * L / n
        assert np.abs(element_stress(u, n, E, L) - f * (L - x_mid) / A).max() < 1e-9
        assert abs(R + f * L) < 1e-6                                     # réaction = -f L


def test_trois_methodes_conditions_essentielles_exercice_6():
    A, L, f = 100.0, 1000.0, 10.0
    u_el, R_el = solve_fixed_free(8, E, A, L, f, "elimination")
    u_lg, R_lg = solve_fixed_free(8, E, A, L, f, "lagrange")
    assert np.allclose(u_lg, u_el) and abs(R_lg - R_el) < 1e-6
    pens = (1e2, 1e6, 1e10, 1e14)
    errs = []
    for pen in pens:
        u_pn, _ = solve_fixed_free(8, E, A, L, f, "penalty", penalty=pen)
        errs.append(np.abs(u_pn - u_el).max())
    # l'erreur décroît en 1/alpha : 3e-4, 3e-8, 3e-12, 5e-16 ; sur ce système à neuf
    # inconnues résolu par LU, la remontée par l'arrondi annoncée par la séance n'apparaît
    # pas même à un conditionnement de 1e21 ; elle exige un grand système ou un solveur
    # itératif, ce que la séance du 19 août montrera
    assert errs[0] > 1e-4 and errs[2] < 1e-11
    K = assemble(8, E, A, L)
    conds = [np.linalg.cond(K + np.diag([pen * K.max()] + [0] * 8)) for pen in pens]
    assert all(c2 / c1 > 1e3 for c1, c2 in zip(conds, conds[1:]))         # cond croît avec alpha
