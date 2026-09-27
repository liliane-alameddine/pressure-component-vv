"""Tests du 14 août : chaque test encode un résultat de la séance."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytic import (kirsch_stresses, kirsch_hole_hoop, kirsch_hole_hoop_biaxial,
                          kirsch_decay, ellipse_kt, polar_to_cartesian)
from src.kirsch_symbolic import certify
from src.stress import principal_stresses

A, S = 10.0, 100.0


def test_certification_symbolique_cinq_niveaux():
    assert certify()


def test_bord_libre_et_points_remarquables():
    th = np.linspace(0.0, 2 * np.pi, 37)
    s_rr, s_tt, s_rt = kirsch_stresses(A, th, A, S)
    assert np.allclose(s_rr, 0.0, atol=1e-9) and np.allclose(s_rt, 0.0, atol=1e-9)
    assert np.allclose(s_tt, kirsch_hole_hoop(th, S))
    assert abs(kirsch_hole_hoop(np.pi / 2, S) - 3 * S) < 1e-9
    assert abs(kirsch_hole_hoop(0.0, S) + S) < 1e-9
    assert abs(np.mean(kirsch_hole_hoop(th[:-1], S)) - S) < 1e-9   # moyenne = sigma


def test_independance_du_rayon_du_trou():
    for a in (0.1, 1.0, 100.0):
        _, s_tt, _ = kirsch_stresses(a, np.pi / 2, a, S)
        assert abs(s_tt - 3 * S) < 1e-9


def test_champ_lointain_uniaxial_exercice_2():
    th = np.linspace(0.0, 2 * np.pi, 50)
    s_rr, s_tt, s_rt = kirsch_stresses(1e6 * A, th, A, S)
    s_xx, s_yy, s_xy = polar_to_cartesian(s_rr, s_tt, s_rt, th)
    assert np.allclose(s_xx, S, atol=1e-6) and np.allclose(s_yy, 0.0, atol=1e-6)
    assert np.allclose(s_xy, 0.0, atol=1e-6)


def test_decroissance_partie_5():
    for x, val in ((1, 3.0), (1.5, 1.518), (2, 1.219), (3, 1.074), (5, 1.022), (10, 1.005),
                   (20, 1.001)):
        assert abs(kirsch_decay(x) - val) < 1e-3
    # exercice 4 : 1 % entre 7 et 8 rayons
    x1 = next(x for x in np.linspace(1, 20, 1901) if kirsch_decay(x) - 1 < 0.01)
    assert 7.0 < x1 < 8.0
    assert abs(x1 - 7.27) < 0.011                                          # 7.6 corrigé, journal


def test_superposition_exercice_3():
    th = np.linspace(0.0, np.pi, 100)
    assert np.allclose(kirsch_hole_hoop_biaxial(th, S, S), 2 * S)           # K_t = 2
    shear = kirsch_hole_hoop_biaxial(th, S, -S)
    assert abs(np.max(np.abs(shear)) - 4 * S) < 1e-9                       # K_t = 4
    assert abs(shear[0] + 4 * S) < 1e-9                                  # compression à 0
    assert abs(float(kirsch_hole_hoop_biaxial(np.pi / 2, S, -S)) - 4 * S) < 1e-9   # traction à 90


def test_concentration_contre_singularite_exercice_5():
    assert abs(ellipse_kt(1.0, 1.0) - 3.0) < 1e-12
    assert abs(ellipse_kt(1.0, 0.25) - 5.0) < 1e-12
    assert abs(ellipse_kt(1.0, 0.01) - 21.0) < 1e-12
    assert abs(ellipse_kt(1.0, 0.1) - 7.32) < 0.01
    assert ellipse_kt(1.0, 1e-12) > 1e6                                     # diverge


def test_principales_au_bord_coherentes_avec_lundi():
    """Sur le bord, l'état est uniaxial circonférentiel : principales (s_tt, 0, 0)."""
    th = np.pi / 2
    s_rr, s_tt, s_rt = kirsch_stresses(A, th, A, S)
    s_xx, s_yy, s_xy = polar_to_cartesian(s_rr, s_tt, s_rt, th)
    p = principal_stresses([[s_xx, s_xy, 0], [s_xy, s_yy, 0], [0, 0, 0]])
    assert abs(p[0] - 3 * S) < 1e-9 and abs(p[2]) < 1e-9
