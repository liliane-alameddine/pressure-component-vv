"""Tests du 13 août : chaque test encode un résultat de la séance."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytic import (lame_constants, lame_stresses, lame_axial, lame_displacement,
                          lame_von_mises, yield_pressure, thin_wall_hoop)
from src.lame_symbolic import certify
from src.stress import von_mises

A, B, P = 50.0, 80.0, 100.0


def test_certification_symbolique_trois_niveaux():
    assert certify()


def test_conditions_aux_limites_et_somme_invariante():
    s_rr_a, s_tt_a = lame_stresses(A, A, B, P)
    s_rr_b, _ = lame_stresses(B, A, B, P)
    assert abs(s_rr_a + P) < 1e-9 and abs(s_rr_b) < 1e-9
    r = np.linspace(A, B, 50)
    s_rr, s_tt = lame_stresses(r, A, B, P)
    assert np.allclose(s_rr + s_tt, 2 * lame_constants(A, B, P)[0])
    assert s_tt[0] == s_tt.max()                                       # maximale à l'alésage
    assert abs(s_tt_a - P * (A**2 + B**2) / (B**2 - A**2)) < 1e-9     # 228.2 MPa


def test_independance_du_materiau_michell():
    """Les contraintes dans le plan ne prennent aucun module ; le déplacement, oui."""
    import inspect
    sig = inspect.signature(lame_stresses).parameters
    assert "E" not in sig and "nu" not in sig
    u1 = lame_displacement(A, A, B, P, E=210000.0)
    u2 = lame_displacement(A, A, B, P, E=420000.0)
    assert abs(u1 / u2 - 2.0) < 1e-12                                  # u en 1/E


def test_exercice_3_trois_conditions_et_pression_d_amorcage():
    s_rr, s_tt = lame_stresses(A, A, B, P)
    for case, s_zz_ref in (("open", 0.0), ("closed", 64.10), ("plane_strain", 38.46)):
        assert abs(float(lame_axial(A, A, B, P, case=case)) - s_zz_ref) < 0.01
        vm = float(lame_von_mises(A, A, B, P, case=case))
        assert abs(vm - von_mises(np.diag([s_rr, s_tt, s_zz_ref]))) < 0.01
    py = {c: yield_pressure(A, B, 350.0, case=c) for c in ("open", "closed", "plane_strain")}
    assert py["closed"] > py["plane_strain"] > py["open"]
    assert 0.003 < (py["closed"] - py["plane_strain"]) / py["closed"] < 0.006   # 0.4 %


def test_von_mises_maximale_a_l_alesage_partie_6():
    r = np.linspace(A, B, 200)
    vm = lame_von_mises(r, A, B, P)
    assert np.all(np.diff(vm) < 0)                                     # strictement décroissante
    C1, C2 = lame_constants(A, B, P)
    k = C2 / r**2
    assert np.allclose(vm**2, 3 * k**2 + C1**2 * (1 - 2 * 0.3)**2)


def test_limite_paroi_mince_partie_4():
    """La table de la séance rapporte l'écart à la valeur mince : (exacte - mince) / mince."""
    for ratio, err_ref in ((1.05, 0.026), (1.10, 0.052), (1.20, 0.109), (1.50, 0.30), (2.0, 0.67)):
        b = A * ratio
        exact = lame_stresses(A, A, b, P)[1]
        thin = thin_wall_hoop(A, b - A, P)
        assert abs((exact - thin) / thin - err_ref) < 0.006
        assert exact > thin                                            # toujours non conservative


def test_plafond_de_pression_partie_5():
    _, s_tt_inf = lame_stresses(A, A, 1.0e6, P)
    assert abs(s_tt_inf - P) / P < 1e-4
    p_inf = yield_pressure(A, 1.0e6, 350.0)
    assert abs(p_inf - 350.0 / np.sqrt(3)) / (350.0 / np.sqrt(3)) < 1e-4   # 0.577 sigma_y
    # exercice 4 : la réponse de la séance situait 95 % de l'asymptote vers b/a = 3 ;
    # le calcul donne b/a = 4.48, et 89 % seulement à b/a = 3 (correction au journal)
    ratios = np.linspace(1.2, 6.0, 500)
    py = np.array([yield_pressure(A, A * x, 350.0) for x in ratios])
    x95 = ratios[np.argmax(py >= 0.95 * 350.0 / np.sqrt(3))]
    assert 4.3 < x95 < 4.7
    assert abs(yield_pressure(A, 3 * A, 350.0) / (350.0 / np.sqrt(3)) - 0.889) < 0.005


def test_composant_du_projet():
    """a = 50, b = 100, p = 20 MPa, acier : les nombres que le projet portera."""
    s_rr, s_tt = lame_stresses(50.0, 50.0, 100.0, 20.0)
    assert abs(s_tt - 33.333) < 1e-3 and abs(s_rr + 20.0) < 1e-9
    assert abs(float(lame_von_mises(50.0, 50.0, 100.0, 20.0)) - 46.26) < 0.01
    assert abs(yield_pressure(50.0, 100.0, 250.0) - 108.07) < 0.01
    assert abs(float(lame_displacement(50.0, 50.0, 100.0, 20.0)) - 9.0794e-3) < 1e-6
