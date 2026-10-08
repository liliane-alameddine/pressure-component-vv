"""Tests du 15 août : chaque test encode un résultat de la séance."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytic import (blocking_stress, temperature_field, heat_flux, thermal_stresses,
                          thermal_axial, thermal_axial_free, combined_stresses,
                          thermal_shock_parameter, lame_von_mises)
from src.thermal_symbolic import certify
from src.stress import von_mises

A, B, TA, TB = 50.0, 80.0, 300.0, 200.0
E, NU, AL, K = 210000.0, 0.3, 12e-6, 45.0


def test_certification_symbolique():
    assert certify()


def test_trois_facteurs_de_blocage_exercice_1():
    s = [blocking_stress(E, NU, AL, 100.0, n) for n in (1, 2, 3)]
    assert np.allclose(s, [-252.0, -360.0, -630.0])
    assert abs(s[2] / s[0] - 2.5) < 1e-12


def test_champ_logarithmique_et_flux():
    assert abs(temperature_field(A, A, B, TA, TB) - TA) < 1e-9
    assert abs(temperature_field(B, A, B, TA, TB) - TB) < 1e-9
    r_mid = 0.5 * (A + B)
    assert abs(temperature_field(r_mid, A, B, TA, TB) - 0.5 * (TA + TB)) > 1.0
    for r in (A, r_mid, B):
        assert abs(heat_flux(r, A, B, TA, TB, K) * r - heat_flux(A, A, B, TA, TB, K) * A) < 1e-9


def test_bords_libres_signe_et_autocontrainte():
    s_rr_a, s_tt_a = thermal_stresses(A, A, B, TA, TB, E, NU, AL)
    s_rr_b, s_tt_b = thermal_stresses(B, A, B, TA, TB, E, NU, AL)
    assert abs(s_rr_a) < 1e-6 and abs(s_rr_b) < 1e-6
    assert s_tt_a < 0.0 < s_tt_b                                    # intérieur chaud comprimé
    r = np.linspace(A, B, 2001)
    _, s_tt = thermal_stresses(r, A, B, TA, TB, E, NU, AL)
    assert abs(np.trapezoid(s_tt, r)) / (abs(s_tt).max() * (B - A)) < 1e-6


def test_independance_de_la_taille():
    r = np.linspace(A, B, 50)
    assert np.allclose(thermal_stresses(r, A, B, TA, TB, E, NU, AL)[1],
                       thermal_stresses(10 * r, 10 * A, 10 * B, TA, TB, E, NU, AL)[1])


def test_exercice_4_et_limite_mince():
    s_a = float(thermal_stresses(A, A, B, TA, TB, E, NU, AL)[1])
    s_b = float(thermal_stresses(B, A, B, TA, TB, E, NU, AL)[1])
    thin = E * AL * 100.0 / (2 * (1 - NU))                          # 180 MPa
    assert abs(thin - 180.0) < 1e-9
    assert abs(s_a + 207.79) < 0.01 and abs(s_b - 152.21) < 0.01   # épais : dissymétrique
    assert abs((s_b - s_a) - 360.0) < 1e-6                           # écart des parois = 2 x mince
    _, s_thin = thermal_stresses(100.0, 100.0, 101.0, TA, TB, E, NU, AL)
    assert abs(float(s_thin) + thin) / thin < 0.02


def test_exercice_5_soulage_ou_aggrave():
    p, T_ref = 100.0, 250.0
    vm_p = float(lame_von_mises(A, A, B, p))
    s = combined_stresses(A, A, B, p, TA, TB, E, NU, AL, T_ref)
    vm_hot_in = von_mises(np.diag([float(x) for x in s]))
    s_inv = combined_stresses(A, A, B, p, TB, TA, E, NU, AL, T_ref)
    vm_hot_out = von_mises(np.diag([float(x) for x in s_inv]))
    assert vm_hot_in < vm_p < vm_hot_out                            # soulage, puis aggrave


def test_parametre_de_choc_thermique_exercice_6():
    steel = thermal_shock_parameter(500.0, 210000.0, 0.30, 12e-6)
    alu = thermal_shock_parameter(300.0, 70000.0, 0.33, 23e-6)
    alumina = thermal_shock_parameter(300.0, 380000.0, 0.22, 8e-6)
    # R en kelvins : acier 139, aluminium 125, alumine 77 ; l'alumine, résistante,
    # est dernière par son module ; l'aluminium perd sur l'acier par sa dilatation
    assert steel > alu > alumina
    assert abs(steel - 138.9) < 0.1 and abs(alumina - 77.0) < 0.1


def test_axial_libre_independant_de_T_ref():
    """Cylindre libre axialement : sigma_zz = sigma_rr + sigma_tt (résultat classique),
    résultante axiale nulle, et plus aucune dépendance à T_ref."""
    r = np.linspace(A, B, 201)
    s_rr, s_tt = thermal_stresses(r, A, B, TA, TB, E, NU, AL)
    s_zz = thermal_axial_free(r, A, B, TA, TB, E, NU, AL)
    assert np.allclose(s_zz, s_rr + s_tt, atol=1e-3)
    assert abs(np.trapezoid(s_zz * r, r)) / (abs(s_zz).max() * B**2) < 1e-5
    ps = [float(thermal_axial(A, A, B, TA, TB, E, NU, AL, T)) for T in (200.0, 250.0, 300.0)]
    assert max(ps) - min(ps) > 200.0                        # déformations planes pures : T_ref pèse


def test_composant_fenetre_de_soulagement():
    """a = 50, b = 100, p = 20 MPa : l'intérieur chaud soulage l'alésage jusqu'à 35 K
    environ, au mieux vers 18 K ; à 50 K il aggrave, la compression thermique dépasse."""
    a, b, p = 50.0, 100.0, 20.0
    vm_p = float(lame_von_mises(a, a, b, p))

    def vm_bore(dT):
        s = combined_stresses(a, a, b, p, 300.0 + dT, 300.0, E, NU, AL)
        return von_mises(np.diag([float(x) for x in s]))
    dts = np.linspace(0.0, 60.0, 601)
    v = np.array([vm_bore(d) for d in dts])
    assert abs(vm_p - 46.26) < 0.01
    assert 17.0 < dts[v.argmin()] < 18.5 and abs(v.min() - 25.4) < 0.2
    assert 34.5 < dts[np.where(v < vm_p)[0].max()] < 35.5
    assert vm_bore(50.0) > vm_p
