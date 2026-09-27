"""Tests du 11 août : chaque test encode un résultat théorique de la séance."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stress import (principal_stresses, principal_directions, invariants, hydrostatic,
                        deviatoric, J2, von_mises, max_shear, tresca, von_mises_field)

SIGMA = np.array([[100., 50., 0.], [50., -20., 0.], [0., 0., 30.]])


def test_exemple_partie_8_principales():
    p = principal_stresses(SIGMA)
    assert np.allclose(p, [40 + np.sqrt(6100), 30.0, 40 - np.sqrt(6100)])   # 118.10, 30, -38.10
    assert abs(p[0] - 118.10) < 0.005 and abs(p[2] + 38.10) < 0.005


def test_invariants_par_composantes_et_par_principales():
    p = principal_stresses(SIGMA)
    I1, I2, I3 = invariants(SIGMA)
    assert abs(I1 - 110.0) < 1e-10 and abs(I1 - p.sum()) < 1e-10
    assert abs(I3 + 135000.0) < 1e-6 and abs(I3 - p.prod()) < 1e-6
    assert abs(I2 - (p[0] * p[1] + p[1] * p[2] + p[2] * p[0])) < 1e-8


def test_invariance_par_rotation_exercice_1():
    th = 0.7
    Q = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    rot = Q @ SIGMA @ Q.T
    assert np.allclose(invariants(rot), invariants(SIGMA))
    assert abs(von_mises(rot) - von_mises(SIGMA)) < 1e-10


def test_directions_orthogonales_et_diagonalisation():
    vals, V = principal_directions(SIGMA)
    assert np.allclose(V.T @ V, np.eye(3))
    assert np.allclose(V.T @ SIGMA @ V, np.diag(vals))


def test_deviateur_et_von_mises_partie_8():
    assert abs(hydrostatic(SIGMA) - 110 / 3) < 1e-12
    assert abs(np.trace(deviatoric(SIGMA))) < 1e-12
    assert abs(J2(SIGMA) - 6133.33) < 0.01
    assert abs(von_mises(SIGMA) - 135.65) < 0.01
    p = principal_stresses(SIGMA)
    vm_p = np.sqrt(0.5 * ((p[0] - p[1])**2 + (p[1] - p[2])**2 + (p[2] - p[0])**2))
    assert abs(von_mises(SIGMA) - vm_p) < 1e-10
    assert abs(max_shear(SIGMA) - np.sqrt(6100)) < 1e-10                    # 78.10


def test_calibration_traction_et_cisaillement_pur_exercice_4():
    assert abs(von_mises(np.diag([250., 0., 0.])) - 250.0) < 1e-9
    tau = 100.0
    assert abs(von_mises(np.diag([tau, 0., -tau])) - tau * np.sqrt(3)) < 1e-9
    # limite en cisaillement / limite en traction : von Mises 1/sqrt(3), Tresca 1/2
    assert abs(tresca(np.diag([tau, 0., -tau])) - 2 * tau) < 1e-9


def test_insensibilite_a_la_pression_exercice_2():
    for p_hyd in (-300.0, 500.0):
        assert abs(von_mises(SIGMA + p_hyd * np.eye(3)) - von_mises(SIGMA)) < 1e-8
        assert np.allclose(deviatoric(SIGMA + p_hyd * np.eye(3)), deviatoric(SIGMA))


def test_piege_contraintes_planes_exercice_3():
    assert abs(max_shear(np.diag([120., 80., 0.])) - 60.0) < 1e-9     # et non 20
    assert abs(max_shear(np.diag([120., 0., -80.])) - 100.0) < 1e-9


def test_exercice_5_paroi_interieure():
    """sigma_z moyenne de sigma_theta et sigma_r : sigma_vm = sqrt(3)/2 (s + p)."""
    s, p = 33.333, 20.0
    sig = np.diag([-p, s, 0.5 * (s - p)])
    assert abs(von_mises(sig) - np.sqrt(3) / 2 * (s + p)) < 1e-9


def test_champ_vectorise_egal_au_scalaire():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 100, (500, 3, 3))
    field = 0.5 * (A + A.transpose(0, 2, 1))
    vf = von_mises_field(field)
    assert np.allclose(vf, [von_mises(t) for t in field])
    assert np.allclose(von_mises_field(np.stack([SIGMA] * 4)), von_mises(SIGMA))
