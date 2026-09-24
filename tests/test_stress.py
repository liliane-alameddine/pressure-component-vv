"""Tests du 10 août : chaque valeur est connue à la main avant d'être calculée."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stress import traction, normal_stress, shear_stress, check_symmetric

SIGMA = np.array([[100., 50., 0.], [50., -20., 0.], [0., 0., 30.]])


def test_exemple_chiffre_partie_7():
    n = [1., 1., 0.]
    t = traction(SIGMA, n)
    assert np.allclose(t, [150 / np.sqrt(2), 30 / np.sqrt(2), 0.0])   # 106.07, 21.21, 0
    assert abs(normal_stress(SIGMA, n) - 90.0) < 1e-9
    assert abs(shear_stress(SIGMA, n) - 60.0) < 1e-9
    assert abs(np.linalg.norm(t)**2 - (90.0**2 + 60.0**2)) < 1e-9     # Pythagore


def test_facette_principale_exercice_1():
    assert abs(normal_stress(SIGMA, [0., 0., 1.]) - 30.0) < 1e-9
    assert abs(shear_stress(SIGMA, [0., 0., 1.])) < 1e-9


def test_traction_simple_45_degres_exercice_3():
    uni = np.diag([200., 0., 0.])
    assert abs(normal_stress(uni, [1., 1., 0.]) - 100.0) < 1e-9
    assert abs(shear_stress(uni, [1., 1., 0.]) - 100.0) < 1e-9        # tau = sigma_0 / 2


def test_action_reaction():
    n = [0.3, -0.5, 0.8]
    assert np.allclose(traction(SIGMA, n), -traction(SIGMA, [-x for x in n]))


def test_normale_non_unitaire_meme_resultat():
    assert abs(normal_stress(SIGMA, [2., 2., 0.]) - normal_stress(SIGMA, [1., 1., 0.])) < 1e-12


def test_surface_libre_exercice_2():
    """Sur une surface libre t(n) = 0 : trois conditions scalaires, celles du bord du perçage."""
    sigma = np.array([[0., 0., 0.], [0., -20., 0.], [0., 0., 30.]])
    assert np.allclose(traction(sigma, [1., 0., 0.]), 0.0)


def test_symetrie_controlee():
    assert check_symmetric(SIGMA)
    assert not check_symmetric([[1., 2., 0.], [3., 1., 0.], [0., 0., 1.]])
