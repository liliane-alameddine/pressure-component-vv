"""Séance du 18 août : la voie variationnelle redonne la matrice du 12 août."""
import numpy as np
import pytest
from src.fem1d import element_stiffness, consistent_load
from src.isoparametric import (shape_linear, shape_quadratic, check_shape_properties,
                               jacobian, jacobian_quadratic_midnode, midnode_valid_range,
                               element_stiffness_isoparametric,
                               element_stiffness_variable_section, element_load_consistent,
                               load_quadratic, barlow_points, strain_error_zeros,
                               rank_sufficiency, condition_number, condition_number_exact,
                               quadrature)


def test_trois_exigences():
    for shape, nodes, deg in ((shape_linear, [-1, 1], 1), (shape_quadratic, [-1, 0, 1], 2)):
        assert max(check_shape_properties(shape, nodes, deg)) < 1e-12
    # la famille quadratique ne reproduit pas le cube : la complétude s'arrête au degré 2
    assert check_shape_properties(shape_quadratic, [-1, 0, 1], 3)[2] > 0.1


def test_recoupement_trois_voies():
    E, A, L = 210000.0, 100.0, 500.0
    ke_ref = element_stiffness(E, A, L)
    for n in (1, 2, 3):
        assert np.allclose(element_stiffness_isoparametric(E, A, 0.0, L, n), ke_ref)
    ke = element_stiffness_isoparametric(E, A, 0.0, L)
    assert np.allclose(ke @ np.ones(2), 0.0) and np.allclose(ke, ke.T)
    assert np.isclose(jacobian(shape_linear, [0.0, L], 0.3), L / 2)
    with pytest.raises(ValueError):
        element_stiffness_isoparametric(E, A, L, 0.0)          # noeuds inversés


def test_charges_coherentes():
    fe = element_load_consistent(10.0, 0.0, 500.0)
    assert np.allclose(fe, consistent_load(1, 10.0, 500.0))
    fq = load_quadratic(10.0, 500.0)
    assert np.allclose(fq, 5000.0 * np.array([1 / 6, 2 / 3, 1 / 6]))
    assert np.isclose(fq[1] / fq[0], 4.0) and np.isclose(fq.sum(), 5000.0)


def test_barlow():
    assert np.allclose(strain_error_zeros([-1, 1]), barlow_points(1))
    assert np.allclose(strain_error_zeros([-1, 0, 1]), barlow_points(2))
    assert np.allclose(np.abs(barlow_points(2)), 1 / np.sqrt(3))


def test_rang():
    assert rank_sufficiency(1, 3, 8, 3)[2] == 2          # Q4 un point : deux modes sablier
    assert rank_sufficiency(4, 3, 16, 3)[2] == 1         # Q8 quatre points : un mode
    assert rank_sufficiency(9, 3, 16, 3)[2] == 0
    assert rank_sufficiency(1, 3, 6, 3)[2] == 0          # T3 un point suffit


def test_conditionnement():
    for n in (10, 40, 160):
        k = condition_number(n)
        assert np.isclose(k, condition_number_exact(n), rtol=1e-8)
    assert abs(condition_number(320) / 320**2 - 4 / np.pi**2) < 1e-3
    assert np.isclose(condition_number(20, E=1.0, A=1.0, L=1.0), condition_number(20))


def test_exercices():
    # ex. 2 : quart central, et non tiers ; 20 mm est hors domaine
    assert midnode_valid_range(0.0, 100.0) == (25.0, 75.0)
    assert jacobian_quadratic_midnode(0.0, 20.0, 100.0, -1.0) < 0.0
    assert jacobian_quadratic_midnode(0.0, 30.0, 100.0, np.linspace(-1, 1, 51)).min() > 0.0
    # ex. 3
    f = lambda xi: 1 + 2 * xi + 3 * xi**2 + 4 * xi**3
    assert np.isclose(quadrature(f, 1), 2.0) and np.isclose(quadrature(f, 2), 4.0)
    # ex. 4 : section moyenne, exacte dès un point
    k1 = element_stiffness_variable_section(210000.0, 100.0, 300.0, 0.0, 500.0, 1)
    assert np.allclose(k1, element_stiffness(210000.0, 200.0, 500.0))
