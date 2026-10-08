"""Séance du 16 août : le recoupement croisé est le test le plus fort du dépôt."""
import numpy as np
from src.analytic import (kirsch_hole_hoop, lame_stresses, thermal_stresses,
                          combined_stresses, yield_pressure)
from src.consolidation import (lame_as_kirsch, kirsch_equibiaxial, lame_kirsch_gap, theta_p,
                               theta_y, yield_window_star, vm_field_star,
                               l2_and_energy_error, observed_orders, exact_bar)
from src.fem1d import solve_fixed_free


def test_lame_kirsch_bord():
    s_rr, s_tt = lame_as_kirsch(10.0, 10.0, 100.0)
    th = np.linspace(0, 2 * np.pi, 73)
    s_k = kirsch_hole_hoop(th, 100.0) + kirsch_hole_hoop(th - np.pi / 2, 100.0)
    assert abs(s_tt - 200.0) / 100 < 1e-5 and abs(s_rr) / 100 < 1e-5
    assert np.allclose(s_k, 200.0)


def test_lame_kirsch_champ_entier():
    r = np.linspace(10.0, 200.0, 200)
    for t in np.linspace(0, np.pi, 7):
        kr, kt, ks = kirsch_equibiaxial(r, t, 10.0, 100.0)
        lr, lt = lame_as_kirsch(r, 10.0, 100.0)
        assert np.allclose(kr, lr, atol=1e-8) and np.allclose(kt, lt, atol=1e-8)
        assert np.allclose(ks, 0.0, atol=1e-10)


def test_ecart_domaine_fini():
    for x in (2.0, 5.0, 10.0):
        lt = lame_stresses(10.0, 10.0, 10.0 * x, 0.0, -100.0)[1]
        assert np.isclose((lt - 200.0) / 200.0, lame_kirsch_gap(x))


def test_traces():
    r = np.linspace(50, 80, 100)
    m = sum(lame_stresses(r, 50, 80, 100.0))
    t = sum(thermal_stresses(r, 50, 80, 300, 200, 210000, 0.3, 12e-6))
    assert np.ptp(m) < 1e-9 and np.ptp(t) > 100.0


def test_acier_aluminium_meme_theta():
    nu, r = 0.3, np.linspace(50, 100, 21)
    th = theta_p(210000., nu, 1.2e-5, 50.0, 20.0)
    dT_al = th * (1 - nu) * 5.0 / (70000. * 2.3e-5)
    s = [np.array(combined_stresses(r, 50, 100, p, dT, 0, E, nu, al)) / p
         for E, al, p, dT in ((210000., 1.2e-5, 20.0, 50.0), (70000., 2.3e-5, 5.0, dT_al))]
    assert np.allclose(s[0], s[1], atol=1e-12)


def test_fenetre_adimensionnelle():
    lo, hi = yield_window_star(2.0, 0.3, 0.0)
    assert lo == 0.0 and np.isclose(250 * hi, yield_pressure(50, 100, 250), rtol=1e-6)
    lo, hi = yield_window_star(2.0, 0.3, theta_y(210000., 0.3, 1.2e-5, 50.0, 250.0))
    assert np.isclose(250 * hi, 144.66, atol=0.02)
    lo, hi = yield_window_star(3.0, 0.3, 1.6)
    assert lo > 0.04                       # le gradient seul plastifie : pression minimale
    r = np.linspace(1, 1.5, 801)
    lo, hi = yield_window_star(1.5, 0.3, 1.6)
    assert np.argmax(vm_field_star(r, 1.5, 0.3, hi, 1.6)) == r.size - 1   # amorçage en r = b


def test_convergence_normes():
    res = np.array([l2_and_energy_error(n, 210000., 100., 1000., 10.) for n in (4, 8, 16, 32)])
    assert np.allclose(observed_orders(res[:, 0]), 2.0, atol=1e-6)
    assert np.allclose(observed_orders(res[:, 1]), 1.0, atol=1e-6)
    assert np.allclose(observed_orders(res[:, 3]), 1.0, atol=1e-6)
    assert res[:, 2].max() < 1e-10            # superconvergence au milieu
    for n in (2, 8, 32):
        u, _ = solve_fixed_free(n, 210000., 100., 1000., 10.)
        x = np.linspace(0, 1000., n + 1)
        assert np.abs(u - exact_bar(x, 210000., 100., 1000., 10.)[0]).max() < 1e-10
