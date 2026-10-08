"""Les huit contrôles de la séance du 10 septembre comme tests ; pytest -q depuis le
dossier de la séance. Les cas lourds (familles) sont marqués lents."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "code"))
from q8_modal import (mesh, assemble, assemble_geometric, clamped_left, modal,
                      euler_bernoulli, timoshenko, effective_masses, critical_load)


@pytest.fixture(scope="module")
def beam():
    xy, conn = mesh(40, 2)
    K, M, Ml = assemble(xy, conn)
    return xy, conn, K, M, Ml, clamped_left(xy)


def flexural(f, Phi, M, free, n=4):
    my, _ = effective_masses(Phi, M, free, 1)
    mx, _ = effective_masses(Phi, M, free, 0)
    return f[[i for i in range(len(f)) if my[i] > mx[i]][:n]]


def test_rigid_modes_on_free_structure(beam):
    xy, conn, K, M, Ml, free = beam
    f, _ = modal(K, M, np.arange(K.shape[0]), n_modes=8)
    assert int(np.sum(f < 1e-4 * f[3])) == 3


def test_frequency_ratios_and_referees(beam):
    xy, conn, K, M, Ml, free = beam
    f, Phi = modal(K, M, free, n_modes=6)
    ff = flexural(f, Phi, M, free)
    assert abs(ff[1] / ff[0] - 6.267) / 6.267 < 0.02
    assert abs(ff[2] / ff[0] - 17.55) / 17.55 < 0.03
    eb, ti = euler_bernoulli(4), timoshenko(4)
    assert np.all(np.abs(ff[:2] - eb[:2]) / eb[:2] < 0.02)
    assert np.all(np.abs(ff - ti) / ti < 0.005)
    assert np.all(np.diff(np.abs(ff - eb) / eb) > 0)        # écart de cisaillement croissant


def test_mass_bracketing(beam):
    xy, conn, K, M, Ml, free = beam
    fc, Pc = modal(K, M, free, n_modes=6)
    fl, Pl = modal(K, Ml, free, n_modes=6)
    assert np.all(flexural(fl, Pl, Ml, free, 3) < flexural(fc, Pc, M, free, 3))
    assert np.all(np.diag(Ml) > 0)                           # HRZ : aucune masse négative


def test_orthogonality_and_effective_mass(beam):
    xy, conn, K, M, Ml, free = beam
    from scipy.linalg import eigh
    Mf = M[np.ix_(free, free)]
    vals, vecs = eigh(K[np.ix_(free, free)], Mf)
    meff, mtot = effective_masses(vecs, M, free, 1)
    assert abs(meff.sum() - mtot) / mtot < 1e-10
    Phi = vecs[:, :6] / np.sqrt(np.einsum("ij,ij->j", vecs[:, :6], Mf @ vecs[:, :6]))
    assert np.allclose(Phi.T @ Mf @ Phi, np.eye(6), atol=1e-8)


def test_prestress_and_euler_load(beam):
    xy, conn, K, M, Ml, free = beam
    f0, _ = modal(K, M, free, n_modes=1)
    Kg = assemble_geometric(xy, conn, 5000.0 / 50.0)
    f1, _ = modal(K + Kg, M, free, n_modes=1)
    assert f1[0] > f0[0]
    p_num, p_euler = critical_load(xy, conn, K, M, free)
    assert abs(p_num - p_euler) / p_euler < 0.01


@pytest.mark.slow
def test_order_2p_free_free():
    fam = []
    for nx, ny in ((10, 1), (20, 2), (40, 4), (80, 8)):
        xy, conn = mesh(nx, ny)
        K, M, Ml = assemble(xy, conn)
        fam.append(modal(K, M, np.arange(K.shape[0]), n_modes=6)[0][3])
    fam = np.array(fam)
    p = np.log((fam[-3] - fam[-2]) / (fam[-2] - fam[-1])) / np.log(2)
    assert np.all(np.diff(fam) < 0) and p > 3.7
