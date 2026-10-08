"""Séance du 16 août : recoupement croisé des arbitres, adimensionnement, convergence en
norme du solveur 1D. Aucune physique nouvelle : on vérifie la cohérence de ce qui existe."""
import numpy as np

from src.analytic import (kirsch_stresses, lame_stresses, lame_axial, thermal_stresses,
                          thermal_axial_free, yield_pressure)
from src.fem1d import solve_fixed_free


# ----------------------------------------------------------------- recoupement croisé
def lame_as_kirsch(r, a, sigma, b=1.0e7):
    """Trou de rayon a dans une plaque infinie en traction équibiaxiale sigma, vu par
    Lamé : cylindre a, b très grand, p_i = 0, p_o = -sigma (traction = pression négative)."""
    return lame_stresses(r, a, b, p_i=0.0, p_o=-sigma)


def kirsch_equibiaxial(r, theta, a, sigma):
    """Même problème vu par Kirsch : deux tractions uniaxiales perpendiculaires."""
    s1 = kirsch_stresses(r, theta, a, sigma)
    s2 = kirsch_stresses(r, theta - np.pi / 2, a, sigma)
    return s1[0] + s2[0], s1[1] + s2[1], s1[2] + s2[2]


def lame_kirsch_gap(b_over_a):
    """Écart relatif de sigma_tt(a) entre Lamé à b fini et Kirsch (2 sigma) :
    2 sigma b^2/(b^2 - a^2) contre 2 sigma, soit a^2/(b^2 - a^2)."""
    x = np.asarray(b_over_a, dtype=float)
    return 1.0 / (x**2 - 1.0)


# ----------------------------------------------------------------- adimensionnement
def theta_p(E, nu, alpha, dT, p):
    """Nombre thermomécanique de la séance, rapporté à la pression."""
    return E * alpha * dT / ((1.0 - nu) * p)


def theta_y(E, nu, alpha, dT, sigma_y):
    """Le même groupe rapporté à la limite d'élasticité. C'est lui qu'il faut quand la
    pression est l'inconnue : Theta_p contient p et ne peut pas servir d'argument à p_y."""
    return E * alpha * dT / ((1.0 - nu) * sigma_y)


def vm_field_star(r_star, b_over_a, nu, p_star, th_y):
    """Von Mises adimensionnée sigma_vm / sigma_y le long de r* = r/a, pour p* = p/sigma_y
    et Theta_y. On choisit a = 1, sigma_y = 1, E alpha / (1 - nu) = 1 : seul le groupe
    compte, ce que le test d'équivalence acier / aluminium contrôle par ailleurs."""
    a, b = 1.0, float(b_over_a)
    E, alpha = 1.0 - nu, 1.0                       # E alpha / (1 - nu) = 1
    dT = th_y                                      # donc Theta_y = dT
    r = np.asarray(r_star, dtype=float)
    pr = lame_stresses(r, a, b, p_star)
    th = thermal_stresses(r, a, b, dT, 0.0, E, nu, alpha)
    s_rr, s_tt = pr[0] + th[0], pr[1] + th[1]
    s_zz = (lame_axial(r, a, b, p_star, case="plane_strain", nu=nu)
            + thermal_axial_free(r, a, b, dT, 0.0, E, nu, alpha))
    return np.sqrt(0.5 * ((s_tt - s_rr)**2 + (s_rr - s_zz)**2 + (s_zz - s_tt)**2))


def yield_window_star(b_over_a, nu, th_y, n_r=801, p_max=3.0, n_p=3001):
    """Intervalle des pressions p/sigma_y sans plastification, pour b/a, nu, Theta_y.
    max_r sigma_vm est convexe en p (maximum de normes de fonctions affines), donc
    l'ensemble admissible est un intervalle [p_lo, p_hi] ; p_lo > 0 quand le gradient
    seul plastifie et que la pression doit d'abord le compenser. Renvoie (p_lo, p_hi),
    ou (nan, nan) si aucune pression ne convient."""
    r = np.linspace(1.0, b_over_a, n_r)
    p = np.linspace(0.0, p_max, n_p)
    g = np.array([vm_field_star(r, b_over_a, nu, pk, th_y).max() for pk in p]) - 1.0
    ok = np.where(g <= 0.0)[0]
    if ok.size == 0:
        return np.nan, np.nan

    def bisect(lo, hi):                            # g(lo) et g(hi) de signes opposés
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            gm = vm_field_star(r, b_over_a, nu, mid, th_y).max() - 1.0
            if (gm <= 0.0) == (vm_field_star(r, b_over_a, nu, lo, th_y).max() <= 1.0):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    i0, i1 = ok[0], ok[-1]
    p_lo = 0.0 if i0 == 0 else bisect(p[i0 - 1], p[i0])
    p_hi = bisect(p[i1], p[i1 + 1])
    return p_lo, p_hi


# ----------------------------------------------------------------- convergence en norme
def exact_bar(x, E, A, L, f):
    return (f / (E * A)) * (L * x - x**2 / 2.0), (f / (E * A)) * (L - x)


def l2_and_energy_error(n_elem, E, A, L, f, n_gauss=3):
    """Erreurs en norme L2 (déplacement) et en norme d'énergie (déformation), par
    quadrature de Gauss élément par élément ; jamais aux noeuds, où l'erreur est nulle.
    Ajout : erreur de contrainte au milieu et au bord des éléments."""
    u_h, _ = solve_fixed_free(n_elem, E, A, L, f)  # le solveur renvoie (u, réaction)
    L_e = L / n_elem
    xi, w = np.polynomial.legendre.leggauss(n_gauss)
    err_l2 = err_en = 0.0
    s_mid = s_end = 0.0
    for e in range(n_elem):
        x0 = e * L_e
        x = x0 + 0.5 * L_e * (xi + 1.0)
        jac = 0.5 * L_e
        s = (x - x0) / L_e
        uh = u_h[e] * (1.0 - s) + u_h[e + 1] * s
        duh = (u_h[e + 1] - u_h[e]) / L_e
        ue, due = exact_bar(x, E, A, L, f)
        err_l2 += np.sum(w * (ue - uh) ** 2) * jac
        err_en += np.sum(w * E * A * (due - duh) ** 2) * jac
        _, d_mid = exact_bar(x0 + 0.5 * L_e, E, A, L, f)
        _, d_end = exact_bar(x0, E, A, L, f)
        s_mid = max(s_mid, abs(E * (d_mid - duh)))
        s_end = max(s_end, abs(E * (d_end - duh)))
    return np.sqrt(err_l2), np.sqrt(err_en), s_mid, s_end


def observed_orders(errors):
    e = np.asarray(errors, dtype=float)
    return np.log2(e[:-1] / e[1:])
