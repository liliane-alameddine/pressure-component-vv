"""Séance du 18 août : fonctions de forme, formulation isoparamétrique, quadrature de
Gauss, vecteur de charge cohérent, points de Barlow, suffisance de rang, conditionnement.
La matrice de rigidité de la barre y est retrouvée par la voie variationnelle ; la voie
élémentaire du 12 août (src/fem1d.py) sert d'arbitre."""
import numpy as np


# ----------------------------------------------------------------- quadrature
def gauss_legendre(n):
    """Points et poids de Gauss-Legendre sur [-1, 1] ; exacte jusqu'au degré 2 n - 1."""
    return np.polynomial.legendre.leggauss(n)


def quadrature(f, n):
    """Intégrale approchée de f sur [-1, 1] par n points de Gauss."""
    xi, w = gauss_legendre(n)
    return float(np.sum(w * f(xi)))


# ----------------------------------------------------------------- fonctions de forme
def shape_linear(xi):
    """Élément à deux noeuds sur [-1, 1] : N et dN/dxi."""
    xi = np.asarray(xi, dtype=float)
    N = np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0])
    dN = np.array([-0.5 * np.ones_like(xi), 0.5 * np.ones_like(xi)])
    return N, dN


def shape_quadratic(xi):
    """Élément à trois noeuds sur [-1, 1], noeud médian en xi = 0 : N et dN/dxi."""
    xi = np.asarray(xi, dtype=float)
    N = np.array([xi * (xi - 1.0) / 2.0, 1.0 - xi**2, xi * (xi + 1.0) / 2.0])
    dN = np.array([xi - 0.5, -2.0 * xi, xi + 0.5])
    return N, dN


def check_shape_properties(shape, nodes_xi, degree, n_sample=11):
    """Les trois exigences : propriété nodale, partition de l'unité, complétude au degré
    donné (reproduction exacte des monômes xi^k, k <= degree). Renvoie les trois écarts."""
    xi_s = np.linspace(-1.0, 1.0, n_sample)
    N, _ = shape(xi_s)                                   # (n_nodes, n_sample)
    Nn, _ = shape(np.asarray(nodes_xi))
    nodal = np.abs(Nn - np.eye(len(nodes_xi))).max()
    partition = np.abs(N.sum(axis=0) - 1.0).max()
    completeness = 0.0
    for k in range(degree + 1):
        reproduced = np.asarray(nodes_xi) ** k @ N        # somme_a N_a(xi) xi_a^k
        completeness = max(completeness, np.abs(reproduced - xi_s**k).max())
    return nodal, partition, completeness


# ----------------------------------------------------------------- isoparamétrique
def jacobian(shape, x_nodes, xi):
    """J(xi) = somme_a dN_a/dxi x_a ; constant pour la barre à deux noeuds (L_e / 2)."""
    _, dN = shape(xi)
    return np.asarray(x_nodes, dtype=float) @ dN


def jacobian_quadratic_midnode(x1, xm, x3, xi):
    """Jacobien de l'élément à trois noeuds à noeud médian décalé : affine en xi,
    J = (x3 - x1)/2 + xi (x1 + x3 - 2 xm). Strictement positif sur [-1, 1] ssi le
    noeud médian est dans le quart central."""
    return (x3 - x1) / 2.0 + np.asarray(xi, dtype=float) * (x1 + x3 - 2.0 * xm)


def midnode_valid_range(x1, x3):
    """Intervalle du noeud médian gardant J > 0 partout : le quart central."""
    L = x3 - x1
    return x1 + L / 4.0, x3 - L / 4.0


def element_stiffness_isoparametric(E, A, x1, x2, n_gauss=1):
    """K_e = int B^T E A B dx par quadrature sur l'élément de référence.
    Un point suffit et est exact pour l'élément linéaire à section constante."""
    xi_pts, w_pts = gauss_legendre(n_gauss)
    ke = np.zeros((2, 2))
    for xi, w in zip(xi_pts, w_pts):
        _, dN = shape_linear(xi)
        J = float(dN @ np.array([x1, x2]))
        if J <= 0.0:
            raise ValueError(f"Jacobien non positif : J = {J}")
        B = dN / J
        ke += w * np.outer(B, B) * E * A * J
    return ke


def element_stiffness_variable_section(E, A1, A2, x1, x2, n_gauss=1):
    """Exercice 4 : section affine A(xi) = N1 A1 + N2 A2. B est constante, l'intégrande
    est affine en xi : un point de Gauss l'intègre exactement et donne la section moyenne."""
    xi_pts, w_pts = gauss_legendre(n_gauss)
    ke = np.zeros((2, 2))
    for xi, w in zip(xi_pts, w_pts):
        N, dN = shape_linear(xi)
        J = float(dN @ np.array([x1, x2]))
        B = dN / J
        A = float(N @ np.array([A1, A2]))
        ke += w * np.outer(B, B) * E * A * J
    return ke


def element_load_consistent(p, x1, x2, n_gauss=2):
    """f_e = int N^T p dx, élément linéaire : p L_e / 2 sur chaque noeud."""
    xi_pts, w_pts = gauss_legendre(n_gauss)
    fe = np.zeros(2)
    for xi, w in zip(xi_pts, w_pts):
        N, dN = shape_linear(xi)
        J = float(dN @ np.array([x1, x2]))
        fe += w * N * p * J
    return fe


def load_quadratic(p, L_e, n_gauss=3):
    """Vecteur de charge cohérent de l'élément à trois noeuds : p L_e [1/6, 2/3, 1/6]."""
    xi_pts, w_pts = gauss_legendre(n_gauss)
    fe = np.zeros(3)
    for xi, w in zip(xi_pts, w_pts):
        N, _ = shape_quadratic(xi)
        fe += w * N * p * (L_e / 2.0)
    return fe


# ----------------------------------------------------------------- Barlow
def barlow_points(degree):
    """Points optimaux d'évaluation des contraintes d'un élément de degré p : les points
    de Gauss d'ordre p (Barlow 1976). Démonstration : l'erreur de déplacement, de degré
    p + 1, s'annule aux p + 1 noeuds ; sa dérivée s'annule aux zéros du polynôme de
    Legendre de degré p."""
    return gauss_legendre(degree)[0]


def strain_error_zeros(nodes_xi):
    """Zéros de la dérivée de prod (xi - xi_a) : points où l'erreur de déformation
    s'annule quand l'erreur de déplacement s'annule aux noeuds."""
    poly = np.poly1d(np.asarray(nodes_xi, dtype=float), r=True)
    return np.sort(np.roots(poly.deriv().coeffs))


# ----------------------------------------------------------------- rang
def rank_sufficiency(n_gauss, n_strain, n_dof, n_rigid):
    """Condition n_gauss * n_strain >= n_dof - n_rigid. Renvoie (rang obtenu, rang
    requis, nombre de modes parasites)."""
    rank = min(n_gauss * n_strain, n_dof - n_rigid)
    required = n_dof - n_rigid
    return rank, required, required - rank


# ----------------------------------------------------------------- conditionnement
def condition_number(n_elem, E=210000.0, A=100.0, L=1000.0):
    """Barre encastrée aux deux extrémités : K_ff = (E A / h) tridiag(-1, 2, -1)."""
    h = L / n_elem
    n_free = n_elem - 1
    K = (E * A / h) * (np.diag(2.0 * np.ones(n_free))
                       + np.diag(-np.ones(n_free - 1), 1)
                       + np.diag(-np.ones(n_free - 1), -1))
    return float(np.linalg.cond(K))


def condition_number_exact(n_elem):
    """Valeurs propres exactes 4 sin^2(k pi / 2n), k = 1..n-1 : kappa = cot^2(pi / 2n)."""
    return float(1.0 / np.tan(np.pi / (2.0 * n_elem)) ** 2)


def digits_lost(kappa):
    return float(np.log10(kappa))
