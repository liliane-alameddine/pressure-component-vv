"""Séance du 12 août : premier solveur éléments finis, barre en traction, écrit de zéro.
Assemblage explicite, conditions essentielles par élimination, par pénalisation ou par
multiplicateur de Lagrange (les trois méthodes de la partie 6.3)."""
import numpy as np


def element_stiffness(E, A, L):
    """Ke = (E A / L) [[1, -1], [-1, 1]] : symétrique, singulière (mode rigide), semi-définie."""
    return (E * A / L) * np.array([[1.0, -1.0], [-1.0, 1.0]])


def assemble(n_elem, E, A, length):
    n_node = n_elem + 1
    L_e = length / n_elem
    K = np.zeros((n_node, n_node))
    for e in range(n_elem):
        k_e = element_stiffness(E, A, L_e)
        dofs = [e, e + 1]
        for i in range(2):
            for j in range(2):
                K[dofs[i], dofs[j]] += k_e[i, j]
    return K


def consistent_load(n_elem, f, length):
    """Charge répartie uniforme f : moitié à chaque nœud de l'élément."""
    n_node = n_elem + 1
    L_e = length / n_elem
    F = np.zeros(n_node)
    for e in range(n_elem):
        F[e] += f * L_e / 2.0
        F[e + 1] += f * L_e / 2.0
    return F


def solve_fixed_free(n_elem, E, A, length, f, method="elimination", penalty=1e8):
    """Barre encastrée en x = 0, libre en x = L, charge répartie f.
    Renvoie u et la réaction en x = 0."""
    K = assemble(n_elem, E, A, length)
    F = consistent_load(n_elem, f, length)
    n = n_elem + 1
    if method == "elimination":
        u = np.zeros(n)
        u[1:] = np.linalg.solve(K[1:, 1:], F[1:])
    elif method == "penalty":
        Kp = K.copy()
        Kp[0, 0] += penalty * K.max()
        u = np.linalg.solve(Kp, F)                        # u[0] = 0 approché
    elif method == "lagrange":
        # [[K, c], [c^T, 0]] [u ; lambda] = [F ; 0], c = e_0 ; lambda = - réaction
        Kl = np.zeros((n + 1, n + 1))
        Kl[:n, :n] = K
        Kl[:n, n] = Kl[n, :n] = np.eye(n)[0]
        sol = np.linalg.solve(Kl, np.r_[F, 0.0])
        u = sol[:n]
    else:
        raise ValueError(method)
    reaction = (K @ u - F)[0]
    return u, reaction


def element_stress(u, n_elem, E, length):
    """Contrainte constante par élément, exacte au point milieu (superconvergence)."""
    L_e = length / n_elem
    return np.array([E * (u[e + 1] - u[e]) / L_e for e in range(n_elem)])
