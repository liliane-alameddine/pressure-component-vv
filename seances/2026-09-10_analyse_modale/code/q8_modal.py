"""Séance du 10 septembre, exécution : analyse modale d'une poutre encastrée-libre en
éléments Q8 (contraintes planes), matrices de masse cohérente et concentrée, arbitre
d'Euler-Bernoulli, masses modales effectives, ordre 2p, raideur géométrique sous
précontrainte, charge critique d'Euler retrouvée par annulation de la première fréquence.
Unités : mm, N, MPa, tonne (rho en t/mm^3) ; les fréquences sortent en Hz."""
import numpy as np
from scipy.linalg import eigh

E, NU, RHO = 210000.0, 0.30, 7.85e-9                     # acier, tonne/mm^3
L, H, T = 1000.0, 50.0, 1.0                              # longueur, hauteur, épaisseur
BETA_L = np.array([1.87510407, 4.69409113, 7.85475744, 10.99554073])
GP, GW = np.polynomial.legendre.leggauss(3)
GP2, GW2 = np.polynomial.legendre.leggauss(2)


def q8_shape(xi, eta):
    """Fonctions de forme du Q8 sérendipité et dérivées par rapport à (xi, eta)."""
    xs = np.array([-1, 1, 1, -1, 0, 1, 0, -1], float)
    ys = np.array([-1, -1, 1, 1, -1, 0, 1, 0], float)
    N = np.zeros(8)
    dN = np.zeros((8, 2))
    for i in range(4):
        N[i] = 0.25 * (1 + xs[i] * xi) * (1 + ys[i] * eta) * (xs[i] * xi + ys[i] * eta - 1)
        dN[i, 0] = 0.25 * xs[i] * (1 + ys[i] * eta) * (2 * xs[i] * xi + ys[i] * eta)
        dN[i, 1] = 0.25 * ys[i] * (1 + xs[i] * xi) * (xs[i] * xi + 2 * ys[i] * eta)
    for i in (4, 6):
        N[i] = 0.5 * (1 - xi**2) * (1 + ys[i] * eta)
        dN[i, 0] = -xi * (1 + ys[i] * eta)
        dN[i, 1] = 0.5 * (1 - xi**2) * ys[i]
    for i in (5, 7):
        N[i] = 0.5 * (1 + xs[i] * xi) * (1 - eta**2)
        dN[i, 0] = 0.5 * xs[i] * (1 - eta**2)
        dN[i, 1] = -eta * (1 + xs[i] * xi)
    return N, dN


def mesh(nx, ny):
    """Maillage Q8 régulier de la poutre [0, L] x [-H/2, H/2] : grille (2nx+1) x (2ny+1)
    dont les centres d'éléments sont omis."""
    idx = -np.ones((2 * nx + 1, 2 * ny + 1), int)
    xy, k = [], 0
    for i in range(2 * nx + 1):
        for j in range(2 * ny + 1):
            if i % 2 == 1 and j % 2 == 1:
                continue
            idx[i, j] = k
            xy.append([i * L / (2 * nx), -H / 2 + j * H / (2 * ny)])
            k += 1
    conn = []
    for ex in range(nx):
        for ey in range(ny):
            i, j = 2 * ex, 2 * ey
            conn.append([idx[i, j], idx[i + 2, j], idx[i + 2, j + 2], idx[i, j + 2],
                         idx[i + 1, j], idx[i + 2, j + 1], idx[i + 1, j + 2], idx[i, j + 1]])
    return np.array(xy), np.array(conn)


def _element_matrices(X, sigma_xx=0.0, ngauss=3):
    """Ke et Kge intégrés à ngauss x ngauss (3 exact, 2 reproduit PLANE183 d'ANSYS) ;
    Me toujours à 3 x 3, sans quoi la masse cohérente du Q8 n'est plus définie positive."""
    D = E / (1 - NU**2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
    S = np.zeros((4, 4)); S[0, 0] = S[2, 2] = sigma_xx
    Ke = np.zeros((16, 16)); Me = np.zeros((16, 16)); Kge = np.zeros((16, 16))

    def at(xi, eta):
        N, dN = q8_shape(xi, eta)
        J = dN.T @ X
        dNx = dN @ np.linalg.inv(J)
        B = np.zeros((3, 16)); Nm = np.zeros((2, 16)); G = np.zeros((4, 16))
        for a in range(8):
            B[0, 2 * a] = dNx[a, 0]; B[1, 2 * a + 1] = dNx[a, 1]
            B[2, 2 * a] = dNx[a, 1]; B[2, 2 * a + 1] = dNx[a, 0]
            Nm[0, 2 * a] = N[a]; Nm[1, 2 * a + 1] = N[a]
            G[0, 2 * a] = dNx[a, 0]; G[1, 2 * a] = dNx[a, 1]               # grad u_x
            G[2, 2 * a + 1] = dNx[a, 0]; G[3, 2 * a + 1] = dNx[a, 1]       # grad u_y
        return B, Nm, G, np.linalg.det(J) * T

    gp, gw = (GP, GW) if ngauss == 3 else (GP2, GW2)
    for xi, wx in zip(gp, gw):
        for eta, wy in zip(gp, gw):
            B, Nm, G, dv = at(xi, eta)
            Ke += B.T @ D @ B * dv * wx * wy
            Kge += G.T @ S @ G * dv * wx * wy
    for xi, wx in zip(GP, GW):
        for eta, wy in zip(GP, GW):
            B, Nm, G, dv = at(xi, eta)
            Me += RHO * Nm.T @ Nm * dv * wx * wy
    return Ke, Me, Kge


def assemble(xy, conn, ngauss=3):
    """K, M cohérente, M concentrée par HRZ (la sommation des lignes donne des masses
    négatives aux nœuds sommets du Q8 ; HRZ met la diagonale à l'échelle de la masse).
    ngauss = 2 reproduit l'intégration 2 x 2 de PLANE183 d'ANSYS, pour la comparaison."""
    n = 2 * len(xy)
    K = np.zeros((n, n)); M = np.zeros((n, n)); Ml = np.zeros((n, n))
    for el in conn:
        dofs = np.ravel([[2 * a, 2 * a + 1] for a in el])
        Ke, Me, _ = _element_matrices(xy[el], 0.0, ngauss)
        d = np.diag(Me)
        K[np.ix_(dofs, dofs)] += Ke
        M[np.ix_(dofs, dofs)] += Me
        Ml[np.ix_(dofs, dofs)] += np.diag(d * Me.sum() / d.sum())
    return K, M, Ml


def assemble_geometric(xy, conn, sigma_xx):
    """Raideur géométrique d'une précontrainte uniaxiale uniforme : dépend de la
    contrainte, non des propriétés élastiques ; positive en traction."""
    n = 2 * len(xy)
    Kg = np.zeros((n, n))
    for el in conn:
        dofs = np.ravel([[2 * a, 2 * a + 1] for a in el])
        Kg[np.ix_(dofs, dofs)] += _element_matrices(xy[el], sigma_xx)[2]
    return Kg


def clamped_left(xy):
    return np.array([d for k in range(len(xy)) for d in (2 * k, 2 * k + 1)
                     if abs(xy[k, 0]) > 1e-9])


def modal(K, M, free, n_modes=6):
    """Problème aux valeurs propres généralisé, symétrique défini positif."""
    vals, vecs = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)],
                      subset_by_index=[0, n_modes - 1])
    return np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi), vecs


def euler_bernoulli(n=4):
    I, A = T * H**3 / 12, T * H
    return BETA_L[:n]**2 * np.sqrt(E * I / (RHO * A * L**4)) / (2 * np.pi)


def effective_masses(Phi, M, free, direction):
    """Masses modales effectives ; leur somme sur tous les modes vaut la masse totale."""
    Mf = M[np.ix_(free, free)]
    r = np.zeros(M.shape[0]); r[direction::2] = 1.0
    rf = r[free]
    Phi = Phi / np.sqrt(np.einsum("ij,ij->j", Phi, Mf @ Phi))
    Lp = Phi.T @ Mf @ rf
    return Lp**2, rf @ Mf @ rf


def critical_load(xy, conn, K, M, free):
    """Compression uniforme qui annule la première fréquence : charge d'Euler."""
    I, A = T * H**3 / 12, T * H
    p_euler = np.pi**2 * E * I / (4 * L**2)
    lo, hi = 0.0, 1.5 * p_euler
    Kf, Mf = K[np.ix_(free, free)], M[np.ix_(free, free)]
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        Kg = assemble_geometric(xy, conn, -mid / A)[np.ix_(free, free)]
        v = eigh(Kf + Kg, Mf, subset_by_index=[0, 0], eigvals_only=True)[0]
        lo, hi = (mid, hi) if v > 0 else (lo, mid)
    return 0.5 * (lo + hi), p_euler


def timoshenko(n=4, ne=2000, kappa=5.0 / 6.0):
    """Fréquences de la poutre encastrée-libre de Timoshenko, cisaillement et inertie de
    rotation compris, par éléments finis 1D linéaires à intégration réduite (2000 éléments,
    convergés à mieux que 1e-6) : l'arbitre qui contient ce qu'Euler-Bernoulli néglige."""
    I, A = T * H**3 / 12, T * H
    G = E / (2 * (1 + NU))
    Le = L / ne
    ndof = 2 * (ne + 1)                                  # (w, theta) par nœud
    K = np.zeros((ndof, ndof)); M = np.zeros((ndof, ndof))
    # flexion : intégration exacte (theta linéaire) ; cisaillement : un point de Gauss
    kb = E * I / Le * np.array([[0, 0, 0, 0], [0, 1, 0, -1], [0, 0, 0, 0], [0, -1, 0, 1]])
    # gamma = w' - theta au point milieu : [-1/Le, -1/2, 1/Le, -1/2]
    bs = np.array([-1 / Le, -0.5, 1 / Le, -0.5])
    ks = kappa * G * A * Le * np.outer(bs, bs)
    me = RHO * Le / 6 * np.array([[2 * A, 0, A, 0], [0, 2 * I, 0, I],
                                  [A, 0, 2 * A, 0], [0, I, 0, 2 * I]])
    for e in range(ne):
        d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(d, d)] += kb + ks
        M[np.ix_(d, d)] += me
    free = np.arange(2, ndof)
    vals = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True,
                subset_by_index=[0, n - 1])
    return np.sqrt(vals) / (2 * np.pi)
