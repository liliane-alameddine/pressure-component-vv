"""Séance du 10 août : le tenseur des contraintes et la relation de Cauchy.
Trois opérations qui serviront tout le projet : vecteur contrainte sur une facette,
contrainte normale, cisaillement. sigma : tenseur 3x3 symétrique en MPa ; n : normale,
normalisée ici (sortante)."""
import numpy as np


def traction(sigma, n):
    """Vecteur contrainte t(n) = sigma . n, relation de Cauchy."""
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    return np.asarray(sigma, float) @ n


def normal_stress(sigma, n):
    """Composante normale sigma_n = n . sigma . n, forme quadratique de la normale."""
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    return float(n @ np.asarray(sigma, float) @ n)


def shear_stress(sigma, n):
    """Module de la composante tangentielle, par Pythagore : tau^2 = |t|^2 - sigma_n^2."""
    t = traction(sigma, n)
    sn = normal_stress(sigma, n)
    return float(np.sqrt(max(t @ t - sn**2, 0.0)))


def check_symmetric(sigma, tol=1e-12):
    """La symétrie est un théorème (équilibre des moments), pas une hypothèse : on la
    contrôle à l'entrée plutôt que de la supposer."""
    sigma = np.asarray(sigma, float)
    return bool(np.allclose(sigma, sigma.T, atol=tol * max(1.0, np.abs(sigma).max())))


# ---------------------------------------------------------------------------------------
# Séance du 11 août : contraintes principales, invariants, déviateur, von Mises, Tresca
# ---------------------------------------------------------------------------------------

def principal_stresses(sigma):
    """Contraintes principales, ordonnées par valeur algébrique décroissante
    (sigma_1 >= sigma_2 >= sigma_3). eigvalsh exploite la symétrie : valeurs réelles."""
    vals = np.linalg.eigvalsh(np.asarray(sigma, dtype=float))
    return vals[::-1]


def principal_directions(sigma):
    """Contraintes et directions principales (colonnes), même ordre décroissant."""
    vals, vecs = np.linalg.eigh(np.asarray(sigma, dtype=float))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def invariants(sigma):
    """I1 = trace, I2 = 1/2 [ (tr sigma)^2 - tr(sigma^2) ], I3 = det."""
    sigma = np.asarray(sigma, dtype=float)
    I1 = np.trace(sigma)
    I2 = 0.5 * (I1**2 - np.trace(sigma @ sigma))
    I3 = np.linalg.det(sigma)
    return float(I1), float(I2), float(I3)


def hydrostatic(sigma):
    """Contrainte moyenne sigma_m = I1 / 3."""
    return float(np.trace(np.asarray(sigma, dtype=float)) / 3.0)


def deviatoric(sigma):
    """Déviateur s = sigma - sigma_m I, de trace nulle : change la forme, pas le volume."""
    sigma = np.asarray(sigma, dtype=float)
    return sigma - hydrostatic(sigma) * np.eye(3)


def J2(sigma):
    """Second invariant du déviateur, 1/2 s : s."""
    s = deviatoric(sigma)
    return 0.5 * float(np.tensordot(s, s))


def von_mises(sigma):
    """Contrainte équivalente de von Mises, sqrt(3 J2) ; le facteur 3 calibre la traction
    simple : sigma_vm = sigma_0."""
    return float(np.sqrt(3.0 * J2(sigma)))


def max_shear(sigma):
    """Cisaillement maximal (sigma_1 - sigma_3) / 2 ; la contrainte principale nulle d'un
    état plan compte dans l'ordonnancement."""
    p = principal_stresses(sigma)
    return float((p[0] - p[-1]) / 2.0)


def tresca(sigma):
    """Contrainte équivalente de Tresca, sigma_1 - sigma_3 = 2 tau_max."""
    return 2.0 * max_shear(sigma)


def von_mises_field(sigma_field):
    """Von Mises sur un champ de tenseurs, tableau [N, 3, 3], sans boucle : servira au
    dépouillement des champs éléments finis."""
    sigma_field = np.asarray(sigma_field, dtype=float)
    s = sigma_field - (np.trace(sigma_field, axis1=1, axis2=2)[:, None, None] / 3.0) * np.eye(3)
    J2f = 0.5 * np.einsum("nij,nij->n", s, s)
    return np.sqrt(3.0 * J2f)
