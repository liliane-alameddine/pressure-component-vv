"""Séance du 13 août : solutions analytiques de référence, premier arbitre du projet.
Convention : traction positive ; une pression interne donne sigma_rr < 0 à l'alésage.
Les contraintes radiale et circonférentielle ne dépendent pas du matériau (Michell) ;
le déplacement et, en déformations planes, la contrainte axiale en dépendent."""
import numpy as np


def lame_constants(a, b, p_i, p_o=0.0):
    """C1 = (p_i a^2 - p_o b^2)/(b^2 - a^2), C2 = (p_i - p_o) a^2 b^2/(b^2 - a^2)."""
    denom = b**2 - a**2
    C1 = (p_i * a**2 - p_o * b**2) / denom
    C2 = (p_i - p_o) * a**2 * b**2 / denom
    return C1, C2


def lame_stresses(r, a, b, p_i, p_o=0.0):
    """sigma_rr = C1 - C2/r^2, sigma_tt = C1 + C2/r^2 ; leur somme vaut 2 C1, constante."""
    C1, C2 = lame_constants(a, b, p_i, p_o)
    r = np.asarray(r, dtype=float)
    return C1 - C2 / r**2, C1 + C2 / r**2


def lame_axial(r, a, b, p_i, p_o=0.0, case="plane_strain", nu=0.3):
    """Contrainte axiale selon la condition d'extrémité : 'open' 0, 'closed' C1,
    'plane_strain' nu (sigma_rr + sigma_tt) = 2 nu C1."""
    s_rr, s_tt = lame_stresses(r, a, b, p_i, p_o)
    if case == "open":
        return np.zeros_like(s_rr)
    if case == "closed":
        return np.full_like(s_rr, (p_i * a**2 - p_o * b**2) / (b**2 - a**2))
    if case == "plane_strain":
        return nu * (s_rr + s_tt)
    raise ValueError("case must be open, closed or plane_strain")


def lame_displacement(r, a, b, p_i, p_o=0.0, E=210000.0, nu=0.3):
    """u = A r + B/r en déformations planes, A = C1/(2(lambda+mu)), B = C2/(2 mu)."""
    C1, C2 = lame_constants(a, b, p_i, p_o)
    r = np.asarray(r, dtype=float)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return (C1 / (2.0 * (lam + mu))) * r + (C2 / (2.0 * mu)) / r


def lame_von_mises(r, a, b, p_i, p_o=0.0, case="plane_strain", nu=0.3):
    """Von Mises le long du rayon : sigma_vm^2 = 3 k^2 + C1^2 (1 - 2 nu)^2 en déformations
    planes, k = C2/r^2, strictement croissante quand r diminue (partie 6)."""
    s_rr, s_tt = lame_stresses(r, a, b, p_i, p_o)
    s_zz = lame_axial(r, a, b, p_i, p_o, case, nu)
    return np.sqrt(0.5 * ((s_tt - s_rr)**2 + (s_rr - s_zz)**2 + (s_zz - s_tt)**2))


def yield_pressure(a, b, sigma_y, case="plane_strain", nu=0.3):
    """Pression d'amorçage de la plastification à l'alésage, critère de von Mises : la
    contrainte équivalente est linéaire en p, donc p_y = sigma_y / sigma_vm(a ; p = 1)."""
    return sigma_y / float(lame_von_mises(a, a, b, 1.0, 0.0, case, nu))


def thin_wall_hoop(a, t, p):
    """Formule des chaudronniers, p a / t : écart de 5.2 % à t = a / 10, moins en dessous."""
    return p * a / t
