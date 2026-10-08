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
    """Formule des chaudronniers, p a / t, valable à 5 % près pour t < a / 10."""
    return p * a / t


# ---------------------------------------------------------------------------------------
# Séance du 14 août : solution de Kirsch, trou circulaire dans une plaque infinie
# ---------------------------------------------------------------------------------------

def kirsch_stresses(r, theta, a, sigma):
    """Contraintes polaires autour d'un trou de rayon a sous traction uniaxiale sigma
    dirigée selon theta = 0 ; bord du trou libre ; traction positive."""
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    ar2 = (a / r) ** 2
    ar4 = ar2 ** 2
    c2, s2 = np.cos(2.0 * theta), np.sin(2.0 * theta)
    s_rr = 0.5 * sigma * (1.0 - ar2) + 0.5 * sigma * (1.0 - 4.0 * ar2 + 3.0 * ar4) * c2
    s_tt = 0.5 * sigma * (1.0 + ar2) - 0.5 * sigma * (1.0 + 3.0 * ar4) * c2
    s_rt = -0.5 * sigma * (1.0 + 2.0 * ar2 - 3.0 * ar4) * s2
    return s_rr, s_tt, s_rt


def kirsch_hole_hoop(theta, sigma):
    """Contrainte circonférentielle sur le bord : sigma (1 - 2 cos 2 theta) ; 3 sigma à
    90 degrés, - sigma dans l'axe de traction ; indépendante du rayon du trou."""
    return sigma * (1.0 - 2.0 * np.cos(2.0 * np.asarray(theta, dtype=float)))


def kirsch_hole_hoop_biaxial(theta, s1, s2):
    """Superposition : tractions s1 selon theta = 0 et s2 selon theta = 90 degrés.
    Équibiaxial (s1 = s2) : 2 s uniforme, K_t = 2 ; cisaillement pur (s2 = -s1) :
    - 4 s1 cos 2 theta, K_t = 4."""
    th = np.asarray(theta, dtype=float)
    return s1 * (1.0 - 2.0 * np.cos(2.0 * th)) + s2 * (1.0 + 2.0 * np.cos(2.0 * th))


def kirsch_decay(r_over_a):
    """sigma_tt / sigma le long de theta = 90 degrés : 1 + a^2/(2 r^2) + 3 a^4/(2 r^4)."""
    x = np.asarray(r_over_a, dtype=float)
    return 1.0 + 0.5 / x**2 + 1.5 / x**4


def ellipse_kt(a, rho):
    """Facteur de concentration approché d'une entaille elliptique, 1 + 2 sqrt(a / rho) ;
    diverge quand rho tend vers zéro : la fissure n'a pas de K_t."""
    return 1.0 + 2.0 * np.sqrt(a / rho)


def polar_to_cartesian(s_rr, s_tt, s_rt, theta):
    """Rotation des composantes polaires vers le repère cartésien : la comparaison avec
    un code éléments finis passe par elle."""
    c, s = np.cos(theta), np.sin(theta)
    s_xx = s_rr * c**2 + s_tt * s**2 - 2.0 * s_rt * s * c
    s_yy = s_rr * s**2 + s_tt * c**2 + 2.0 * s_rt * s * c
    s_xy = (s_rr - s_tt) * s * c + s_rt * (c**2 - s**2)
    return s_xx, s_yy, s_xy


# ---------------------------------------------------------------------------------------
# Séance du 15 août : champ thermique permanent et contraintes thermiques du cylindre
# delta_T_0 = T_a - T_b, positif si l'intérieur est plus chaud ; traction positive.
# ---------------------------------------------------------------------------------------

def blocking_stress(E, nu, alpha, dT, n_blocked):
    """Contrainte d'un corps chauffé bloqué dans 1, 2 ou 3 directions :
    - E alpha dT, - E alpha dT / (1 - nu), - E alpha dT / (1 - 2 nu)."""
    factor = {1: 1.0, 2: 1.0 / (1.0 - nu), 3: 1.0 / (1.0 - 2.0 * nu)}[n_blocked]
    return -E * alpha * dT * factor


def temperature_field(r, a, b, T_a, T_b):
    """Profil permanent logarithmique : T_b + (T_a - T_b) ln(b/r) / ln(b/a)."""
    r = np.asarray(r, dtype=float)
    return T_b + (T_a - T_b) * np.log(b / r) / np.log(b / a)


def heat_flux(r, a, b, T_a, T_b, k):
    """Flux radial k (T_a - T_b) / (r ln(b/a)) ; le produit flux fois r est constant."""
    r = np.asarray(r, dtype=float)
    return k * (T_a - T_b) / (r * np.log(b / a))


def thermal_stresses(r, a, b, T_a, T_b, E, nu, alpha):
    """Contraintes thermiques radiale et circonférentielle en déformations planes ;
    autocontrainte, les deux surfaces sont libres d'effort."""
    r = np.asarray(r, dtype=float)
    K = E * alpha * (T_a - T_b) / (2.0 * (1.0 - nu) * np.log(b / a))
    C = (a**2 / (b**2 - a**2)) * np.log(b / a)
    s_rr = K * (-np.log(b / r) + C * (b**2 / r**2 - 1.0))
    s_tt = K * (1.0 - np.log(b / r) - C * (b**2 / r**2 + 1.0))
    return s_rr, s_tt


def thermal_axial(r, a, b, T_a, T_b, E, nu, alpha, T_ref):
    """Contrainte axiale en déformations planes : nu (s_rr + s_tt) - E alpha (T - T_ref) ;
    varie dans l'épaisseur, contrairement au cas de Lamé."""
    s_rr, s_tt = thermal_stresses(r, a, b, T_a, T_b, E, nu, alpha)
    dT = temperature_field(r, a, b, T_a, T_b) - T_ref
    return nu * (s_rr + s_tt) - E * alpha * dT


def combined_stresses(r, a, b, p, T_a, T_b, E, nu, alpha, T_ref=None):
    """Superposition pression plus thermique, licite par linéarité et couplage faible.
    Axial de pression en déformations planes (12 août) ; axial thermique libre, sans
    résultante (protocole du 15 août), sauf si T_ref est donné : alors déformations
    planes pures, dépendantes de T_ref."""
    pr = lame_stresses(r, a, b, p)
    th = thermal_stresses(r, a, b, T_a, T_b, E, nu, alpha)
    s_rr, s_tt = pr[0] + th[0], pr[1] + th[1]
    s_zz_th = (thermal_axial_free(r, a, b, T_a, T_b, E, nu, alpha) if T_ref is None
               else thermal_axial(r, a, b, T_a, T_b, E, nu, alpha, T_ref))
    s_zz = lame_axial(r, a, b, p, case="plane_strain", nu=nu) + s_zz_th
    return s_rr, s_tt, s_zz


def thermal_shock_parameter(sigma_f, E, nu, alpha):
    """Écart de température admissible R = sigma_f (1 - nu) / (E alpha)."""
    return sigma_f * (1.0 - nu) / (E * alpha)


def thermal_axial_free(r, a, b, T_a, T_b, E, nu, alpha):
    """Contrainte axiale thermique d'un cylindre libre axialement (déformation plane
    généralisée) : on superpose à la solution en déformations planes une contrainte
    uniforme qui annule la résultante axiale ; le résultat ne dépend plus de T_ref."""
    r = np.asarray(r, dtype=float)
    rr = np.linspace(a, b, 4001)
    s_rr, s_tt = thermal_stresses(rr, a, b, T_a, T_b, E, nu, alpha)
    zz = nu * (s_rr + s_tt) - E * alpha * temperature_field(rr, a, b, T_a, T_b)
    c0 = -np.trapezoid(zz * rr, rr) / np.trapezoid(rr, rr)
    s_rr, s_tt = thermal_stresses(r, a, b, T_a, T_b, E, nu, alpha)
    return nu * (s_rr + s_tt) - E * alpha * temperature_field(r, a, b, T_a, T_b) + c0
