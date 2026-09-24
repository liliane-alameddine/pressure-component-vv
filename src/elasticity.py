"""Séance du 12 août : déformation linéarisée, loi de Hooke isotrope, forme découplée,
contraintes planes et déformations planes, contrainte axiale du cylindre selon sa
condition d'extrémité. Unités : MPa, mm."""
import numpy as np




def small_strain(grad_u):
    """epsilon = 1/2 (grad u + grad u^T), partie symétrique du gradient de déplacement."""
    g = np.asarray(grad_u, float)
    return 0.5 * (g + g.T)


def green_lagrange(grad_u):
    """E = 1/2 (grad u + grad u^T + grad u^T grad u), mesure exacte, nulle pour une
    rotation rigide."""
    g = np.asarray(grad_u, float)
    return 0.5 * (g + g.T + g.T @ g)


def lame_constants(E, nu):
    """G (mu), lambda, K. Bornes : -1 < nu < 0.5 ; K et lambda divergent en nu = 0.5."""
    G = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    return G, lam, K


def hooke_isotropic(eps, E, nu):
    """sigma = lambda tr(eps) I + 2 mu eps, en 3D."""
    G, lam, _ = lame_constants(E, nu)
    eps = np.asarray(eps, float)
    return lam * np.trace(eps) * np.eye(3) + 2.0 * G * eps


def hooke_decoupled(eps, E, nu):
    """Forme découplée : sigma_m = K eps_v, s = 2 G e ; doit redonner hooke_isotropic."""
    G, _, K = lame_constants(E, nu)
    eps = np.asarray(eps, float)
    eps_v = np.trace(eps)
    e = eps - eps_v / 3.0 * np.eye(3)
    return K * eps_v * np.eye(3) + 2.0 * G * e


def plane_strain_sigma_zz(sigma_xx, sigma_yy, nu):
    """Déformations planes : eps_zz = 0 => sigma_zz = nu (sigma_xx + sigma_yy)."""
    return nu * (sigma_xx + sigma_yy)


def plane_stress_eps_zz(sigma_xx, sigma_yy, E, nu):
    """Contraintes planes : sigma_zz = 0 => eps_zz = -(nu/E)(sigma_xx + sigma_yy)."""
    return -(nu / E) * (sigma_xx + sigma_yy)


def plane_equivalent_constants(E, nu):
    """Une solution en contraintes planes se transpose en déformations planes avec
    E' = E / (1 - nu^2), nu' = nu / (1 - nu)."""
    return E / (1.0 - nu**2), nu / (1.0 - nu)


def cylinder_axial_stress(p, a, b, nu, end_condition):
    """Contrainte axiale uniforme d'un cylindre sous pression interne selon l'extrémité :
    'open' (contraintes planes) 0 ; 'closed' p a^2/(b^2-a^2) ; 'plane_strain'
    2 nu p a^2/(b^2-a^2). Les cas fermé et déformations planes coïncident si nu = 1/2."""
    ratio = p * a**2 / (b**2 - a**2)
    return {"open": 0.0, "closed": ratio, "plane_strain": 2.0 * nu * ratio}[end_condition]
