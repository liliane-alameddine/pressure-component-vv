"""Séance du 15 août, partie 8.2 : certification symbolique du champ thermique, cinq
niveaux, plus un sixième ajouté : la loi de Duhamel-Neumann redonne sigma_zz."""
import sympy as sp


def certify():
    r, a, b, E, nu, al, dT0 = sp.symbols("r a b E nu alpha DeltaT", positive=True)
    T = sp.log(b / r)
    assert sp.simplify(sp.diff(r * sp.diff(T, r), r) / r) == 0            # 1 harmonique
    K = E * al * dT0 / (2 * (1 - nu) * sp.log(b / a))
    C = (a**2 / (b**2 - a**2)) * sp.log(b / a)
    s_rr = K * (-sp.log(b / r) + C * (b**2 / r**2 - 1))
    s_tt = K * (1 - sp.log(b / r) - C * (b**2 / r**2 + 1))
    assert sp.simplify(sp.diff(s_rr, r) + (s_rr - s_tt) / r) == 0         # 2 équilibre
    assert sp.simplify(s_rr.subs(r, a)) == 0 and sp.simplify(s_rr.subs(r, b)) == 0  # 3
    assert sp.simplify(sp.integrate(s_tt, (r, a, b))) == 0                 # 4 résultante
    prod = sp.simplify(s_tt.subs(r, a) * s_tt.subs(r, b))
    # 5 : changement de signe entre les parois, vérifié numériquement pour b/a = 1.6
    val = prod.subs({a: 50, b: 80, E: 1, nu: sp.Rational(3, 10), al: 1, dT0: 1})
    assert float(val) < 0
    # 6 : compatibilité, la déformation circonférentielle eps_tt = u/r dérivée de la loi
    #     de Duhamel-Neumann en déformations planes satisfait d(r eps_tt)/dr = eps_rr
    Tf = dT0 * sp.log(b / r) / sp.log(b / a)
    s_zz = nu * (s_rr + s_tt) - E * al * Tf
    e_rr = (s_rr - nu * (s_tt + s_zz)) / E + al * Tf
    e_tt = (s_tt - nu * (s_rr + s_zz)) / E + al * Tf
    assert sp.simplify(sp.diff(r * e_tt, r) - e_rr) == 0
    return True


if __name__ == "__main__":
    certify()
    print("Arbitre thermique certifié : six niveaux.")
