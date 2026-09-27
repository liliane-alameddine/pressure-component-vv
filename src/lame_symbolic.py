"""Séance du 13 août, partie 7.2 : la certification symbolique de l'arbitre, trois niveaux.
Tant que ces trois assertions ne passent pas, aucune comparaison n'a de sens."""
import sympy as sp


def certify():
    r, a, b, p, nu = sp.symbols("r a b p nu", positive=True)
    C1, C2 = sp.symbols("C1 C2", real=True)
    s_rr = C1 - C2 / r**2
    s_tt = C1 + C2 / r**2
    # niveau 1 : l'équilibre radial est satisfait pour tout C1, C2
    residual = sp.diff(s_rr, r) + (s_rr - s_tt) / r
    assert sp.simplify(residual) == 0
    # niveau 2 : les conditions aux limites déterminent C1 et C2
    C1s = p * a**2 / (b**2 - a**2)
    C2s = p * a**2 * b**2 / (b**2 - a**2)
    s_rr_s = s_rr.subs({C1: C1s, C2: C2s})
    assert sp.simplify(s_rr_s.subs(r, a) + p) == 0
    assert sp.simplify(s_rr_s.subs(r, b)) == 0
    # niveau 3 : le déplacement A r + B/r satisfait l'équation d'Euler-Cauchy
    A, B = sp.symbols("A B", real=True)
    u = A * r + B / r
    ode = sp.diff(u, r, 2) + sp.diff(u, r) / r - u / r**2
    assert sp.simplify(ode) == 0
    # niveau 3 bis : la dérivation complète refait par la machine ; les constantes
    # élastiques se factorisent en (lambda + 2 mu), comme à la partie 2.3
    lam, mu = sp.symbols("lambda mu", positive=True)
    ufun = sp.Function("u")(r)
    e_rr, e_tt = sp.diff(ufun, r), ufun / r
    srr = lam * (e_rr + e_tt) + 2 * mu * e_rr
    stt = lam * (e_rr + e_tt) + 2 * mu * e_tt
    eq = sp.simplify(sp.diff(srr, r) + (srr - stt) / r)
    assert sp.simplify(eq / (lam + 2 * mu) - (sp.diff(ufun, r, 2) + sp.diff(ufun, r) / r
                                             - ufun / r**2)) == 0
    return True


if __name__ == "__main__":
    certify()
    print("Arbitre de Lamé certifié : équilibre, conditions aux limites, Euler-Cauchy, "
          "factorisation par (lambda + 2 mu).")
