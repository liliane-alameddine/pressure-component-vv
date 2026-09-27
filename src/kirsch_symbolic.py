"""Séance du 14 août, partie 9.2 : certification symbolique de Kirsch, quatre niveaux :
équilibre polaire, bord libre, champ lointain uniaxial, valeurs remarquables ; un
cinquième ajouté, la fonction d'Airy dont le champ dérive est biharmonique."""
import sympy as sp


def certify():
    r, th, a, sig = sp.symbols("r theta a sigma", positive=True)
    ar2 = (a / r) ** 2
    s_rr = sig / 2 * (1 - ar2) + sig / 2 * (1 - 4 * ar2 + 3 * ar2**2) * sp.cos(2 * th)
    s_tt = sig / 2 * (1 + ar2) - sig / 2 * (1 + 3 * ar2**2) * sp.cos(2 * th)
    s_rt = -sig / 2 * (1 + 2 * ar2 - 3 * ar2**2) * sp.sin(2 * th)
    eq_r = sp.diff(s_rr, r) + sp.diff(s_rt, th) / r + (s_rr - s_tt) / r
    eq_t = sp.diff(s_rt, r) + sp.diff(s_tt, th) / r + 2 * s_rt / r
    assert sp.simplify(eq_r) == 0 and sp.simplify(eq_t) == 0            # niveau 1
    assert sp.simplify(s_rr.subs(r, a)) == 0 and sp.simplify(s_rt.subs(r, a)) == 0  # 2
    assert sp.simplify(sp.limit(s_rr, r, sp.oo) - sig * sp.cos(th)**2) == 0      # 3
    assert sp.simplify(sp.limit(s_tt, r, sp.oo) - sig * sp.sin(th)**2) == 0
    assert sp.simplify(s_tt.subs({r: a, th: sp.pi / 2}) - 3 * sig) == 0          # 4
    assert sp.simplify(s_tt.subs({r: a, th: 0}) + sig) == 0
    # niveau 5 : la fonction d'Airy de Kirsch est biharmonique et redonne le champ
    phi = (sig / 4 * (r**2 - 2 * a**2 * sp.log(r))
           - sig / 4 * (r**2 - 2 * a**2 + a**4 / r**2) * sp.cos(2 * th))
    lap = lambda f: sp.diff(f, r, 2) + sp.diff(f, r) / r + sp.diff(f, th, 2) / r**2
    assert sp.simplify(lap(lap(phi))) == 0
    assert sp.simplify(sp.diff(phi, r) / r + sp.diff(phi, th, 2) / r**2 - s_rr) == 0
    assert sp.simplify(sp.diff(phi, r, 2) - s_tt) == 0
    assert sp.simplify(-sp.diff(sp.diff(phi, th) / r, r) - s_rt) == 0
    return True


if __name__ == "__main__":
    certify()
    print("Arbitre de Kirsch certifié : équilibre, bord libre, champ lointain, valeurs "
          "remarquables, fonction d'Airy biharmonique.")
