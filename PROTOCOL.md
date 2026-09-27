# Protocole du projet : composant épais perforé sous chargement thermomécanique

Ouvert le 10 août 2026. Ce fichier décrit ce qui est calculé et comment ; il est gelé avant
chaque campagne et n'est modifié qu'avec une ligne datée dans DEVIATIONS.md.

## Composant
Cylindre épais, rayon intérieur a = 50 mm, rayon extérieur b = 100 mm, percé d'un trou
circulaire de rayon c = 5 mm, d'axe parallèle à l'axe du cylindre, centré à r = 75 mm
(mi-épaisseur). Chargement : pression interne p = 20 MPa sur l'alésage ; gradient thermique
radial (à préciser au jour de la thermique). Hypothèse cinématique : déformations planes (cylindre long, sections axiales empêchées),
condition d'extrémité du cas 3 : eps_zz = 0, sigma_zz = nu (sigma_rr + sigma_thth), soit
2 nu p a^2 / (b^2 - a^2) uniforme ; les cas ouvert et fermé sont écartés, voir 12 août.
Petites perturbations : ni grande rotation ni grand déplacement pour ce composant.

## Matériau
Acier : E = 210 000 MPa, nu = 0.30, alpha = 1.2e-5 /K, limite d'élasticité sigma_y = 250 MPa
(loi de comportement à préciser au jour de la plasticité).

## Grandeurs d'intérêt (à mesurer)
Contrainte circonférentielle à l'alésage sigma_theta(a) ; déplacement radial u_r(a) ;
contrainte maximale au bord du perçage et facteur de concentration K_t ; pression de
première plastification ; chaque grandeur avec son ordre de convergence et son indice.

## Solveur
Conditions essentielles imposées par élimination (partition libre / imposé) ; les
réactions se lisent par R = K U - F. Pénalisation et multiplicateurs de Lagrange sont
implémentés dans src/fem1d.py à titre de comparaison.

## Critère d'amorçage de la plastification
Critère de von Mises : le matériau est un acier, métal dont la plastification procède du
glissement des dislocations et est insensible à la pression hydrostatique (11 août).
Tresca est calculé en parallèle comme borne conservative, plus sévère de 13 % au plus.

## Unités et conventions
mm, N, MPa, K. Contraintes positives en traction. Tenseur des contraintes sigma_ij :
premier indice direction de la force, second indice normale de la facette ; symétrique.
Normales sortantes unitaires. Vecteur contrainte t(n) = sigma . n.

## Références analytiques (arbitres)
Lamé, établi et certifié symboliquement le 13 août (src/analytic.py, src/lame_symbolic.py) :
sigma_rr = C1 - C2/r^2, sigma_tt = C1 + C2/r^2, C1 = p a^2/(b^2 - a^2), C2 = C1 b^2 ;
sigma_zz = 2 nu C1 en déformations planes. Sur le composant sans perçage : sigma_tt(a) =
33.33 MPa, sigma_vm(a) = 46.26 MPa, u_r(a) = 9.079e-3 mm, pression d'amorçage 108.07 MPa.
Test de diagnostic gratuit : sigma_rr et sigma_tt ne doivent pas changer avec E (Michell).
Kirsch (trou dans une plaque) et thermoélasticité du cylindre : jours 14 et 15.

## Jours et étiquettes
Chaque séance close est étiquetée seance-AAAA-MM-JJ ; la liste est dans CHANGELOG.md.
