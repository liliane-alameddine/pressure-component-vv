# Protocole du projet : composant épais perforé sous chargement thermomécanique

Ouvert le 10 août 2026. Ce fichier décrit ce qui est calculé et comment ; il est gelé avant
chaque campagne et n'est modifié qu'avec une ligne datée dans DEVIATIONS.md.

## Composant
Cylindre épais, rayon intérieur a = 50 mm, rayon extérieur b = 100 mm, percé d'un trou
circulaire de rayon c = 5 mm, d'axe parallèle à l'axe du cylindre, centré à r = 75 mm
(mi-épaisseur). Chargement : pression interne p = 20 MPa sur l'alésage ; gradient thermique radial
permanent, écart T_a - T_b en paramètre (0 à 50 K), intérieur chaud. Hypothèse cinématique : déformations planes (cylindre long, sections axiales empêchées),
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
Kirsch, établi et certifié symboliquement le 14 août (cinq niveaux, fonction d'Airy
comprise) : sigma_tt(a, theta) = sigma (1 - 2 cos 2 theta), K_t = 3 en uniaxial, 2 en
équibiaxial, 4 en cisaillement pur ; indépendant du rayon du trou ; perturbation 2.2 % à
cinq rayons, 1 % à 7.27 rayons. Domaine de calcul du perçage : domaine fini avec les
tractions exactes de Kirsch appliquées au bord (option 2 de la partie 5), de sorte que
l'erreur mesurée ne contienne que l'erreur numérique. Le perçage est une concentration,
non une singularité : la convergence existe, plus lente qu'en champ lisse.
Thermoélasticité du cylindre, établie et certifiée le 15 août (six niveaux) : T(r)
logarithmique ; sigma_rr et sigma_tt thermiques autoéquilibrées, bords libres,
indépendantes de la taille ; intérieur chaud comprimé en régime permanent. Axial
thermique : cylindre libre axialement, sigma_zz = sigma_rr + sigma_tt, résultante nulle,
indépendant de toute température de référence ; l'axial de pression reste en
déformations planes. Superposition par linéarité et couplage faible (thermique puis
mécanique). Trois arbitres exacts : l'étape 1 est close.

## Méta-modèles (étapes 9 et 9 bis)
Étape 9 : chaos polynomial ajusté sur la campagne paramétrique, degré borné par le nombre
de niveaux du plan, validé sur des calculs complets retenus hors grille (Q2 de validation).
Étape 9 bis, ajoutée le 29 septembre : réseau de neurones entraîné sur la même campagne,
jugé par la même règle sur les mêmes cas retenus ; courbe de précision en fonction du
nombre de calculs d'entraînement ; comparaison au chaos polynomial en précision, en coût
et en extrapolation. Un méta-modèle n'est jamais appelé validé sans cas retenus.

## Méthode global/local (étape ajoutée le 8 octobre)
Modèle global : composant complet maillé grossièrement autour du perçage. Sous-modèle :
disque autour du perçage, maillé finement, piloté par les déplacements du modèle global
interpolés sur sa frontière de coupe. Arbitre : calcul fin complet du composant, et
Kirsch pour la partie mécanique. Mesures : erreur du sous-modèle en fonction de la
distance de coupe (2, 3, 5 rayons), confrontée à la décroissance de Kirsch (2.2 % à cinq
rayons) ; variante pilotée en tractions comparée à la variante en déplacements ; gain de
coût par rapport au calcul fin complet. Règle : la coupe doit se trouver là où le champ
global est déjà convergé ; sinon l'erreur du global passe dans le local.
Ouverture : fissures radiales à l'alésage et au perçage (séance du 13 octobre), traitées
dans le sous-modèle, qui est l'endroit où la rupture se calcule dans l'industrie.

## Limites déclarées
Régime thermique permanent seulement ; le cas dimensionnant réel d'une paroi sous
pression est le refroidissement brutal de l'intérieur (choc thermique sous pression), où
la paroi intérieure passe en traction ; il n'est pas traité.

## Jours et étiquettes
Chaque séance close est étiquetée seance-AAAA-MM-JJ ; la liste est dans CHANGELOG.md.
