# Séance du 10 septembre, note de théorie : analyse modale, ordre 2p, raideur centrifuge

## 1. Le problème aux valeurs propres généralisé

Travaux virtuels étendus à la dynamique : int rho u'' . v + int sigma(u) : epsilon(v) = travail
extérieur. Discrétisé par les mêmes fonctions de forme que la statique : M U'' + K U = F(t),
avec M = int rho N^T N (fonctions de forme) et K = int B^T C B (leurs dérivées). Régime
libre, U = Phi cos(omega t) : (K - omega^2 M) Phi = 0, problème aux valeurs propres
généralisé. Après conditions essentielles, K et M symétriques définies positives : valeurs
propres réelles positives, vecteurs orthogonaux au sens de M et de K, Phi_i^T M Phi_j = 0.
Normalisation par la masse : Phi^T M Phi = 1, alors Phi^T K Phi = omega^2. Un vecteur propre
n'a pas d'amplitude, seule sa forme a un sens.

## 2. Masse cohérente et masse concentrée

Cohérente : même quadrature que la rigidité, pleine par élément. Concentrée : diagonale.
Pour le Q8, la sommation des lignes donne des masses négatives aux nœuds sommets ; on
emploie HRZ, la diagonale de la matrice cohérente mise à l'échelle de la masse de
l'élément. Encadrement : omega_concentrée <= omega_exacte <= omega_cohérente ; l'écart
entre les deux est une estimation d'erreur sans raffinement.

## 3. Ordre 2p et convergence par le haut

Une valeur propre est un quotient de Rayleigh, stationnaire au voisinage du mode : une
erreur d'ordre h^p sur le mode donne une erreur d'ordre h^2p sur la valeur propre, par le
même mécanisme que le doublement d'ordre sur l'énergie de déformation (orthogonalité de
Galerkin). Le modèle conforme en déplacement est trop raide : toutes les fréquences
calculées majorent les exactes et convergent par le haut, de façon monotone. Condition du
2p : un mode régulier. Un coin encastré en élasticité plane porte une singularité de
contrainte ; le mode n'est plus dans H^3 et l'ordre observé tombe vers 2. Résultat du jour :
ordre 3.95 et facteur 16 sur la poutre libre-libre, ordre 1.87 sur la poutre encastrée.

## 4. L'arbitre analytique et sa limite

Euler-Bernoulli encastrée-libre : omega_n = (beta_n L)^2 sqrt(EI / rho A L^4),
beta L = 1.87510, 4.69409, 7.85476, 10.99554. Rapports universels f2/f1 = 6.267,
f3/f1 = 17.55. Le calcul 2D contient le cisaillement et l'inertie de rotation
qu'Euler-Bernoulli néglige : écart croissant avec le mode, -0.09, -1.21, -2.93, -5.27 %
pour L/H = 20. L'arbitre qui les contient est Timoshenko, ici par éléments finis 1D à
intégration réduite convergés : écart 0.11 à 0.30 % sur quatre modes.

## 5. Masse modale effective

L_i = Phi_i^T M r, M_eff,i = L_i^2 avec la normalisation par la masse ; la somme sur tous
les modes vaut la masse totale dans la direction de r (à 1e-10 ici). Critère industriel :
90 % de la masse avec les modes retenus ; ici quatre modes suffisent, le premier porte 62 %.
Un mode de masse effective nulle n'est pas sans importance : il n'est pas excité par une
accélération d'ensemble dans cette direction.

## 6. Précontrainte, raideur géométrique, Campbell

K_totale = K_élastique + K_géométrique(sigma), avec K_g = int G^T S G, G les gradients complets
des déplacements et S la contrainte existante : positive en traction, négative en
compression, indépendante des propriétés élastiques. Calcul en deux temps, comme le
thermomécanique : statique sous précontrainte, puis modal avec K_totale. Traction : f1 de
41.74 à 56.21 Hz sous 5 kN. Compression : f1 s'annule à P = 5392 N contre Euler
pi^2 E I / 4 L^2 = 5397 N, écart 0.10 % ; le flambement est une fréquence propre nulle. Pour
une machine tournante, la force centrifuge est la précontrainte ; les fréquences montent
avec le régime, les ordres d'excitation sont des droites, et leurs intersections sont les
régimes critiques du diagramme de Campbell.

## 7. Ce que la séance n'établit pas

Aucune dynamique transitoire, aucune réponse forcée, aucun amortissement, aucune fatigue
vibratoire ; un calcul modal est le premier calcul d'une chaîne dynamique.
