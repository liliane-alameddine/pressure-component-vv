# pressure-component-vv

Étude de vérification et validation d'un composant épais perforé sous chargement
thermomécanique, exécutée séance par séance à partir du 10 août 2026 : PROTOCOL.md dit ce
qui est calculé, DEVIATIONS.md ce qui a changé et pourquoi, CHANGELOG.md les étiquettes.

| Séance | Objet | Dossier | État |
|---|---|---|---|
| 10 août | tenseur des contraintes, Cauchy, module src/stress.py | src, tests | faite, 7 tests |
| 11 août | contraintes principales, von Mises, Tresca, champ vectorisé | src, tests | faite, 10 tests |
| 12 août | déformation, équilibre, Hooke, hypothèses planes, premier solveur EF 1D | src, tests | faite, 8 tests |
| 13 août | Lamé, certification symbolique, plafond de pression, trois figures | src, tests, figures | faite, 8 tests |
| 14 août | Kirsch, concentration, superposition, décroissance, trois figures | src, tests, figures | faite, 8 tests |
| 15 août | thermoélasticité, axial libre, fenêtre de soulagement, choc thermique | src, tests, figures | faite, 10 tests ; étape 1 close |
| 18 août | fonctions de forme, isoparamétrique, Gauss, recoupement des trois voies, Barlow, rang, conditionnement | src, tests, seances/2026-08-18_discretisation | faite, 7 tests |
| 16 août | recoupement Lamé/Kirsch, adimensionnement, fenêtre d'amorçage, convergence en norme, protocole gelé | src, tests, seances/2026-08-16_consolidation | faite, 7 tests |
| 14 à 16 octobre | étape 9 : substitut par chaos polynomial, propagation, optimisation robuste | | à venir |
| après le 16 octobre | étape 9 bis : méta-modèle par réseau de neurones, comparé au chaos polynomial | | à venir, ajoutée le 29 septembre |
| fin août, après le perçage | étape global/local : sous-modèle du perçage piloté par le modèle global, erreur selon la distance de coupe | | à venir, ajoutée le 8 octobre |
| 13 octobre | fissures radiales dans le sous-modèle, facteur d'intensité des contraintes | | à venir |
| 10 septembre | analyse modale, ordre 2p, raideur centrifuge | seances/2026-09-10_analyse_modale | faite, 6 tests |

Lancer : `pytest -q` à la racine exécute tous les tests (dossier tests/). Chaîne complète,
dans l'ordre arbitres, tests, figures :
`python src/lame_symbolic.py && python src/kirsch_symbolic.py && python src/thermal_symbolic.py && python -m pytest -q &&
python src/figures_lame.py && python src/figures_kirsch.py && python src/figures_thermal.py`.
