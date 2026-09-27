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
| 14 août | Kirsch, concentration de contraintes | | à venir |
| 10 septembre | analyse modale, ordre 2p, raideur centrifuge | seances/2026-09-10_analyse_modale | faite, 6 tests |

Lancer : `pytest -q` à la racine exécute tous les tests (dossier tests/). Chaîne de Lamé,
dans l'ordre arbitre, tests, figures :
`python src/lame_symbolic.py && python -m pytest -q && python src/figures_lame.py`.
