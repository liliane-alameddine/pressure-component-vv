# Vérification croisée ANSYS

Fichier poutre_modale_plane183.inp : même poutre, même maillage 40 x 2, même élément Q8
(PLANE183), contraintes planes avec épaisseur, masse cohérente, six modes.

| Mode | Python 2x2 (Hz) | Python 3x3 (Hz) | ANSYS (à remplir) |
|---|---|---|---|
| 1 flexion | 41.734 | 41.739 | |
| 2 flexion | 258.595 | 258.625 | |
| 3 flexion | 711.447 | 711.548 | |
| 4 axial | 1293.970 | 1293.997 | |
| 5 flexion | 1360.610 | 1360.861 | |
| 6 flexion | 2183.875 | 2184.419 | |

Puis relancer avec LUMPM,ON : les fréquences doivent descendre sous les valeurs cohérentes,
ce qui est l'encadrement de la partie 4.7 de la séance. Ce que l'accord établit : deux
implémentations du même élément sur le même maillage donnent les mêmes valeurs propres ;
ce n'est pas une validation, l'arbitre reste Timoshenko.
