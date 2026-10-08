# Séance du 10 septembre : analyse modale, ordre 2p, raideur centrifuge

Poutre encastrée-libre 1000 x 50 x 1 mm, acier, éléments Q8 en contraintes planes.
Exécution des huit contrôles de la séance : modes rigides, rapports de fréquences, arbitres
Euler-Bernoulli et Timoshenko, encadrement par les deux masses, orthogonalité, masses
effectives, ordre 2p sur deux familles, précontrainte jusqu'à la charge d'Euler.

Lancer : `python code/run_modal.py` (30 s, écrit results/), `python code/figures.py`,
`pytest -q` (six tests, 20 s). ANSYS : ansys/poutre_modale_plane183.inp, table à remplir.

Résultats : f1 = 41.739 Hz (Timoshenko 41.693, Euler-Bernoulli 41.776) ; ordre observé 3.95
et facteur 16 sur la poutre libre-libre, 1.87 sur la poutre encastrée (singularité de coin) ;
charge d'Euler retrouvée à 0.10 % par annulation de f1. Détail dans results/ et
theorie/note_analyse_modale.md.

Ce que ce dossier n'établit pas : dynamique transitoire, réponse forcée, amortissement.
