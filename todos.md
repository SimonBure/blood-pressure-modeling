# Todo admin
- asso doctorant dopamines (sport)
- PC en attente livraison
- souscrire mutuelle alan 

# Todo scientifique
## Anesthésie
- Se renseigner sur les moindres carrés récursifs (comprendre la forme 'régresseur', les variables instrumentales)
- Regarder comment faire l'inférence Bayesienne (si t'as le temps le faire pour un système simple 'toy problem')
- étudier convexité et non-convexité du problème --> plot fonction de coût vs. variation 1 paramètre, le reste fixé.
- étudier importance du schéma de discrétisation --> comparer != + implicite vs explicite
- sensibilité et réduction du nombre de paramètres ?
- estimation poids total db physionet + amsterdam utiles ~ 1-2TB.

- modèle à 2 compartiments et comparaison fits

- interpolation pour combler les trous de données pour patients 21


## Exploration DB
- analyse et exploration MIMIC IV / III --> recherche + grande DB avec entrées drugs + sorties PA (+ covariables cliniques). Chercher dans les différents modules
- vitaldb
- hirid
- elcu
- amsterdamdb
- comprendre les datasets 
- entraîner petit modèle sur les données démo 

- méthode ia inférence paramètres ?
## Codes
- simuler du temps réel --> prédiction du modèle ?

## Bibliographie
- physiologie nor --> effets, élimination nor ? --> reins ?
- Efficacité contrôleurs actuels BP ?
- Efficacité prédiction machine learning ?
- Google Deep Learning pour la météo.
- bests AI and DL models for biological data (small datasets, noisy data...) ?
- best controller models for MAP problem (only positive input) ?
- modèles globaux de la circulation sanguine
- modèles bayésiens pour la biologie
- all available controllers ? Holes ? Design a more adapted controller ?

# Besoins

# Questions
- questions anesthésistes --> protocole et règles internes pour injection bolus NOR / contrôle MAP ? motifs hypotension ? --> seuil bas impératif ? seuil sur la dynamique ? Remplacer anesthésiste sur long-terme ? Meilleurs soins ? Lissés les réponses MAP pour éviter les surdosages et surtensions aussi dangeureux ? --> pas d'impératif pour avoir une réponse lissée

- contrainte sur la valeur de E_0 --> prendre celle mesurée ?

# Prompts


# Exploration
- Simulation complète système vasculaire / cardiaque / circulatoire corps humain avec/sans pathologies
- Suivi d'autres variables vitales
- next-token prediction pour PA
- deep reinforcement learning pour PA
- apprentissage efficace sur petit jeu de données ?
- machine/deep/reinforcement learning for biology --> best models for noisy data ? Create algo see state of the art ?
- PINN pour ci-dessus
- Regarder modèle IA / math pour biology état de l'art (dernier modèle très bon entraînement sur jeux de données bruitées / petits / bons pour modéliser des effets très incomplet)
- Whole patient biological model --> avec toutes les donnéees.
- MIT Computational Physiology Lab exchange student
- Contact hopitaux pour obtenir base de données monitoring BP + injection drogue norépinéphrine / nicardipine --> Papa + Maman + Hermione + Jean-Yves + base Lariboisière
- état de l'art en optimisation --> creux à remplir
- modèle hybride contrôle + ML
