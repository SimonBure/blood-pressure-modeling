# Todo admin
- asso doctorant dopamines (sport)
- PC en attente livraison
- contacter Fabrice pour assister à une opération
- souscrire mutuelle alan 

# Todo scientifique
## Anesthésie
- Finir l'identification avec 3 compartiments 
  - Tester 2 compartiments 
- Se renseigner sur les moindres carrés récursifs (comprendre la forme 'régresseur', les variables instrumentales)
- Regarder comment faire l'inférence Bayesienne (si t'as le temps le faire pour un système simple 'toy problem')
- inférer pour le reste des patients

- étudier convexité et non-convexité du problème --> plot fonction de coût vs. variation 1 paramètre, le reste fixé.

- étudier importance du schéma de discrétisation --> comparer != + implicite vs explicite

- sensibilité et réduction du nombre de paramètres ?

- estimation poids total db physionet + amsterdam utiles ~ 1-2TB.


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
- exponents beta & gamma as model parameters to be optimized ? --> yes and easy
- casadi optimized traj vs pkpd model --> why not the same thing ? --> code pb in casadi --> injections INOR + lag ?
- negative parameters values --> add > 0 constraints on parameters ? --> yes
- cost computation adding BP differences AND Cc differences altogether ? Meaningless addition of different units ? --> don't optimize on NOR concentration (Cc), just BP.
- questions anesthésistes --> protocole et règles internes pour injection bolus NOR / contrôle MAP ? motifs hypotension ? --> seuil bas impératif ? seuil sur la dynamique ? Remplacer anesthésiste sur long-terme ? Meilleurs soins ? Lissés les réponses MAP pour éviter les surdosages et surtensions aussi dangeureux ? --> pas d'impératif pour avoir une réponse lissée

# Prompts
Let's clean our code, and separate the functions into different modules. 1 module for plottnig, 1 module with utils (loading functions, saving functions, printing functions).
Task: create 2 python modules --> utils.py + plots.py that will store all relevant functions from pkpd.py AND opti_patient_by_patient.py.
Keep in pkpd --> only methods related to pkpkd class and same for opti --> only functions for NL optimization with casadi.
Ask for clarifications if needed.

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
