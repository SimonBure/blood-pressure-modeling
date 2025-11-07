# Todo admin
- asso doctorant dopamines (sport)
- PC en attente livraison
- mutuelle santé --> prendre la mienne ? --> appeler papa/maman

# Todo scientifique
## Anesthésie
- regarder inférence paramètres pour +100 points
- inférer pour le reste des patients

- étudier convexité et non-convexité du problème --> plot fonction de coût vs. variation 1 paramètre, le reste fixé.

- étudier importance du schéma de discrétisation --> comparer != + implicite vs explicite

- sensibilité et réduction du nombre de paramètres ?

- estimation poids total db physionet + amsterdam utiles ~ 1-2TB.

- présentation thèse pour les médecins le 12 novembre (20 minutes)
  - résumé thèse. real-time monitoring BP during surgery + real-time patient adaptation
  - étude clinique NICU


## Exploration DB
- analyse et exploration MIMIC IV / III --> recherche + grande DB avec entrées drugs + sorties PA. Chercher dans les différents modules.

- méthode ia inférence paramètres ?

# Besoins
- De quoi ont besoin les médecins et chirurgiens ? / Pour quoi doit-on créer un modèle ?

# Questions
- exponents beta & gamma as model parameters to be optimized ?
- casadi optimized traj vs pkpd model --> why not the same thing ?
- negative parameters values --> add > 0 constraints on parameters ?
- cost computation adding BP differences AND Cc differences altogether ? Meaningless addition of different units ? 

# Prompts
initial guesses for parameters --> population values
initial guesses for state variables --> pkpd simulated traj of the patient

- initial conditions constraints on E ('effect' --> basal blood pressure of a specific patient) is staticly fixed at 57.09 but we want it to be patient specific


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
