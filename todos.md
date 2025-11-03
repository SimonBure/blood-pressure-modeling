# Todo admin
- asso doctorant dopamines (sport)
- PC en attente livraison
- mutuelle santé --> prendre la mienne ? --> appeler papa/maman
- activer carte étudiant
- activer imagine R sur pass navigo
- formulaire BD amsterdam par un praticien

# Todo scientifique
## Anesthésie
- simuler modèle jonas linéarisé + sans retard avec paramètres du papier --> euler implicite
   - CI ? --> t0 sur jeu de données JJ --> 0 pour les compartiments (absence drogue) et 0 cas statique, E_0 cas Windkessel
   - INOR ? --> protocole injection norépinéphrine à formaliser en fonction --> adapter unités bolus en débit.
   
- réinférer les paramètres globaux par optimisation sous-contrainte --> minimisation moindres carrés sur distance entre la sortie réelle (PA) et modélisée --> librairie cascadi.
  - contraintes = dynamique du modèle : x_(i+1) = x_i + dt * f_theta (x_(i+1), u_i).
  
- inférence paramètres des patients individuels par optimisation également --> cascadi. Guess initiaux paramètres pour descente de gradient --> fourchette donnée par écart-type TABLE 2 JJ
  
- étudier convexité et non-convexité du problème --> plot fonction de coût vs. variation 1 paramètre, le reste fixé.

- étudier importance du schéma de discrétisation --> comparer != + implicite vs explicite

- sensibilité et réduction du nombre de paramètres ?

- Optimisation paramètres population : un modèle pour tous les patients. Comparaison des données de chaque patient sur le modèle population global --> fonction coût loop sur les datas des patients en plus des points temporels et ajoute toutes les différences carrées entre data d'un patient et le modèle global.
Étapes :
  - Load toutes les datas --> dict patient spécifique
  - 

- estimation poids total db physionet + amsterdam utiles ~ 1-2TB.
- poly automatique florent


## Exploration DB
- analyse et exploration MIMIC IV / III --> recherche + grande DB avec entrées drugs + sorties PA. Chercher dans les différents modules.

- méthode ia inférence paramètres ?

# Besoins

# Questions


# Prompts
You are a professional data scientist with decade of expertise both in academic research and  and professor, specialized in giving precise, meaningful and clear as water courses on your domains of expertise.
My goals: after simulation of the PKPD model and by comparing the simulated versus the actual data (norepinephrine concentration / blood pressure e.g. figure 2 and 3) I want to compute the parameters value --> parameters inference. First: using values for drug plasmatic concentration/blood pressure taken as mean of those values for all individuals. Second, individual parameter inferences using the individual datas.
Write a small introduction to parameter inference statistical and AI based. What are the main approaches, algorithms, techniques ?


# Exploration
- Simulation complète système vasculaire / cardiaque / circulatoire corps humain avec/sans pathologies
- next-token prediction pour PA
- deep reinforcement learning pour PA
- apprentissage efficace sur court jeu de données
- PINN pour ci-dessus
- MIT Computational Physiology Lab exchange student
- Contact hopitaux pour obtenir base de données monitoring BP + injection drogue norépinéphrine / nicardipine --> Papa + Maman + Hermione + Jean-Yves + base Lariboisière
- état de l'art en optimisation --> creux à remplir

