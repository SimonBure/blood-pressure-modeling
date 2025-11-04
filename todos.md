# Todo admin
- asso doctorant dopamines (sport)
- PC en attente livraison
- mutuelle santé --> prendre la mienne ? --> appeler papa/maman
- activer carte étudiant

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


- présentation thèse pour les médecins le 12 novembre (20 minutes)
  - Chirurgie / NICU
  - 


## Exploration DB
- analyse et exploration MIMIC IV / III --> recherche + grande DB avec entrées drugs + sorties PA. Chercher dans les différents modules.

- méthode ia inférence paramètres ?

# Besoins

# Questions


# Prompts
You are a NLP optimization expert, totally familiar with casadi python lib.

Context & Goal: I want to estimate the best parameter values of my pkpd multi-compartment population model that is defined in details in pkpd.py or in paper biblio/jonas_pkpd.pdf. I want to use casadi Direct Multiple Shooting NLP solver in a similar way that of example script test_casadi.py. We want 1 model at the population level, that could be shared by all patients.

Task : Write a detailed prompt for coding agent with every details written down. Ask user for ambiguity clearance or critical question to answer before coding.

Specifications: update and write code in opti_popuplation.py.
Reuse maximum of pkpd.py functions --> load_observation, load_injections, INOR for drug injection protocol depending on patient etc.
Follow following casadi code structure for optimization NLP problem : 
- Define time parameters : max time is ~2200 sec. Time step : 1 time point every 30 sec.
- Define opti variables --> all pk parameters except T_lag and exponents gamma & beta (we are first working with linear version of pkpd model), all pd parameters (emax + windkessel) + compartment state variables (A_d, A_c, A_p, E emax + windkessel)
- Define all initial conditions constraints + traj constraints with euler implicit schemes like in pkpd.py
- Compute cost as sum square of differences between scheme and data obs for all patients. (obsid = 0 -->  Cc - obs ; obsid=1 --> E - obs). Data output from load_observation method will be a dict with key = patient_id and for every patient_id key, a dict with NOR plasmatic concentration AND blood_pressure available time series for this patient. Time sampling for those 2 values are inconsistant between patients and time points will be different from our 30sec grid. Time alignment problem between observation and simulated values for plasmatic concentration Cc and effect on blood pressure E. for any time point k, linear interpolation between two closest data points for the relevant observation (Cc or blood pressure) to solve the time alignment problem. If k is before first observation or after last observtion, just take exact first or last observation as the data to compute the cost contribution.
- Define initial guesses for optimization variables. PKPD parameters are initialized with true values from pkpd model. State trajectories variable will be initialized with the simulated trajectory from patient 23, stored in codes/res/patient_23/ --> in all explicit .npy files.

Output: text in a terminal-friendly format, structured in smallest task with implementation details en examples to maximize coding agent performances.

Let's change strategy to start in a simpler way. Let's optimize the parameters patient by patient. For each patient_id in patients_ids :
  - generate ~100 data points evenly distributed on curve (but we will the output by putting them on a graph with the simulated curve to compare) using previously done simulations pkpd.py. .npy files in codes/res/patient_id/linear_no_lag/files.npy. Each .npy file contains temporal trajectory from a state variable (Ac, Ad, Ab, BP_emax, BP_windkessel)
  - casadi code : optimization of all pk-pd parameters for a given patient. Time points --> same as number of generated data to ease future cost computation.
  - opti variables are all parameters to optimize + state variables of the model.
  - compute cost with sum squared differences between model and data (using as always the appropriate variable from obs)
  - prints optimal parameters values + plots of simulated "true" curve + generated data + curve with estimated parameters. You can reuse pkpd class with the estimated parameters.
If ambiguity ask user for clarity. Reuse knowledge from previous interactions + code in pkpd.py if relevant.
Output : text prompt for a coding agent. Structured in small tasks + details for task. 


# Exploration
- Simulation complète système vasculaire / cardiaque / circulatoire corps humain avec/sans pathologies
- Suivi d'autres variables vitales
- next-token prediction pour PA
- deep reinforcement learning pour PA
- apprentissage efficace sur petit jeu de données
- PINN pour ci-dessus
- Regarder modèle IA / math pour biology état de l'art (dernier modèle très bon entraînement sur jeux de données bruitées / petits / bons pour modéliser des effets très incomplet)
- Whole patient biological model --> avec toutes les donnéees.
- MIT Computational Physiology Lab exchange student
- Contact hopitaux pour obtenir base de données monitoring BP + injection drogue norépinéphrine / nicardipine --> Papa + Maman + Hermione + Jean-Yves + base Lariboisière
- état de l'art en optimisation --> creux à remplir

