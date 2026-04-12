# Projet Data/Auto : Modélisation, Comparaison et Réglage de Régulateurs PID

Ce projet a trois objectifs :
1. **Création d'un Modèle de Procédé (Machine Learning)** : Extraire des données historisées d'un procédé industriel, les prétraiter, puis entraîner un modèle (ex: RandomForestRegressor) pour prédire la variable de procédé (PV) d'une boucle de régulation.
2. **Comparaison de PID (Jumeau Numérique)** : Simuler le comportement d'un régulateur PID basé sur des paramètres configurés ou historisés et le comparer au comportement réel du PID de l'automate industriel.
3. **Réglage de PID (Tuner)** : Optimiser automatiquement les paramètres Kp/Ti/Td en simulant la réponse du procédé via le modèle ML et un solveur mathématique (scipy.optimize).

---
## 🔄 Flux de Données (Diagramme Logique)

```
┌─────────────────────────────────────────────────────────────────────┐
│  PIPELINE 1 : pid_model_builder_main.py                            │
│                                                                     │
│  SQL (History) ─→ extract_data ─→ preprocess ─→ create_lags ─→     │
│  split_chrono ─→ scale_data ─→ train_model ─→ evaluate ─→          │
│  save(model.joblib, scalers.joblib)                                 │
│                                                                     │
│  Config: pid_model_builder.ini                                      │
│  Sortie: Pid_Model/trained.joblib + scalers.joblib + graphique      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PIPELINE 2 : pid_comparator_main.py                               │
│                                                                     │
│  SQL (History) ─→ extract_data ─→ preprocess ─→                     │
│  PIDController.update(SP_real, PV_real) en boucle ─→                │
│  MV_simulée vs MV_réelle ─→ graphique comparatif                   │
│                                                                     │
│  Config: pid_comparator.ini (Kp/Ti/Td depuis tags ou fallback)      │
│  Sortie: trace/pid_comparison_plot.png                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PIPELINE 3 : pid_tuner_main.py                                    │
│                                                                     │
│  DÉPEND DE: Pipeline 1 (modèle ML entraîné)                        │
│                                                                     │
│  SQL (état initial) ─→ create_lags ─→ dernier point ─→              │
│  scipy.minimize( évaluer_perf(Kp,Ti,Td) ) ─→                       │
│    └─→ En boucle: PID.update + model.predict ─→ score IAE/ISE      │
│  Kp/Ti/Td optimaux                                                  │
│                                                                     │
│  Config: pid_tuner.ini                                              │
│  Sortie: paramètres PID optimisés (console/log)                     │
└─────────────────────────────────────────────────────────────────────┘
```

---
## 📂 Structure du Projet
Légende :
[PMB] 	: Utilisé/Généré par pid_model_builder_main.py
[PC] 	: Utilisé/Généré par pid_comparator_main.py
[PT]    : Utilisé/Généré par pid_tuner_main.py
[SHARED]: Utilisé par les deux scripts principaux

pid_project/
├── pid_model_builder_main.py       # [PMB] Création du modèle ML de prédiction
├── pid_comparator_main.py          # [PC]  Comparaison Simulateur PID vs Automate réel
├── pid_tuner_main.py               # [PT]  (WIP) Outil de réglage des paramètres PID
├── pid_model_builder.ini           # [PMB] Configuration pour le model builder
├── pid_comparator.ini              # [PC]  Configuration pour le comparateur PID
├── README.md                       # Documentation générale du projet
├── requirements.txt                # [SHARED] Dépendances Python (à créer/maintenir)
├── src/                            # [SHARED] Code source (Logique métier)
│   ├── __init__.py                 # [SHARED] Initialisateur de package Python
│   ├── config_loader.py            # [SHARED] Chargement de configuration et initialisation du logging
│   ├── data_processing.py          # [SHARED] Prétraitement, fractionnement, normalisation des données
│   ├── db_utils.py                 # [SHARED] Fonctions d'interaction avec la base de données
│   ├── feature_engineering.py      # [PMB]  Création des variables décalées
│   ├── modeling.py                 # [PMB]  Entraînement, évaluation, sauvegarde/chargement du modèle
│   ├── pid_logic/                  # [PC] Logique spécifique au PID
│   │   ├── __init__.py             # [PC] Initialisateur de package Python
│   │   └── pid_controller.py       # [PC] Classe du simulateur de régulateur PID
│   └── plotting_utils.py           # [SHARED] Utilitaires de génération de graphiques
├── trace/                            # Répertoire pour les logs et graphiques générés
│   ├── pid_model_builder_log.txt   # [PMB] Log du model builder
│   ├── pid_comparator_log.txt      # [PC] Log du comparateur PID
│   ├── pid_model_pv_prediction_plot.png # [PMB] Graphique de prédiction du modèle
│   └── pid_comparison_plot.png     # [PC] Graphique de comparaison des PIDs
└── Pid_Model/                        # Répertoire pour le modèle et les scalers sauvegardés
    └── trained.joblib              # [PMB] Modèle de procédé entraîné
    └── scalers.joblib              # [PMB] Scalers (Normalisateurs) pour les features et la cible du modèle

---
## ⚙️ Prérequis et Installation

*   Python 3.9+ (testé avec 3.14)
*   Accès à une instance SQL Server (ou la base de données configurée).
*   **Driver ODBC pour SQL Server** (par exemple, "ODBC Driver 17 for SQL Server" ou "SQL Server") installé sur Windows.

### 1. Préparation de l'environnement (Recommandé)
Dans un terminal VS Code (ou PowerShell) à la racine du projet :
```powershell
# Autoriser l'exécution des scripts si PowerShell bloque l'activation (Erreur PSSecurityException)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement virtuel
.venv\Scripts\activate

# Installer les bibliothèques requises
pip install -r requirements.txt
```

---
## 🛠️ Configuration (.ini)

La configuration du projet est gérée via des fichiers `.ini` distincts pour chaque fonctionnalité principale :

*   **`pid_model_builder.ini`**: Pour la création du modèle de procédé.
    *   Définit la connexion SQL, les tags à extraire pour entraîner l'IA (PV, MV, SP), la période d'extraction et les hyperparamètres du modèle (RandomForest).
*   **`pid_comparator.ini`**: Pour la comparaison du PID simulé vs automate.
    *   `[Database]`: Connexion BDD.
    *   `[Tags]`: Tags pour PV, MV, SP, état d'activation du PID automate (optionnel), paramètres PID (Kp, Ti, Td si historisés et suivis par le simulateur).
    *   `[DataPeriod]`: Période d'extraction et résolution.
    *   `[PIDSimParams]`: Paramètres de secours pour Kp, Ti, Td du PID simulé, période d'échantillonnage de la simulation (`Tsamp`), limites MV, action directe/inverse.
    *   `[Output]`: Chemin de sauvegarde pour le graphique de comparaison, configuration des logs.

---
## 🚀 Utilisation

1. **Configurer les fichiers `.ini`** :
   Modifiez `pid_model_builder.ini` ou `pid_comparator.ini` avec les identifiants de votre base de données et les tags de votre automate.

2. **Exécuter les pipelines** :
    *   Placez-vous à la racine du projet (`pid_project/`) dans un terminal.
    *   Pour créer le modèle de procédé :
        ```bash
        python pid_model_builder_main.py
        ```
    *   Pour exécuter la comparaison de PID :
        ```powershell
        python pid_comparator_main.py
        ```

---
## 📊 Sorties et Résultats

*   **Fichiers de log** : Dans `trace/` (par exemple, `pid_model_builder_log.txt`, `pid_comparator_log.txt`), contenant les détails de chaque étape.
*   **Graphiques** :
    *   `trace/pid_model_pv_prediction_plot.png`: (par `pid_model_builder`) Compare PV réelle vs. PV prédite.
    *   `trace/pid_comparison_plot.png`: (par `pid_comparator`) Compare PV/SP, MV automate/simulée, et état d'activation des PIDs.
*   **Modèle et Scalers** (par `pid_model_builder`) :
    *   `Pid_Model/trained.joblib`: Le modèle de procédé entraîné.
    *   `Pid_Model/scalers.joblib`: Les normalisateurs (scalers) pour les entrées (X) et la sortie (Y).

---
## 🚑 Dépannage (Troubleshooting)

*   **`ImportError` / `ModuleNotFoundError`** :
    *   Assurez-vous d'exécuter les scripts `*_main.py` depuis la racine du projet (`pid_project/`).
    *   Vérifiez que votre environnement virtuel est activé et que toutes les dépendances de `requirements.txt` sont installées.
*   **Erreur `pyodbc.InterfaceError: ('IM002', '[IM002] Source de données introuvable et nom de pilote non spécifié...')`** :
    *   Cela signifie que le pilote (driver) spécifié dans la clé `odbc_driver` du fichier `.ini` n'est pas installé sur votre ordinateur. 
    *   Solution : Modifiez le nom pour qu'il corresponde à un pilote installé (ex: `SQL Server`) ou téléchargez le pilote "ODBC Driver 17 for SQL Server" depuis le site de Microsoft.
*   **`KeyError` dans la configuration** :
    *   Indique souvent une section ou une option manquante dans le fichier `.ini`. Le log vous précisera quelle clé est manquante.
*   **`NameError: name 'configparser' is not defined`** (ou similaire pour d'autres modules) :
    *   Assurez-vous que le module est bien importé (`import configparser`) au début du fichier Python où l'erreur se produit.
*   **Problèmes de chemins de fichiers** :
    *   Les chemins relatifs dans les fichiers `.ini` (pour `log_file_base_name`, `plot_save_path`, etc.) sont relatifs au répertoire à partir duquel le script est exécuté (normalement la racine `pid_project/`).

---
## ⚠️ Pièges Courants dans les fichiers .ini

### pid_tuner.ini — Section [SCENARIO_PARAMS]
| Paramètre | Obligatoire | Par défaut | Piège |
|-----------|-------------|------------|-------|
| `sp_initial_offset` | Oui | 0.0 | Si `sp_step_value_offset` est aussi à 0, le solveur n'a aucun échelon à optimiser et retourne le point de départ ! |
| `sp_step_time_seconds` | Oui | 15.0 | Doit être < `simulation_duration_seconds` sinon l'échelon n'est jamais appliqué |
| `sp_step_value_offset` | Oui | 2.0 | C'est **l'amplitude de l'échelon**. Mettre une valeur réaliste pour le procédé (ex: 2.0 unités). **Jamais 0** pour l'optimisation ! |

### pid_tuner.ini — Section [SIMULATION_PARAMS]
| Paramètre | Obligatoire | Par défaut | Piège |
|-----------|-------------|------------|-------|
| `tsamp_simulation_ms` | Oui | 100 | Doit correspondre à `ww_resolution_ms` des données d'entraînement |
| `mv_min` / `mv_max` | Oui | 0/100 | Si trop étroit, le PID sature et tous les jeux de paramètres donnent le même score |
| `simulation_duration_seconds` | Oui | 60 | Augmenter si le procédé est lent (Ti > 500s) |

### pid_model_builder.ini — Section [ModelFeatures]
| Paramètre | Obligatoire | Par défaut | Piège |
|-----------|-------------|------------|-------|
| `pv_lags` / `mv_lags` / `sp_lags` | Oui | 0 | Trop de lags par rapport aux données = DataFrame vide après dropna |
| `train_split_ratio` + `validation_split_ratio` | Oui | 0.7 + 0.15 | La somme doit être **< 1.0** |

### Tous les .ini — Section [Database]
| Paramètre | Obligatoire | Piège |
|-----------|-------------|-------|
| `db_password` | Si auth SQL | **Stocké en clair.** Pour la production, utiliser `trusted_connection = yes` (Windows Auth) |
| `odbc_driver` | Oui | `SQL Server` (ancien) vs `ODBC Driver 17 for SQL Server` (nouveau). Doit correspondre au driver installé |

### Cohérence entre fichiers .ini
- Les sections `[Tags]`, `[DataPeriod]`, `[ModelFeatures]` doivent être **identiques** entre `pid_model_builder.ini` et `pid_tuner.ini` (mêmes tags, mêmes lags)
- Si le modèle est entraîné avec 10 lags PV mais que le tuner en attend 5, le scaler crashera (colonnes manquantes)

---
## 🔮 Améliorations Futures
- Utiliser `differential_evolution` (optimisation globale) au lieu de `L-BFGS-B` (local) pour le tuner
- Ajouter des métriques combinées (IAE + pénalité de dépassement + pénalité de variation de MV)
- Externaliser les mots de passe via variables d'environnement ou un fichier `.env`
- Ajouter des tests unitaires pour `PIDController` et `evaluate_closed_loop_performance`
- Utiliser `pid_single_simulation.py` comme moteur de simulation unifié pour le tuner