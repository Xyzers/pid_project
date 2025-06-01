
# Modélisation et Comparaison de Régulateurs PID

Ce projet a deux objectifs principaux :
1.  **Création d'un Modèle de Procédé** : Extraire des données historisées d'un procédé industriel, les prétraiter, puis entraîner un modèle de Machine Learning (par exemple, RandomForestRegressor) pour prédire la variable de procédé (PV - Process Variable) d'une boucle de régulation PID.
2.  **Comparaison de PID** : Simuler le comportement d'un régulateur PID basé sur des paramètres configurés ou historisés et le comparer au comportement réel du PID de l'automate, en utilisant les mêmes données de PV et SP (Setpoint) historisées.

Les modèles entraînés, les normalisateurs (scalers) et les graphiques d'analyse sont sauvegardés pour une utilisation ultérieure.

## Structure du Projet
Légende :
[PMB] 	: Utilisé/Généré par pid_model_builder_main.py
[PC] 	: Utilisé/Généré par pid_comparator_main.py
[SHARED]: Utilisé par les deux scripts principaux

pid_project/
├── pid_model_builder_main.py       # [PMB] Script principal pour la création du modèle de procédé
├── pid_comparator_main.py          # [PC] Script principal pour la comparaison de PID
├── pid_model_builder.ini           # [PMB] Fichier de configuration pour le model builder
├── pid_comparator.ini              # [PC] Fichier de configuration pour le comparateur PID
├── README.md                       # Documentation générale du projet
├── requirements.txt                # [SHARED] Dépendances Python (à créer/maintenir)
├── src/                            # [SHARED] Code source modulaire
│   ├── __init__.py                 # [SHARED] Initialisateur de package Python
│   ├── config_loader.py            # [SHARED] Chargement de configuration et initialisation du logging
│   ├── data_processing.py          # [SHARED] Prétraitement, fractionnement, normalisation des données
│   ├── db_utils.py                 # [SHARED] Fonctions d'interaction avec la base de données
│   ├── feature_engineering.py      # [PMB] Création des variables décalées
│   ├── modeling.py                 # [PMB] Entraînement, évaluation, sauvegarde/chargement du modèle
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
    └── scalers.joblib              # [PMB] Scalers pour les features et la cible du modèle

## Prérequis

*   Python 3.9+ (testé avec 3.10)
*   Bibliothèques Python listées dans `requirements.txt`. Principales bibliothèques :
    *   pandas
    *   scikit-learn
    *   SQLAlchemy
    *   pyodbc (ou le driver de base de données approprié pour votre SGBD)
    *   matplotlib
    *   joblib
*   Accès à une instance SQL Server (ou la base de données configurée).
*   Driver ODBC pour SQL Server (par exemple, "ODBC Driver 17 for SQL Server") correctement installé et configuré.

### Fichier `requirements.txt` (Suggestion)

Il est fortement recommandé de créer et maintenir un fichier `requirements.txt`. Vous pouvez le générer avec `pip freeze > requirements.txt` dans votre environnement virtuel après avoir installé toutes les dépendances. Un contenu de base pourrait être :
* pandas>=1.3
* scikit-learn>=1.0
* SQLAlchemy>=1.4
* pyodbc>=4.0 # Si vous utilisez SQL Server avec pyodbc
* matplotlib>=3.4
* joblib>=1.1


## Configuration

La configuration du projet est gérée via des fichiers `.ini` distincts pour chaque fonctionnalité principale :

*   **`pid_model_builder.ini`**: Pour la création du modèle de procédé.
    *   `[Database]`: Connexion BDD.
    *   `[Tags]`: Tags pour PV, MV, SP, paramètres PID (Kp, Ti, Td si historisés), et perturbations.
    *   `[DataPeriod]`: Période d'extraction et résolution.
    *   `[ModelFeatures]`: Lags pour les features, type de scaler, ratios de split.
    *   `[ModelParams]`: Hyperparamètres du modèle ML.
    *   `[ModelOutput]`: Chemins de sauvegarde pour modèle, scalers, graphique de prédiction.
    *   `[Output]`: Configuration des logs (nom de base).
*   **`pid_comparator.ini`**: Pour la comparaison du PID simulé vs automate.
    *   `[Database]`: Connexion BDD.
    *   `[Tags]`: Tags pour PV, MV, SP, état d'activation du PID automate (optionnel), paramètres PID (Kp, Ti, Td si historisés et suivis par le simulateur).
    *   `[DataPeriod]`: Période d'extraction et résolution.
    *   `[PIDSimParams]`: Paramètres de secours pour Kp, Ti, Td du PID simulé, période d'échantillonnage de la simulation (`Tsamp`), limites MV, action directe/inverse.
    *   `[Output]`: Chemin de sauvegarde pour le graphique de comparaison, configuration des logs.

**Avant d'exécuter les scripts, assurez-vous que les fichiers `.ini` correspondants sont correctement configurés.**

## Utilisation

1.  **Configurer l'environnement** :
    *   Assurez-vous que Python et les prérequis sont installés.
    *   Il est fortement recommandé de travailler dans un environnement virtuel Python.
        ```bash
        python -m venv .venv
        source .venv/bin/activate  # Linux/macOS
        # .venv\Scripts\activate    # Windows
        ```
    *   Installez les dépendances : `pip install -r requirements.txt`.

2.  **Configurer les fichiers `.ini`** :
    *   Modifiez `pid_model_builder.ini` et/ou `pid_comparator.ini` avec les paramètres spécifiques à votre environnement et à votre procédé.

3.  **Exécuter les pipelines** :
    *   Placez-vous à la racine du projet (`pid_project/`) dans un terminal.
    *   Pour créer le modèle de procédé :
        ```bash
        python pid_model_builder_main.py
        ```
    *   Pour exécuter la comparaison de PID :
        ```bash
        python pid_comparator_main.py
        ```
    *   Si vous utilisez un IDE comme VS Code avec un interpréteur Python configuré, vous pouvez généralement exécuter les fichiers directement.

## Sorties des Scripts

*   **Fichiers de log** : Dans `trace/` (par exemple, `pid_model_builder_log.txt`, `pid_comparator_log.txt`), contenant les détails de chaque étape.
*   **Graphiques** :
    *   `trace/pid_model_pv_prediction_plot.png`: (par `pid_model_builder`) Compare PV réelle vs. PV prédite.
    *   `trace/pid_comparison_plot.png`: (par `pid_comparator`) Compare PV/SP, MV automate/simulée, et état d'activation des PIDs.
*   **Modèle et Scalers** (par `pid_model_builder`) :
    *   `Pid_Model/trained.joblib`: Le modèle de procédé entraîné.
    *   `Pid_Model/scalers.joblib`: Les normalisateurs (scalers) pour les features (X) et la cible (Y).

## Description des Modules (`src/`)

*   `config_loader.py`: Charge les fichiers `.ini` et initialise le système de logging.
*   `db_utils.py`: Gère la connexion à la base de données et la construction des requêtes SQL.
*   `data_processing.py`: Prétraitement des données (nettoyage, conversion, ffill/bfill), fractionnement chronologique, et normalisation.
*   `feature_engineering.py`: (Utilisé par `pid_model_builder`) Crée des variables décalées (lags) pour modélisation.
*   `modeling.py`: (Utilisé par `pid_model_builder`) Instanciation, entraînement, évaluation du modèle ML, sauvegarde/chargement du modèle et des scalers.
*   `pid_logic/pid_controller.py`: Contient la classe `PIDController` pour la simulation du régulateur PID, incluant la gestion de l'état actif/inactif et le transfert sans à-coup (bumpless).
*   `plotting_utils.py`: Fonctions pour générer les divers graphiques de visualisation.

## Dépannage

*   **`ImportError` / `ModuleNotFoundError`** :
    *   Assurez-vous d'exécuter les scripts `*_main.py` depuis la racine du projet (`pid_project/`).
    *   Vérifiez que votre environnement virtuel est activé et que toutes les dépendances de `requirements.txt` sont installées.
*   **Erreur de connexion à la base de données / SQL** :
    *   Vérifiez les informations d'identification, noms de serveur/base, driver ODBC dans le fichier `.ini` pertinent.
    *   Assurez-vous que le driver ODBC est correctement installé.
    *   Vérifiez les noms des tags et de la table dans le `.ini`.
*   **`KeyError` dans la configuration** :
    *   Indique souvent une section ou une option manquante dans le fichier `.ini`. Le message d'erreur du log devrait indiquer la clé manquante.
*   **`NameError: name 'configparser' is not defined`** (ou similaire pour d'autres modules) :
    *   Assurez-vous que le module est bien importé (`import configparser`) au début du fichier Python où l'erreur se produit.
*   **Problèmes de chemins de fichiers** :
    *   Les chemins relatifs dans les fichiers `.ini` (pour `log_file_base_name`, `plot_save_path`, etc.) sont relatifs au répertoire à partir duquel le script est exécuté (normalement la racine `pid_project/`).
