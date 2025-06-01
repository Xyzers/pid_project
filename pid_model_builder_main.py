# main.py
import sys
from pathlib import Path
import logging # Importer logging ici aussi
import configparser
import sqlalchemy # Pour les types d'exception
import winsound # Si vous voulez garder le Beep

# Importer les fonctions depuis vos modules
from src.config_loader import load_config_and_setup_logging
from src.db_utils import get_db_engine, build_sql_query, extract_data
from src.data_processing import preprocess_timeseries_data, split_data_chronological, scale_data
from src.feature_engineering import create_lagged_features
from src.modeling import train_model, evaluate_model, save_model_and_scalers
from src.plotting import plot_predictions

# logger sera initialisé après la configuration du logging
logger = logging.getLogger(__name__) # ou getLogger("pid_application")

def run_pipeline():
    # Déterminer le chemin du script et le nom de base pour la config/log
    try:
        # Si exécuté comme script, __file__ est défini
        executed_script_path = Path(__file__).resolve()
    except NameError:
         # Tombe ici si exécuté dans un environnement interactif (comme un notebook)
         # où __file__ n'est pas défini. sys.argv[0] peut être une alternative.
        executed_script_path = Path(sys.argv[0]).resolve()

    # Le nom de base pour le fichier .ini et potentiellement pour le log par défaut
    # Ce nom est utilisé pour trouver le fichier .ini, e.g., pid_model_builder.ini
    # Le fichier .ini lui-même est attendu dans le même répertoire que main.py ou un chemin spécifié
    config_base_name = "pid_model_builder"
    config_file_actual_name = f"{config_base_name}.ini" # Assurez-vous que pid_model_builder.ini est au bon endroit

    config = None
    try:
        # Le nom du fichier log sera déterminé dans load_config_and_setup_logging
        config = load_config_and_setup_logging(script_stem_for_config_file=config_base_name, 
                                               config_file_name=config_file_actual_name)
    except FileNotFoundError:
        # Message déjà affiché par load_config_and_setup_logging
        sys.exit(1)
    except Exception as e_cfg:
        # Utiliser print car le logger pourrait ne pas être pleinement fonctionnel
        print(f"ERREUR CRITIQUE: Échec config/logging: {e_cfg}", file=sys.stderr)
        # Tenter de logger si possible (si au moins un handler console est actif)
        if logging.getLogger().hasHandlers():
             logging.critical(f"Échec config/logging: {e_cfg}", exc_info=True)
        sys.exit(1)

    # Maintenant que le logging est configuré, on peut utiliser logger partout
    logger.info(f"--- Démarrage du script (fichier exécuté: {executed_script_path.name}) ---")
    # Si le nom du script exécuté est différent du nom de base de la config, logguer l'info
    if executed_script_path.stem != config_base_name:
        logger.info(f"--- Utilisation du nom de base '{config_base_name}' pour le fichier de configuration .ini ---")

    # Le reste de votre logique principale, en appelant les fonctions importées
    logger.info("--- Démarrage de la phase de création de modèle de procédé ---")
    db_engine = None # Pour le finally
    try:
        required_sections = ['Database', 'Tags', 'DataPeriod', 'ModelFeatures', 'ModelParams', 'ModelOutput']
        missing_sections = [s for s in required_sections if s not in config]
        if missing_sections:
            logger.error(f"Sections de config manquantes: {', '.join(missing_sections)}. Arrêt.")
            sys.exit(1)

        db_engine = get_db_engine(config['Database'])
        table_name = config['Database'].get('table_name', 'History')
        sql_query, column_aliases = build_sql_query(config['Tags'], config['DataPeriod'], table_name)
        raw_df = extract_data(db_engine, sql_query)

        if raw_df.empty:
            logger.error("Aucune donnée extraite. Arrêt.")
            sys.exit(1)

        essential_cols_for_builder = ['PV_real', 'MV_real', 'SP_real'] # Adaptez ces alias à ce que build_sql_query retourne

        clean_df = preprocess_timeseries_data(
            raw_df,
            column_aliases, # Assurez-vous que 'column_aliases' est la liste des alias retournés par build_sql_query
            essential_value_cols=essential_cols_for_builder
)
        if clean_df.empty:
            logger.error("Données vides après pré-traitement. Arrêt.")
            sys.exit(1)
        
        # Logguer les statistiques descriptives pour Kp_hist, Ti_hist, Td_hist si présentes
        for col_hist_param in ['Kp_hist', 'Ti_hist', 'Td_hist']:
            if col_hist_param in clean_df.columns:
                logger.info(f"Stats pour {col_hist_param} dans clean_df:\n{clean_df[col_hist_param].describe()}")
            else:
                # Ceci était déjà loggué dans le script original mais peut être utile de le garder
                logger.info(f"Colonne {col_hist_param} non présente dans clean_df pour statistiques (tag optionnel).")


        X, y = create_lagged_features(clean_df, config['ModelFeatures'], target_col='PV_real')
        if X.empty or y.empty:
            logger.error("Features (X) ou cible (y) vides après lags. Arrêt.")
            sys.exit(1)

        train_ratio = config['ModelFeatures'].getfloat('train_split_ratio', 0.7)
        val_ratio = config['ModelFeatures'].getfloat('validation_split_ratio', 0.15)
        if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and (train_ratio + val_ratio) < 1):
            logger.error(f"Ratios de division invalides: train={train_ratio}, val={val_ratio}.")
            raise ValueError("Ratios de division invalides.")

        X_train, X_val, X_test, y_train, y_val, y_test_original_series = \
            split_data_chronological(X, y, train_ratio, val_ratio)
        
        y_test_index_for_plot_and_df = y_test_original_series.index

        if X_test.empty or y_test_original_series.empty:
            logger.warning("Ensemble de test vide après division.")

        scaler_type_cfg = config['ModelFeatures'].get('scaler_type', 'MinMaxScaler')
        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, _, scaler_X_obj, scaler_y_obj = \
            scale_data(X_train, X_val, X_test, y_train, y_val, y_test_original_series, scaler_type_cfg)
        
        if X_train_s.shape[0] == 0: # y_test_s_unused
            logger.error("Données d'entraînement vides après scaling. Arrêt.")
            sys.exit(1)

        model = train_model(X_train_s, y_train_s, config['ModelParams'])
        
        _, _, _, predictions_df = evaluate_model(model, X_test_s, y_test_original_series.values, scaler_y_obj, y_test_index_for_plot_and_df)
        
        plot_file_path_str = config['ModelOutput'].get('plot_save_path', None)
        plot_file_path = Path(plot_file_path_str) if plot_file_path_str else None

        if not predictions_df.empty:
            plot_predictions(predictions_df, plot_file_path)
        else:
            logger.warning("DataFrame prédictions vide, graphique non généré.")

        save_model_and_scalers(model, scaler_X_obj, scaler_y_obj, config['ModelOutput'])
        
        logger.info("Script de création de modèle de procédé terminé avec succès.")

    except FileNotFoundError as fnf_err: 
        logger.error(f"Erreur fichier non trouvé: {fnf_err}", exc_info=True)
        sys.exit(1)
    except ValueError as val_err: 
        logger.error(f"Erreur valeur/configuration: {val_err}", exc_info=True)
        sys.exit(1)
    except sqlalchemy.exc.SQLAlchemyError as db_err: 
        logger.error(f"Erreur base de données: {db_err}", exc_info=True)
        sys.exit(1)
    except configparser.Error as cfg_parser_err: 
        logger.error(f"Erreur fichier .ini: {cfg_parser_err}", exc_info=True)
        sys.exit(1)
    except KeyError as key_err: 
        logger.error(f"Clé config manquante: '{key_err}'. Vérifiez .ini.", exc_info=True)
        sys.exit(1)
    except NotImplementedError as ni_err: 
        logger.error(f"Fonctionnalité non implémentée: {ni_err}", exc_info=True)
        sys.exit(1)
    except ImportError as imp_err: 
        logger.error(f"Erreur importation - paquet manquant: {imp_err}", exc_info=True)
        sys.exit(1)
    except Exception as e_main: 
        logger.error(f"Erreur inattendue: {e_main}", exc_info=True)
        sys.exit(1)
    finally:
        if db_engine:
            try:
                db_engine.dispose()
                logger.info("Moteur BD fermé.")
            except Exception as e_dispose:
                logger.error(f"Erreur fermeture moteur BD: {e_dispose}", exc_info=True)
        logger.info(f"--- Fin du script (fichier exécuté: {executed_script_path.name}) ---")
        try:
            winsound.Beep(1000, 500) # Optionnel, dépend de si winsound est dispo/voulu
        except ImportError:
            logger.info("Module winsound non trouvé, pas de bip de fin.")
        except RuntimeError: # Sur les systèmes sans son
             logger.info("Impossible de jouer le son de fin (pas de périphérique audio ou winsound non dispo).")


if __name__ == "__main__":
    run_pipeline()