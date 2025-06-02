# pid_tuner_main.py (version remaniée pour débogage intensif)

# Imports Python Standard Library en premier
import sys
import logging # Sera initialisé plus tard
import configparser
from pathlib import Path

# Un print initial pour voir si le fichier est au moins lu par Python
print("DEBUG_GLOBAL: Le fichier pid_tuner_main.py est en cours d'interprétation.", file=sys.stderr)
print(f"DEBUG_GLOBAL: Version Python: {sys.version}", file=sys.stderr)
print(f"DEBUG_GLOBAL: Exécutable Python: {sys.executable}", file=sys.stderr)
print(f"DEBUG_GLOBAL: Répertoire courant (CWD): {Path.cwd()}", file=sys.stderr)
# print(f"DEBUG_GLOBAL: sys.path: {sys.path}", file=sys.stderr) # Peut être très long

# Vérification de base du dossier 'src'
src_path_check = Path("./src")
if not src_path_check.is_dir():
    print(f"ERREUR CRITIQUE: Le dossier 'src' ({src_path_check.resolve()}) est introuvable. "
          f"Lancez le script depuis la racine du projet.", file=sys.stderr)
    sys.exit("Arrêt prématuré: Dossier 'src' manquant.")
else:
    print(f"DEBUG_GLOBAL: Dossier 'src' trouvé: {src_path_check.resolve()}", file=sys.stderr)

# Imports des bibliothèques tierces
try:
    print("DEBUG_IMPORT: Tentative d'import de pandas...", file=sys.stderr)
    import pandas as pd
    print("DEBUG_IMPORT: pandas importé.", file=sys.stderr)

    print("DEBUG_IMPORT: Tentative d'import de numpy...", file=sys.stderr)
    import numpy as np
    print("DEBUG_IMPORT: numpy importé.", file=sys.stderr)

    print("DEBUG_IMPORT: Tentative d'import de matplotlib.pyplot...", file=sys.stderr)
    import matplotlib.pyplot as plt
    print("DEBUG_IMPORT: matplotlib.pyplot importé.", file=sys.stderr)
except ImportError as e_import_tiers:
    print(f"ERREUR CRITIQUE D'IMPORTATION (bibliothèque tierce): {e_import_tiers}", file=sys.stderr)
    sys.exit(f"Arrêt: Échec de l'importation d'une bibliothèque tierce ({e_import_tiers}).")

# Imports des modules personnalisés 'src'
try:
    print("DEBUG_IMPORT: Tentative d'import de src.config_loader...", file=sys.stderr)
    from src.config_loader import load_config_and_setup_logging
    print("DEBUG_IMPORT: src.config_loader importé.", file=sys.stderr)

    print("DEBUG_IMPORT: Tentative d'import de src.db_utils...", file=sys.stderr)
    from src.db_utils import get_db_engine, build_sql_query, extract_data
    print("DEBUG_IMPORT: src.db_utils importé.", file=sys.stderr)

    print("DEBUG_IMPORT: Tentative d'import de src.data_processing...", file=sys.stderr)
    from src.data_processing import preprocess_timeseries_data
    print("DEBUG_IMPORT: src.data_processing importé.", file=sys.stderr)

    print("DEBUG_IMPORT: Tentative d'import de src.modeling...", file=sys.stderr)
    from src.modeling import load_model_and_scalers
    print("DEBUG_IMPORT: src.modeling importé.", file=sys.stderr)

    print("DEBUG_IMPORT: Tentative d'import de src.pid_simulation.pid_single_simulation...", file=sys.stderr)
    from src.pid_simulation.pid_single_simulation import evaluer_performance_pid, prepare_initial_model_input
    print("DEBUG_IMPORT: src.pid_simulation.pid_single_simulation importé.", file=sys.stderr)
except ImportError as e_import_src:
    print(f"ERREUR CRITIQUE D'IMPORTATION (module src): {e_import_src}", file=sys.stderr)
    print("Vérifiez que tous les modules src, leurs __init__.py sont corrects, "
          "et qu'il n'y a pas d'erreurs de syntaxe ou d'imports circulaires dans ces modules.", file=sys.stderr)
    sys.exit(f"Arrêt: Échec de l'importation d'un module src ({e_import_src}).")
except Exception as e_import_autre: # Attrape d'autres erreurs potentielles pendant les imports de src
    print(f"ERREUR INATTENDUE PENDANT L'IMPORTATION d'un module src: {e_import_autre}", file=sys.stderr)
    sys.exit(f"Arrêt: Erreur inattendue lors de l'importation d'un module src ({e_import_autre}).")

# Initialisation du logger global (sera configuré plus tard)
logger = None

def run_pid_tuning_simulation(): # Nom de fonction plus descriptif
    global logger # Pour pouvoir assigner au logger global

    print("DEBUG_MAIN: Entrée dans run_pid_tuning_simulation()", file=sys.stderr)
    config = None # Définir config en dehors du try pour qu'il soit accessible dans le finally (si besoin)

    # --- Étape 1: Chargement de la configuration et initialisation du Logging ---
    try:
        script_name_stem = "pid_tuner_simulation"
        config_file_name = f"{script_name_stem}.ini"
        print(f"DEBUG_MAIN: Fichier de configuration cible: {config_file_name}", file=sys.stderr)

        config_path_check = Path(config_file_name)
        if not config_path_check.is_file():
            print(f"ERREUR CRITIQUE: Le fichier de configuration '{config_path_check.resolve()}' est introuvable.", file=sys.stderr)
            sys.exit(f"Arrêt: Fichier config '{config_file_name}' manquant.")
        else:
            print(f"DEBUG_MAIN: Fichier config '{config_path_check.resolve()}' trouvé.", file=sys.stderr)

        config = load_config_and_setup_logging(
            script_stem_for_config_file=script_name_stem,
            config_file_name=config_file_name
        )
        # Le logger est maintenant configuré, obtenons une instance pour ce module
        logger = logging.getLogger(__name__) # Utilisation de __name__ est une convention
        print("DEBUG_MAIN: Configuration et logging initialisés par load_config_and_setup_logging.", file=sys.stderr)
        logger.info(f"Configuration '{config_file_name}' chargée et logging actif.")

    except configparser.Error as e_cfg_parse:
        print(f"ERREUR CRITIQUE (ConfigParser): Échec de l'analyse du fichier '{config_file_name}': {e_cfg_parse}", file=sys.stderr)
        sys.exit(f"Arrêt: Erreur d'analyse de '{config_file_name}'.")
    except Exception as e_cfg_log_setup:
        print(f"ERREUR CRITIQUE (Setup): Échec inattendu lors du chargement de la config ou de l'init. du logging: {e_cfg_log_setup}", file=sys.stderr)
        sys.exit(f"Arrêt: Erreur setup config/logging ({e_cfg_log_setup}).")

    if config is None: # Double sécurité
        print("ERREUR CRITIQUE: L'objet de configuration est None après tentative de chargement.", file=sys.stderr)
        sys.exit("Arrêt: Échec critique de l'initialisation de la configuration.")

    # --- Début de la logique principale de simulation ---
    try:
        # --- Vérifications de configuration essentielles ---
        print("DEBUG_MAIN: Vérification des sections de configuration requises...", file=sys.stderr)
        required_sections = {
            "MODEL_PATHS": ['model_file', 'scalers_file'],
            "DATABASE": ['db_host', 'db_name', 'db_user', 'db_password', 'table_name'],
            "TIME_SETTINGS": ['historical_data_start_time', 'initial_data_duration_seconds', 'ww_resolution_ms'],
            "TAGS": [], # On vérifie juste la présence de la section
            "ModelFeatures": [], # Idem
            "PID_PARAMS_TEST": ['kp_test', 'ti_test', 'td_test'],
            "SCENARIO_PARAMS": ['scenario_type', 'sp_step_time_seconds', 'sp_step_value_offset'],
            "PERFORMANCE_METRICS": ['metric_type'],
            "OUTPUT": ['plot_save_path']
        }
        for section, keys in required_sections.items():
            if section not in config:
                logger.critical(f"Section de configuration [{section}] manquante dans '{config_file_name}'.")
                sys.exit(f"Arrêt: Section [{section}] manquante.")
            for key in keys:
                if key not in config[section]:
                    logger.critical(f"Clé '{key}' manquante dans la section [{section}] du fichier '{config_file_name}'.")
                    sys.exit(f"Arrêt: Clé '{key}' manquante dans [{section}].")
        print("DEBUG_MAIN: Vérification des sections de configuration terminée.", file=sys.stderr)


        # --- 1. Charger le modèle ML et les scalers ---
        logger.info("Chargement du modèle ML et des scalers...")
        print("DEBUG_MAIN: Avant chargement du modèle ML.", file=sys.stderr)
        model_path = Path(config['MODEL_PATHS']['model_save_path'])
        scalers_path = Path(config['MODEL_PATHS']['scalers_save_path'])

        if not model_path.is_file(): # Redondant si config_path_check est fait, mais bonne pratique
            logger.critical(f"Fichier modèle non trouvé (vérification post-config): {model_path}")
            sys.exit(f"Arrêt: Fichier modèle '{model_path}' non trouvé (vérification post-config).")
        if not scalers_path.is_file():
            logger.critical(f"Fichier scalers non trouvé (vérification post-config): {scalers_path}")
            sys.exit(f"Arrêt: Fichier scalers '{scalers_path}' non trouvé (vérification post-config).")

        model, scaler_X, scaler_y = load_model_and_scalers(config['MODEL_PATHS'])
        model_objects = {'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y}
        logger.info("Modèle ML et scalers chargés.")
        print("DEBUG_MAIN: Modèle ML et scalers chargés.", file=sys.stderr)

        # --- 2. Acquérir et préparer les données initiales ---
        logger.info("Acquisition des données initiales...")
        print("DEBUG_MAIN: Avant acquisition des données initiales.", file=sys.stderr)
        db_engine = get_db_engine(config['DATABASE'])
        
        hist_start_str = config['TIME_SETTINGS']['historical_data_start_time']
        initial_duration_s = config['TIME_SETTINGS'].getfloat('initial_data_duration_seconds')
        hist_start_dt = pd.to_datetime(hist_start_str)
        hist_end_dt = hist_start_dt + pd.to_timedelta(initial_duration_s, unit='s')

        temp_time_settings_for_query = configparser.SectionProxy(config, 'TIME_SETTINGS_QUERY_TEMP')
        temp_time_settings_for_query['start_time'] = hist_start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        temp_time_settings_for_query['end_time'] = hist_end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        temp_time_settings_for_query['ww_resolution_ms'] = config['TIME_SETTINGS']['ww_resolution_ms']
        
        sql_query_initial, col_aliases_initial = build_sql_query(
            config['TAGS'], temp_time_settings_for_query, config['DATABASE']['table_name']
        )
        initial_raw_df = extract_data(db_engine, sql_query_initial)
        db_engine.dispose()

        if initial_raw_df.empty:
            logger.error("Aucune donnée initiale extraite de la BDD.")
            sys.exit("Arrêt: Données initiales (BDD) vides.")
        
        initial_df = preprocess_timeseries_data(initial_raw_df, col_aliases_initial)
        if initial_df.empty:
            logger.error("Données initiales vides après pré-traitement.")
            sys.exit("Arrêt: Données initiales (pré-traitées) vides.")
            
        initial_model_input_scaled = prepare_initial_model_input(
            initial_df, config['ModelFeatures'], scaler_X, col_aliases_initial
        )
        
        pv_col_name = col_aliases_initial.get('pv_tag_name', 'PV_real')
        sp_col_name = col_aliases_initial.get('sp_tag_name', 'SP_real')
        mv_col_name = col_aliases_initial.get('mv_tag_name', 'MV_real')

        last_pv = initial_df[pv_col_name].iloc[-1]
        last_sp = initial_df[sp_col_name].iloc[-1]
        last_mv = initial_df[mv_col_name].iloc[-1]
        
        initial_disturbance_values_unscaled = {}
        for i in range(1, 5):
            dist_tag_config_key = f'disturbance_tag_{i}'
            dist_feature_alias = f'Dist{i}_real'
            if dist_tag_config_key in config['TAGS'] and config['TAGS'][dist_tag_config_key]:
                actual_dist_col_name = col_aliases_initial.get(dist_tag_config_key)
                if actual_dist_col_name and actual_dist_col_name in initial_df.columns:
                    initial_disturbance_values_unscaled[dist_feature_alias] = initial_df[actual_dist_col_name].iloc[-1]
                    logger.info(f"Valeur initiale pour {dist_feature_alias}: {initial_disturbance_values_unscaled[dist_feature_alias]}")

        initial_simulation_conditions = {
            'current_model_input_scaled': initial_model_input_scaled,
            'last_initial_pv': last_pv, 'last_initial_sp': last_sp, 'last_initial_mv': last_mv,
            'disturbance_values_unscaled': initial_disturbance_values_unscaled
        }
        logger.info("Données initiales pour simulation préparées.")
        print("DEBUG_MAIN: Données initiales pour simulation préparées.", file=sys.stderr)

        # --- 3. Définir les paramètres PID et Scénario ---
        kp_test = config['PID_PARAMS_TEST'].getfloat('kp_test')
        ti_test = config['PID_PARAMS_TEST'].getfloat('ti_test')
        td_test = config['PID_PARAMS_TEST'].getfloat('td_test')
        test_pid_params = [kp_test, ti_test, td_test]

        test_scenario = {
            'type': config['SCENARIO_PARAMS']['scenario_type'],
            'sp_initial_offset': config['SCENARIO_PARAMS'].getfloat('sp_initial_offset', 0.0),
            'sp_step_time_s': config['SCENARIO_PARAMS'].getfloat('sp_step_time_seconds'),
            'sp_step_value_offset': config['SCENARIO_PARAMS'].getfloat('sp_step_value_offset'),
        }
        logger.info(f"Test avec PID={test_pid_params}, Scénario={test_scenario}")
        print(f"DEBUG_MAIN: Test avec PID={test_pid_params}, Scénario={test_scenario}", file=sys.stderr)

        # --- 4. Exécuter la simulation ---
        logger.info("Appel à evaluer_performance_pid...")
        print("DEBUG_MAIN: Avant appel à evaluer_performance_pid.", file=sys.stderr)
        
        performance_score, sim_data_df = evaluer_performance_pid(
            test_pid_params, config, test_scenario, model_objects,
            initial_simulation_conditions, return_sim_data=True
        )
        metric_display = config.get('PERFORMANCE_METRICS', 'metric_type', fallback='IAE') # Fallback pour l'affichage
        logger.info(f"Simulation terminée. Score ({metric_display}): {performance_score:.4f}")
        print(f"DEBUG_MAIN: Retour de evaluer_performance_pid, score: {performance_score}, "
              f"données sim: {'Non vides' if sim_data_df is not None and not sim_data_df.empty else 'Vides/None'}", file=sys.stderr)

        # --- 5. Générer le graphique ---
        if sim_data_df is not None and not sim_data_df.empty:
            logger.info("Génération du graphique...")
            print("DEBUG_MAIN: Avant génération du graphique.", file=sys.stderr)
            plt.figure(figsize=(16, 9))
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(sim_data_df.index, sim_data_df['PV_sim'], label='PV Simulée (ML)', color='blue', lw=2)
            ax1.plot(sim_data_df.index, sim_data_df['SP_sim'], label='SP Simulée', color='green', ls='--', lw=2)
            ax1.set_ylabel('Valeur Procédé / Consigne'); ax1.legend(loc='best'); ax1.grid(True, which='both', ls=':', lw=0.5)
            plt.setp(ax1.get_xticklabels(), visible=False)

            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.plot(sim_data_df.index, sim_data_df['MV_sim'], label='MV Simulée (PID)', color='red', ls='-.', lw=2)
            ax2.set_ylabel('Valeur Commande (%)'); ax2.set_xlabel('Temps de simulation (s)'); ax2.legend(loc='best'); ax2.grid(True, which='both', ls=':', lw=0.5)

            plt.suptitle(f"Simulation Test PID: Kp={kp_test}, Ti={ti_test}, Td={td_test}\nScore {metric_display}: {performance_score:.2f}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            plot_save_path_str = config['OUTPUT']['plot_save_path'] # Clé vérifiée plus haut
            plot_path = Path(plot_save_path_str).resolve()
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path)
            logger.info(f"Graphique sauvegardé : '{plot_path}'")
            print(f"DEBUG_MAIN: Graphique sauvegardé. Tentative d'affichage...", file=sys.stderr)
            try:
                plt.show()
                print("DEBUG_MAIN: plt.show() appelé.", file=sys.stderr)
            except Exception as e_plot:
                logger.warning(f"Affichage interactif du graphique échoué: {e_plot}")
                print(f"DEBUG_MAIN: Erreur plt.show(): {e_plot}", file=sys.stderr)
        else:
            print("DEBUG_MAIN: sim_data_df est vide ou None, pas de graphique.", file=sys.stderr)
            logger.warning("Aucune donnée de simulation à plotter.")

    except FileNotFoundError as e_fnf:
        print(f"ERREUR (FileNotFoundError) dans run_pid_tuning_simulation: {e_fnf}", file=sys.stderr)
        if logger: logger.critical(f"Fichier non trouvé: {e_fnf}", exc_info=True)
        sys.exit(f"Arrêt: Fichier non trouvé ({e_fnf}).")
    except KeyError as e_key:
        print(f"ERREUR (KeyError) dans run_pid_tuning_simulation: {e_key}. Clé de config ou DataFrame manquante?", file=sys.stderr)
        if logger: logger.critical(f"Clé manquante: {e_key}", exc_info=True)
        sys.exit(f"Arrêt: Clé manquante ({e_key}).")
    except ValueError as e_val:
        print(f"ERREUR (ValueError) dans run_pid_tuning_simulation: {e_val}. Type de valeur incorrect dans config?", file=sys.stderr)
        if logger: logger.critical(f"Erreur de valeur: {e_val}", exc_info=True)
        sys.exit(f"Arrêt: Erreur de valeur ({e_val}).")
    except Exception as e_runtime:
        print(f"ERREUR INATTENDUE dans run_pid_tuning_simulation: {e_runtime}", file=sys.stderr)
        if logger: logger.critical(f"Erreur d'exécution inattendue: {e_runtime}", exc_info=True)
        sys.exit(f"Arrêt: Erreur inattendue ({e_runtime}).")
    finally:
        print("DEBUG_MAIN: Fin de run_pid_tuning_simulation() (bloc finally).", file=sys.stderr)

if __name__ == "__main__":
    print("DEBUG_GLOBAL: Le script est exécuté en tant que __main__.", file=sys.stderr)
    try:
        print("DEBUG_GLOBAL: Appel de run_pid_tuning_simulation().", file=sys.stderr)
        run_pid_tuning_simulation()
        print("DEBUG_GLOBAL: run_pid_tuning_simulation() terminé (apparent).", file=sys.stderr)
        sys.exit(0) # Sortie explicite avec code 0 pour succès
    except SystemExit as e_sys_exit:
        # Permet aux sys.exit() de la fonction principale de terminer le script
        # avec le code approprié ou le message.
        print(f"DEBUG_GLOBAL: Script terminé par SystemExit code/message: '{e_sys_exit.code}'.", file=sys.stderr)
        raise # Re-lever pour que l'interpréteur quitte avec ce code
    except Exception as e_script_level:
        # Au cas où une erreur se produirait en dehors du try/except de run_pid_tuning_simulation
        print(f"ERREUR CRITIQUE NON GÉRÉE au niveau du script: {e_script_level}", file=sys.stderr)
        if logger: # Si le logger a été initialisé
            logger.critical(f"Erreur critique non gérée au niveau du script: {e_script_level}", exc_info=True)
        else: # Afficher le traceback manuellement si le logger n'est pas dispo
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit("Arrêt: Erreur critique non gérée au niveau du script.")
    finally:
        print("DEBUG_GLOBAL: Fin du bloc if __name__ == '__main__'.", file=sys.stderr)