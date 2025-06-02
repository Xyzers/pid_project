# pid_tuner_main.py 

import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import configparser # Import direct

# Imports depuis le dossier src
from src.config_loader import load_config_and_setup_logging
from src.db_utils import get_db_engine, build_sql_query, extract_data
from src.data_processing import preprocess_timeseries_data
from src.modeling import load_model_and_scalers # S'assurer que cette fonction existe et retourne model, scaler_X, scaler_y
from src.pid_simulation.pid_single_simulation import evaluer_performance_pid, prepare_initial_model_input

logger = logging.getLogger(__name__)

def main():
    script_name_stem = "pid_tuner_simulation" # Pour le nom du .ini et du log
    config_file_name = f"{script_name_stem}.ini" # Attendu à la racine

    try:
        # Charger la config .ini et configurer le logging
        # load_config_and_setup_logging s'attend à ce que le .ini soit dans le même dossier que le script appelant
        config = load_config_and_setup_logging(
            script_stem_for_config_file=script_name_stem, # Utilisé pour le nom du log
            config_file_name=config_file_name # Nom direct du fichier .ini
        )
        logger.info(f"Configuration '{config_file_name}' chargée.")
    except Exception as e:
        print(f"Erreur critique lors du chargement de la configuration ou du logging: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # --- 1. Charger le modèle ML et les scalers ---
        logger.info("Chargement du modèle ML et des scalers...")
        model_path = Path(config['MODEL_PATHS']['model_file'])
        scalers_path = Path(config['MODEL_PATHS']['scalers_file'])

        if not model_path.is_file() or not scalers_path.is_file():
            raise FileNotFoundError(f"Fichier modèle ou scalers non trouvé. Vérifiez les chemins: {model_path}, {scalers_path}")

        # La fonction load_model_and_scalers doit retourner model, scaler_X, scaler_y
        # Le scaler_X doit être celui qui a été fitté sur les features d'entrée du modèle (avec lags)
        # Le scaler_y doit être celui qui a été fitté sur la target (PV)
        model, scaler_X, scaler_y = load_model_and_scalers(config['MODEL_PATHS'])
        model_objects = {'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y}
        logger.info("Modèle ML et scalers chargés.")

        # --- 2. Acquérir et préparer les données initiales pour la simulation ---
        logger.info("Acquisition des données initiales pour les lags du modèle...")
        db_engine = get_db_engine(config['DATABASE'])
        
        hist_start_str = config['TIME_SETTINGS']['historical_data_start_time']
        initial_duration_s = config['TIME_SETTINGS'].getfloat('initial_data_duration_seconds')
        hist_start_dt = pd.to_datetime(hist_start_str)
        hist_end_dt = hist_start_dt + pd.to_timedelta(initial_duration_s, unit='s')

        # Créer une section de config temporaire pour build_sql_query si elle attend une section
        temp_time_settings_for_query = configparser.SectionProxy(config, 'TIME_SETTINGS_QUERY_TEMP')
        temp_time_settings_for_query['start_time'] = hist_start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Format avec ms
        temp_time_settings_for_query['end_time'] = hist_end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        temp_time_settings_for_query['ww_resolution_ms'] = config['TIME_SETTINGS']['ww_resolution_ms']
        
        # S'assurer que config['TAGS'] contient les bons alias pour build_sql_query
        # Les clés de config['TAGS'] sont utilisées par build_sql_query pour les alias
        sql_query_initial, col_aliases_initial = build_sql_query(
            config['TAGS'], # Doit contenir pv_tag_name, mv_tag_name etc. comme clés
            temp_time_settings_for_query,
            config['DATABASE']['table_name']
        )
        initial_raw_df = extract_data(db_engine, sql_query_initial)
        db_engine.dispose()

        if initial_raw_df.empty:
            logger.error("Aucune donnée initiale extraite de la base. Vérifiez la configuration et la base de données.")
            sys.exit(1)
        
        # Le preproces_timeseries_data retourne un df avec les alias comme noms de colonnes
        initial_df = preprocess_timeseries_data(initial_raw_df, col_aliases_initial)
        if initial_df.empty:
            logger.error("Données initiales vides après pré-traitement.")
            sys.exit(1)
            
        # Préparer le premier input pour le modèle ML
        # col_aliases_initial contient le mapping entre 'pv_tag_name' et le vrai nom du tag
        # prepare_initial_model_input a besoin de ces alias pour savoir quelles colonnes renommer en PV_real etc.
        initial_model_input_scaled = prepare_initial_model_input(
            initial_df, 
            config['ModelFeatures'], # La section [ModelFeatures] de pid_model_builder.ini doit être ici
            scaler_X,
            col_aliases_initial # Passer les alias pour aider au renommage interne
        )
        
        # Extraire les dernières valeurs non-scalées pour démarrer la simulation
        # Utiliser les alias pour extraire les bonnes colonnes
        pv_col_name = col_aliases_initial.get('pv_tag_name', 'PV_real') # PV_real si déjà renommé
        sp_col_name = col_aliases_initial.get('sp_tag_name', 'SP_real')
        mv_col_name = col_aliases_initial.get('mv_tag_name', 'MV_real')

        # Vérifier si les colonnes existent après le renommage potentiel dans prepare_initial_model_input
        # ou directement dans initial_df si on n'a pas encore fait de renommage généralisé.
        # Pour plus de robustesse, on utilise les alias pour accéder à initial_df
        last_pv = initial_df[pv_col_name].iloc[-1]
        last_sp = initial_df[sp_col_name].iloc[-1]
        last_mv = initial_df[mv_col_name].iloc[-1]
        
        # Gérer les perturbations initiales si elles sont utilisées par le modèle
        initial_disturbance_values_unscaled = {}
        for i in range(1, 5):
            dist_tag_config_key = f'disturbance_tag_{i}' # Clé dans [TAGS]
            dist_feature_alias = f'Dist{i}_real' # Nom que create_lagged_features et update_model_input attendent

            if dist_tag_config_key in config['TAGS'] and config['TAGS'][dist_tag_config_key]:
                # Récupérer le nom actuel de la colonne de perturbation depuis initial_df via son alias
                actual_dist_col_name = col_aliases_initial.get(dist_tag_config_key)
                if actual_dist_col_name and actual_dist_col_name in initial_df.columns:
                    initial_disturbance_values_unscaled[dist_feature_alias] = initial_df[actual_dist_col_name].iloc[-1]
                    logger.info(f"Valeur initiale pour {dist_feature_alias}: {initial_disturbance_values_unscaled[dist_feature_alias]}")
                else:
                    logger.warning(f"Tag de perturbation {dist_tag_config_key} configuré mais non trouvé dans les données initiales sous l'alias {actual_dist_col_name}.")

        initial_simulation_conditions = {
            'current_model_input_scaled': initial_model_input_scaled,
            'last_initial_pv': last_pv,
            'last_initial_sp': last_sp,
            'last_initial_mv': last_mv,
            'disturbance_values_unscaled': initial_disturbance_values_unscaled
        }
        logger.info("Données initiales pour la simulation préparées.")

        # --- 3. Définir les paramètres PID et le scénario de test ---
        kp_test = config['PID_PARAMS_TEST'].getfloat('kp_test')
        ti_test = config['PID_PARAMS_TEST'].getfloat('ti_test')
        td_test = config['PID_PARAMS_TEST'].getfloat('td_test')
        test_pid_params = [kp_test, ti_test, td_test]

        test_scenario = {
            'type': config['SCENARIO_PARAMS'].get('scenario_type'),
            'sp_initial_offset': config['SCENARIO_PARAMS'].getfloat('sp_initial_offset', 0.0),
            'sp_step_time_s': config['SCENARIO_PARAMS'].getfloat('sp_step_time_seconds'),
            'sp_step_value_offset': config['SCENARIO_PARAMS'].getfloat('sp_step_value_offset'),
        }
        logger.info(f"Test avec PID={test_pid_params}, Scénario={test_scenario}")

        # --- 4. Exécuter la simulation et obtenir le score ET les données ---
        performance_score, sim_data_df = evaluer_performance_pid(
            test_pid_params,
            config, # Passer l'objet config complet
            test_scenario,
            model_objects,
            initial_simulation_conditions, # Passer les conditions initiales préparées
            return_sim_data=True # Important pour obtenir les données pour le plot
        )
        logger.info(f"Simulation terminée. Score de performance ({config['PERFORMANCE_METRICS']['metric_type']}): {performance_score:.4f}")

        # --- 5. Générer le graphique des résultats ---
        if not sim_data_df.empty:
            plt.figure(figsize=(16, 9))
            
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(sim_data_df.index, sim_data_df['PV_sim'], label='PV Simulée (modèle ML)', color='blue', linewidth=2)
            ax1.plot(sim_data_df.index, sim_data_df['SP_sim'], label='SP Simulée', color='green', linestyle='--', linewidth=2)
            ax1.set_ylabel('Valeur Procédé / Consigne')
            ax1.legend(loc='best')
            ax1.grid(True, which='both', linestyle=':', linewidth=0.5)
            plt.setp(ax1.get_xticklabels(), visible=False) # Masquer les labels x pour le subplot du haut

            ax2 = plt.subplot(2, 1, 2, sharex=ax1) # Partager l'axe X avec ax1
            ax2.plot(sim_data_df.index, sim_data_df['MV_sim'], label='MV Simulée (PID)', color='red', linestyle='-.', linewidth=2)
            ax2.set_ylabel('Valeur Commande (%)')
            ax2.set_xlabel('Temps de simulation (s)')
            ax2.legend(loc='best')
            ax2.grid(True, which='both', linestyle=':', linewidth=0.5)

            plt.suptitle(f"Simulation Test PID: Kp={kp_test}, Ti={ti_test}, Td={td_test}\nScore IAE: {performance_score:.2f}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuster pour le suptitle

            plot_save_path_str = config['OUTPUT'].get('plot_save_path')
            if plot_save_path_str:
                plot_path = Path(plot_save_path_str).resolve()
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_path)
                logger.info(f"Graphique de simulation sauvegardé : '{plot_path}'")
            
            try:
                plt.show()
            except Exception as e_plot:
                logger.warning(f"Affichage du graphique interactif échoué (environnement non graphique?): {e_plot}")
        else:
            logger.warning("Aucune donnée de simulation à plotter (simulation échouée ou score infini).")

    except FileNotFoundError as fnf_e:
        logger.error(f"Erreur de fichier non trouvé: {fnf_e}")
    except KeyError as ke_e:
        logger.error(f"Erreur de clé manquante dans la configuration ou les données: {ke_e}. Assurez-vous que toutes les sections et clés nécessaires (ex: [ModelFeatures] depuis pid_model_builder.ini) sont présentes dans pid_tuner_simulation.ini ou correctement mappées.", exc_info=True)