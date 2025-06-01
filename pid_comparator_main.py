# pid_comparator_main.py
#!/usr/bin/env python
# coding: utf-8

import sys
import logging # logging est configuré par config_loader
import pandas as pd
import numpy as np
import winsound # Si vous voulez garder le Beep

# Imports depuis src
from src.config_loader import load_config_and_setup_logging
from src.db_utils import get_db_engine, build_sql_query, extract_data
from src.data_processing import preprocess_timeseries_data 
from src.pid_logic.pid_controller import PIDController
from src.plotting_utils import plot_pid_comparison 

logger = logging.getLogger(__name__) 

def run_pid_comparison():
    script_name_stem = "pid_comparator" 
    config = None
    try:
        config = load_config_and_setup_logging(script_name_stem, config_file_name=f"{script_name_stem}.ini")
    except Exception as e_cfg:
        print(f"ERREUR CRITIQUE lors du chargement de la configuration ou du logging: {e_cfg}", file=sys.stderr)
        if logging.getLogger().hasHandlers():
             logging.critical("Échec critique config/logging", exc_info=True)
        sys.exit(1)

    logger.info(f"--- Démarrage {script_name_stem}.py ---", extra={'important_phase': True})
    db_engine = None
    try:
        db_engine = get_db_engine(config['Database'])
        table_name_hist = config['Database'].get('table_name', 'History')
        
        required_aliases = ['PV_real', 'MV_real', 'SP_real'] 
        
        sql_query_str, col_aliases_from_sql = build_sql_query(
            config['Tags'], 
            config['DataPeriod'], 
            table_name_hist,
            required_tags_aliases=required_aliases 
        )
        raw_hist_df = extract_data(db_engine, sql_query_str)

        if raw_hist_df.empty:
            logger.error("Aucune donnée historique extraite. Arrêt.")
            sys.exit(1)

        essential_cols_for_sim = ['PV_real', 'SP_real'] 
        
        hist_data_df = preprocess_timeseries_data(
            raw_hist_df, 
            col_aliases_from_sql,
            essential_value_cols=essential_cols_for_sim 
        )
        
        if hist_data_df.empty or hist_data_df[essential_cols_for_sim].isnull().values.any():
            logger.error(f"Données manquantes dans les colonnes essentielles ({', '.join(essential_cols_for_sim)}) après pré-traitement. Arrêt.")
            sys.exit(1)

        tsamp_pid_ms = config['PIDSimParams'].getfloat('tsamp_pid_sim_ms')
        if tsamp_pid_ms <= 0: raise ValueError("tsamp_pid_sim_ms doit être > 0.")
        tsamp_pid_s = tsamp_pid_ms / 1000.0

        fb_kp = config['PIDSimParams'].getfloat('fallback_kp')
        fb_ti = config['PIDSimParams'].getfloat('fallback_ti')
        if fb_ti <= 0: raise ValueError("fallback_ti doit être > 0.")
        fb_td = config['PIDSimParams'].getfloat('fallback_td')
        
        mv_min_val = config['PIDSimParams'].getfloat('mv_min')
        mv_max_val = config['PIDSimParams'].getfloat('mv_max')
        direct_act = config['PIDSimParams'].getboolean('direct_action')

        pid_structure_str = config['PIDSimParams'].get('pid_structure', 'parallel_isa').lower()
        derivative_action_str = config['PIDSimParams'].get('derivative_action', 'on_pv').lower()
        logger.info(f"Options de simulation PID lues depuis .ini: Structure='{pid_structure_str}', Dérivée='{derivative_action_str}'")
        
        first_row = hist_data_df.iloc[0]
        
        init_kp = first_row.get('Kp_real', fb_kp) if 'Kp_real' in first_row and not pd.isna(first_row['Kp_real']) else fb_kp
        init_ti_val = first_row.get('Ti_real', fb_ti) if 'Ti_real' in first_row and not pd.isna(first_row['Ti_real']) else fb_ti
        init_ti = init_ti_val if init_ti_val > 0 else fb_ti 
        init_td = first_row.get('Td_real', fb_td) if 'Td_real' in first_row and not pd.isna(first_row['Td_real']) else fb_td
        
        init_mv = first_row.get('MV_real', (mv_min_val + mv_max_val) / 2.0)
        if pd.isna(init_mv): 
            logger.warning(f"MV_real initiale est NaN. Utilisation de la moyenne de mv_min/mv_max: {(mv_min_val + mv_max_val) / 2.0}")
            init_mv = (mv_min_val + mv_max_val) / 2.0

        pid_simulator = PIDController(
            Kp=init_kp, 
            Ti=init_ti, 
            Td=init_td, 
            Tsamp=tsamp_pid_s,
            mv_min=mv_min_val, 
            mv_max=mv_max_val, 
            direct_action=direct_act, 
            initial_mv=init_mv,
            pid_structure=pid_structure_str,
            derivative_action=derivative_action_str
        )
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # DÉFINITION DE pid_enable_col_name ET LOGIQUE D'INITIALISATION DE L'ÉTAT ACTIF
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pid_enable_col_name = 'PID_Enable_real' # Nom de la colonne pour l'état d'activation du PID

        init_pid_enable_from_tag = pid_simulator.is_active # Valeur par défaut (celle du constructeur PIDController)
        
        if pid_enable_col_name in first_row and not pd.isna(first_row[pid_enable_col_name]):
            init_pid_enable_from_tag = bool(first_row[pid_enable_col_name])
            logger.info(f"État initial PID (depuis tag '{pid_enable_col_name}' sur première ligne): {'Actif' if init_pid_enable_from_tag else 'Inactif'}")
        elif pid_enable_col_name not in hist_data_df.columns:
            logger.warning(f"Tag d'activation PID ('{pid_enable_col_name}') non trouvé dans les colonnes de données. "
                           f"Le PID simulé démarrera avec son état par défaut ('{pid_simulator.is_active}').")
            # init_pid_enable_from_tag conserve la valeur par défaut de pid_simulator.is_active
        else: # La colonne existe mais la valeur sur la première ligne est NaN
             logger.warning(f"Tag d'activation PID ('{pid_enable_col_name}') est NaN pour la première ligne. "
                           f"Le PID simulé démarrera avec son état par défaut ('{pid_simulator.is_active}').")
             # init_pid_enable_from_tag conserve la valeur par défaut de pid_simulator.is_active
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        pid_simulator.set_initial_state(
            pv_initial=first_row['PV_real'],
            sp_initial=first_row['SP_real'],
            mv_initial=init_mv, 
            active_initial=init_pid_enable_from_tag # Utilisation de la valeur déterminée ci-dessus
        )
        # Log après que set_initial_state ait pu ajuster self.mv et self.is_active
        logger.info(f"PID simulé état après set_initial_state: Kp={pid_simulator.Kp_param:.2f}, Ti={pid_simulator.Ti_param:.2f}, Td={pid_simulator.Td_param:.2f}, MV_init={pid_simulator.mv:.2f}, Actif_init={pid_simulator.is_active}")

        sim_mv_list = []
        sim_pid_active_list = [] 
        
        # Bloc de débogage pour pid_enable_col_name (devrait maintenant fonctionner)
        try:
            logger.debug(f"Avant la boucle, pid_enable_col_name = '{pid_enable_col_name}'")
        except NameError: # Ne devrait plus arriver
            logger.error("ERREUR DE DÉBOGAGE INTERNE: pid_enable_col_name n'est PAS défini avant la boucle !")
            
        logger.info(f"Simulation PID (Tsamp={tsamp_pid_s:.3f}s) avec {len(hist_data_df)} points...")
        for timestamp, data_row in hist_data_df.iterrows():
            pv_r = data_row['PV_real']
            sp_r = data_row['SP_real']
            mv_r = data_row['MV_real'] 
            if pd.isna(mv_r): 
                logger.debug(f"MV_real est NaN à {timestamp}. Bumpless transfer peut être affecté si transition I->A.")

            if pd.isna(pv_r) or pd.isna(sp_r):
                logger.warning(f"PV ou SP NaN à {timestamp}. Le PID simulé maintiendra sa dernière MV et son état.")
                sim_mv_list.append(pid_simulator.mv) 
                sim_pid_active_list.append(pid_simulator.is_active)
                if pid_simulator.previous_pv is None and not pd.isna(pv_r):
                     pid_simulator.previous_pv = pv_r 
                continue

            # Utilisation de pid_enable_col_name dans la boucle
            pid_enabled_r_current = pid_simulator.is_active 
            if pid_enable_col_name in data_row and not pd.isna(data_row[pid_enable_col_name]):
                pid_enabled_r_current = bool(data_row[pid_enable_col_name])
            elif pid_enable_col_name not in hist_data_df.columns: 
                 pid_enabled_r_current = True # Ou pid_simulator.is_active si on veut garder l'état initial par défaut
                                              # Le comportement actuel est de le forcer à True si la colonne n'existe pas du tout.

            mv_for_bumpless = mv_r if not pd.isna(mv_r) else pid_simulator.last_active_mv 

            if pid_simulator.is_active != pid_enabled_r_current:
                pid_simulator.set_active_state(pid_enabled_r_current, sp_r, pv_r, mv_for_bumpless)

            current_kp = data_row.get('Kp_real', fb_kp); current_kp = fb_kp if pd.isna(current_kp) else current_kp
            current_ti_val = data_row.get('Ti_real', fb_ti); current_ti_val = fb_ti if pd.isna(current_ti_val) else current_ti_val
            current_ti = current_ti_val if current_ti_val > 0 else fb_ti
            current_td = data_row.get('Td_real', fb_td); current_td = fb_td if pd.isna(current_td) else current_td
            pid_simulator.set_parameters(current_kp, current_ti, current_td)
            
            sim_mv_output = pid_simulator.update(sp_r, pv_r)
            sim_mv_list.append(sim_mv_output)
            sim_pid_active_list.append(pid_simulator.is_active)

        hist_data_df['MV_simulated'] = sim_mv_list
        hist_data_df['PID_Active_simulated'] = sim_pid_active_list
        
        logger.info("Simulation PID terminée.", extra={'important_phase': True})

        plot_path = config['Output'].get('plot_save_path', None)
        plot_pid_comparison(hist_data_df, config['DataPeriod'], plot_path)

        if 'MV_real' in hist_data_df.columns and 'MV_simulated' in hist_data_df.columns:
            comp_df = hist_data_df[['MV_real', 'MV_simulated']].dropna() 
            if not comp_df.empty:
                mse_overall = np.mean((comp_df['MV_real'] - comp_df['MV_simulated'])**2)
                mae_overall = np.mean(np.abs(comp_df['MV_real'] - comp_df['MV_simulated']))
                logger.info(f"Comparaison MV (globale, lignes avec MV_real et MV_sim non-NaN): MSE={mse_overall:.4f}, MAE={mae_overall:.4f}")

    except FileNotFoundError as e_fnf:
        logger.error(f"Fichier de configuration non trouvé: {e_fnf}", exc_info=True)
        sys.exit(1)
    except ValueError as e_val:
        logger.error(f"Erreur de valeur ou de configuration: {e_val}", exc_info=True)
        sys.exit(1)
    except KeyError as e_key:
        logger.error(f"Erreur de clé (souvent un tag manquant dans .ini ou dans les données): {e_key}", exc_info=True)
        sys.exit(1)
    except Exception as e_main_exc:
        logger.error(f"Erreur inattendue dans l'exécution principale: {e_main_exc}", exc_info=True)
        sys.exit(1)
    finally:
        if db_engine:
            db_engine.dispose()
            logger.info("Moteur de base de données fermé.")
        
        logger.info(f"--- Fin {script_name_stem}.py ---", extra={'important_phase': True})
        
        try:
            winsound.Beep(1000, 500) 
        except ImportError:
            logger.info("Module winsound non trouvé, pas de bip de fin.")
        except RuntimeError: 
            logger.info("Impossible de jouer le son de fin (pas de périphérique audio ou winsound non dispo).")
        except Exception as e_sound: 
            logger.warning(f"Erreur lors de la tentative de jouer le son de fin: {e_sound}")

if __name__ == "__main__":
    run_pid_comparison()