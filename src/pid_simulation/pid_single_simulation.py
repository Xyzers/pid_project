# src/pid_simulation/pid_single_simulation.py

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import configparser # Import direct de configparser

# Ajustement des imports relatifs pour les modules dans src/
from ..db_utils import get_db_engine, build_sql_query, extract_data
from ..data_processing import preprocess_timeseries_data
# from ..modeling import load_model_and_scalers # Sera chargé dans le main script
from ..pid_logic.pid_controller import PIDController
from ..feature_engineering import create_lagged_features

logger = logging.getLogger(__name__)

def prepare_initial_model_input(initial_df: pd.DataFrame, 
                                config_model_features: configparser.SectionProxy, 
                                scaler_X,
                                tag_aliases: dict):
    """
    Prépare le DataFrame d'entrée initial pour le modèle ML avec les lags requis.
    """
    df_for_lags = initial_df.copy()

    # Renommer les colonnes en PV_real, MV_real, SP_real, DistX_real si nécessaire
    # pour correspondre à ce que create_lagged_features attend et à la config ModelFeatures
    rename_map = {}
    if tag_aliases.get('pv_tag_name') and 'PV_real' not in df_for_lags.columns:
        rename_map[tag_aliases['pv_tag_name']] = 'PV_real'
    if tag_aliases.get('mv_tag_name') and 'MV_real' not in df_for_lags.columns:
        rename_map[tag_aliases['mv_tag_name']] = 'MV_real'
    if tag_aliases.get('sp_tag_name') and 'SP_real' not in df_for_lags.columns:
        rename_map[tag_aliases['sp_tag_name']] = 'SP_real'
    
    for i in range(1, 5): # Supposons jusqu'à 4 perturbations
        tag_key = f'disturbance_tag_{i}'
        dist_alias_col = f'Dist{i}_real' # Nom attendu par create_lagged_features
        if tag_aliases.get(tag_key) and dist_alias_col not in df_for_lags.columns:
            rename_map[tag_aliases[tag_key]] = dist_alias_col
            
    df_for_lags.rename(columns=rename_map, inplace=True)
    
    # S'assurer que les colonnes pour les lags (MV_real, SP_real, etc.) existent
    # create_lagged_features s'occupera des lags pour les colonnes présentes.

    X_with_lags, _ = create_lagged_features(df_for_lags, config_model_features, target_col='PV_real')

    if X_with_lags.empty:
        raise ValueError("Le DataFrame est vide après la création des features décalées pour l'initialisation du modèle.")

    last_features_row = X_with_lags.iloc[[-1]]
    
    # S'assurer de l'ordre des colonnes pour la normalisation
    if hasattr(scaler_X, 'feature_names_in_'):
        try:
            last_features_row = last_features_row[scaler_X.feature_names_in_]
        except KeyError as e:
            missing_cols = set(scaler_X.feature_names_in_) - set(last_features_row.columns)
            raise ValueError(f"Colonnes manquantes pour la normalisation par scaler_X: {missing_cols}. Colonnes disponibles: {last_features_row.columns}. Erreur originale: {e}")

    scaled_features_row = scaler_X.transform(last_features_row)
    return scaled_features_row

def update_model_input(current_input_scaled: np.ndarray, 
                       pv_sim_scaled: float, 
                       mv_sim_scaled: float, 
                       sp_sim_scaled: float,
                       config_model_features: configparser.SectionProxy, 
                       disturbance_values_scaled: dict = None):
    """
    Met à jour le vecteur d'entrée du modèle ML.
    """
    updated_input = current_input_scaled.copy().ravel()
    current_idx = 0

    pv_lags = config_model_features.getint('pv_lags', 0)
    if pv_lags > 0:
        updated_input[current_idx+1 : current_idx+pv_lags] = updated_input[current_idx : current_idx+pv_lags-1]
        updated_input[current_idx] = pv_sim_scaled
    current_idx += pv_lags

    mv_lags = config_model_features.getint('mv_lags', 0)
    if mv_lags > 0:
        updated_input[current_idx+1 : current_idx+mv_lags] = updated_input[current_idx : current_idx+mv_lags-1]
        updated_input[current_idx] = mv_sim_scaled
    current_idx += mv_lags

    sp_lags = config_model_features.getint('sp_lags', 0)
    if sp_lags > 0:
        updated_input[current_idx+1 : current_idx+sp_lags] = updated_input[current_idx : current_idx+sp_lags-1]
        updated_input[current_idx] = sp_sim_scaled
    current_idx += sp_lags

    if disturbance_values_scaled:
        i = 1
        while True:
            dist_lag_key = f'disturbance_{i}_lags'
            dist_alias_model_feature = f'Dist{i}_lag1' # Ce nom doit correspondre aux features du scaler_X
            
            if dist_lag_key not in config_model_features:
                break
            
            num_dist_lags = config_model_features.getint(dist_lag_key, 0)
            if num_dist_lags > 0:
                # Le nom de la perturbation dans disturbance_values_scaled doit correspondre
                # à la clé utilisée (ex: 'Dist1_real' ou un alias)
                # Pour l'update, on cherche la *nouvelle valeur* de la perturbation (non laggée)
                dist_key_in_dict = f'Dist{i}_real' # Supposons que c'est la clé dans disturbance_values_scaled
                if dist_key_in_dict in disturbance_values_scaled:
                    dist_val_scaled = disturbance_values_scaled[dist_key_in_dict]
                    updated_input[current_idx+1 : current_idx+num_dist_lags] = \
                        updated_input[current_idx : current_idx+num_dist_lags-1]
                    updated_input[current_idx] = dist_val_scaled
                else:
                    logger.debug(f"Aucune nouvelle valeur pour perturbation {dist_key_in_dict} dans update_model_input, ses lags ne seront pas mis à jour avec une nouvelle observation.")
                current_idx += num_dist_lags
            i += 1
            
    return updated_input.reshape(1, -1)

def evaluer_performance_pid(params_pid: list, 
                            config: configparser.ConfigParser, # Passer l'objet config chargé
                            scenario_config: dict, 
                            model_objects: dict,
                            initial_sim_data: dict, # Données pour initialiser la simulation
                            return_sim_data=False): # Flag pour retourner les données de simulation
    """
    Évalue la performance d'un jeu de paramètres PID.
    Retourne le score, et optionnellement les données de simulation.
    """
    try:
        kp, ti, td = params_pid
        
        model = model_objects['model']
        scaler_X = model_objects['scaler_X']
        scaler_y = model_objects['scaler_y']
        
        # Utiliser les données initiales pré-calculées
        current_model_input_scaled = initial_sim_data['current_model_input_scaled']
        last_initial_pv = initial_sim_data['last_initial_pv']
        last_initial_sp = initial_sim_data['last_initial_sp']
        last_initial_mv = initial_sim_data['last_initial_mv']
        disturbance_values_unscaled = initial_sim_data.get('disturbance_values_unscaled', {})

        tsamp_s = config['SIMULATION_PARAMS'].getfloat('tsamp_simulation_ms') / 1000.0
        pid_sim = PIDController(
            Kp=kp, Ti=ti, Td=td, Tsamp=tsamp_s,
            mv_min=config['SIMULATION_PARAMS'].getfloat('mv_min'),
            mv_max=config['SIMULATION_PARAMS'].getfloat('mv_max'),
            direct_action=config['SIMULATION_PARAMS'].getboolean('direct_action'),
            pid_structure=config['SIMULATION_PARAMS'].get('pid_structure'),
            derivative_action=config['SIMULATION_PARAMS'].get('derivative_action')
        )
        
        current_sp_sim = last_initial_sp + scenario_config.get('sp_initial_offset', 0.0)
        pid_sim.set_initial_state(
            pv_initial=last_initial_pv,
            sp_initial=current_sp_sim,
            mv_initial=last_initial_mv,
            active_initial=True
        )
        current_pv_sim = last_initial_pv
        
        simulation_duration_s = config['SIMULATION_PARAMS'].getfloat('simulation_duration_seconds')
        num_steps = int(simulation_duration_s / tsamp_s)
        
        sim_results = {'Time': [], 'PV_sim': [], 'MV_sim': [], 'SP_sim': []}
        accumulated_absolute_error = 0.0

        # Normaliser les valeurs de perturbation une fois si elles sont constantes
        disturbance_values_scaled_for_model = {}
        if disturbance_values_unscaled and scaler_X and hasattr(scaler_X, 'feature_names_in_'):
            # Créer un dataframe temporaire avec les noms attendus par scaler_X pour les perturbations
            temp_df_dist = pd.DataFrame(columns=scaler_X.feature_names_in_)
            for dist_key, dist_val in disturbance_values_unscaled.items():
                 # dist_key est 'Dist1_real', etc. Il faut trouver où cette feature (non laggée)
                 # se situerait dans l'input de scaler_X si elle y était directement (ce qui n'est pas le cas,
                 # ce sont les lags qui y sont). C'est un point délicat.
                 # L'hypothèse de update_model_input est que disturbance_values_scaled contient
                 # les *nouvelles valeurs scalées* des perturbations.
                 # On va scaler ici en supposant qu'on a un scaler pour ces features individuelles
                 # ou que scaler_y est une approximation acceptable (MAUVAISE HYPOTHÈSE GÉNÉRALE)
                 # Pour une meilleure solution, les scalers de chaque feature (PV, MV, SP, Dist1, Dist2..)
                 # devraient être accessibles (par exemple, si scaler_X est un ColumnTransformer).
                 # Si scaler_y est pour PV, l'utiliser pour autre chose est une approximation.
                 logger.warning(f"Approximation: Utilisation de scaler_y pour normaliser la perturbation '{dist_key}'. Ceci peut être incorrect.")
                 disturbance_values_scaled_for_model[dist_key] = scaler_y.transform(np.array([[dist_val]]))[0,0]


        for step in range(num_steps):
            current_time_s = step * tsamp_s
            
            if scenario_config['type'] == 'sp_step' and \
               current_time_s >= scenario_config.get('sp_step_time_s', float('inf')):
                current_sp_sim = last_initial_sp + scenario_config.get('sp_initial_offset', 0.0) + \
                                 scenario_config.get('sp_step_value_offset', 0.0)
            
            current_mv_sim = pid_sim.update(current_sp_sim, current_pv_sim)
            
            # Approximations pour la normalisation (idem, scaler_y est pour PV)
            pv_sim_for_update_scaled = scaler_y.transform(np.array([[current_pv_sim]]))[0,0]
            mv_sim_for_update_scaled = scaler_y.transform(np.array([[current_mv_sim]]))[0,0] # Approximation
            sp_sim_for_update_scaled = scaler_y.transform(np.array([[current_sp_sim]]))[0,0] # Approximation

            current_model_input_scaled = update_model_input(
                current_model_input_scaled,
                pv_sim_for_update_scaled,
                mv_sim_for_update_scaled,
                sp_sim_for_update_scaled,
                config['ModelFeatures'],
                disturbance_values_scaled_for_model 
            )
            
            pv_pred_scaled = model.predict(current_model_input_scaled)
            current_pv_sim_descaled = scaler_y.inverse_transform(pv_pred_scaled.reshape(-1, 1)).ravel()[0]

            if not np.isfinite(current_pv_sim_descaled) or \
               abs(current_pv_sim_descaled) > 1e7: # Seuil d'instabilité
                logger.warning(f"Instabilité détectée: PV_sim={current_pv_sim_descaled} pour PID {params_pid}. Arrêt.")
                if return_sim_data:
                    # Retourner les données partielles et un score élevé
                    sim_results_df = pd.DataFrame(sim_results).set_index('Time')
                    return float('inf'), sim_results_df
                return float('inf')

            current_pv_sim = current_pv_sim_descaled
            
            if return_sim_data:
                sim_results['Time'].append(current_time_s)
                sim_results['PV_sim'].append(current_pv_sim)
                sim_results['MV_sim'].append(current_mv_sim)
                sim_results['SP_sim'].append(current_sp_sim)

            error = current_sp_sim - current_pv_sim
            accumulated_absolute_error += abs(error) * tsamp_s

        performance_score = float('inf')
        metric_type = config['PERFORMANCE_METRICS'].get('metric_type', 'IAE').upper()
        if metric_type == 'IAE':
            performance_score = accumulated_absolute_error
        
        if return_sim_data:
            sim_results_df = pd.DataFrame(sim_results).set_index('Time')
            return performance_score, sim_results_df
        
        return performance_score

    except Exception as e:
        logger.error(f"Erreur dans evaluer_performance_pid pour PID {params_pid}: {e}", exc_info=True)
        if return_sim_data:
            return float('inf'), pd.DataFrame() # Retourner un DataFrame vide en cas d'erreur majeure
        return float('inf')