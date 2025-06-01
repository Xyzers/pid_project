# src/feature_engineering.py
import pandas as pd
import logging
import configparser

logger = logging.getLogger(__name__)

def create_lagged_features(df: pd.DataFrame, config_features: configparser.SectionProxy, target_col='PV_real'): # Assurez-vous que target_col a la bonne valeur par défaut ici aussi
    if df.empty:
        logger.warning("DataFrame vide pour create_lagged_features.")
        return pd.DataFrame(), pd.Series(dtype='float64')

    lagged_df = df.copy()
    if target_col not in lagged_df.columns:
        logger.error(f"Colonne cible '{target_col}' non trouvée dans le DataFrame.")
        raise KeyError(f"Colonne cible '{target_col}' non trouvée dans le DataFrame '{list(lagged_df.columns)}'.")
    
    # La colonne 'target_PV' est créée pour stocker la cible y avant que les lignes avec NaN ne soient supprimées
    lagged_df['target_PV_column_for_y'] = lagged_df[target_col] # Nom plus explicite
    
    feature_cols = []
    
    # UTILISER LES NOMS DE COLONNES AVEC LE SUFFIXE _real
    base_features_lags_config = {
        'PV_real': config_features.getint('pv_lags', 0),
        'MV_real': config_features.getint('mv_lags', 0),
        'SP_real': config_features.getint('sp_lags', 0),
        'Kp_hist': config_features.getint('kp_hist_lags', 0), # Garder Kp_hist, etc. si ces alias sont utilisés directement
        'Ti_hist': config_features.getint('ti_hist_lags', 0),
        'Td_hist': config_features.getint('td_hist_lags', 0)
    }

    # Gestion des tags de perturbation
    i = 1
    while True:
        dist_lag_key = f'disturbance_{i}_lags' # Clé dans le fichier .ini, ex: disturbance_1_lags
        # L'alias de la colonne dans le DataFrame sera DistX_real, ex: Dist1_real
        # Cet alias est construit dans db_utils.py
        column_alias_for_disturbance = f'Dist{i}_real' 
        
        if dist_lag_key in config_features: # Vérifier si la config de lag existe
            num_lags = config_features.getint(dist_lag_key, 0)
            if num_lags > 0: # Seulement si on veut des lags
                if column_alias_for_disturbance in df.columns: # Vérifier si la colonne existe dans le df
                    base_features_lags_config[column_alias_for_disturbance] = num_lags
                else:
                    logger.warning(f"Lags configurés pour '{dist_lag_key}' ({num_lags} lags), "
                                   f"mais la colonne correspondante '{column_alias_for_disturbance}' est inexistante dans le DataFrame. Elle sera ignorée.")
            i += 1
        else:
            # Arrêter si la clé de configuration de lag n'est plus trouvée (ex: pas de disturbance_3_lags)
            break 
            
    for col_name_from_config, num_lags in base_features_lags_config.items():
        # col_name_from_config est maintenant par ex. 'PV_real', 'Dist1_real'
        if col_name_from_config not in lagged_df.columns:
            if num_lags > 0: # Ne logguer un warning que si des lags étaient attendus
                logger.warning(f"Colonne '{col_name_from_config}' (configurée pour {num_lags} lags) non trouvée dans le DataFrame. Ignorée.")
            continue # Passer à la prochaine feature de base

        if num_lags > 0:
            logger.info(f"Création de {num_lags} lag(s) pour {col_name_from_config}.")
            for lag_idx in range(1, num_lags + 1):
                lagged_col_name = f'{col_name_from_config}_lag_{lag_idx}' # Ex: PV_real_lag_1
                lagged_df[lagged_col_name] = lagged_df[col_name_from_config].shift(lag_idx)
                feature_cols.append(lagged_col_name)
        # else: logger.info(f"Aucun lag configuré pour {col_name_from_config}.") # Optionnel: log pour 0 lags

    # On ne supprime les NaN qu'APRÈS avoir créé toutes les colonnes de lags
    initial_rows = len(lagged_df)
    # DropNA basé sur les colonnes de features créées. Si feature_cols est vide, cela ne fait rien.
    # Et aussi sur la colonne cible pour s'assurer que y n'a pas de NaN si elle provenait d'un shift aussi (non applicable ici car y = target_col non shiftée)
    # Mais la création des lags introduit des NaN au début des colonnes de features.
    if feature_cols: # Seulement si des features ont été créées (et donc des NaNs potentiellement introduits)
        lagged_df.dropna(subset=feature_cols, inplace=True)
    
    rows_dropped = initial_rows - len(lagged_df)
    if rows_dropped > 0:
        logger.info(f"{rows_dropped} lignes supprimées après création des lags (NaNs dus aux shifts).")

    if lagged_df.empty and initial_rows > 0: # Si le df devient vide après dropna mais n'était pas vide avant
        logger.error("DataFrame vide après création des lags et dropna. Vérifiez le nombre de lags par rapport à la longueur des données.")
        return pd.DataFrame(), pd.Series(dtype='float64')

    X = lagged_df[feature_cols] if feature_cols else pd.DataFrame(index=lagged_df.index) # Retourner un DF vide avec le bon index si pas de features
    y = lagged_df['target_PV_column_for_y'] # Utiliser la colonne cible sauvegardée
    
    logger.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
    if X.empty and feature_cols: # Si feature_cols n'était pas vide mais X l'est (après dropna)
        logger.error("X est vide bien que des features étaient attendues. "
                     "Cela peut arriver si tous les échantillons ont été supprimés par dropna.")
    elif not feature_cols:
        logger.warning("Aucune colonne de feature (lag) n'a été créée. X sera vide.")

    logger.info(f"Colonnes des features finales: {X.columns.tolist()}")
    
    return X, y