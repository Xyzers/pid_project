# src/data_processing.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_timeseries_data(df: pd.DataFrame, column_aliases_from_sql: list, 
                               essential_value_cols: list = None, 
                               datetime_col: str = 'DateTime'):
    if df.empty:
        logger.info("DataFrame initial vide, aucun pré-traitement à effectuer.")
        return df # Retourner un df vide, le delta_t sera None

    if datetime_col not in df.columns:
        logger.error(f"Colonne DateTime '{datetime_col}' non trouvée.")
        raise KeyError(f"Colonne DateTime '{datetime_col}' non trouvée.")

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)

    for col_alias in column_aliases_from_sql:
        if col_alias == datetime_col: continue # Déjà traité
        if col_alias in df.columns:
            df[col_alias] = pd.to_numeric(df[col_alias], errors='coerce')
        else:
            # C'est normal si un tag optionnel n'a pas été ramené par SQL car il n'y avait pas de valeur
            logger.debug(f"Colonne '{col_alias}' (alias SQL) non trouvée dans le DataFrame après pivot, ou tag optionnel sans données.")

    # Remplissage avant de vérifier les colonnes essentielles
    # (car une colonne essentielle pourrait avoir des NaNs initiaux)
    cols_to_fill = [col for col in column_aliases_from_sql if col != datetime_col and col in df.columns]
    if cols_to_fill:
        df[cols_to_fill] = df[cols_to_fill].ffill().bfill()
    
    if essential_value_cols:
        for col in essential_value_cols:
            if col not in df.columns:
                logger.error(f"Colonne essentielle '{col}' non trouvée dans le DataFrame après ffill/bfill.")
                raise ValueError(f"Colonne essentielle '{col}' manquante.")
            if df[col].isnull().any():
                logger.error(f"NaNs persistants dans la colonne essentielle '{col}' après remplissage. La suite du traitement peut échouer.")
                # Vous pourriez choisir de supprimer ces lignes ici ou de laisser l'appelant gérer
                # df.dropna(subset=[col], inplace=True) # Optionnel
    
    # Vérifier si le DataFrame est devenu vide après un éventuel dropna sur les colonnes essentielles
    if df.empty:
        logger.warning("DataFrame vide après pré-traitement (potentiellement après suppression de lignes avec NaNs dans colonnes essentielles).")
        return df # delta_t sera None

    # delta_t n'est pas retourné par votre preprocess_data original, mais peut être utile
    # Si vous ne le voulez pas, retirez cette partie
    delta_t = None
    if len(df.index) > 1:
        delta_t = df.index.to_series().diff().median()
        logger.info(f"Pas de temps médian des données nettoyées: {delta_t}")
    elif len(df.index) == 1:
         logger.info("Une seule ligne de données après pré-traitement, pas de delta_t calculable.")
    else:
         logger.info("Aucune donnée après pré-traitement, pas de delta_t.")

    return df # Ne pas retourner delta_t si votre signature de fonction originale ne le fait pas

def split_data_chronological(X: pd.DataFrame, y: pd.Series, train_ratio: float, val_ratio: float):
    # ... (votre code pour split_data_chronological)
    n_samples = len(X)
    if n_samples == 0: 
        logger.error("Impossible de diviser les données : X est vide.")
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype='float64')
        return empty_df, empty_df, empty_df, empty_series, empty_series, empty_series

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test < 0 :
        logger.error(f"Division invalide: train={train_ratio}, val={val_ratio} sur {n_samples} échantillons.")
        raise ValueError("Division des données impossible.")
    if n_test == 0: 
        logger.warning("L'ensemble de test sera de taille 0.")

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train:n_train + n_val], y.iloc[n_train:n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]
    
    logger.info(f"Données divisées: Train {len(X_train)} ({train_ratio*100:.1f}%), "
                f"Validation {len(X_val)} ({val_ratio*100:.1f}%), "
                f"Test {len(X_test)} ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    if len(X_test) == 0 and (train_ratio + val_ratio) < 1.0 and n_samples > 0 and n_test > 0: # Correction ici
         logger.warning("Ensemble de test vide malgré calculs indiquant données.")

    return X_train, X_val, X_test, y_train, y_val, y_test
    # ...
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler_type='MinMaxScaler'):
    # ... (votre code pour scale_data)
    # N'oubliez pas d'importer MinMaxScaler, StandardScaler de sklearn.preprocessing
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    if X_train.empty or y_train.empty:
        logger.warning("X_train ou y_train vide. Retour de données non scalées ou vides.")
        # ... (gestion des cas vides)
        empty_array_X = np.array([]).reshape(0, X_train.shape[1] if X_train.shape[1] > 0 else 1)
        empty_array_y = np.array([])
        scaler_X_dummy = MinMaxScaler() if scaler_type == 'MinMaxScaler' else StandardScaler()
        scaler_y_dummy = MinMaxScaler() if scaler_type == 'MinMaxScaler' else StandardScaler()
        return X_train.values, X_val.values, X_test.values, \
               y_train.values.ravel(), y_val.values.ravel(), y_test.values.ravel(), \
               scaler_X_dummy, scaler_y_dummy

    if scaler_type == 'StandardScaler':
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
    else:
        logger.warning(f"Scaler '{scaler_type}' non reconnu. Utilisation de MinMaxScaler.")
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val) if not X_val.empty else np.array([]).reshape(0, X_train_scaled.shape[1])
    X_test_scaled = scaler_X.transform(X_test) if not X_test.empty else np.array([]).reshape(0, X_train_scaled.shape[1])

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)) if not y_val.empty else np.array([])
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)) if not y_test.empty else np.array([]) # y_test et non y_test_original_series
    
    logger.info(f"Données mises à l'échelle avec {scaler_X.__class__.__name__}.")
    return X_train_scaled, X_val_scaled, X_test_scaled, \
           y_train_scaled.ravel(), y_val_scaled.ravel(), y_test_scaled.ravel(), \
           scaler_X, scaler_y    
    # ...
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled.ravel(), y_val_scaled.ravel(), y_test_scaled.ravel(), scaler_X, scaler_y