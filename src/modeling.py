# src/modeling.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump, load
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def train_model(X_train_scaled, y_train_scaled, config_model_params: dict):
    # ... (votre code pour train_model)
    if X_train_scaled.shape[0] == 0 or y_train_scaled.shape[0] == 0:
        logger.error("Données d'entraînement vides. Impossible d'entraîner.")
        raise ValueError("Données d'entraînement vides.")

    model_type = config_model_params.get('model_type', 'RandomForestRegressor')
    
    if model_type == 'RandomForestRegressor':
        n_estimators = config_model_params.getint('n_estimators', 100)
        max_depth_str = config_model_params.get('max_depth', 'None')
        max_depth = None if max_depth_str.lower() == 'none' else int(max_depth_str)
        random_state = config_model_params.getint('random_state', 42)
        min_samples_leaf = config_model_params.getint('min_samples_leaf', 1) # Default était 1
        min_samples_split = config_model_params.getint('min_samples_split', 2) # Default était 2

        logger.info(f"Entraînement RandomForest: n_est={n_estimators}, depth={max_depth}, leaf={min_samples_leaf}, split={min_samples_split}, state={random_state}")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
            min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
            n_jobs=-1
        )
    else:
        logger.error(f"Type de modèle '{model_type}' non supporté.")
        raise NotImplementedError(f"Modèle '{model_type}' non supporté.")

    model.fit(X_train_scaled, y_train_scaled)
    logger.info("Entraînement du modèle terminé.")
    return model    
    # ...
    return model

def evaluate_model(model, X_test_scaled, y_test_original, scaler_y, y_test_index):
    # ... (votre code pour evaluate_model)
    if X_test_scaled.shape[0] == 0:
        logger.warning("Ensemble de test vide. Évaluation ignorée.")
        empty_pred_df_index = y_test_index if y_test_index is not None and not y_test_index.empty else pd.Index([])
        return np.nan, np.nan, np.nan, pd.DataFrame(columns=['Actual_PV', 'Predicted_PV'], index=empty_pred_df_index)

    if not isinstance(y_test_original, np.ndarray):
        logger.warning("y_test_original n'est pas un ndarray. Tentative de conversion.")
        y_test_original = np.array(y_test_original)

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred_descaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    if len(y_test_original) != len(y_pred_descaled):
        logger.error(f"Discordance longueur y_test_original ({len(y_test_original)}) et y_pred_descaled ({len(y_pred_descaled)}).")
        return np.nan, np.nan, np.nan, pd.DataFrame(columns=['Actual_PV', 'Predicted_PV'], index=y_test_index if y_test_index is not None else pd.Index([]))

    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_descaled))
    r2 = r2_score(y_test_original, y_pred_descaled)
    mae = mean_absolute_error(y_test_original, y_pred_descaled)

    logger.info(f"Évaluation modèle (test): RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

    final_index_for_df = y_test_index
    if final_index_for_df is None or len(final_index_for_df) != len(y_test_original):
        logger.warning("Index y_test_index invalide. Utilisation d'un RangeIndex.")
        final_index_for_df = pd.RangeIndex(start=0, stop=len(y_test_original), step=1)
    
    try:
        predictions_df = pd.DataFrame({'Actual_PV': y_test_original, 'Predicted_PV': y_pred_descaled}, index=final_index_for_df)
    except Exception as e_df:
        logger.error(f"Erreur création DataFrame prédictions: {e_df}. Index par défaut.")
        predictions_df = pd.DataFrame({'Actual_PV': y_test_original, 'Predicted_PV': y_pred_descaled})

    return rmse, r2, mae, predictions_df
    # ...
    return rmse, r2, mae, predictions_df

def save_model_and_scalers(model, scaler_X, scaler_y, config_output: dict):
    # ... (votre code pour save_model_and_scalers)
    model_path_str = config_output.get('model_save_path', './trained_model.joblib')
    scalers_path_str = config_output.get('scalers_save_path', './scalers.joblib')

    model_path = Path(model_path_str).resolve()
    scalers_path = Path(scalers_path_str).resolve()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scalers_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        dump(model, model_path)
        logger.info(f"Modèle sauvegardé dans '{model_path}'.")
    except Exception as e:
        logger.error(f"Erreur sauvegarde modèle '{model_path}': {e}", exc_info=True)
        raise

    try:
        scalers_dict = {'scaler_X': scaler_X, 'scaler_y': scaler_y}
        dump(scalers_dict, scalers_path)
        logger.info(f"Scalers sauvegardés dans '{scalers_path}'.")
    except Exception as e:
        logger.error(f"Erreur sauvegarde scalers '{scalers_path}': {e}", exc_info=True)
        raise
    # ...

def load_model_and_scalers(config_output: dict):
    # ... (votre code pour load_model_and_scalers)
    print(f"DEBUG_MODELING: model_paths_config reçu: {model_paths_config}")
    model_path_str = config_output.get('model_save_path')
    scalers_path_str = config_output.get('scalers_save_path')

    if not model_path_str or not scalers_path_str:
        logger.error("Chemins 'model_save_path' ou 'scalers_save_path' non définis.")
        raise ValueError("Chemins modèle/scalers non configurés.")

    model_path = Path(model_path_str).resolve()
    scalers_path = Path(scalers_path_str).resolve()

    if not model_path.is_file():
        logger.error(f"Fichier modèle '{model_path}' non trouvé.")
        raise FileNotFoundError(f"Fichier modèle '{model_path}' non trouvé.")
    if not scalers_path.is_file():
        logger.error(f"Fichier scalers '{scalers_path}' non trouvé.")
        raise FileNotFoundError(f"Fichier scalers '{scalers_path}' non trouvé.")

    model = load(model_path)
    scalers_dict = load(scalers_path)
    scaler_X = scalers_dict['scaler_X']
    scaler_y = scalers_dict['scaler_y']
    logger.info(f"Modèle et scalers chargés de '{model_path}' et '{scalers_path}'.")
    return model, scaler_X, scaler_y
    # ...
    return model, scaler_X, scaler_y