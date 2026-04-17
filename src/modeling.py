# src/modeling.py
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
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
    logger.debug(f"config_output reçu: {dict(config_output) if hasattr(config_output, 'items') else config_output}")
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


def evaluate_model_comprehensive(model, X_test_scaled, y_test_original, scaler_y, y_test_index=None):
    """
    Évaluation complète du modèle avec métriques détaillées et temps d'exécution.
    
    Returns:
        dict: Dictionnaire avec toutes les métriques (RMSE, R², MAE, MAPE, quantiles, temps inférence)
        np.ndarray: Prédictions descalées
        pd.DataFrame: DataFrame avec prédictions et valeurs réelles
    """
    if X_test_scaled.shape[0] == 0:
        logger.warning("Ensemble de test vide. Évaluation ignorée.")
        empty_metrics = {
            'rmse': np.nan, 'r2': np.nan, 'mae': np.nan, 'mape': np.nan,
            'median_error': np.nan, 'q75_error': np.nan, 'q95_error': np.nan,
            'inference_time_ms': np.nan, 'num_samples': 0
        }
        return empty_metrics, np.array([]), pd.DataFrame()

    if not isinstance(y_test_original, np.ndarray):
        logger.warning("y_test_original n'est pas un ndarray. Tentative de conversion.")
        y_test_original = np.array(y_test_original)

    # Mesure du temps d'inférence
    start_predict = time.time()
    y_pred_scaled = model.predict(X_test_scaled)
    inference_time = time.time() - start_predict
    
    y_pred_descaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    if len(y_test_original) != len(y_pred_descaled):
        logger.error(f"Discordance longueur y_test ({len(y_test_original)}) et prédictions ({len(y_pred_descaled)}).")
        empty_metrics = {
            'rmse': np.nan, 'r2': np.nan, 'mae': np.nan, 'mape': np.nan,
            'median_error': np.nan, 'q75_error': np.nan, 'q95_error': np.nan,
            'inference_time_ms': np.nan, 'num_samples': 0
        }
        return empty_metrics, np.array([]), pd.DataFrame()

    # Calcul des métriques
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_descaled))
    r2 = r2_score(y_test_original, y_pred_descaled)
    mae = mean_absolute_error(y_test_original, y_pred_descaled)
    
    # MAPE (Mean Absolute Percentage Error)
    mask_nonzero = y_test_original != 0
    if mask_nonzero.sum() > 0:
        mape = np.mean(np.abs((y_test_original[mask_nonzero] - y_pred_descaled[mask_nonzero]) / 
                              np.abs(y_test_original[mask_nonzero]))) * 100
    else:
        mape = np.nan

    # Analyse des erreurs (quantiles)
    errors = np.abs(y_test_original - y_pred_descaled)
    median_error = np.median(errors)
    q75_error = np.percentile(errors, 75)
    q95_error = np.percentile(errors, 95)

    metrics = {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape,
        'median_error': median_error,
        'q75_error': q75_error,
        'q95_error': q95_error,
        'inference_time_ms': inference_time * 1000,
        'num_samples': len(y_test_original)
    }

    logger.info(
        f"Évaluation complète: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, "
        f"Inférence={inference_time*1000:.2f}ms"
    )

    # Création du DataFrame de prédictions
    final_index_for_df = y_test_index
    if final_index_for_df is None or len(final_index_for_df) != len(y_test_original):
        logger.warning("Index invalide. Utilisation d'un RangeIndex.")
        final_index_for_df = pd.RangeIndex(start=0, stop=len(y_test_original), step=1)
    
    try:
        predictions_df = pd.DataFrame({
            'Actual_PV': y_test_original,
            'Predicted_PV': y_pred_descaled,
            'Error': errors,
            'Abs_Error_Pct': (errors / np.abs(y_test_original)) * 100
        }, index=final_index_for_df)
    except Exception as e_df:
        logger.error(f"Erreur création DataFrame: {e_df}. Index par défaut.")
        predictions_df = pd.DataFrame({
            'Actual_PV': y_test_original,
            'Predicted_PV': y_pred_descaled,
            'Error': errors
        })

    return metrics, y_pred_descaled, predictions_df


def tune_hyperparameters_grid_search(X_train_scaled, y_train_scaled, config_model_params: dict):
    """
    Optimisation des hyperparamètres RandomForest via GridSearchCV.
    
    Args:
        X_train_scaled: Features d'entraînement normalisées
        y_train_scaled: Cible d'entraînement normalisée
        config_model_params: Section [ModelParams] de la config
    
    Returns:
        tuple: (best_model, best_params, grid_search_results_df)
    """
    logger.info("=== Démarrage de l'optimisation GridSearchCV ===")
    
    # Extraction des paramètres de recherche depuis la config
    n_estimators_range = config_model_params.get('n_estimators_range', '50,100,200')
    max_depth_range = config_model_params.get('max_depth_range', '10,15,20,None')
    min_samples_leaf_range = config_model_params.get('min_samples_leaf_range', '1,2,5')
    min_samples_split_range = config_model_params.get('min_samples_split_range', '2,5')
    cv_folds = config_model_params.getint('cv_folds', 5)
    
    # Parse des ranges
    def parse_range_str(range_str):
        """Convertit '1,2,3' ou '1,2,None' en liste avec les bons types."""
        return [None if x.strip().lower() == 'none' else int(x.strip()) 
                for x in range_str.split(',')]
    
    param_grid = {
        'n_estimators': parse_range_str(n_estimators_range),
        'max_depth': parse_range_str(max_depth_range),
        'min_samples_leaf': parse_range_str(min_samples_leaf_range),
        'min_samples_split': parse_range_str(min_samples_split_range)
    }
    
    logger.info(f"Grille de recherche:\n{param_grid}")
    
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    start_time = time.time()
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_folds,
        scoring='r2',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    grid_search.fit(X_train_scaled, y_train_scaled)
    tuning_time = time.time() - start_time
    
    logger.info(f"✓ Optimisation terminée en {tuning_time:.2f}s")
    logger.info(f"✓ Meilleurs paramètres: {grid_search.best_params_}")
    logger.info(f"✓ Meilleur score CV (R²): {grid_search.best_score_:.4f}")
    
    # Création d'un DataFrame avec tous les résultats
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df[[
        'param_n_estimators', 'param_max_depth', 'param_min_samples_leaf', 'param_min_samples_split',
        'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score'
    ]]
    results_df.columns = [
        'n_est', 'max_depth', 'min_leaf', 'min_split',
        'mean_r2_test', 'std_r2_test', 'mean_r2_train', 'rank'
    ]
    results_df = results_df.sort_values('rank')
    
    logger.info(f"\nTop 10 configurations:\n{results_df.head(10).to_string()}")
    
    return grid_search.best_estimator_, grid_search.best_params_, results_df


def generate_performance_report(metrics: dict, config_model_params: dict, tuning_results_df=None):
    """
    Génère un rapport structuré de performance du modèle.
    
    Args:
        metrics: Dictionnaire des métriques complètes
        config_model_params: Configuration du modèle
        tuning_results_df: Résultats de GridSearchCV (optionnel)
    
    Returns:
        str: Rapport formaté
    """
    report = "\n" + "="*70 + "\n"
    report += "RAPPORT DE PERFORMANCE DU MODÈLE RANDOM FOREST\n"
    report += "="*70 + "\n\n"
    
    # Métriques de précision
    report += "📊 MÉTRIQUES DE PRÉCISION\n"
    report += "-" * 70 + "\n"
    report += f"  R² Score (Coefficient de détermination):  {metrics.get('r2', np.nan):.4f}\n"
    report += f"  RMSE (Root Mean Squared Error):          {metrics.get('rmse', np.nan):.4f}\n"
    report += f"  MAE (Mean Absolute Error):               {metrics.get('mae', np.nan):.4f}\n"
    report += f"  MAPE (Mean Absolute Percentage Error):   {metrics.get('mape', np.nan):.2f}%\n"
    report += f"\n"
    
    # Analyse des erreurs
    report += "📈 ANALYSE DES ERREURS\n"
    report += "-" * 70 + "\n"
    report += f"  Erreur médiane:                         {metrics.get('median_error', np.nan):.4f}\n"
    report += f"  75e percentile (Q75):                    {metrics.get('q75_error', np.nan):.4f}\n"
    report += f"  95e percentile (Q95):                    {metrics.get('q95_error', np.nan):.4f}\n"
    report += f"\n"
    
    # Performance en temps
    report += "⏱️  PERFORMANCE EN TEMPS\n"
    report += "-" * 70 + "\n"
    report += f"  Temps d'inférence:                       {metrics.get('inference_time_ms', np.nan):.2f} ms\n"
    report += f"  Nombre d'échantillons testés:            {metrics.get('num_samples', 0)}\n"
    report += f"\n"
    
    # Interprétation
    report += "💡 INTERPRÉTATION\n"
    report += "-" * 70 + "\n"
    
    r2 = metrics.get('r2', np.nan)
    if r2 > 0.9:
        quality = "Excellent ✓✓✓"
    elif r2 > 0.8:
        quality = "Bon ✓✓"
    elif r2 > 0.7:
        quality = "Acceptable ✓"
    else:
        quality = "À améliorer"
    report += f"  Qualité du modèle:                       {quality} (R² = {r2:.4f})\n"
    
    mape = metrics.get('mape', np.nan)
    if mape < 5:
        accuracy_label = "Très haute ✓"
    elif mape < 10:
        accuracy_label = "Haute ✓"
    elif mape < 20:
        accuracy_label = "Modérée"
    else:
        accuracy_label = "Faible"
    report += f"  Précision (MAPE):                        {accuracy_label} ({mape:.2f}%)\n"
    
    report += "="*70 + "\n"
    
    return report