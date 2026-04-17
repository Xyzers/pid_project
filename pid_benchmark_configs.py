#!/usr/bin/env python3
"""
Script de comparaison comparative de configurations RandomForest
Teste plusieurs configurations et génère un rapport récapitulatif.
"""

import sys
from pathlib import Path
import logging
import time
import pandas as pd
import numpy as np

from src.config_loader import load_config_and_setup_logging
from src.db_utils import get_db_engine, build_sql_query, extract_data
from src.data_processing import preprocess_timeseries_data, split_data_chronological, scale_data
from src.feature_engineering import create_lagged_features
from src.modeling import train_model, evaluate_model_comprehensive
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def benchmark_configurations(X_train_s, y_train_s, X_test_s, y_test_original, scaler_y, y_test_index, configurations):
    """
    Teste plusieurs configurations et compare leurs performances.
    
    Args:
        configurations: List[dict] avec keys n_estimators, max_depth, etc.
    
    Returns:
        pd.DataFrame avec résultats comparatifs
    """
    results = []
    
    logger.info(f"\n=== Benchmarking de {len(configurations)} configurations ===\n")
    
    for idx, config in enumerate(configurations, 1):
        logger.info(f"Configuration {idx}/{len(configurations)}: {config}")
        
        try:
            # Temps d'entraînement
            start_train = time.time()
            model = RandomForestRegressor(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', None),
                min_samples_leaf=config.get('min_samples_leaf', 1),
                min_samples_split=config.get('min_samples_split', 2),
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_s, y_train_s)
            train_time = time.time() - start_train
            
            # Évaluation
            metrics, _, _ = evaluate_model_comprehensive(
                model, X_test_s, y_test_original, scaler_y, y_test_index
            )
            
            result = {
                'n_estimators': config.get('n_estimators', 100),
                'max_depth': config.get('max_depth', 'None'),
                'min_samples_leaf': config.get('min_samples_leaf', 1),
                'min_samples_split': config.get('min_samples_split', 2),
                'train_time_s': train_time,
                'inference_time_ms': metrics['inference_time_ms'],
                'r2': metrics['r2'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape'],
                'status': 'OK'
            }
            results.append(result)
            logger.info(f"  ✓ Train: {train_time:.2f}s, R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
        except Exception as e:
            logger.error(f"  ✗ Erreur: {e}")
            result = {
                'n_estimators': config.get('n_estimators', 100),
                'max_depth': config.get('max_depth', 'None'),
                'min_samples_leaf': config.get('min_samples_leaf', 1),
                'min_samples_split': config.get('min_samples_split', 2),
                'status': f'ERROR: {str(e)[:50]}'
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)
    return results_df


def run_benchmark():
    """Pipeline de benchmarking."""
    
    config_base_name = "pid_model_builder"
    config_file_actual_name = f"{config_base_name}.ini"

    config = None
    try:
        config = load_config_and_setup_logging(
            script_stem_for_config_file=config_base_name,
            config_file_name=config_file_actual_name
        )
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e_cfg:
        print(f"ERREUR CRITIQUE: {e_cfg}", file=sys.stderr)
        sys.exit(1)

    logger.info("=== Démarrage du Benchmarking de Configurations ===")

    db_engine = None
    try:
        # Extraction et prétraitement
        logger.info("Extraction et prétraitement des données...")
        
        db_engine = get_db_engine(config['Database'])
        table_name = config['Database'].get('table_name', 'History')
        sql_query, column_aliases = build_sql_query(config['Tags'], config['DataPeriod'], table_name)
        raw_df = extract_data(db_engine, sql_query)

        if raw_df.empty:
            logger.error("Aucune donnée extraite.")
            sys.exit(1)

        essential_cols = ['PV_real', 'MV_real', 'SP_real']
        clean_df = preprocess_timeseries_data(raw_df, column_aliases, essential_value_cols=essential_cols)
        
        if clean_df.empty:
            logger.error("Données vides après prétraitement.")
            sys.exit(1)

        # Création des features
        X, y = create_lagged_features(clean_df, config['ModelFeatures'], target_col='PV_real')
        if X.empty or y.empty:
            logger.error("Features vides.")
            sys.exit(1)

        # Division et scaling
        train_ratio = config['ModelFeatures'].getfloat('train_split_ratio', 0.7)
        val_ratio = config['ModelFeatures'].getfloat('validation_split_ratio', 0.15)
        X_train, X_val, X_test, y_train, y_val, y_test_original = \
            split_data_chronological(X, y, train_ratio, val_ratio)

        scaler_type = config['ModelFeatures'].get('scaler_type', 'MinMaxScaler')
        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler_X, scaler_y = \
            scale_data(X_train, X_val, X_test, y_train, y_val, y_test_original, scaler_type)

        logger.info(f"✓ Données prêtes: {X_train_s.shape[0]} samples train, {X_test_s.shape[0]} test")

        # Configurations à tester
        configurations = [
            {'n_estimators': 50, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2},
            {'n_estimators': 100, 'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 5},
            {'n_estimators': 150, 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5},
            {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2},
            {'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 3, 'min_samples_split': 2},  # Config actuelle
        ]

        # Benchmarking
        results_df = benchmark_configurations(
            X_train_s, y_train_s, X_test_s, y_test_original, scaler_y, y_test_original.index, configurations
        )

        # Afficher les résultats
        logger.info("\n" + "="*100)
        logger.info("RÉSUMÉ COMPARATIF")
        logger.info("="*100)
        logger.info(results_df.to_string())

        # Sauvegarder
        benchmark_path = Path('./trace/benchmark_results.csv')
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(benchmark_path, index=False)
        logger.info(f"\n✓ Résultats sauvegardés: {benchmark_path}")

        # Recommandation
        if 'r2' in results_df.columns:
            best_idx = results_df['r2'].idxmax()
            best_config = results_df.loc[best_idx]
            logger.info(f"\n💡 Meilleure configuration (R²): {best_config.to_dict()}")

    except Exception as e:
        logger.error(f"ERREUR: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if db_engine:
            db_engine.dispose()
        logger.info("\n=== Fin du Benchmarking ===")


if __name__ == '__main__':
    run_benchmark()
