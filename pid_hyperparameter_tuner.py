#!/usr/bin/env python3
"""
Script de tuning interactif des hyperparamètres RandomForest
Utilise GridSearchCV pour trouver les meilleurs hyperparamètres.
"""

import sys
from pathlib import Path
import logging
import configparser
import pandas as pd
import numpy as np

# Importer les fonctions depuis les modules
from src.config_loader import load_config_and_setup_logging
from src.db_utils import get_db_engine, build_sql_query, extract_data
from src.data_processing import preprocess_timeseries_data, split_data_chronological, scale_data
from src.feature_engineering import create_lagged_features
from src.modeling import (
    tune_hyperparameters_grid_search,
    evaluate_model_comprehensive,
    generate_performance_report
)

logger = logging.getLogger(__name__)


def run_hyperparameter_tuning():
    """Pipeline de tuning des hyperparamètres."""
    
    executed_script_path = Path(__file__).resolve()
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
        print(f"ERREUR CRITIQUE: Échec config/logging: {e_cfg}", file=sys.stderr)
        if logging.getLogger().hasHandlers():
            logging.critical(f"Échec config/logging: {e_cfg}", exc_info=True)
        sys.exit(1)

    logger.info(f"--- Démarrage du tuning d'hyperparamètres ---")
    logger.info(f"Fichier script: {executed_script_path.name}")

    db_engine = None
    try:
        # Vérifier les sections requises
        required_sections = ['Database', 'Tags', 'DataPeriod', 'ModelFeatures', 'ModelParams', 'HyperparameterTuning']
        missing_sections = [s for s in required_sections if s not in config]
        if missing_sections:
            logger.error(f"Sections de config manquantes: {', '.join(missing_sections)}. Arrêt.")
            sys.exit(1)

        # Extraction et prétraitement des données
        logger.info("=== Phase 1 : Extraction et Prétraitement des Données ===")
        
        db_engine = get_db_engine(config['Database'])
        table_name = config['Database'].get('table_name', 'History')
        sql_query, column_aliases = build_sql_query(config['Tags'], config['DataPeriod'], table_name)
        raw_df = extract_data(db_engine, sql_query)

        if raw_df.empty:
            logger.error("Aucune donnée extraite. Arrêt.")
            sys.exit(1)

        essential_cols = ['PV_real', 'MV_real', 'SP_real']
        clean_df = preprocess_timeseries_data(raw_df, column_aliases, essential_value_cols=essential_cols)
        
        if clean_df.empty:
            logger.error("Données vides après pré-traitement. Arrêt.")
            sys.exit(1)

        logger.info(f"✓ Données extraites: {len(clean_df)} lignes")

        # Création des features
        logger.info("=== Phase 2 : Création des Features ===")
        
        X, y = create_lagged_features(clean_df, config['ModelFeatures'], target_col='PV_real')
        if X.empty or y.empty:
            logger.error("Features (X) ou cible (y) vides. Arrêt.")
            sys.exit(1)

        logger.info(f"✓ Features créées: X.shape={X.shape}, y.shape={y.shape}")

        # Division chronologique
        logger.info("=== Phase 3 : Division Chronologique ===")
        
        train_ratio = config['ModelFeatures'].getfloat('train_split_ratio', 0.7)
        val_ratio = config['ModelFeatures'].getfloat('validation_split_ratio', 0.15)

        X_train, X_val, X_test, y_train, y_val, y_test_original = \
            split_data_chronological(X, y, train_ratio, val_ratio)
        
        logger.info(f"✓ Ensemble d'entraînement: {X_train.shape[0]} samples")
        logger.info(f"✓ Ensemble de validation: {X_val.shape[0]} samples")
        logger.info(f"✓ Ensemble de test: {X_test.shape[0]} samples")

        # Scaling
        logger.info("=== Phase 4 : Normalisation des Données ===")
        
        scaler_type = config['ModelFeatures'].get('scaler_type', 'MinMaxScaler')
        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler_X, scaler_y = \
            scale_data(X_train, X_val, X_test, y_train, y_val, y_test_original, scaler_type)
        
        logger.info(f"✓ Données normalisées (scaler: {scaler_type})")

        # Optimisation des hyperparamètres
        logger.info("=== Phase 5 : Optimisation GridSearchCV ===")
        
        best_model, best_params, results_df = tune_hyperparameters_grid_search(
            X_train_s, y_train_s, config['ModelParams']
        )

        # Évaluation sur les ensembles de validation et test
        logger.info("=== Phase 6 : Évaluation du Meilleur Modèle ===")
        
        # Validation
        metrics_val, _, _ = evaluate_model_comprehensive(
            best_model, X_val_s, y_val, scaler_y, y_val.index
        )
        logger.info(f"Validation - R²: {metrics_val['r2']:.4f}, RMSE: {metrics_val['rmse']:.4f}")

        # Test
        metrics_test, _, _ = evaluate_model_comprehensive(
            best_model, X_test_s, y_test_original, scaler_y, y_test_original.index
        )
        logger.info(f"Test - R²: {metrics_test['r2']:.4f}, RMSE: {metrics_test['rmse']:.4f}")

        # Sauvegarde des résultats
        logger.info("=== Phase 7 : Sauvegarde des Résultats ===")
        
        # Sauvegarder les résultats de tuning
        tuning_results_path = Path(config['HyperparameterTuning'].get(
            'tuning_results_path', './trace/hyperparameter_tuning_results.csv'
        ))
        tuning_results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(tuning_results_path, index=False)
        logger.info(f"✓ Résultats de tuning sauvegardés: {tuning_results_path}")

        # Générer et sauvegarder le rapport
        report = generate_performance_report(metrics_test, config['ModelParams'], results_df)
        report_path = Path(config['HyperparameterTuning'].get(
            'tuning_report_path', './trace/hyperparameter_tuning_report.txt'
        ))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"✓ Rapport de performance sauvegardé: {report_path}")

        # Afficher le rapport
        logger.info(report)

        # Recommandations
        logger.info("=== RECOMMANDATIONS POUR LA CONFIG ===")
        logger.info(f"\nMettez à jour votre fichier {config_file_actual_name} [ModelParams] avec:")
        logger.info(f"  n_estimators = {best_params['n_estimators']}")
        logger.info(f"  max_depth = {best_params['max_depth']}")
        logger.info(f"  min_samples_leaf = {best_params['min_samples_leaf']}")
        logger.info(f"  min_samples_split = {best_params['min_samples_split']}")

    except Exception as e:
        logger.error(f"ERREUR PIPELINE: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if db_engine:
            db_engine.dispose()
        logger.info("=== Fin du tuning d'hyperparamètres ===")


if __name__ == '__main__':
    run_hyperparameter_tuning()
