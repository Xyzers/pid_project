# pid_tuner_main.py

# 1. Imports Python Standard Library
import sys
import logging # Sera initialisé plus tard
import configparser
import time
import winsound
import warnings
from pathlib import Path

# Vérification de base du dossier 'src'
src_path_check = Path("./src")
if not src_path_check.is_dir():
    print(f"ERREUR CRITIQUE: Le dossier 'src' est introuvable. "
          f"Veuillez lancer le script depuis la racine du projet.", file=sys.stderr)
    sys.exit("Arrêt prématuré: Dossier 'src' manquant.")

# 2. Imports des bibliothèques tierces
try:
    import pandas as pd
    import numpy as np
    import joblib
except ImportError as e:
    print(f"ERREUR CRITIQUE D'IMPORT: {e}. Avez-vous installé requirements.txt ?", file=sys.stderr)
    sys.exit(1)

# 3. Imports des modules internes du projet
from src.config_loader import load_config_and_setup_logging
from src.pid_logic.pid_controller import PIDController
from src.db_utils import get_db_engine, build_sql_query, extract_data
from src.data_processing import preprocess_timeseries_data
from src.feature_engineering import create_lagged_features

logger = logging.getLogger(__name__)


def _clip_params_to_bounds(params, bounds):
    clipped = []
    for val, (vmin, vmax) in zip(params, bounds):
        clipped.append(max(vmin, min(vmax, val)))
    return clipped


def _force_model_single_thread_for_tuning(process_model):
    """Force n_jobs=1 pour éviter le parallélisme imbriqué pendant le tuning global."""
    try:
        if hasattr(process_model, "get_params") and hasattr(process_model, "set_params"):
            params = process_model.get_params(deep=True)
            if "n_jobs" in params:
                process_model.set_params(n_jobs=1)
                logger.info("Mode robuste: modèle configuré en mono-thread (n_jobs=1) pendant le tuning.")
                return
        if hasattr(process_model, "n_jobs"):
            process_model.n_jobs = 1
            logger.info("Mode robuste: modèle configuré en mono-thread (n_jobs=1) pendant le tuning.")
    except Exception as e:
        logger.warning(f"Impossible de forcer n_jobs=1 sur le modèle chargé: {e}")


def _read_optimizer_settings(config):
    optimizer_cfg = config['OPTIMIZER'] if config.has_section('OPTIMIZER') else None

    settings = {
        'kp_min': optimizer_cfg.getfloat('kp_min', 0.1) if optimizer_cfg else 0.1,
        'kp_max': optimizer_cfg.getfloat('kp_max', 10.0) if optimizer_cfg else 10.0,
        'ti_min': optimizer_cfg.getfloat('ti_min', 10.0) if optimizer_cfg else 10.0,
        'ti_max': optimizer_cfg.getfloat('ti_max', 2000.0) if optimizer_cfg else 2000.0,
        'td_min': optimizer_cfg.getfloat('td_min', 0.0) if optimizer_cfg else 0.0,
        'td_max': optimizer_cfg.getfloat('td_max', 100.0) if optimizer_cfg else 100.0,
        'lbfgs_maxiter': optimizer_cfg.getint('lbfgs_maxiter', 80) if optimizer_cfg else 80,
        'lbfgs_maxfun': optimizer_cfg.getint('lbfgs_maxfun', 250) if optimizer_cfg else 250,
        'lbfgs_restarts': optimizer_cfg.getint('lbfgs_restarts', 4) if optimizer_cfg else 4,
        'de_enabled': optimizer_cfg.getboolean('de_enabled', True) if optimizer_cfg else True,
        'de_force_if_flat': optimizer_cfg.getboolean('de_force_if_flat', False) if optimizer_cfg else False,
        'de_always_refine': optimizer_cfg.getboolean('de_always_refine', False) if optimizer_cfg else False,
        'de_maxiter': optimizer_cfg.getint('de_maxiter', 8) if optimizer_cfg else 8,
        'de_popsize': optimizer_cfg.getint('de_popsize', 6) if optimizer_cfg else 6,
        'de_polish': optimizer_cfg.getboolean('de_polish', False) if optimizer_cfg else False,
        'de_max_runtime_seconds': optimizer_cfg.getint('de_max_runtime_seconds', 300) if optimizer_cfg else 300,
        'de_workers': optimizer_cfg.getint('de_workers', -1) if optimizer_cfg else -1,
        'de_updating': optimizer_cfg.get('de_updating', 'deferred').strip().lower() if optimizer_cfg else 'deferred',
        'suppress_nested_parallel_warning': optimizer_cfg.getboolean('suppress_nested_parallel_warning', True) if optimizer_cfg else True,
    }

    if settings['de_updating'] not in ('immediate', 'deferred'):
        logger.warning(
            f"Valeur de_updating invalide '{settings['de_updating']}', fallback sur 'deferred'."
        )
        settings['de_updating'] = 'deferred'

    if settings['de_workers'] != 1 and settings['de_updating'] == 'immediate':
        logger.warning(
            "Incompatibilité détectée: de_updating='immediate' exige de_workers=1. "
            "Fallback sur de_updating='deferred' pour conserver le parallélisme multi-cœur."
        )
        settings['de_updating'] = 'deferred'

    if settings['suppress_nested_parallel_warning']:
        warnings.filterwarnings(
            "ignore",
            message=r".*Loky-backed parallel loops cannot be called in a multiprocessing.*",
            category=UserWarning,
        )

    return settings


def run_model_reactivity_check(process_model, scaler_X, scaler_y, initial_state_df):
    """Vérifie si le modèle réagit aux variations des lags MV/SP/PV.

    Ce test ne remplace pas l'optimisation, il sert uniquement de diagnostic
    quand la surface de coût est plate.
    """
    base_features = initial_state_df.copy()
    base_pred_scaled = process_model.predict(scaler_X.transform(base_features))[0]
    base_pred = scaler_y.inverse_transform([[base_pred_scaled]])[0][0]

    tests = [
        ('MV_real_lag_1', 1.0),
        ('SP_real_lag_1', 1.0),
        ('PV_real_lag_1', 1.0),
    ]

    logger.info("--- Diagnostic réactivité modèle (perturbation 1 pas) ---")
    logger.info(f"Prédiction de base PV: {base_pred:.6f}")

    for col_name, delta in tests:
        if col_name not in base_features.columns:
            logger.warning(f"Colonne absente pour diagnostic: {col_name}")
            continue

        varied = base_features.copy()
        varied[col_name] = varied[col_name] + delta
        pred_scaled = process_model.predict(scaler_X.transform(varied))[0]
        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        effect = pred - base_pred

        logger.info(
            f"Test {col_name} +{delta:.3f} -> PV_pred={pred:.6f} (delta={effect:+.6f})"
        )

    logger.info("--- Fin diagnostic réactivité modèle ---")


def run_pid_sensitivity_protocol(process_model, scaler_X, scaler_y, config, initial_state_df, initial_guess, bounds):
    """
    Protocole terrain: évalue une grille de 10 jeux PID pour vérifier
    que la fonction objectif est réellement sensible aux paramètres.

    Returns:
        tuple[bool, list]: (is_sensitive, scored_trials)
    """
    kp0, ti0, td0 = initial_guess
    (kp_min, kp_max), (ti_min, ti_max), (td_min, td_max) = bounds

    # 10 essais simples et lisibles pour diagnostic maintenance.
    trial_params = [
        [kp0, ti0, td0],
        [min(kp_max, kp0 * 1.5), ti0, td0],
        [max(kp_min, kp0 * 0.7), ti0, td0],
        [kp0, min(ti_max, ti0 * 1.5), td0],
        [kp0, max(ti_min, ti0 * 0.7), td0],
        [kp0, ti0, min(td_max, td0 + 5.0)],
        [kp0, ti0, min(td_max, td0 + 20.0)],
        [kp_min, ti_max, td_min],
        [kp_max, ti_min, td_max],
        [0.5 * (kp_min + kp_max), 0.5 * (ti_min + ti_max), 0.5 * (td_min + td_max)],
    ]

    scored_trials = []
    seen = set()
    for idx, params in enumerate(trial_params, start=1):
        params = _clip_params_to_bounds(params, bounds)
        key = (round(params[0], 8), round(params[1], 8), round(params[2], 8))
        if key in seen:
            continue
        seen.add(key)
        score = evaluate_closed_loop_performance(params, process_model, scaler_X, scaler_y, config, initial_state_df)
        scored_trials.append((idx, params, score))

    min_score = min(t[2] for t in scored_trials)
    max_score = max(t[2] for t in scored_trials)
    spread = max_score - min_score
    rel_spread = spread / (abs(min_score) + 1e-12)
    is_sensitive = rel_spread > 0.05

    logger.info("--- Protocole sensibilité (10 essais PID) ---")
    for idx, params, score in scored_trials:
        logger.info(
            f"Essai {idx:02d} | Kp={params[0]:.3f} Ti={params[1]:.3f} Td={params[2]:.3f} | Score={score:.6f}"
        )
    logger.info(f"Spread score absolu: {spread:.6f} | Spread relatif: {100.0 * rel_spread:.2f}%")
    if is_sensitive:
        logger.info("Diagnostic: objectif sensible aux paramètres PID (OK pour optimisation).")
    else:
        logger.warning("Diagnostic: objectif peu sensible (surface plate). L'optimisation locale peut rester bloquée.")

    return is_sensitive, scored_trials

def evaluate_closed_loop_performance(params, process_model, scaler_X, scaler_y, config, initial_state_df) -> float:
    """
    Fonction objectif pour l'optimiseur (Scipy).
    Prend un tableau de paramètres [Kp, Ti, Td] et retourne un score d'erreur.
    Utilise le modèle ML (process_model) pour simuler la réponse du procédé.

    Le scénario implémenté est un échelon de consigne (SP step) :
    - Phase 1 (avant sp_step_time) : SP = SP_initiale + sp_initial_offset
    - Phase 2 (après sp_step_time) : SP = SP_phase1 + sp_step_value_offset

    Returns:
        float: Score d'erreur (IAE ou ISE selon la config). Plus petit = meilleur.
    """
    kp, ti, td = params

    # --- 1. Lecture des paramètres depuis la config ---
    tsamp_ms = config['SIMULATION_PARAMS'].getfloat('tsamp_simulation_ms', 100)
    tsamp_s = tsamp_ms / 1000.0
    mv_min = config['SIMULATION_PARAMS'].getfloat('mv_min', 0.0)
    mv_max = config['SIMULATION_PARAMS'].getfloat('mv_max', 100.0)
    direct_action = config['PID_PARAMS'].getboolean('direct_action', True)
    pid_structure = config['PID_PARAMS'].get('pid_structure', 'parallel_isa')
    derivative_action = config['PID_PARAMS'].get('derivative_action', 'on_pv')

    # On récupère l'état t-1 pour amorcer le PID
    current_pv = initial_state_df['PV_real_lag_1'].values[0]
    current_sp = initial_state_df['SP_real_lag_1'].values[0]
    current_mv = initial_state_df['MV_real_lag_1'].values[0]

    pid = PIDController(Kp=kp, Ti=ti, Td=td, Tsamp=tsamp_s,
                        mv_min=mv_min, mv_max=mv_max,
                        direct_action=direct_action, initial_mv=current_mv,
                        pid_structure=pid_structure,
                        derivative_action=derivative_action)
    pid.set_initial_state(current_pv, current_sp, current_mv, active_initial=True)

    # --- 2. Configuration du Scénario (Échelon de consigne) ---
    sp_initial_offset = config['SCENARIO_PARAMS'].getfloat('sp_initial_offset', 0.0)
    sp_step_time_s = config['SCENARIO_PARAMS'].getfloat('sp_step_time_seconds', 15.0)
    sp_step_value = config['SCENARIO_PARAMS'].getfloat('sp_step_value_offset', 2.0)

    sp_before_step = current_sp + sp_initial_offset
    sp_after_step = sp_before_step + sp_step_value

    duration_s = config['SIMULATION_PARAMS'].getint('simulation_duration_seconds', 60)
    steps = int(duration_s / tsamp_s)

    # Choix de la métrique depuis la config
    metric_type = config['PERFORMANCE_METRICS'].get('metric_type', 'IAE').upper()

    score = 0.0
    current_features = initial_state_df.copy()

    # Identifier toutes les familles de lags présentes (PV, MV, SP, Dist1, Dist2...)
    lag_families = {}
    for col in current_features.columns:
        if '_lag_' in col:
            base_name = col.rsplit('_lag_', 1)[0]  # ex: 'PV_real', 'Dist1_real'
            lag_num = int(col.rsplit('_lag_', 1)[1])
            if base_name not in lag_families:
                lag_families[base_name] = 0
            lag_families[base_name] = max(lag_families[base_name], lag_num)

    # --- 3. Boucle de Simulation Temporelle ---
    for step in range(steps):
        current_time_s = step * tsamp_s

        # a. Calcul de la SP avec le scénario d'échelon
        if current_time_s < sp_step_time_s:
            target_sp = sp_before_step
        else:
            target_sp = sp_after_step

        # b. L'IA prédit la prochaine PV (Mesure)
        X_scaled = scaler_X.transform(current_features)
        predicted_pv_scaled = process_model.predict(X_scaled)[0]
        predicted_pv = scaler_y.inverse_transform([[predicted_pv_scaled]])[0][0]

        # Détection d'instabilité numérique
        if not np.isfinite(predicted_pv) or abs(predicted_pv) > 1e7:
            return 1e12  # Score pénalisant très élevé

        # c. Le PID réagit et calcule la nouvelle MV (Vanne)
        new_mv = pid.update(target_sp, predicted_pv)

        # d. Calcul de l'erreur selon la métrique choisie
        error = target_sp - predicted_pv
        if metric_type == 'ISE':
            score += (error ** 2) * tsamp_s
        else:  # IAE par défaut
            score += abs(error) * tsamp_s

        # e. Décalage de TOUS les lags (PV, MV, SP, perturbations...)
        next_features = current_features.copy()
        for base_name, max_lag in lag_families.items():
            for i in range(max_lag, 1, -1):
                col_i = f'{base_name}_lag_{i}'
                col_prev = f'{base_name}_lag_{i-1}'
                if col_i in next_features.columns and col_prev in next_features.columns:
                    next_features[col_i] = next_features[col_prev]

        # f. Injection des nouvelles valeurs à l'instant t-1 (lag_1)
        next_features['PV_real_lag_1'] = predicted_pv
        next_features['MV_real_lag_1'] = new_mv
        next_features['SP_real_lag_1'] = target_sp
        # Note : les perturbations restent à leur dernière valeur connue (scénario constant)

        current_features = next_features

    return score

def run_pid_tuner():
    """Fonction principale d'orchestration du réglage PID."""
    script_name_stem = "pid_tuner"
    config = None
    
    try:
        # Chargement de pid_tuner.ini (il faudra le créer !)
        config = load_config_and_setup_logging(script_name_stem, config_file_name=f"{script_name_stem}.ini")
    except FileNotFoundError:
        print(f"ATTENTION : Le fichier '{script_name_stem}.ini' est introuvable. "
              f"Veuillez le créer en vous inspirant de pid_comparator.ini.", file=sys.stderr)
        sys.exit(1)
    except Exception as e_cfg:
        print(f"ERREUR CRITIQUE config/logging: {e_cfg}", file=sys.stderr)
        sys.exit(1)

    logger.info(f"--- Démarrage {script_name_stem}_main.py (Tuner PID) ---", extra={'important_phase': True})
    
    logger.info("Étape 1 : Chargement du Modèle de Procédé (Machine Learning)...")
    try:
        model_path = config['MODEL_PATHS']['model_save_path']
        scalers_path = config['MODEL_PATHS']['scalers_save_path']
        
        logger.info(f"Tentative de chargement du modèle depuis : {model_path}")
        process_model = joblib.load(model_path)
        _force_model_single_thread_for_tuning(process_model)
        
        logger.info(f"Tentative de chargement des scalers depuis : {scalers_path}")
        scalers_dict = joblib.load(scalers_path)
        scaler_X = scalers_dict['scaler_X']
        scaler_y = scalers_dict['scaler_y']
        
        logger.info("Modèle ML et scalers chargés avec succès !")
    except Exception as e:
        logger.error(f"Impossible de charger le modèle. Avez-vous bien lancé pid_model_builder_main.py avant ? Erreur : {e}")
        sys.exit(1)
        
    logger.info("Étape 2 : Connexion à la base de données et extraction de l'état initial...")
    db_engine = None
    initial_state_df = None
    try:
        # Vérification des blocs requis
        required = ['Database', 'Tags', 'DataPeriod', 'ModelFeatures']
        missing = [s for s in required if s not in config]
        if missing:
            logger.error(f"Il manque les sections {missing} dans pid_tuner.ini.")
            logger.error("Copiez-les depuis pid_model_builder.ini pour que le Tuner puisse démarrer.")
            sys.exit(1)

        db_engine = get_db_engine(config['Database'])
        table_name = config['Database'].get('table_name', 'History')
        
        # Extraction et Lags (comme dans le Builder)
        sql_query, col_aliases = build_sql_query(config['Tags'], config['DataPeriod'], table_name)
        raw_df = extract_data(db_engine, sql_query)
        
        if raw_df.empty:
            raise ValueError("Aucune donnée n'a été extraite de SQL. Vérifiez les dates dans [DataPeriod].")
            
        clean_df = preprocess_timeseries_data(raw_df, col_aliases, essential_value_cols=['PV_real', 'MV_real', 'SP_real'])
        
        if clean_df.empty:
            raise ValueError("Les données sont vides après le nettoyage (probablement trop de valeurs manquantes).")
            
        X, _ = create_lagged_features(clean_df, config['ModelFeatures'], target_col='PV_real')
        
        if X.empty:
            raise ValueError("Le tableau de features (X) est vide après la création des Lags.")
            
        # On isole la TOUTE DERNIÈRE ligne extraite, c'est notre point de départ "Présent" !
        initial_state_df = X.iloc[-1:].copy()
        logger.info("État initial extrait avec succès ! La simulation est prête à démarrer.")

        # Diagnostic rapide: vérifier si le modèle ML réagit aux entrées MV/SP.
        run_model_reactivity_check(process_model, scaler_X, scaler_y, initial_state_df)
        
    except Exception as e:
        print(f"\n❌ ERREUR RÉELLE DÉTECTÉE LORS DE L'ÉTAPE 2 : {e}\n", file=sys.stderr)
        logger.error(f"Erreur lors de l'extraction de l'état initial : {e}", exc_info=True)
        sys.exit(1)
    finally:
        if db_engine:
            db_engine.dispose()
        
    logger.info("Étape 3 : Lancement de l'algorithme d'optimisation (Test du solveur)...")
    try:
        from scipy.optimize import minimize, differential_evolution
    except ImportError:
        logger.error("ERREUR CRITIQUE: 'scipy' n'est pas installé. Tapez 'pip install scipy' dans le terminal.")
        sys.exit(1)

    optimizer_settings = _read_optimizer_settings(config)

    # Limites de recherche (Bounds) configurables via [OPTIMIZER]
    kp_min = optimizer_settings['kp_min']
    kp_max = optimizer_settings['kp_max']
    ti_min = optimizer_settings['ti_min']
    ti_max = optimizer_settings['ti_max']
    td_min = optimizer_settings['td_min']
    td_max = optimizer_settings['td_max']
    bounds = ((kp_min, kp_max), (ti_min, ti_max), (td_min, td_max))
    
    # Point de départ : utiliser les valeurs du test PID s'il existe, sinon valeurs raisonnables
    if config.has_section('PID_PARAMS_TEST'):
        initial_guess = [
            config['PID_PARAMS_TEST'].getfloat('kp_test', 1.0),
            config['PID_PARAMS_TEST'].getfloat('ti_test', 500.0),
            config['PID_PARAMS_TEST'].getfloat('td_test', 0.0),
        ]
    else:
        initial_guess = [1.0, 500.0, 0.0]
    
    original_guess = initial_guess[:]
    initial_guess = _clip_params_to_bounds(initial_guess, bounds)
    if initial_guess != original_guess:
        logger.warning(
            "Point de depart hors bornes detecte. "
            f"Original={original_guess} -> Borne={initial_guess}"
        )

    logger.info(
        f"Bornes solveur: Kp[{kp_min}, {kp_max}], Ti[{ti_min}, {ti_max}], Td[{td_min}, {td_max}]"
    )
    logger.info(f"Point de départ du solveur : Kp={initial_guess[0]}, Ti={initial_guess[1]}, Td={initial_guess[2]}")
    
    # Diagnostic rapide : vérifier que la fonction objective est sensible aux paramètres PID
    baseline_score = evaluate_closed_loop_performance(initial_guess, process_model, scaler_X, scaler_y, config, initial_state_df)
    logger.info(f"Score au point de départ : {baseline_score:.6f}")
    
    test_params = [initial_guess[0] * 2, initial_guess[1] * 0.5, 10.0]
    test_score = evaluate_closed_loop_performance(test_params, process_model, scaler_X, scaler_y, config, initial_state_df)
    logger.info(f"Score avec test params [{test_params[0]:.1f}, {test_params[1]:.1f}, {test_params[2]:.1f}] : {test_score:.6f}")
    
    score_diff = abs(baseline_score - test_score)
    logger.info(f"Différence de score détectée : {score_diff:.6f}")
    
    if score_diff < 1e-10:
        logger.warning("⚠️ ATTENTION: La fonction objectif semble insensible aux paramètres PID !")
        logger.warning("  Causes possibles : sp_step_value_offset=0, modèle ML insensible, ou durée trop courte.")

    is_sensitive, scored_trials = run_pid_sensitivity_protocol(
        process_model,
        scaler_X,
        scaler_y,
        config,
        initial_state_df,
        initial_guess,
        bounds,
    )

    if not is_sensitive:
        best_trial = min(scored_trials, key=lambda t: t[2])
        _, best_params, best_score = best_trial
        logger.warning("Objectif trop plat: optimisation numerique ignoree pour eviter un faux optimum.")
        logger.info("Etape 4 : Affichage des parametres optimaux (issus du protocole sensibilité)...")
        logger.info("==================================================")
        logger.info("Diagnostic: optimisation sautee (surface de cout quasi constante).")
        logger.info("==================================================")
        logger.info(f"Kp retenu : {best_params[0]:.3f}")
        logger.info(f"Ti retenu : {best_params[1]:.3f} secondes")
        logger.info(f"Td retenu : {best_params[2]:.3f} secondes")
        logger.info(f"Score retenu : {best_score:.6f}")
        logger.info("Action recommandee: enrichir les donnees d'entrainement et relancer le model builder.")
        logger.info("==================================================")
        return
    
    # Lancement du solveur local (L-BFGS-B) en multi-start pour éviter les minima locaux.
    logger.info("Lancement de l'optimisation L-BFGS-B (multi-start)...")
    lbfgs_maxiter = optimizer_settings['lbfgs_maxiter']
    lbfgs_maxfun = optimizer_settings['lbfgs_maxfun']
    lbfgs_restarts = optimizer_settings['lbfgs_restarts']

    # Construire les points de départ: guess initial + meilleurs essais protocole.
    sorted_trials = sorted(scored_trials, key=lambda t: t[2])
    start_points = [initial_guess]
    for _, p, _ in sorted_trials:
        p_clipped = _clip_params_to_bounds(p, bounds)
        if p_clipped not in start_points:
            start_points.append(p_clipped)
        if len(start_points) >= max(1, lbfgs_restarts):
            break

    candidate_results = []
    for idx, start_pt in enumerate(start_points, start=1):
        logger.info(
            f"L-BFGS-B start {idx}/{len(start_points)}: "
            f"Kp={start_pt[0]:.3f}, Ti={start_pt[1]:.3f}, Td={start_pt[2]:.3f}"
        )
        res_i = minimize(
            evaluate_closed_loop_performance,
            start_pt,
            args=(process_model, scaler_X, scaler_y, config, initial_state_df),
            bounds=bounds,
            method='L-BFGS-B',
            options={
                'ftol': 1e-8,
                'gtol': 1e-7,
                'maxiter': lbfgs_maxiter,
                'maxfun': lbfgs_maxfun,
            }
        )
        logger.info(
            f"L-BFGS-B result {idx}: score={res_i.fun:.6f}, nit={getattr(res_i, 'nit', 0)}, nfev={getattr(res_i, 'nfev', 0)}"
        )
        candidate_results.append(res_i)

    result = min(candidate_results, key=lambda r: r.fun)
    logger.info(
        f"Meilleur L-BFGS-B multi-start: score={result.fun:.6f}, "
        f"Kp={result.x[0]:.3f}, Ti={result.x[1]:.3f}, Td={result.x[2]:.3f}"
    )

    # Filet de sécurité: si L-BFGS-B reste peu concluant, on peut tenter un solveur global.
    de_always_refine = optimizer_settings['de_always_refine']
    if (getattr(result, 'nit', 0) <= 1) or de_always_refine:
        de_enabled = optimizer_settings['de_enabled']
        de_force_if_flat = optimizer_settings['de_force_if_flat']
        de_maxiter = optimizer_settings['de_maxiter']
        de_popsize = optimizer_settings['de_popsize']
        de_polish = optimizer_settings['de_polish']
        de_max_runtime_s = optimizer_settings['de_max_runtime_seconds']
        de_workers = optimizer_settings['de_workers']
        de_updating = optimizer_settings['de_updating']

        if not de_enabled:
            logger.warning("DE désactivé (de_enabled=false). Fin sur résultat L-BFGS-B.")
        elif (not is_sensitive) and (not de_force_if_flat):
            logger.warning("Objectif peu sensible: DE ignoré (de_force_if_flat=false) pour éviter un run trop long.")
        else:
            logger.warning("L-BFGS-B peu concluant. Tentative d'optimisation globale (differential_evolution)...")
            logger.info(
                f"DE config: maxiter={de_maxiter}, popsize={de_popsize}, polish={de_polish}, "
                f"max_runtime={de_max_runtime_s}s, workers={de_workers}, updating={de_updating}"
            )

            de_start_t = time.perf_counter()

            def _de_callback(_xk, convergence):
                elapsed = time.perf_counter() - de_start_t
                if elapsed > de_max_runtime_s:
                    logger.warning(
                        f"Arrêt DE par timeout: {elapsed:.1f}s > {de_max_runtime_s}s (convergence={convergence:.6g})."
                    )
                    return True
                return False

            de_result = differential_evolution(
                evaluate_closed_loop_performance,
                bounds=bounds,
                args=(process_model, scaler_X, scaler_y, config, initial_state_df),
                maxiter=de_maxiter,
                popsize=de_popsize,
                polish=de_polish,
                seed=42,
                workers=de_workers,
                updating=de_updating,
                callback=_de_callback,
            )
            logger.info(
                f"Résultat global: success={de_result.success}, score={de_result.fun:.6f}, "
                f"Kp={de_result.x[0]:.3f}, Ti={de_result.x[1]:.3f}, Td={de_result.x[2]:.3f}"
            )
            if de_result.fun < result.fun:
                logger.info("Le solveur global a trouvé une meilleure solution. Elle remplace le résultat local.")
                result = de_result
            elif not is_sensitive:
                logger.warning("Même le solveur global n'apporte pas mieux: la fonction objectif est probablement trop plate.")
    
    # Sécurité métier: conserver aussi la meilleure solution vue par le protocole.
    best_trial = min(scored_trials, key=lambda t: t[2])
    _, best_protocol_params, best_protocol_score = best_trial
    if best_protocol_score < result.fun:
        logger.warning(
            "La meilleure solution du protocole sensibilité est meilleure que le résultat solveur. "
            "On conserve la solution protocole."
        )

        class _SimpleResult:
            pass

        fallback_result = _SimpleResult()
        fallback_result.x = np.array(best_protocol_params, dtype=float)
        fallback_result.fun = float(best_protocol_score)
        fallback_result.success = result.success
        fallback_result.message = "Solution protocole retenue (meilleure que solveur)."
        fallback_result.nit = getattr(result, 'nit', 0)
        fallback_result.nfev = getattr(result, 'nfev', 0)
        result = fallback_result

    logger.info("Étape 4 : Affichage des paramètres optimaux...")
    best_kp, best_ti, best_td = result.x
    logger.info("==================================================")
    if result.success:
        logger.info("🎉 OPTIMISATION TERMINÉE AVEC SUCCÈS ! 🎉")
    else:
        logger.warning(f"⚠️ OPTIMISATION TERMINÉE AVEC AVERTISSEMENT: {result.message}")
    logger.info("==================================================")
    logger.info(f"👉 Kp optimal trouvé : {best_kp:.3f}")
    logger.info(f"👉 Ti optimal trouvé : {best_ti:.3f} secondes")
    logger.info(f"👉 Td optimal trouvé : {best_td:.3f} secondes")
    logger.info(f"Score de l'erreur minimale : {result.fun:.6f}")
    logger.info(f"Nombre d'itérations : {result.nit}")
    logger.info(f"Nombre d'évaluations de la fonction : {result.nfev}")
    logger.info("=="*25)

if __name__ == "__main__":
    try:
        run_pid_tuner()
    finally:
        try:
            winsound.Beep(1000, 500)
        except ImportError:
            logger.info("Module winsound non trouvé, pas de bip de fin.")
        except RuntimeError:
            logger.info("Impossible de jouer le son de fin (pas de périphérique audio ou winsound non dispo).")
        except Exception as e_sound:
            logger.warning(f"Erreur lors de la tentative de jouer le son de fin: {e_sound}")
