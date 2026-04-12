# tests/test_basic.py - Tests simples exécutables avec: python tests/test_basic.py
# Lancer depuis la racine du projet : python tests/test_basic.py

import sys; sys.path.insert(0, '.')


def test_pid_controller_basic():
    """Test: le PID produit une MV différente de l'initiale après un échelon."""
    from src.pid_logic.pid_controller import PIDController
    pid = PIDController(Kp=1.0, Ti=100.0, Td=0.0, Tsamp=0.1,
                        mv_min=0.0, mv_max=100.0, direct_action=True, initial_mv=50.0)
    pid.set_initial_state(pv_initial=50.0, sp_initial=50.0, mv_initial=50.0, active_initial=True)
    mv = pid.update(sp=55.0, pv=50.0)  # Échelon SP de +5
    assert mv != 50.0, f"Le PID n'a pas réagi à l'échelon ! MV={mv}"
    assert 50.0 < mv <= 100.0, f"MV hors plage attendue: {mv}"
    print("  test_pid_controller_basic PASSED")


def test_pid_anti_windup():
    """Test: l'anti-windup empêche la MV de dépasser les limites."""
    from src.pid_logic.pid_controller import PIDController
    pid = PIDController(Kp=10.0, Ti=1.0, Td=0.0, Tsamp=0.1,
                        mv_min=0.0, mv_max=100.0, direct_action=True, initial_mv=50.0)
    pid.set_initial_state(50.0, 50.0, 50.0, True)
    for _ in range(1000):
        mv = pid.update(sp=200.0, pv=50.0)  # Erreur énorme
    assert mv <= 100.0, f"Anti-windup casse ! MV={mv} > 100"
    assert mv >= 0.0, f"Anti-windup casse ! MV={mv} < 0"
    print("  test_pid_anti_windup PASSED")


def test_pid_direct_vs_inverse():
    """Test: action directe et inverse produisent des MV opposées."""
    from src.pid_logic.pid_controller import PIDController
    pid_direct = PIDController(Kp=1.0, Ti=100.0, Td=0.0, Tsamp=0.1,
                               mv_min=0.0, mv_max=100.0, direct_action=True, initial_mv=50.0)
    pid_direct.set_initial_state(50.0, 50.0, 50.0, True)
    mv_direct = pid_direct.update(55.0, 50.0)

    pid_inverse = PIDController(Kp=1.0, Ti=100.0, Td=0.0, Tsamp=0.1,
                                mv_min=0.0, mv_max=100.0, direct_action=False, initial_mv=50.0)
    pid_inverse.set_initial_state(50.0, 50.0, 50.0, True)
    mv_inverse = pid_inverse.update(55.0, 50.0)

    assert (mv_direct - 50.0) * (mv_inverse - 50.0) < 0, \
        f"Direct et inverse devraient aller en sens opposes ! direct={mv_direct}, inverse={mv_inverse}"
    print("  test_pid_direct_vs_inverse PASSED")


def test_pid_bumpless_transfer():
    """Test: le passage Inactif->Actif ne provoque pas de saut de MV."""
    from src.pid_logic.pid_controller import PIDController
    pid = PIDController(Kp=2.0, Ti=50.0, Td=0.0, Tsamp=0.1,
                        mv_min=0.0, mv_max=100.0, direct_action=True, initial_mv=30.0)
    pid.set_initial_state(45.0, 50.0, 30.0, active_initial=False)

    # Le PID est inactif, MV devrait rester gelée
    mv_inactive = pid.update(50.0, 45.0)
    assert abs(mv_inactive - 30.0) < 0.01, f"MV devrait etre gelee a 30.0 en inactif, got {mv_inactive}"

    # Passage Inactif -> Actif
    pid.set_active_state(True, sp_at_transition=50.0, pv_at_transition=45.0, mv_real_at_transition=30.0)
    mv_after_activation = pid.update(50.0, 45.0)

    # Le bumpless transfer doit garder la MV proche de 30.0 (pas de saut brutal)
    assert abs(mv_after_activation - 30.0) < 5.0, \
        f"Bumpless transfer casse ! MV saute de 30.0 a {mv_after_activation:.2f}"
    print("  test_pid_bumpless_transfer PASSED")


def test_feature_engineering_lag_count():
    """Test: create_lagged_features cree le bon nombre de colonnes."""
    import pandas as pd
    import configparser
    from src.feature_engineering import create_lagged_features

    df = pd.DataFrame({
        'PV_real': range(50),
        'MV_real': range(50),
        'SP_real': [42.0] * 50,
    }, index=pd.date_range('2026-01-01', periods=50, freq='100ms'))

    config = configparser.ConfigParser()
    config.add_section('ModelFeatures')
    config.set('ModelFeatures', 'pv_lags', '3')
    config.set('ModelFeatures', 'mv_lags', '3')
    config.set('ModelFeatures', 'sp_lags', '2')
    config.set('ModelFeatures', 'kp_hist_lags', '0')
    config.set('ModelFeatures', 'ti_hist_lags', '0')
    config.set('ModelFeatures', 'td_hist_lags', '0')

    X, y = create_lagged_features(df, config['ModelFeatures'])
    expected_cols = 3 + 3 + 2  # PV*3 + MV*3 + SP*2
    assert X.shape[1] == expected_cols, \
        f"Attendu {expected_cols} colonnes, obtenu {X.shape[1]}: {X.columns.tolist()}"
    assert len(y) == len(X), "X et y doivent avoir la meme longueur"
    assert X.isnull().sum().sum() == 0, "Pas de NaN attendu apres dropna"
    print("  test_feature_engineering_lag_count PASSED")


def test_feature_engineering_lag_names():
    """Test: les noms de colonnes de lags suivent le pattern attendu."""
    import pandas as pd
    import configparser
    from src.feature_engineering import create_lagged_features

    df = pd.DataFrame({
        'PV_real': range(20),
        'MV_real': range(20),
        'SP_real': [10.0] * 20,
    }, index=pd.date_range('2026-01-01', periods=20, freq='100ms'))

    config = configparser.ConfigParser()
    config.add_section('ModelFeatures')
    config.set('ModelFeatures', 'pv_lags', '2')
    config.set('ModelFeatures', 'mv_lags', '2')
    config.set('ModelFeatures', 'sp_lags', '1')
    config.set('ModelFeatures', 'kp_hist_lags', '0')
    config.set('ModelFeatures', 'ti_hist_lags', '0')
    config.set('ModelFeatures', 'td_hist_lags', '0')

    X, _ = create_lagged_features(df, config['ModelFeatures'])
    expected_names = ['PV_real_lag_1', 'PV_real_lag_2',
                      'MV_real_lag_1', 'MV_real_lag_2',
                      'SP_real_lag_1']
    assert X.columns.tolist() == expected_names, \
        f"Noms attendus: {expected_names}, obtenus: {X.columns.tolist()}"
    print("  test_feature_engineering_lag_names PASSED")


if __name__ == '__main__':
    print("=" * 50)
    print("  Tests unitaires PID Project")
    print("=" * 50)
    test_pid_controller_basic()
    test_pid_anti_windup()
    test_pid_direct_vs_inverse()
    test_pid_bumpless_transfer()
    test_feature_engineering_lag_count()
    test_feature_engineering_lag_names()
    print("\n  Tous les tests passes !")
