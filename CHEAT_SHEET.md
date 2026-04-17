# ⚡ CHEAT SHEET - Optimisation Hyperparamètres (Rapide)

## 🎯 OBJECTIF EN UNE LIGNE
Trouver les meilleurs paramètres RandomForest pour votre modèle (meilleur R², RMSE, temps)

---

## 🚀 QUICK START (5 MIN)

### Option 1: Benchmark Rapide
```bash
python pid_benchmark_configs.py
```
✅ **Résultat:** Compare 5 configurations, identifie la meilleure  
⏱️ **Temps:** 2-10 min  
📊 **Output:** `./trace/benchmark_results.csv`

### Option 2: Tuning Exhaustif
```bash
python pid_hyperparameter_tuner.py
```
✅ **Résultat:** Teste TOUTES combinaisons, optimal complet  
⏱️ **Temps:** 15-60 min  
📊 **Output:** `./trace/hyperparameter_tuning_results.csv` + rapport

---

## 📋 3 ÉTAPES POUR OPTIMISER

### 1️⃣ Diagnostic (5-10 min)
```bash
python pid_benchmark_configs.py
cat ./trace/benchmark_results.csv
```
→ Voir laquelle des 5 configs est meilleure

### 2️⃣ Affiner (30 min)
Éditer `pid_model_builder.ini` → Section `[HyperparameterTuning]`
```bash
python pid_hyperparameter_tuner.py
```
→ Affiner autour de la meilleure du step 1

### 3️⃣ Mettre à Jour Config (1 min)
```ini
[ModelParams]
n_estimators = ???       # Copier du rapport
max_depth = ???
min_samples_leaf = ???
min_samples_split = ???
```
```bash
python pid_model_builder_main.py
```
→ Entraîner avec meilleurs paramètres

---

## 🎛️ PARAMÈTRES À TESTER

| Paramètre | Petit | Moyen | Grand |
|-----------|-------|-------|-------|
| `n_estimators` | 50-100 | 150-200 | 300+ |
| `max_depth` | 10-15 | 20-25 | 30+ / None |
| `min_samples_leaf` | 1 | 2-3 | 5+ |
| `min_samples_split` | 2 | 5 | 10+ |

**Recommandation:** Commencer par **Moyen**

---

## 📊 MÉTRIQUES À REGARDER

| Métrique | Bon | Excellent |
|----------|------|-----------|
| **R²** | 0.8+ | 0.9+ |
| **RMSE** | Bas | Plus bas |
| **MAPE** | < 10% | < 5% |
| **Inférence** | < 50ms | < 20ms |

---

## 📈 EXEMPLE DE RÉSULTAT

```
AVANT:
  n_est=100, depth=12 → R²=0.8234, RMSE=14.32, Inférence=8.5ms

APRÈS:
  n_est=150, depth=20 → R²=0.8934, RMSE=9.18, Inférence=12.3ms
  Améliorations: +13% R², -36% RMSE, +45% latence (acceptable)
```

---

## 🔍 DIAGNOSTIC RAPIDE

### ❌ RMSE Élevé, R² Faible
→ **Causes:** Underfitting, features insuffisantes, données mauvaises  
→ **Actions:** Augmenter n_estimators, augmenter max_depth, diminuer min_samples_leaf

### ❌ R² Excellent Test mais Faible Validation
→ **Cause:** Overfitting  
→ **Actions:** Augmenter min_samples_leaf, diminuer max_depth, augmenter min_samples_split

### ⚠️ Inférence Lente (> 100ms)
→ **Cause:** Trop d'arbres ou profondeur trop grande  
→ **Actions:** Réduire n_estimators, limiter max_depth

---

## 🛠️ CONFIG À COPIER

### Configuration "Légère" (Rapide)
```ini
n_estimators = 50
max_depth = 10
min_samples_leaf = 1
min_samples_split = 2
```
⏱️ Inférence: ~4ms | 📊 R²: ~0.82

### Configuration "Équilibrée" (RECOMMANDÉE)
```ini
n_estimators = 150
max_depth = 20
min_samples_leaf = 2
min_samples_split = 5
```
⏱️ Inférence: ~12ms | 📊 R²: ~0.89

### Configuration "Lourde" (Précision)
```ini
n_estimators = 300
max_depth = 25
min_samples_leaf = 1
min_samples_split = 2
```
⏱️ Inférence: ~25ms | 📊 R²: ~0.91

---

## 📁 FICHIERS CLÉS

```
pid_hyperparameter_tuner.py      ← Tuning complet GridSearchCV
pid_benchmark_configs.py          ← Benchmark rapide comparatif
src/modeling.py                   ← Fonctions d'optimisation (NE PAS ÉDITER)
pid_model_builder.ini             ← CONFIG (À ÉDITER pour personnaliser)
./trace/benchmark_results.csv     ← Résultats benchmarking
./trace/hyperparameter_tuning_*   ← Résultats tuning complet
HYPERPARAMETER_TUNING_GUIDE.md    ← Guide détaillé
```

---

## ⏱️ TEMPS ESTIMÉS

| Action | Temps |
|--------|-------|
| Benchmark 5 configs | 2-10 min |
| GridSearchCV 160 combos | 15-60 min |
| Entraîner final | 5-15 min |
| **Total recommandé** | **25-90 min** |

---

## 💡 TIPS

1. **Commencer par benchmark** → rapide, identifie direction
2. **Puis affiner avec tuning** → complet, optimal
3. **Ou ignorer tuning si** → R² déjà > 0.90
4. **Réduire cv_folds de 5→3** → GridSearchCV 2x plus rapide
5. **Augmenter cv_folds 5→10** → Résultats plus robustes

---

## 🆘 PROBLÈMES COURANTS

| Problème | Solution |
|----------|----------|
| **GridSearchCV trop lent** | Réduire cv_folds, réduire ranges |
| **R² excellent mais MAPE mauvais** | Données à grandes variations → utiliser MinMaxScaler |
| **Underfitting** | ↑ n_estimators, ↑ max_depth |
| **Overfitting** | ↑ min_samples_leaf, ↓ max_depth |
| **Inférence lente** | ↓ n_estimators, ↓ max_depth |

---

## 🎯 PROCHAINES ÉTAPES

```bash
# 1. Essayer le benchmark
python pid_benchmark_configs.py

# 2. Voir les résultats
cat ./trace/benchmark_results.csv

# 3. Lancer le tuning (optionnel, si temps)
python pid_hyperparameter_tuner.py

# 4. Copier meilleurs paramètres dans pid_model_builder.ini

# 5. Entraîner
python pid_model_builder_main.py
```

---

**Questions?** Lire `HYPERPARAMETER_TUNING_GUIDE.md` pour détails complets.
