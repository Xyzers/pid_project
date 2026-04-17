# 🚀 Guide Complet : Optimisation des Hyperparamètres RandomForest

## 📋 Vue d'ensemble

Votre projet dispose maintenant de **3 outils puissants** pour optimiser et évaluer les hyperparamètres RandomForest :

1. **`pid_hyperparameter_tuner.py`** - Optimisation GridSearchCV complète
2. **`pid_benchmark_configs.py`** - Comparaison de configurations
3. **`evaluate_model_comprehensive()`** - Évaluation détaillée intégrée

---

## ⚙️ Configuration

### Fichier: `pid_model_builder.ini`

Une nouvelle section a été ajoutée :

```ini
[HyperparameterTuning]
# Activer l'optimisation automatique (true/false)
enable_tuning = false

# Plages de recherche (séparées par des virgules)
n_estimators_range = 50,100,200,300
max_depth_range = 10,15,20,25,None
min_samples_leaf_range = 1,2,3,5
min_samples_split_range = 2,5,10

# Nombre de folds pour la validation croisée
cv_folds = 5

# Fichiers de sortie
tuning_results_path = ./trace/hyperparameter_tuning_results.csv
tuning_report_path = ./trace/hyperparameter_tuning_report.txt
```

---

## 🎯 Utilisation des Outils

### 1️⃣ Tuning Complet avec GridSearchCV

**Quand l'utiliser:** Vous voulez tester automatiquement TOUTES les combinaisons de paramètres.

**Commande:**
```bash
python pid_hyperparameter_tuner.py
```

**Processus:**
1. Extrait les données de la BD
2. Prépare les features
3. Lance GridSearchCV avec la grille configurée dans l'ini
4. Teste TOUTES les combinaisons (ex: 4×5×4×2 = 160 modèles)
5. Sauvegarde les résultats en CSV et génère un rapport

**Sortie:**
- `./trace/hyperparameter_tuning_results.csv` - Détails de chaque configuration
- `./trace/hyperparameter_tuning_report.txt` - Rapport synthétique

**Exemple de résultat:**
```
n_est  max_depth  min_leaf  min_split  mean_r2_test  std_r2_test  rank
200    20         2         5          0.8934        0.0145       1
150    20         2         5          0.8912        0.0152       2
300    25         2         5          0.8901        0.0168       3
100    15         1         2          0.8756        0.0201       4
```

**Temps estimé:** 15-60 minutes (dépend de la taille des données et du nombre de folds)

---

### 2️⃣ Benchmarking Rapide

**Quand l'utiliser:** Vous voulez comparer rapidement quelques configurations sans exhaustivité.

**Commande:**
```bash
python pid_benchmark_configs.py
```

**Configurations testées par défaut:**
```python
[
    {'n_estimators': 50, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 5},
    {'n_estimators': 150, 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5},
    {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2},
    {'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 3, 'min_samples_split': 2},
]
```

**Sortie:**
```
Configuration 1/5: {'n_estimators': 50, 'max_depth': 10, ...}
  ✓ Train: 5.23s, R²: 0.8234, RMSE: 12.45
Configuration 2/5: {'n_estimators': 100, ...}
  ✓ Train: 8.15s, R²: 0.8567, RMSE: 10.32
...

RÉSUMÉ COMPARATIF
n_estimators  max_depth  train_time_s  r2      rmse
50            10         5.23          0.8234  12.45
100           15         8.15          0.8567  10.32
150           20         12.04         0.8645  9.98
200           None       18.67         0.8234  12.23
300           12         22.34         0.8901  8.76
```

**Temps estimé:** 2-10 minutes

---

### 3️⃣ Évaluation Détaillée dans le Pipeline Principal

L'évaluation complète est **automatiquement intégrée** lors du `pid_model_builder_main.py` normal.

**Métriques calculées automatiquement:**

```python
metrics, predictions, df = evaluate_model_comprehensive(
    model, X_test_scaled, y_test_original, scaler_y, y_test_index
)
```

**Résultat = dictionnaire avec:**
- ✅ **R²** - Coefficient de détermination (0-1, plus haut = mieux)
- ✅ **RMSE** - Racine de l'erreur quadratique moyenne
- ✅ **MAE** - Erreur absolue moyenne
- ✅ **MAPE** - Erreur absolue moyenne en pourcentage
- ✅ **Median_Error** - Erreur médiane (robustesse)
- ✅ **Q75_Error, Q95_Error** - Quantiles (pire cas)
- ✅ **Inference_Time_ms** - Temps de prédiction

---

## 📊 Interprétation des Résultats

### Qualité du Modèle

| R² | Qualité | Action |
|---|---------|--------|
| > 0.90 | Excellent ✓✓✓ | Déployer en prod |
| 0.80-0.90 | Bon ✓✓ | OK, monitorer |
| 0.70-0.80 | Acceptable ✓ | À améliorer |
| < 0.70 | Faible | Retravailler les features |

### Précision (MAPE)

| MAPE | Précision | Interprétation |
|------|-----------|----------------|
| < 5% | Très haute | Prédictions très fiables |
| 5-10% | Haute | Bon pour la plupart des usages |
| 10-20% | Modérée | À surveiller de près |
| > 20% | Faible | Améliorer la qualité des données |

### Temps d'Inférence

| Temps | Verdict |
|------|---------|
| < 10ms | Très rapide ✓ |
| 10-50ms | Rapide |
| 50-100ms | Acceptable |
| > 100ms | Lent (réduire n_estimators ou max_depth) |

---

## 🔍 Guide d'Interprétation des Paramètres

### `n_estimators` (nombre d'arbres)

```python
Petit (50-100)      → Rapide, peu d'overfitting, peut manquer de précision
Moyen (150-300)     → Bon compromis (RECOMMANDÉ)
Grand (500+)        → Lent, risque d'overfitting sans bénéfice
```

**Recommandation:** Commencer à 150, augmenter si R² stagne.

---

### `max_depth` (profondeur maximale)

```python
Petit (10-15)       → Risque underfitting (manque de détails)
Moyen (15-25)       → Bon compromis (RECOMMANDÉ)
Grands (25+)        → Overfitting, peut être très lent
None                → Croissance libre (dangereux si mal réglé)
```

**Recommandation:** Fixer à 20 initialement.

---

### `min_samples_leaf` (points minimum par feuille)

```python
1                   → Arbres profonds, peut overfitter
2-3                 → Bon compromis (RECOMMANDÉ)
5+                  → Arbres peu profonds, perte de précision
```

**Recommandation:** 2 ou 3 généralement optimal.

---

### `min_samples_split` (points minimum pour splitter)

```python
2 (défaut)          → Flexible, peut overfitter
5                   → Bon compromis (RECOMMANDÉ)
10+                 → Très restrictif
```

**Recommandation:** 5 généralement bon.

---

## 🎬 Workflow Recommandé

### Phase 1 : Diagnostic Rapide (5-10 min)
```bash
python pid_benchmark_configs.py
```
→ Identifiez la meilleure config approximative

### Phase 2 : Optimisation Fine (30-60 min)
Mettez à jour `pid_model_builder.ini` :
```ini
[HyperparameterTuning]
enable_tuning = true
# Affinez les ranges autour du meilleur résultat de Phase 1
n_estimators_range = 100,150,200
max_depth_range = 18,20,22
min_samples_leaf_range = 2,3
min_samples_split_range = 4,5,6
```

```bash
python pid_hyperparameter_tuner.py
```

### Phase 3 : Mise en Production
Copiez les meilleurs paramètres dans `pid_model_builder.ini` :
```ini
[ModelParams]
n_estimators = 150
max_depth = 20
min_samples_leaf = 2
min_samples_split = 5
```

```bash
python pid_model_builder_main.py
```

---

## 📈 Exemple : Avant / Après

### Avant optimisation :
```
Configuration: n_est=100, depth=12, leaf=1, split=2
R² = 0.7845
RMSE = 14.32
Inference = 8.5ms
Status = Acceptable mais à améliorer
```

### Après optimisation :
```
Configuration: n_est=150, depth=20, leaf=2, split=5
R² = 0.8934  (+13%)
RMSE = 9.18  (-36%)
Inference = 12.3ms (acceptable pour +13% de précision)
Status = Excellent ✓✓✓
```

---

## 💡 Tips & Tricks

### 1. Réduire le Temps de Tuning
```ini
[HyperparameterTuning]
cv_folds = 3          # Au lieu de 5 (plus rapide, moins fiable)
# Réduire les ranges
n_estimators_range = 100,200
max_depth_range = 15,20,25
```

### 2. Améliorer la Fiabilité
```ini
cv_folds = 10         # Plus de folds = plus robuste
# Élargir les ranges
n_estimators_range = 50,100,150,200,300
max_depth_range = 10,15,20,25,None
```

### 3. Focus Performance (moins de latence)
```python
# Modifier pid_benchmark_configs.py pour tester configs rapides:
configurations = [
    {'n_estimators': 50, 'max_depth': 10, ...},   # 5ms
    {'n_estimators': 100, 'max_depth': 12, ...},  # 8ms
    {'n_estimators': 150, 'max_depth': 15, ...},  # 12ms
]
```

### 4. Focus Précision (moins d'erreur)
```ini
n_estimators_range = 200,300,400,500
max_depth_range = 20,25,30
min_samples_leaf_range = 1,2
min_samples_split_range = 2,3
```

---

## 🐛 Dépannage

### Q: GridSearchCV est trop lent
**R:** Réduire `cv_folds` de 5 à 3, ou réduire les ranges de paramètres

### Q: RMSE bon mais MAPE mauvais (> 20%)
**R:** Vos données ont probablement de grandes variations de magnitude. Considérez scaling par rapport à la plage.

### Q: Overfitting (R² train >> R² test)
**R:** Augmentez `min_samples_leaf`, réduisez `n_estimators`, limitez `max_depth`

### Q: Underfitting (R² faible partout)
**R:** Augmentez `n_estimators`, augmentez `max_depth`, réduisez `min_samples_leaf`

---

## 📝 Fichiers Modifiés/Créés

- ✅ `src/modeling.py` - Nouvelles fonctions d'optimisation
- ✅ `pid_model_builder.ini` - Nouvelle section [HyperparameterTuning]
- ✅ `pid_hyperparameter_tuner.py` - Script de tuning GridSearchCV
- ✅ `pid_benchmark_configs.py` - Script de benchmarking comparatif
- ✅ `HYPERPARAMETER_TUNING_GUIDE.md` - Ce guide

---

## 🎓 Ressources

- [Scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [RandomForestRegressor Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Hyperparameter Tuning Guide](https://towardsdatascience.com/hyperparameter-tuning-c5619e7e3f69)

---

**Questions ?** Consultez les logs dans `./trace/pid_model_builder_log.txt`
