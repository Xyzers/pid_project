Faire un patch “accélération Ryzen” dans pid_tuner_main.py:
- activer le multi-cœur dans differential_evolution (workers=-1),
- passer updating='deferred' (requis pour paralléliser proprement),

Patch le tuner pour mode robuste :
- Forcer le modèle en mono-thread pendant le tuning (n_jobs=1 au moment du chargement).
- Garder le parallélisme au niveau solveur global uniquement.
- Optionnel : masquer ce warning précis pour ne pas polluer les logs.
