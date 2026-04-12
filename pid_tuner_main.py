# pid_tuner_main.py (version remaniée pour débogage intensif)

# Imports Python Standard Library en premier
import sys
import logging # Sera initialisé plus tard
import configparser
from pathlib import Path

# Vérification de base du dossier 'src'
src_path_check = Path("./src")
if not src_path_check.is_dir():
    print(f"ERREUR CRITIQUE: Le dossier 'src' est introuvable. "
          f"Veuillez lancer le script depuis la racine du projet.", file=sys.stderr)
    sys.exit("Arrêt prématuré: Dossier 'src' manquant.")

# Imports des bibliothèques tierces
try:
    import pandas as pd
except ImportError as e:
    print(f"ERREUR CRITIQUE D'IMPORT: {e}. Avez-vous installé requirements.txt ?", file=sys.stderr)
    sys.exit(1)
