# src/plotting.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def plot_predictions(predictions_df: pd.DataFrame, plot_save_path: Path = None):
    # ... (votre code pour plot_predictions)
    if predictions_df.empty:
        logger.warning("DataFrame prédictions vide. Graphique non généré.")
        return

    plt.figure(figsize=(15, 7))
    plt.plot(predictions_df.index, predictions_df['Actual_PV'], label='PV Réelle (Test)', color='blue', alpha=0.8, marker='.', linestyle='-')
    plt.plot(predictions_df.index, predictions_df['Predicted_PV'], label='PV Prédite (Test)', color='red', linestyle='--', alpha=0.8, marker='.')
    plt.title('Comparaison PV Réelle vs. PV Prédite (Test)')
    plt.xlabel('Temps')
    plt.ylabel('Valeur de PV')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if plot_save_path:
        try:
            save_path = Path(plot_save_path).resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Graphique sauvegardé dans '{save_path}'.")
        except Exception as e:
            logger.error(f"Erreur sauvegarde graphique : {e}")
    
    try:
        plt.show()
    except Exception as e:
        logger.warning(f"Impossible d'afficher graphique (plt.show()): {e}.")
    # ...