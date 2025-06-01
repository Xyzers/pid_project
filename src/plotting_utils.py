# Dans src/plotting_utils.py

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging
import numpy as np # Pour np.any et np.isfinite

logger = logging.getLogger(__name__)

# ... votre plot_predictions existant ...

def plot_pid_comparison(hist_data_df: pd.DataFrame, config_data_period: dict,
                        plot_save_path_str: str = None, suptitle_prefix="Comparaison PID"):
    if hist_data_df.empty:
        logger.warning("DataFrame vide pour plot_pid_comparison. Graphique non généré.")
        return

    # Identifier les colonnes de perturbation présentes
    disturbance_cols = sorted([col for col in hist_data_df.columns if col.startswith('Dist') and col.endswith('_real')])
    num_disturbance_plots = 0
    valid_disturbance_cols_to_plot = []
    if disturbance_cols:
        for dist_col in disturbance_cols:
            # Vérifier si la colonne a des données valides à tracer
            if dist_col in hist_data_df and pd.api.types.is_numeric_dtype(hist_data_df[dist_col]) and \
               np.any(np.isfinite(hist_data_df[dist_col].dropna())): # Au moins une valeur finie non-NaN
                valid_disturbance_cols_to_plot.append(dist_col)
        num_disturbance_plots = len(valid_disturbance_cols_to_plot)


    # Déterminer si le graphique d'activation PID est nécessaire
    plot_pid_enable_graph = (
        ('PID_Enable_real' in hist_data_df.columns and hist_data_df['PID_Enable_real'].nunique() > 0) or
        ('PID_Active_simulated' in hist_data_df.columns and hist_data_df['PID_Active_simulated'].nunique() > 0)
    )
    num_fixed_plots = 2 # PV/SP, MV
    if plot_pid_enable_graph:
        num_fixed_plots = 3

    total_num_plots = num_fixed_plots + num_disturbance_plots

    if total_num_plots == 0 : # Rien à tracer
        logger.warning("Aucune donnée valide à tracer pour la comparaison PID.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    # Ajuster la hauteur en fonction du nombre de subplots
    fig, axs = plt.subplots(total_num_plots, 1, figsize=(18, 4 * total_num_plots), sharex=True)
    
    # S'assurer que axs est toujours une liste, même si total_num_plots = 1
    if total_num_plots == 1:
        axs = [axs]

    title_suffix = ""
    if config_data_period:
        start_time = config_data_period.get('start_time', '')
        end_time = config_data_period.get('end_time', '')
        if start_time and end_time:
            title_suffix = f" - {start_time} à {end_time}"
    
    fig.suptitle(f'{suptitle_prefix}{title_suffix}', fontsize=16)

    current_ax_idx = 0

    # Plot PV/SP
    if 'PV_real' in hist_data_df.columns:
        axs[current_ax_idx].plot(hist_data_df.index, hist_data_df['PV_real'], label='PV Réelle (Automate)', color='deepskyblue', linewidth=2)
    if 'SP_real' in hist_data_df.columns:
        axs[current_ax_idx].plot(hist_data_df.index, hist_data_df['SP_real'], label='SP Réelle (Automate)', color='limegreen', linestyle='--', linewidth=2)
    axs[current_ax_idx].set_ylabel('PV / SP')
    axs[current_ax_idx].legend()
    axs[current_ax_idx].grid(True, which='both', linestyle='--', linewidth=0.5)
    current_ax_idx += 1

    # Plot MV
    if 'MV_real' in hist_data_df.columns:
        axs[current_ax_idx].plot(hist_data_df.index, hist_data_df['MV_real'], label='MV Réelle (Automate)', color='black', linestyle='-', linewidth=2)
    if 'MV_simulated' in hist_data_df.columns:
        axs[current_ax_idx].plot(hist_data_df.index, hist_data_df['MV_simulated'], label='MV Simulée (Python)', color='red', linestyle='-.', linewidth=1.5, alpha=0.8)
    axs[current_ax_idx].set_ylabel('Sortie MV (%)')
    axs[current_ax_idx].legend()
    axs[current_ax_idx].grid(True, which='both', linestyle='--', linewidth=0.5)
    current_ax_idx += 1

    # Plot PID Enable Status (si nécessaire)
    if plot_pid_enable_graph:
        if 'PID_Enable_real' in hist_data_df.columns:
            axs[current_ax_idx].plot(hist_data_df.index, hist_data_df['PID_Enable_real'].astype(float), label='PID Automate Actif (Réel)', linestyle=':', drawstyle='steps-post', color='darkorange', linewidth=2)
        if 'PID_Active_simulated' in hist_data_df.columns:
            axs[current_ax_idx].plot(hist_data_df.index, hist_data_df['PID_Active_simulated'].astype(float), label='PID Simulé Actif', linestyle='-.', drawstyle='steps-post', color='purple', linewidth=1.5, alpha=0.8)
        axs[current_ax_idx].set_ylabel('État Activation PID')
        axs[current_ax_idx].set_yticks([0, 1])
        axs[current_ax_idx].set_yticklabels(['Inactif', 'Actif'])
        axs[current_ax_idx].legend()
        axs[current_ax_idx].grid(True, which='both', linestyle='--', linewidth=0.5)
        current_ax_idx += 1

    # Plot des perturbations
    disturbance_colors = ['saddlebrown', 'teal', 'indigo', 'olive', 'maroon'] # Ajoutez plus de couleurs si besoin
    for i, dist_col in enumerate(valid_disturbance_cols_to_plot):
        if current_ax_idx < total_num_plots: # Vérification de sécurité
            axs[current_ax_idx].plot(hist_data_df.index, hist_data_df[dist_col], label=dist_col.replace('_real',''), color=disturbance_colors[i % len(disturbance_colors)], linewidth=1.5)
            axs[current_ax_idx].set_ylabel(f'{dist_col.replace("_real","")} Value')
            axs[current_ax_idx].legend()
            axs[current_ax_idx].grid(True, which='both', linestyle='--', linewidth=0.5)
            current_ax_idx += 1
        else:
            logger.warning(f"Pas assez d'axes pour tracer la perturbation {dist_col}. Max axes: {total_num_plots}, current_idx: {current_ax_idx}")


    # Label X sur le dernier plot utilisé
    if current_ax_idx > 0:
        axs[current_ax_idx -1].set_xlabel('Temps')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if plot_save_path_str:
        try:
            path_obj = Path(plot_save_path_str).resolve()
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path_obj)
            logger.info(f"Graphique de comparaison PID sauvegardé: '{path_obj}'.")
        except Exception as e:
            logger.error(f"Erreur sauvegarde graphique comparaison PID : {e}")
    
    try:
        plt.show()
    except Exception as e:
        logger.warning(f"Impossible d'afficher graphique comparaison PID (plt.show()): {e}.")