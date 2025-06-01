# src/config_loader.py
import configparser
import logging
import sys
from pathlib import Path

# Le logger principal pour l'application
logger = logging.getLogger() # Ou un nom de logger spécifique

# Définition du formateur conditionnel (copiez la classe ici ou importez-la)
class ConditionalTimestampFormatter(logging.Formatter):
    def __init__(self, fmt_default_no_time, fmt_important_with_time, datefmt=None):
        super().__init__(fmt_default_no_time, datefmt)
        self.formatter_default_no_time = logging.Formatter(fmt_default_no_time, datefmt)
        self.formatter_important_with_time = logging.Formatter(fmt_important_with_time, datefmt)

    def format(self, record):
        if hasattr(record, 'important_phase') and record.important_phase:
            return self.formatter_important_with_time.format(record)
        else:
            return self.formatter_default_no_time.format(record)

def load_config_and_setup_logging(script_stem_for_config_file: str, config_file_name: str = None):
    """
    Charge la configuration depuis <config_file_name> (ou <script_stem_for_config_file>.ini)
    et configure le logging.
    """
    if config_file_name:
        config_file_path = Path(config_file_name).resolve()
    else:
        config_file_path = Path(f"{script_stem_for_config_file}.ini").resolve()

    config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    config.optionxform = str # Conserve la casse des clés
    if not config_file_path.is_file():
        print(f"ERREUR: Fichier de configuration '{config_file_path}' non trouvé. "
              f"Vérifiez que le fichier existe à cet emplacement.")
        raise FileNotFoundError(f"Fichier de configuration '{config_file_path.name}' non trouvé.")
    try:
        config.read(config_file_path, encoding='utf-8')
    except configparser.Error as e:
        print(f"ERREUR: Erreur lors de la lecture du fichier de configuration '{config_file_path.name}': {e}")
        raise

    app_logger = logging.getLogger()

    if app_logger.hasHandlers():
        for handler in app_logger.handlers[:]:
            try:
                handler.close()
            except Exception as e_close:
                print(f"Note: Erreur lors de la fermeture d'un handler existant: {e_close}", file=sys.stderr)
            app_logger.removeHandler(handler)

    app_logger.setLevel(logging.INFO)

    # Définition des formats de log
    # Format AVEC timestamp pour les phases importantes
    console_fmt_with_time = '%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'
    file_fmt_with_time = '%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(pathname)s - %(message)s'
    
    # Format SANS timestamp pour les logs par défaut (pour réduire la "pollution")
    console_fmt_no_time = '%(levelname)s - [%(module)s] - %(message)s'
    file_fmt_no_time = '%(levelname)s - [%(module)s:%(lineno)d] - %(pathname)s - %(message)s'

    # Handler pour la console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ConditionalTimestampFormatter(
        fmt_default_no_time=console_fmt_no_time,
        fmt_important_with_time=console_fmt_with_time
    )
    console_handler.setFormatter(console_formatter)
    app_logger.addHandler(console_handler)

    log_file_base_stem_from_config = None
    if config.has_section('Output') and config.has_option('Output', 'log_file_base_name'):
        custom_log_stem = config.get('Output', 'log_file_base_name').strip()
        if custom_log_stem:
            log_file_base_stem_from_config = custom_log_stem
            # Ce message peut être considéré comme important
            app_logger.info(
                f"Utilisation du nom de base personnalisé '{log_file_base_stem_from_config}' pour le fichier de log (depuis la configuration).",
                extra={'important_phase': True} 
            )
        else:
            app_logger.warning("La clé 'log_file_base_name' dans [Output] est vide. Utilisation du nom de base par défaut pour le log.")
    
    final_log_file_base_stem = log_file_base_stem_from_config if log_file_base_stem_from_config else script_stem_for_config_file
    if not log_file_base_stem_from_config:
        # Ce message peut aussi être important
        app_logger.info(
            f"Aucun 'log_file_base_name' valide spécifié dans [Output] du fichier .ini. "
            f"Utilisation du nom de base par défaut '{final_log_file_base_stem}' pour le log.",
            extra={'important_phase': True}
        )

    log_file_name_with_ext = f"{final_log_file_base_stem}.txt"
    
    try:
        log_file_path_obj = Path(log_file_name_with_ext).resolve()
        log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path_obj, mode='w', encoding='utf-8')
        file_formatter = ConditionalTimestampFormatter(
            fmt_default_no_time=file_fmt_no_time,
            fmt_important_with_time=file_fmt_with_time
        )
        file_handler.setFormatter(file_formatter)
        app_logger.addHandler(file_handler)
        
        app_logger.info(
            f"Journalisation configurée. Logs également sauvegardés dans : {log_file_path_obj}",
            extra={'important_phase': True} # Marquer ce log comme important pour inclure le timestamp
        )
    except Exception as e:
        print(f"AVERTISSEMENT CRITIQUE: Échec de la configuration de la journalisation vers le fichier '{log_file_name_with_ext}': {e}", file=sys.stderr)
        if app_logger.hasHandlers():
            app_logger.critical(f"Échec de la configuration de la journalisation vers le fichier '{log_file_name_with_ext}': {e}", exc_info=True, extra={'important_phase': True})

    app_logger.info(
        f"Configuration chargée depuis '{config_file_path}'.",
        extra={'important_phase': True} # Marquer ce log comme important
    )
    return config