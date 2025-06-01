# src/db_utils.py
import sqlalchemy
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def get_db_engine(config_db):
    # ... (votre code pour get_db_engine)
    # Assurez-vous que les messages de log utilisent `logger.info`, `logger.error`, etc.
    db_host = config_db.get('db_host')
    db_name = config_db.get('db_name')
    odbc_driver = config_db.get('odbc_driver', 'ODBC Driver 17 for SQL Server')

    conn_str_params = {
        "drivername": "mssql+pyodbc",
        "host": db_host,
        "database": db_name,
        "query": {}
    }
    conn_str_params["query"]["driver"] = odbc_driver

    db_user = config_db.get('db_user', None)
    db_password = config_db.get('db_password', None)
    trusted_connection_config = config_db.getboolean('trusted_connection', False)

    if db_user and db_password:
        conn_str_params["username"] = db_user
        conn_str_params["password"] = db_password
        logger.info(f"Connexion à SQL Server '{db_host}/{db_name}' avec l'utilisateur '{db_user}'.")
        if trusted_connection_config:
            logger.warning("Authentification par utilisateur/mot de passe fournie. L'option 'trusted_connection=yes' sera ignorée.")
    elif trusted_connection_config:
        logger.info(f"Connexion à SQL Server '{db_host}/{db_name}' avec Windows Authentication (Trusted Connection).")
        conn_str_params["query"]["Trusted_Connection"] = "yes"
    else:
        logger.error("Informations d'authentification non configurées.")
        raise ValueError("Authentification pour la base de données non configurée.")

    try:
        connection_url_obj = sqlalchemy.URL.create(**conn_str_params)
        engine = sqlalchemy.create_engine(connection_url_obj)
        with engine.connect() as connection:
            logger.info("Connexion à la base de données réussie et testée.")
        return engine
    except Exception as e:
        logger.error(f"Erreur lors de la création du moteur de base de données ou de la connexion: {e}")
        if 'connection_url_obj' in locals():
             logger.error(f"Chaîne de connexion SQLAlchemy tentée (sans mot de passe): {connection_url_obj.render_as_string(hide_password=True)}")
        else:
            logger.error(f"Les paramètres de connexion étaient: {conn_str_params}")
        raise
    # ...
    return engine

def build_sql_query(config_tags, config_period, table_name, required_tags_aliases=None):
    """
    Construit la requête SQL dynamiquement.
    required_tags_aliases: liste optionnelle d'alias qui doivent être présents.
    """
    logger.info("Construction de la requête SQL...")
    
    tags_to_fetch = {} # Dictionnaire alias -> tag_name_sql
    
    # Tags principaux, on peut les standardiser ou les rendre configurables via alias
    # Pour cet exemple, on suppose que les alias PV, MV, SP sont attendus par le reste du code
    # et que les noms de tag réels sont sous pv_tag_name, etc.
    core_tag_mapping = { 
        "PV_real": config_tags.get('pv_tag_name'),
        "MV_real": config_tags.get('mv_tag_name'),
        "SP_real": config_tags.get('sp_tag_name'),
    }
    # Tags optionnels spécifiques au comparateur ou au builder
    optional_tag_mapping = {
        "Kp_real": config_tags.get('kp_hist_tag_name'),
        "Ti_real": config_tags.get('ti_hist_tag_name'),
        "Td_real": config_tags.get('td_hist_tag_name'),
        "PID_Enable_real": config_tags.get('pid_enable_tag_name') # Pour le comparateur
    }
    
    all_potential_tags = {**core_tag_mapping, **optional_tag_mapping}

    # Ajout des tags de perturbation pour le model builder
    i = 1
    while True:
        dist_tag_config_key = f'disturbance_tag_{i}'
        tag_name = config_tags.get(dist_tag_config_key)
        if tag_name:
            all_potential_tags[f'Dist{i}_real'] = tag_name # Utiliser un alias cohérent
            i += 1
        else:
            # Arrêter si la clé n'existe pas, ou si elle existe mais est vide
            if dist_tag_config_key not in config_tags or not config_tags.get(dist_tag_config_key):
                 break
            i+=1 # Permet des "trous" si un tag est vide mais d'autres suivent

    active_tags_for_select = {alias: tag for alias, tag in all_potential_tags.items() if tag}

    if required_tags_aliases:
        for req_alias in required_tags_aliases:
            if req_alias not in active_tags_for_select:
                # Tenter de trouver le tag_name correspondant à l'alias requis si l'alias n'est pas directement dans active_tags_for_select
                # Cela dépend de comment vous structurez vos alias dans le .ini
                # Pour simplifier ici, on s'attend à ce que l'alias soit la clé dans config_tags
                # ou que le tag_name soit sous pv_tag_name etc.
                key_for_req_tag = f"{req_alias.replace('_real', '').lower()}_tag_name" # ex: pv_tag_name
                if config_tags.get(key_for_req_tag):
                     active_tags_for_select[req_alias] = config_tags.get(key_for_req_tag)
                else:
                    logger.error(f"Tag requis avec alias '{req_alias}' (ou son tag_name correspondant) non trouvé ou non configuré dans [Tags].")
                    raise ValueError(f"Tag requis '{req_alias}' non configuré.")
    
    # Vérification minimale pour PV, MV, SP (si ce sont les alias standardisés)
    if not all(alias in active_tags_for_select for alias in ["PV_real", "MV_real", "SP_real"]):
         raise ValueError("pv_tag_name, mv_tag_name, et sp_tag_name (résultant en alias PV_real, MV_real, SP_real) sont requis dans [Tags]")


    tag_names_in_query_list = [f"'{tag}'" for tag in active_tags_for_select.values()]
    select_cases_list = [f"MAX(CASE WHEN TagName = '{tag_sql}' THEN Value END) AS {alias}"
                         for alias, tag_sql in active_tags_for_select.items()]
    
    select_clause_str = ",\n        ".join(select_cases_list)
    start_time_str = config_period.get('start_time')
    end_time_str = config_period.get('end_time')    
    resolution_int = config_period.getint('ww_resolution_ms')

    query_str = f"""
    SELECT DateTime, {select_clause_str}
    FROM {table_name}
    WHERE TagName IN ({', '.join(tag_names_in_query_list)})
      AND DateTime >= '{start_time_str}' AND DateTime <= '{end_time_str}'
      AND wwRetrievalMode = 'Cyclic' AND wwResolution = {resolution_int} AND wwVersion = 'Latest'
    GROUP BY DateTime
    ORDER BY DateTime ASC;
    """
    logger.info("Requête SQL construite:\n" + query_str)
    return query_str, list(active_tags_for_select.keys()) # Retourne les alias utilisés
    
def extract_data(engine, query):
    # ... (votre code pour extract_data)
    try:
        df = pd.read_sql_query(sqlalchemy.text(query), engine)
        logger.info(f"Données extraites avec succès. Forme: {df.shape}")
        if df.empty:
            logger.warning("Aucune donnée retournée par la requête SQL.")
        return df
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des données: {e}")
        raise
    # ...
    return df