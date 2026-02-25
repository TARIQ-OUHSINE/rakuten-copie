#Import fichier de configuration du logging
from src.conf.logging_config import setup_logging

# Import packages
import logging
from datetime import date,datetime
import pandas as pd
import yaml

# Import fonctions 
import src.visualization.visualize as viz

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Erreur : Le fichier {config_path} est introuvable.")
    except yaml.YAMLError as e:
        print(f"Erreur de lecture du fichier YAML : {e}")


def main() -> None:

    setup_logging(log_level=logging.DEBUG, log_file="logs/app.log")
    logger = logging.getLogger(__name__)
    config = load_config()

    logger.info(f"Démarrage de l'application {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    # Chargement des données
    X_train = pd.read_csv(r"data\raw\X_train_update.csv",sep=",", encoding="utf-8", index_col=0)
    X_test = pd.read_csv(r"data\raw\X_test_update.csv",sep=",", encoding="utf-8", index_col=0)
    y_train = pd.read_csv(r"data\raw\Y_train.csv",sep=",", encoding="utf-8", index_col=0)

    data_train = pd.concat([X_train, y_train], axis=1)

    print(config)
    if config.get("VIZ", True):
        
        print("5 premièeres lignes de X_train :")
        print(X_train.head())

        print("Description de y_train :")
        print(y_train.describe())
        
        viz.histogram(data_train, "prdtypecode", save_path="reports/figures/histogram_product_type.png")
        


if __name__ == "__main__":
    main()
