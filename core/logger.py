# core/logger.py
import logging
from config import Config

def setup_logger():
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.INFO)

    # Log dans un fichier
    file_handler = logging.FileHandler(Config.LOG_DIR / "trading.log")
    file_handler.setLevel(logging.INFO)

    # Log dans la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Initialisation globale
logger = setup_logger()
