# config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Charger automatiquement le fichier .env
load_dotenv()

class Config:
    """Gestion centralisée de la configuration du bot"""

    # --- Connexion Binance ---
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

    # Testnet activé par défaut pour éviter les pertes au début
    BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "True").lower() == "true"

    # --- Trading ---
    TRADING_PAIR = os.getenv("TRADING_PAIR", "BTCUSDT")
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))  # 2% du capital
    STOP_LOSS = float(os.getenv("STOP_LOSS", 0.02))            # 2% de perte max
    TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", 0.04))        # 4% de gain
    MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", 0.10))      # -10% capital

    # --- Divers ---
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)

    @classmethod
    def check_api_keys(cls):
        """Vérifie que les clés API sont bien définies"""
        if not cls.BINANCE_API_KEY or not cls.BINANCE_API_SECRET:
            raise ValueError("❌ Les clés API Binance ne sont pas configurées. "
                             "Ajoute-les dans le fichier .env !")

    @classmethod
    def summary(cls):
        """Affiche un résumé de la configuration"""
        return {
            "TRADING_PAIR": cls.TRADING_PAIR,
            "RISK_PER_TRADE": cls.RISK_PER_TRADE,
            "STOP_LOSS": cls.STOP_LOSS,
            "TAKE_PROFIT": cls.TAKE_PROFIT,
            "MAX_DRAWDOWN": cls.MAX_DRAWDOWN,
            "BINANCE_TESTNET": cls.BINANCE_TESTNET
        }
