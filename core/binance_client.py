# core/binance_client.py
from binance.client import Client
from tenacity import retry, stop_after_attempt, wait_fixed
from config import Config
import logging

logger = logging.getLogger("TradingBot")

class BinanceClient:
    def __init__(self):
        Config.check_api_keys()

        # Initialisation du client Binance
        self.client = Client(
            api_key=Config.BINANCE_API_KEY,
            api_secret=Config.BINANCE_API_SECRET,
            testnet=Config.BINANCE_TESTNET
        )

        mode = "TESTNET" if Config.BINANCE_TESTNET else "RÉEL"
        logger.info(f"Connexion Binance établie en mode {mode}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_klines(self, symbol=None, interval="1m", limit=100):
        """Récupère les bougies (OHLCV) avec retry automatique"""
        symbol = symbol or Config.TRADING_PAIR
        logger.info(f"Récupération des données de {symbol} en {interval}")
        return self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

    def get_account_balance(self, asset="USDT"):
        """Récupère le solde disponible en USDT"""
        balances = self.client.get_account()['balances']
        for balance in balances:
            if balance['asset'] == asset:
                return float(balance['free'])
        return 0.0

    def place_order(self, symbol, side, quantity):
        """Passe un ordre au marché"""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity
            )
            logger.info(f"Ordre {side} exécuté : {order}")
            return order
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'ordre : {e}")
            return None
