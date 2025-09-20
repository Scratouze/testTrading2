# core/trade_executor.py

from config import Config
from core.logger import logger
from core.binance_client import BinanceClient
from risk.risk_manager import RiskManager


class TradeExecutor:
    def __init__(self):
        self.binance = BinanceClient()

    def _format_quantity(self, quantity, symbol="BTCUSDT"):
        """
        Binance exige une quantité avec le bon nombre de décimales.
        Exemple :
          - BTC → 6 décimales
          - USDT → 2 décimales
        """
        if "BTC" in symbol:
            return round(quantity, 6)
        return round(quantity, 2)

    def execute_trade(self, action, price, balance):
        """
        Passe un ordre Market + OCO (Stop Loss & Take Profit)
        """
        logger.info(f"--- Exécution d'un trade {action} ---")

        # Initialiser le gestionnaire de risque
        risk = RiskManager(initial_balance=balance)

        # Taille de position
        position_size = risk.calculate_position_size()
        quantity = self._format_quantity(position_size / price, Config.TRADING_PAIR)

        logger.info(f"Quantité finale arrondie : {quantity} {Config.TRADING_PAIR.split('USDT')[0]}")

        # Déterminer SL et TP
        stop_loss, take_profit = risk.stop_loss_take_profit(price)

        try:
            # --- 1) Ordre au marché ---
            order = self.binance.client.create_order(
                symbol=Config.TRADING_PAIR,
                side=action,
                type="MARKET",
                quantity=quantity
            )
            logger.info(f"Ordre principal exécuté : {order}")

            # --- 2) OCO SL/TP ---
            opposite_side = "SELL" if action == "BUY" else "BUY"

            oco_order = self.binance.client.create_oco_order(
                symbol=Config.TRADING_PAIR,
                side=opposite_side,
                quantity=quantity,
                stopPrice=round(stop_loss, 2),
                stopLimitPrice=round(stop_loss * 0.999, 2),
                stopLimitTimeInForce="GTC",
                price=round(take_profit, 2)
            )
            logger.info(f"Ordre OCO SL/TP placé : {oco_order}")

            return {
                "status": "success",
                "market_order": order,
                "oco_order": oco_order
            }

        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du trade : {e}")
            return {"status": "error", "message": str(e)}
