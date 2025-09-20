# risk/risk_manager.py

from config import Config
from core.logger import logger

class RiskManager:
    def __init__(self, initial_balance):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.loss_streak = 0  # Compte les pertes consécutives

    def calculate_position_size(self):
        """
        Calcule la taille de position en fonction du capital disponible.
        Exemple : 2% du capital par trade
        """
        position_size = self.current_balance * Config.RISK_PER_TRADE
        logger.info(f"Taille de position calculée : {position_size:.2f} USDT")
        return position_size

    def stop_loss_take_profit(self, entry_price):
        """
        Détermine les niveaux de stop loss et take profit.
        """
        stop_loss = entry_price * (1 - Config.STOP_LOSS)
        take_profit = entry_price * (1 + Config.TAKE_PROFIT)

        logger.info(f"Stop Loss : {stop_loss:.2f} | Take Profit : {take_profit:.2f}")
        return stop_loss, take_profit

    def update_balance(self, profit_loss):
        """
        Met à jour le capital après un trade.
        """
        self.current_balance += profit_loss

        # Suivi des pertes consécutives
        if profit_loss < 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0

        logger.info(f"Nouveau solde : {self.current_balance:.2f} USDT")
        return self.current_balance

    def should_stop_bot(self):
        """
        Vérifie si le bot doit s'arrêter :
        - si drawdown trop important
        - ou série de pertes trop longue
        """
        # 1) Drawdown max
        max_loss_allowed = self.initial_balance * (1 - Config.MAX_DRAWDOWN)
        if self.current_balance < max_loss_allowed:
            logger.warning("⚠️ Drawdown maximum atteint, arrêt du bot !")
            return True

        # 2) Pertes consécutives
        if self.loss_streak >= 3:
            logger.warning("⚠️ 3 pertes consécutives, mise en pause du bot.")
            return True

        return False
