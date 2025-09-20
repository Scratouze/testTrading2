# backtest/backtest_runner.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from core.binance_client import BinanceClient
from config import Config
from strategies.sma_rsi_macd import StrategySmaRsiMacd
from core.logger import logger


class BacktestRunner:
    def __init__(self, starting_balance=1000):
        self.binance = BinanceClient()
        self.strategy = StrategySmaRsiMacd()
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.trades = []

    def fetch_historical_data(self, limit=1000, interval="1h"):
        """
        Récupère les données historiques depuis Binance Testnet
        """
        logger.info(f"Téléchargement de {limit} bougies en {interval} pour {Config.TRADING_PAIR}")
        klines = self.binance.get_klines(symbol=Config.TRADING_PAIR, interval=interval, limit=limit)

        df = pd.DataFrame(klines, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        # Conversion des types
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        return df[["time", "open", "high", "low", "close", "volume"]]

    # backtest/backtest_runner.py

    def simulate_backtest(self, df):
        df = self.strategy._add_indicators(df)

        position = None
        entry_price = 0
        stop_loss_price = None
        take_profit_price = None
        balance_history = []
        time_history = []

        logger.info("Démarrage du backtest avec SL/TP...")

        for i in range(len(df)):
            current_price = df.iloc[i]["close"]
            current_high = df.iloc[i]["high"]
            current_low = df.iloc[i]["low"]
            current_time = df.iloc[i]["time"]

            partial_df = df.iloc[:i + 1]
            signal_data = self.strategy.generate_signal(partial_df)
            signal = signal_data["action"]

            # Skip si indicateurs pas prêts
            if signal_data["reason"].startswith("Indicateurs non calculés"):
                continue

            # Si pas de position ouverte et signal BUY
            if position is None and signal == "BUY":
                position = "LONG"
                entry_price = current_price
                # définir SL et TP à partir de l'entrée
                stop_loss_price = entry_price * (1 - Config.STOP_LOSS)
                take_profit_price = entry_price * (1 + Config.TAKE_PROFIT)
                logger.info(
                    f"[{current_time}] OUVERTURE LONG @ {entry_price:.2f}, SL=@{stop_loss_price:.2f}, TP=@{take_profit_price:.2f}")

            # Si position ouverte, vérifier SL/TP
            elif position == "LONG":
                # d’abord vérifier si TP atteint
                if current_high >= take_profit_price:
                    profit = take_profit_price - entry_price
                    self.current_balance += profit
                    self.trades.append(profit)
                    logger.info(
                        f"[{current_time}] TAKE PROFIT atteint @ {take_profit_price:.2f} | Profit: {profit:.2f}")
                    position = None
                    # reset SL/TP
                    stop_loss_price = None
                    take_profit_price = None

                # sinon vérifier SL
                elif current_low <= stop_loss_price:
                    profit = stop_loss_price - entry_price
                    self.current_balance += profit
                    self.trades.append(profit)
                    logger.info(f"[{current_time}] STOP LOSS atteint @ {stop_loss_price:.2f} | Perte: {profit:.2f}")
                    position = None
                    stop_loss_price = None
                    take_profit_price = None

                # sinon vérifier signal SELL pour sortir
                elif signal == "SELL":
                    profit = current_price - entry_price
                    self.current_balance += profit
                    self.trades.append(profit)
                    logger.info(f"[{current_time}] SIGNAL SELL @ {current_price:.2f} | Profit: {profit:.2f}")
                    position = None
                    stop_loss_price = None
                    take_profit_price = None

            # Toujours enregistrer le capital et le temps pour courbe
            balance_history.append(self.current_balance)
            time_history.append(current_time)

        # Si position encore ouverte à la fin, la clore
        if position == "LONG" and entry_price is not None:
            profit = df.iloc[-1]["close"] - entry_price
            self.current_balance += profit
            self.trades.append(profit)
            logger.info(f"[FIN] Clôture finale LONG @ {df.iloc[-1]['close']:.2f} | Profit: {profit:.2f}")

        return time_history, balance_history

    def run(self, limit=1000, interval="1h"):
        """
        Lance un backtest complet
        """
        df = self.fetch_historical_data(limit=limit, interval=interval)
        time_history, balance_history = self.simulate_backtest(df)

        total_profit = self.current_balance - self.starting_balance
        win_trades = len([t for t in self.trades if t > 0])
        loss_trades = len([t for t in self.trades if t <= 0])
        total_trades = len(self.trades)
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0

        logger.info("===== Résultats du Backtest =====")
        logger.info(f"Capital initial : {self.starting_balance:.2f} USDT")
        logger.info(f"Capital final   : {self.current_balance:.2f} USDT")
        logger.info(f"Profit net      : {total_profit:.2f} USDT")
        logger.info(f"Trades gagnants : {win_trades}")
        logger.info(f"Trades perdants : {loss_trades}")
        logger.info(f"Taux de réussite : {win_rate:.2f}%")

        # Affichage graphique
        if len(time_history) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(time_history, balance_history, label="Évolution du capital", color="blue")
            plt.xlabel("Temps")
            plt.ylabel("Capital (USDT)")
            plt.title("Courbe du capital - Backtest")
            plt.legend()
            plt.grid()
            plt.show()
        else:
            logger.warning("Aucune donnée à afficher pour le graphique.")


if __name__ == "__main__":
    backtest = BacktestRunner(starting_balance=1000)
    backtest.run(limit=1500, interval="1h")
