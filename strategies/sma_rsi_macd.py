# strategies/sma_rsi_macd.py

import pandas as pd
import pandas_ta as ta
from core.logger import logger

class StrategySmaRsiMacd:
    def __init__(self):
        self.sma_fast = 50
        self.sma_slow = 200
        self.rsi_length = 14
        # On rend plus flexibles les seuils pour plus de signaux
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

    def _prepare_dataframe(self, klines):
        df = pd.DataFrame(klines, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df[["time", "open", "high", "low", "close", "volume"]]

    def _add_indicators(self, df):
        df["SMA_Fast"] = ta.sma(df["close"], length=self.sma_fast)
        df["SMA_Slow"] = ta.sma(df["close"], length=self.sma_slow)
        df["RSI"] = ta.rsi(df["close"], length=self.rsi_length)
        macd = ta.macd(df["close"], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        df = pd.concat([df, macd], axis=1)
        return df

    def generate_signal(self, klines):
        df = self._prepare_dataframe(klines)
        df = self._add_indicators(df)

        latest = df.iloc[-1]
        current_price = latest["close"]

        # Vérification des indicateurs valides
        if pd.isna(latest["SMA_Fast"]) or pd.isna(latest["SMA_Slow"]) or pd.isna(latest["RSI"]) or pd.isna(latest["MACD_12_26_9"]):
            return {
                "action": "HOLD",
                "price": current_price,
                "reason": "Indicateurs non calculés (pas assez de données)"
            }

        # Conditions ACHAT
        if (
            latest["SMA_Fast"] > latest["SMA_Slow"] and
            latest["RSI"] < 60 and
            latest["MACD_12_26_9"] > latest["MACDs_12_26_9"]
        ):
            return {
                "action": "BUY",
                "price": current_price,
                "reason": "SMA haussière, RSI < 60, MACD positif"
            }

        # Conditions VENTE
        elif (
            latest["SMA_Fast"] < latest["SMA_Slow"] and
            latest["RSI"] > 40 and
            latest["MACD_12_26_9"] < latest["MACDs_12_26_9"]
        ):
            return {
                "action": "SELL",
                "price": current_price,
                "reason": "SMA baissière, RSI > 40, MACD négatif"
            }

        # Sinon HOLD
        return {
            "action": "HOLD",
            "price": current_price,
            "reason": "Pas de signal clair"
        }
